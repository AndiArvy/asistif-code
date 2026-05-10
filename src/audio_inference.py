# audio_inference.py
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import websockets
import numpy as np
import requests
import re
from faster_whisper import WhisperModel
import json
from tts_cache import get_tts_cache
global system_is_busy
system_is_busy = False
http_session = requests.Session()

# --- IMPORT LOGIKA VLM DARI FILE TERPISAH ---
import vision_reasoning as vision_reasoning
from vision_reasoning import process_vlm_reasoning

print("Memuat model Whisper...")
model = WhisperModel(
    os.getenv("WHISPER_MODEL_SIZE", "deepdml/faster-whisper-large-v3-turbo-ct2"), #4s
    device=os.getenv("WHISPER_DEVICE", "cuda"),
    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16"),
    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "8")),
)
print("Model siap!")
# Panggil ini tepat setelah Anda berhasil me-load Whisper!
print("Memuat model visual...")
# vision_reasoning.init_vision_models()
print("Sistem siap menerima perintah!")

# --- PENGATURAN DETEKSI KALIMAT (VAD) ---
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "5000"))
# Nilai 50 sering memicu jeda panjang. Turunkan agar kalimat diproses lebih cepat.
SILENCE_CHUNKS_LIMIT = int(os.getenv("SILENCE_CHUNKS_LIMIT", "20"))

# --- FLAG UNTUK PAUSE AUDIO SAAT WHISPER INFERENCE ---
global whisper_is_inferencing
whisper_is_inferencing = False


async def wait_until_layout_ready():
    """Tahan listener mic sampai layout remote berhasil dipetakan."""
    if not vision_reasoning.is_layout_ready:
        print("Mic nonaktif. Menunggu layout remote berhasil diambil...")

    while not vision_reasoning.is_layout_ready:
        await asyncio.sleep(0.5)


# --- HELPER FUNCTIONS UNTUK BACKGROUND POST REQUESTS ---
async def send_log_async(text_result):
    """Mengirim log ke server secara background (non-blocking)"""
    try:
        await asyncio.to_thread(
            http_session.post,
            "http://localhost:8080/send_log",
            json={"sender": "User", "text": text_result},
            timeout=2,
        )
    except Exception as e:
        print("Gagal kirim log:", e)


async def preload_tts_async():
    """Preload TTS cache ke server secara background (non-blocking)"""
    try:
        await asyncio.to_thread(
            http_session.post,
            "http://localhost:8080/preload_tts",
            timeout=30
        )
    except Exception as e:
        print("Gagal preload TTS:", e)


# --- FUNGSI HELPER UNTUK WHISPER AGAR TIDAK MEMBLOKIR ASYNC ---
def run_whisper_transcription(audio_data):
    segments_generator, info = model.transcribe(
        audio_data,
        language="id",
        task="transcribe",
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=int(os.getenv("WHISPER_MIN_SILENCE_MS", "250"))
        ),
        initial_prompt=(
            "Percakapan asistif navigasi remote AC: 'Tolong nyalain AC-nya, saya kepanasan. "
            "Apakah benar tombol yang ini? Coba raba bagian tengah, lalu tekan tombol power. "
            "Suhunya turunkan ke 24 derajat atau naikkan sedikit. AC-nya masih menyala, "
            "pindah ke mode dingin, atur kipasnya. Sepertinya bukan tombol yang ini, "
            "geser ke ujung kanan bawah. Oke, matikan AC-nya sekarang.'"
        ),
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
        best_of=int(os.getenv("WHISPER_BEST_OF", "1")),
        temperature=0.0,
    )
    # Ubah generator ke list di dalam thread terpisah ini
    return list(segments_generator)

async def run_vlm_task(text_result):
    global system_is_busy
    try:
        # Jalankan VLM
        await asyncio.to_thread(process_vlm_reasoning, text_result)
    except Exception as e:
        print(f"Error saat memproses VLM: {e}")
    finally:
        # BUKA KUNCI MIC SETELAH VLM SELESAI
        print(">>> VLM Selesai. Mic siap menerima perintah baru. <<<")
        system_is_busy = False


async def listen_to_mic():
    global system_is_busy
    uri = "ws://localhost:8080/audio_feed"

    while True:
        await wait_until_layout_ready()

        try:
            print("Mencoba terhubung ke mic HP via WebSocket...")
            async with websockets.connect(uri) as websocket:
                print("Berhasil terhubung! Mulai bicara satu kalimat utuh...")

                audio_buffer = bytearray()
                silence_counter = 0
                is_speaking = False

                while True:
                    if not vision_reasoning.is_layout_ready:
                        print("Layout belum siap/direset. Mic dinonaktifkan sementara.")
                        audio_buffer.clear()
                        break

                    chunk = await websocket.recv()
                    
                    # --- [MODIFIKASI] BUANG AUDIO JIKA SISTEM (WHISPER ATAU VLM) SIBUK ---
                    # Ini mencegah WebSocket terputus, tapi memastikan suara baru tidak diproses
                    if system_is_busy:
                        continue

                    chunk_arr = np.frombuffer(chunk, dtype=np.int16)
                    volume = np.max(np.abs(chunk_arr))

                    if volume > SILENCE_THRESHOLD:
                        is_speaking = True
                        silence_counter = 0
                        audio_buffer.extend(chunk)
                    else:
                        if is_speaking:
                            silence_counter += 1
                            audio_buffer.extend(chunk)
                        else:
                            audio_buffer = bytearray(chunk)

                    if is_speaking and silence_counter > SILENCE_CHUNKS_LIMIT:
                        audio_data = (
                            np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        )

                        if len(audio_data) > (16000 * 0.5):
                            print("Memproses 1 kalimat utuh...")

                            # --- [MODIFIKASI] KUNCI MIC SEKARANG! ---
                            system_is_busy = True
                            is_valid_command = False # Penanda apakah ini perintah beneran atau noise

                            try:
                                # Jalankan Whisper
                                segments = await asyncio.to_thread(run_whisper_transcription, audio_data)
                                
                                if len(segments) > 0:
                                    avg_no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)

                                    if avg_no_speech_prob > 0.6:
                                        print(f"(Mengabaikan noise. Probabilitas: {avg_no_speech_prob:.2f})")
                                    else:
                                        text_result = "".join([s.text for s in segments]).strip()
                                        clean_text = re.sub(r"[^\w\s]", "", text_result).strip().lower()

                                        halusinasi = [
                                            "terima kasih", "terimakasih", "terima kasih banyak",
                                            "terima kasih telah menonton", "terimakasih telah menonton", ""
                                        ]

                                        if clean_text not in halusinasi and text_result:
                                            is_valid_command = True
                                            print(f"[Pengguna]: {text_result}")

                                            # --- KIRIM LOG SECARA BACKGROUND (NON-BLOCKING) ---
                                            asyncio.create_task(send_log_async(text_result))

                                            # --- [MODIFIKASI] JALANKAN VLM MENGGUNAKAN WRAPPER ---
                                            # Mic akan tetap terkunci sampai fungsi run_vlm_task ini selesai!
                                            asyncio.create_task(run_vlm_task(text_result))

                            except Exception as e:
                                print(f"Terjadi kesalahan di Whisper: {e}")
                                
                            finally:
                                # --- JIKA BUKAN PERINTAH VALID (CUMA NOISE), LANGSUNG BUKA KUNCI ---
                                # Tapi jika is_valid_command True, biarkan system_is_busy tetap True 
                                # agar VLM yang berjalan di background bisa menyelesaikannya.
                                if not is_valid_command:
                                    system_is_busy = False

                            # Reset state setelah pemrosesan awal (Whisper) selesai
                            is_speaking = False
                            silence_counter = 0
                            audio_buffer.clear()
                            await asyncio.sleep(0.5)

        except websockets.exceptions.ConnectionClosed:
            print("Koneksi terputus. Menunggu server...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(2)

async def listen_to_commands():
    """Mendengarkan sinyal Tap Layar dari server.py"""
    uri = "ws://localhost:8080/command_feed"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    # Beri perintah default reset layout jika ditekan
                    teks_sinyal = data.get("text", "ambil ulang layout")
                    asyncio.create_task(
                        asyncio.to_thread(process_vlm_reasoning, teks_sinyal)
                    )
        # ... catch block ...
        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(2)
            
async def auto_scan_layout():
    """Looping background untuk mendeteksi remote secara otomatis saat baru mulai atau setelah di-reset"""
    print("Pemindai otomatis aktif. Menunggu remote masuk frame...")
    while True:
        # Jika layout belum ada, coba tangkap frame secara diam-diam (silent=True)
        if not vision_reasoning.is_layout_ready:
            sukses = await asyncio.to_thread(vision_reasoning.auto_setup_layout, silent=True)
            if sukses:
                print(">>> Layout remote otomatis tertangkap dan diproses! <<<")
                
        # Polling setiap 3 detik (jangan terlalu cepat agar PC tidak berat)
        await asyncio.sleep(3)

async def ensure_tts_cache_preloaded():
    """Ensure TTS cache is available and trigger server preload"""
    try:
        cache = get_tts_cache()
        info = cache.get_cache_info()
        print(f"[TTS Cache] Status: {info['total_entries']} entries, {info['total_size_mb']} MB")
        
        # --- TRIGGER SERVER PRELOAD SECARA BACKGROUND (NON-BLOCKING) ---
        asyncio.create_task(preload_tts_async())
    except Exception as e:
        print(f"[TTS Cache] Warning: {e}")

async def main():
    # Ensure TTS cache is ready before starting main loops
    print("[System] Initializing TTS cache system...")
    await ensure_tts_cache_preloaded()
    vision_reasoning.start_background_task_monitor()
    
    print("[System] Starting main service loops...")
    await asyncio.gather(
        listen_to_mic(),
        listen_to_commands(),
        auto_scan_layout()  # Tambahkan scanner ini di sini
    )

if __name__ == "__main__":
    asyncio.run(main())
