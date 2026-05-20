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
    os.getenv("WHISPER_MODEL_SIZE", "deepdml/faster-whisper-large-v3-turbo-ct2"),
    device=os.getenv("WHISPER_DEVICE", "cuda"),
    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16"),
    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "8")),
)
print("Model siap!")
print("Memuat model visual...")
print("Sistem siap menerima perintah!")

# --- HOLD TO SPEAK STATE ---
hold_to_speak_active = False
hold_audio_buffer = bytearray()
draining = False

# --- FLAG UNTUK PAUSE AUDIO SAAT WHISPER INFERENCE ---
global whisper_is_inferencing
whisper_is_inferencing = False


async def wait_until_layout_ready():
    """Tahan listener mic sampai layout remote berhasil dipetakan."""
    if not vision_reasoning.is_layout_ready:
        print("Menunggu layout remote berhasil diambil...")

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
            min_silence_duration_ms=int(os.getenv("WHISPER_MIN_SILENCE_MS", "500"))
        ),
        initial_prompt = (
            "Ini tombol apa ya? Tolong nyalakan AC-nya. Suhu sekarang berapa derajat? "
            "Apakah posisi jempol saya sudah benar di tombol power? Udah bener belum di sini? "
            "Tolong turunkan suhu jadi 24 derajat atau naikkan sedikit. "
            "Matikan mode swing, ubah kipas ke fan, ganti mode cool atau dry. "
            "Cek status layar AC. Geser ke atas, bawah, kiri, kanan."
        ),
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
        best_of=int(os.getenv("WHISPER_BEST_OF", "1")),
        temperature=0.0,
    )
    return list(segments_generator)


async def run_vlm_task(text_result):
    global system_is_busy
    try:
        await asyncio.to_thread(process_vlm_reasoning, text_result)
    except Exception as e:
        print(f"Error saat memproses VLM: {e}")
    finally:
        print(">>> VLM Selesai. Siap menerima perintah baru. <<<")
        system_is_busy = False


async def process_buffered_audio():
    """Process the hold-to-speak audio buffer with Whisper"""
    global system_is_busy, hold_audio_buffer

    if len(hold_audio_buffer) < 2048:
        print("[Hold] Audio terlalu pendek, diabaikan.")
        hold_audio_buffer.clear()
        return

    system_is_busy = True
    is_valid_command = False

    audio_data = (
        np.frombuffer(hold_audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    )
    hold_audio_buffer.clear()

    try:
        segments = await asyncio.to_thread(run_whisper_transcription, audio_data)

        if len(segments) > 0:
            avg_no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)

            if avg_no_speech_prob > 0.5:
                print(f"(Mengabaikan noise. Probabilitas: {avg_no_speech_prob:.2f})")
            else:
                text_result = "".join([s.text for s in segments]).strip()
                clean_text = re.sub(r"[^\w\s]", "", text_result).strip().lower()

                halusinasi = [
                    "terima kasih", "terimakasih", "terima kasih banyak",
                    "terima kasih telah menonton", "terimakasih telah menonton", "kembali"
                ]

                if clean_text not in halusinasi and text_result:
                    is_valid_command = True
                    print(f"[Pengguna]: {text_result}")

                    asyncio.create_task(send_log_async(text_result))
                    asyncio.create_task(run_vlm_task(text_result))

    except Exception as e:
        print(f"Terjadi kesalahan di Whisper: {e}")

    finally:
        if not is_valid_command:
            system_is_busy = False


async def listen_to_mic():
    """Hold-to-speak: buffer audio only while user holds the button"""
    global hold_to_speak_active, hold_audio_buffer, draining
    uri = "ws://localhost:8080/audio_feed"

    while True:
        await wait_until_layout_ready()

        try:
            print("Mencoba terhubung ke mic HP via WebSocket...")
            async with websockets.connect(uri) as websocket:
                print("Berhasil terhubung! Tekan & tahan tombol untuk bicara...")

                while True:
                    if not vision_reasoning.is_layout_ready:
                        print("Layout belum siap/direset. Mic dinonaktifkan sementara.")
                        hold_audio_buffer.clear()
                        break

                    chunk = await websocket.recv()

                    if system_is_busy:
                        continue

                    if hold_to_speak_active or draining:
                        hold_audio_buffer.extend(chunk)

        except websockets.exceptions.ConnectionClosed:
            print("Koneksi terputus. Menunggu server...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(2)


async def listen_to_commands():
    """Mendengarkan sinyal hold-to-speak dari server.py via command_feed"""
    global hold_to_speak_active, hold_audio_buffer, draining
    uri = "ws://localhost:8080/command_feed"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)

                    msg_type = data.get("type", "")
                    action = data.get("action", "")

                    if msg_type == "hold_action" and action == "start_listening":
                        hold_audio_buffer.clear()
                        hold_to_speak_active = True
                        print("[Hold] MULAI mendengarkan...")

                    elif msg_type == "hold_action" and action == "stop_listening":
                        hold_to_speak_active = False
                        draining = True
                        print("[Hold] BERHENTI, menguras sisa audio...")
                        await asyncio.sleep(0.4)
                        draining = False
                        print("[Hold] Memproses audio...")
                        asyncio.create_task(process_buffered_audio())

                    elif data.get("text"):
                        # Legacy/alternative command format
                        teks_sinyal = data.get("text", "ambil ulang layout")
                        asyncio.create_task(
                            asyncio.to_thread(process_vlm_reasoning, teks_sinyal)
                        )

        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(2)


async def auto_scan_layout():
    """Looping background untuk mendeteksi remote secara otomatis saat baru mulai atau setelah di-reset"""
    print("Pemindai otomatis aktif. Menunggu remote masuk frame...")
    while True:
        if not vision_reasoning.is_layout_ready:
            sukses = await asyncio.to_thread(vision_reasoning.auto_setup_layout, silent=True)
            if sukses:
                print(">>> Layout remote otomatis tertangkap dan diproses! <<<")
        await asyncio.sleep(3)


async def ensure_tts_cache_preloaded():
    """Ensure TTS cache is available and trigger server preload"""
    try:
        cache = get_tts_cache()
        info = cache.get_cache_info()
        print(f"[TTS Cache] Status: {info['total_entries']} entries, {info['total_size_mb']} MB")

        asyncio.create_task(preload_tts_async())
    except Exception as e:
        print(f"[TTS Cache] Warning: {e}")


async def main():
    print("[System] Initializing TTS cache system...")
    await ensure_tts_cache_preloaded()
    vision_reasoning.start_background_task_monitor()

    print("[System] Starting main service loops...")
    await asyncio.gather(
        listen_to_mic(),
        listen_to_commands(),
        auto_scan_layout()
    )

if __name__ == "__main__":
    asyncio.run(main())
