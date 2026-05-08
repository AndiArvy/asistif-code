import asyncio
import json
import os
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import base64
import io
from gtts import gTTS
from av.audio.resampler import AudioResampler
from pathlib import Path
from tts_cache import get_tts_cache

# Menyimpan koneksi antarmuka web HP
frontend_clients = set()

# --- VARIABEL GLOBAL ---
pcs = set()  # Menyimpan koneksi WebRTC aktif
latest_jpeg = None  # Menyimpan frame video terakhir
audio_clients = set()  # Menyimpan client Python yang mendengarkan audio
tts_cache = get_tts_cache()  # Initialize TTS cache


# --- 1. RUTE HALAMAN UTAMA (UI HP) ---
async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)


# --- 2. JEMBATAN VIDEO (IP WEBCAM) ---
async def video_feed(request):
    my_boundary = "frame-boundary"
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={"Content-Type": f"multipart/x-mixed-replace;boundary={my_boundary}"},
    )
    await response.prepare(request)

    while True:
        if latest_jpeg is not None:
            part = (
                (
                    f"--{my_boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(latest_jpeg)}\r\n\r\n"
                ).encode("utf-8")
                + latest_jpeg
                + b"\r\n"
            )
            try:
                await response.write(part)
            except Exception:
                break
        await asyncio.sleep(0.03)
    return response


# --- 3. JEMBATAN AUDIO (WEBSOCKET) ---
async def audio_feed(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    audio_clients.add(ws)
    print("-> Ada script Python yang terhubung ke aliran Audio!")

    try:
        async for msg in ws:
            pass  # Biarkan terbuka untuk mengirim data searah
    finally:
        audio_clients.remove(ws)
        print("-> Script Python terputus dari aliran Audio.")
    return ws


# --- 4. PROSES WEBRTC (SIGNALING & TERIMA DATA) ---
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Status WebRTC:", pc.connectionState)
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        # TANGKAP VIDEO
        if track.kind == "video":
            print("Menerima Video Track dari HP!")

            async def consume_video():
                global latest_jpeg
                while True:
                    try:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        ret, buffer = cv2.imencode(
                            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 99]
                        )
                        if ret:
                            latest_jpeg = buffer.tobytes()
                    except Exception:
                        break

            asyncio.ensure_future(consume_video())

        # TANGKAP AUDIO
        elif track.kind == "audio":
            print("Menerima Audio Track dari mic HP!")

            # Buat alat untuk mengubah format audio menjadi standar Whisper
            # s16 = 16-bit integer, mono = 1 channel, 16000 = 16kHz
            resampler = AudioResampler(format="s16", layout="mono", rate=16000)

            async def consume_audio():
                while True:
                    try:
                        frame = await track.recv()

                        # Ubah frame audio bawaan HP ke format Whisper
                        resampled_frames = resampler.resample(frame)

                        for resampled_frame in resampled_frames:
                            # Ambil byte yang sudah rapi
                            raw_audio = resampled_frame.to_ndarray().tobytes()

                            # Kirim ke websocket STT
                            for ws in list(audio_clients):
                                await ws.send_bytes(raw_audio)
                    except Exception as e:
                        print("Stream audio berhenti:", e)
                        break

            asyncio.ensure_future(consume_audio())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# --- 5. WEBSOCKET UNTUK MENGIRIM PERINTAH KE HP ---
async def frontend_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    frontend_clients.add(ws)
    try:
        async for msg in ws:
            pass
    finally:
        frontend_clients.remove(ws)
    return ws


# --- 6. API UNTUK MENERIMA TEKS DARI FILE LLM ---
async def trigger_tts(request):
    params = await request.json()
    teks = params.get("text", "")
    use_cache = params.get("use_cache", True)
    cache_file = params.get("cache_file", None)

    if teks:
        audio_b64 = None
        
        # Check if cached file is provided
        if use_cache and cache_file and os.path.exists(cache_file):
            print(f"[TTS] Loading cached audio from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"[TTS] Error loading cache: {e}, will regenerate...")
                audio_b64 = None
        
        # If not cached or cache failed, generate new TTS
        if not audio_b64:
            print(f"[TTS] Generating new TTS for: {teks}")
            try:
                tts = gTTS(text=teks, lang="id")
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                
                audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
                
                # Cache the generated audio
                try:
                    cache_filename = tts_cache._text_to_hash(teks) + ".mp3"
                    cache_filepath = os.path.join(tts_cache.cache_dir, cache_filename)
                    
                    with open(cache_filepath, 'wb') as f:
                        f.write(fp.getvalue())
                    
                    tts_cache.add_to_cache(teks, cache_filepath)
                    print(f"[TTS Cache] Cached audio: {cache_filename}")
                except Exception as e:
                    print(f"[TTS Cache] Error caching: {e}")
                    
            except Exception as e:
                print(f"[TTS] Error generating TTS: {e}")
                return web.Response(text="TTS generation failed", status=500)
        
        # Send audio to frontend clients
        if audio_b64:
            for ws in list(frontend_clients):
                try:
                    await ws.send_json({"type": "audio", "audio": audio_b64, "text": teks})
                except Exception as e:
                    print(f"[TTS] Error sending to client: {e}")

    return web.Response(text="Suara berhasil dikirim ke HP")



# --- 7. API UNTUK MENERIMA LOG DARI SCRIPT PYTHON LAIN ---
async def send_log(request):
    params = await request.json()
    sender = params.get("sender", "System")
    text = params.get("text", "")

    # Teruskan log ini ke HP via WebSocket frontend
    for ws in list(frontend_clients):
        await ws.send_json({"type": "log", "sender": sender, "text": text})

    return web.Response(text="Log terkirim ke HP")


# --- 8. API & WEBSOCKET UNTUK PERINTAH MANUAL (TAP LAYAR) ---
command_clients = set()

async def command_feed(request):
    """Websocket untuk mendengarkan perintah dari server ke audio_inference.py"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    command_clients.add(ws)
    print("-> Sistem AI terhubung ke jalur Tap Layar.")
    try:
        async for msg in ws:
            pass
    finally:
        command_clients.remove(ws)
    return ws

async def manual_trigger(request):
    """Menerima sinyal Tap Layar dari HP dan meneruskannya ke AI"""
    # Teruskan sinyal ke audio_inference.py
    for ws in list(command_clients):
        await ws.send_json({"text": "ambil_layout"})
        
    # Tampilkan di Log Chat UI
    for ws in list(frontend_clients):
        await ws.send_json({"type": "log", "sender": "User", "text": "[Tap Layar] Mengambil frame layout..."})
        
    return web.Response(text="Sinyal tap diterima")

# --- 9. API UNTUK MENGAMBIL 1 FRAME SAJA (SNAPSHOT) ---
async def snapshot(request):
    """Mengirimkan 1 frame gambar JPEG terbaru ke AI tanpa membuka stream"""
    if latest_jpeg is not None:
        return web.Response(body=latest_jpeg, content_type="image/jpeg")
    return web.Response(status=404, text="Kamera belum siap")


# --- 8. PRE-GENERATE COMMON TTS PHRASES ---
async def preload_common_tts():
    """Pre-generate and cache common TTS phrases on startup"""
    common_phrases = [
        # Navigation & Guidance
        "Remote terlihat. Tahan posisi Anda sebentar...",
        "Posisinya pas. Silakan tekan tombolnya.",
        "Remote tidak terlihat di kamera. Arahkan kembali.",
        "Jempol Anda belum terlihat di atas remote. Silakan raba perlahan.",
        "Layout direset. Letakkan remote di depan kamera.",
        "Pemetaan remote berhasil. Mau saya bantu apa?",
        
        # Directional guidance
        "Geser jempol Anda ke atas.",
        "Geser jempol Anda ke bawah.",
        "Geser jempol Anda ke kiri.",
        "Geser jempol Anda ke kanan.",
        "Geser jempol Anda ke atas lalu ke kanan.",
        "Geser jempol Anda ke bawah lalu ke kiri.",
        
        # Task confirmations
        "Memulai pemetaan remote.",
        "Mengambil gambar jernih...",
        "Memetakan fungsi tombol AC...",
        "Menganalisis posisi dan perintah...",
    ]
    
    print("[TTS Preload] Starting pre-generation of common phrases...")
    for phrase in common_phrases:
        try:
            # Check if already cached
            cached = tts_cache.get_cached_file(phrase)
            if cached:
                print(f"[TTS Preload] Already cached: {phrase[:40]}...")
            else:
                print(f"[TTS Preload] Generating: {phrase[:40]}...")
                tts = gTTS(text=phrase, lang="id")
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                
                # Save to cache
                cache_filename = tts_cache._text_to_hash(phrase) + ".mp3"
                cache_filepath = os.path.join(tts_cache.cache_dir, cache_filename)
                
                with open(cache_filepath, 'wb') as f:
                    f.write(fp.getvalue())
                
                tts_cache.add_to_cache(phrase, cache_filepath)
                print(f"[TTS Preload] Cached: {phrase[:40]}...")
        except Exception as e:
            print(f"[TTS Preload] Error with '{phrase}': {e}")
    
    cache_info = tts_cache.get_cache_info()
    print(f"[TTS Preload] Complete! Cache info: {cache_info}")


# --- JALANKAN SERVER ---
if __name__ == "__main__":
    app = web.Application()
    
    # Add startup callback to pre-generate TTS
    app.on_startup.append(lambda app: preload_common_tts())

    # Rute Inti WebRTC
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    # Rute AI Inference (Jembatan Data Mentah)
    app.router.add_get("/video_feed", video_feed)
    app.router.add_get("/audio_feed", audio_feed)
    
    
    # TAMBAHKAN RUTE SNAPSHOT INI:
    app.router.add_get("/snapshot", snapshot)

    # --- PASTIKAN 3 BARIS INI ADA UNTUK LOG & TTS ---
    app.router.add_get("/frontend_ws", frontend_ws)  # Pintasan WebSocket dari HP
    app.router.add_post("/send_log", send_log)  # Penerima teks dari Whisper
    app.router.add_post(
        "/trigger_tts", trigger_tts
    )  # Aktifkan jika fungsi ini sudah dibuat

    app.router.add_get("/command_feed", command_feed)
    # app.router.add_post("/manual_trigger", manual_trigger)
    
    # Add preload endpoint that can be triggered manually
    async def preload_tts_endpoint(request):
        await preload_common_tts()
        return web.Response(text="TTS preload completed")
    
    app.router.add_post("/preload_tts", preload_tts_endpoint)
    
    print("Server Utama WebRTC: http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
