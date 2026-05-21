from __future__ import annotations

import asyncio
import json
import os
import cv2
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription
import base64
import io
from gtts import gTTS
from av.audio.resampler import AudioResampler
from pathlib import Path
from typing import Any, Optional, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from tts_cache import TTSCache
from tts_cache import get_tts_cache

# Menyimpan koneksi antarmuka web HP
frontend_clients: set = set()

# --- VARIABEL GLOBAL ---
pcs: set = set()  # Menyimpan koneksi WebRTC aktif
latest_jpeg: bytes | None = None  # Menyimpan frame video terakhir
frame_event: asyncio.Event = asyncio.Event()  # Event-driven: sinyal frame baru
audio_clients: set = set()  # Menyimpan client Python yang mendengarkan audio
tts_cache: TTSCache = get_tts_cache()  # Initialize TTS cache
TTS_PLAYBACK_RATE: float = float(os.getenv("TTS_PLAYBACK_RATE", "1.2"))
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "99"))


# --- 1. RUTE HALAMAN UTAMA (UI HP) ---
async def index(request: web.Request) -> web.Response:
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)


# --- 2. JEMBATAN VIDEO (IP WEBCAM) ---
async def video_feed(request: web.Request) -> web.StreamResponse:
    my_boundary = "frame-boundary"
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={"Content-Type": f"multipart/x-mixed-replace;boundary={my_boundary}"},
    )
    await response.prepare(request)

    try:
        while True:
            await frame_event.wait()
            frame_event.clear()

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
                await response.write(part)
    except Exception:
        pass
    return response


# --- 3. JEMBATAN AUDIO (WEBSOCKET) ---
async def audio_feed(request: web.Request) -> web.WebSocketResponse:
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
async def offer(request: web.Request) -> web.Response:
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
    def on_track(track: Any) -> None:
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
                            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                        )
                        if ret:
                            latest_jpeg = buffer.tobytes()
                            frame_event.set()
                    except Exception:
                        break

            asyncio.create_task(consume_video())

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

            asyncio.create_task(consume_audio())

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
async def frontend_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    frontend_clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "hold_action":
                        for ws_cmd in list(command_clients):
                            await ws_cmd.send_json(data)
                except Exception as e:
                    print(f"[frontend_ws] Error: {e}")
    finally:
        frontend_clients.remove(ws)
    return ws


# --- 6. API UNTUK MENERIMA TEKS DARI FILE LLM ---
async def trigger_tts(request: web.Request) -> web.Response:
    params = await request.json()
    teks = params.get("text", "")
    use_cache = params.get("use_cache", True)
    cache_file = params.get("cache_file", None)

    if teks:
        audio_b64 = None
        
        # Check if cached file is provided
        if use_cache and cache_file:
            # --- VALIDASI PATH TRAVERSAL: pastikan cache_file di dalam cache_dir ---
            cache_file_resolved = Path(cache_file).resolve()
            cache_dir_resolved = Path(tts_cache.get_cache_dir()).resolve()
            try:
                cache_file_resolved.relative_to(cache_dir_resolved)
            except ValueError:
                print(f"[TTS] Security: rejected path traversal attempt: {cache_file}")
                cache_file = None

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
                    cache_filename = tts_cache.hash_text(teks) + ".mp3"
                    cache_filepath = os.path.join(tts_cache.get_cache_dir(), cache_filename)
                    
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
                    await ws.send_json(
                        {
                            "type": "audio",
                            "audio": audio_b64,
                            "text": teks,
                            "playback_rate": TTS_PLAYBACK_RATE,
                        }
                    )
                except Exception as e:
                    print(f"[TTS] Error sending to client: {e}")

    return web.Response(text="Suara berhasil dikirim ke HP")



# --- 7. API UNTUK MENERIMA LOG DARI SCRIPT PYTHON LAIN ---
async def send_log(request: web.Request) -> web.Response:
    params = await request.json()
    sender = params.get("sender", "System")
    text = params.get("text", "")

    # Teruskan log ini ke HP via WebSocket frontend
    for ws in list(frontend_clients):
        try:
            await ws.send_json({"type": "log", "sender": sender, "text": text})
        except Exception:
            pass

    return web.Response(text="Log terkirim ke HP")


# --- 8. API & WEBSOCKET UNTUK PERINTAH MANUAL (TAP LAYAR) ---
command_clients = set()

async def command_feed(request: web.Request) -> web.WebSocketResponse:
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

async def manual_trigger(request: web.Request) -> web.Response:
    """Menerima sinyal Tap Layar dari HP dan meneruskannya ke AI"""
    # Teruskan sinyal ke audio_inference.py
    for ws in list(command_clients):
        await ws.send_json({"text": "ambil_layout"})
        
    # Tampilkan di Log Chat UI
    for ws in list(frontend_clients):
        await ws.send_json({"type": "log", "sender": "User", "text": "[Tap Layar] Mengambil frame layout..."})
        
    return web.Response(text="Sinyal tap diterima")

# --- 9. API UNTUK MENGAMBIL 1 FRAME SAJA (SNAPSHOT) ---
async def snapshot(request: web.Request) -> web.Response:
    """Mengirimkan 1 frame gambar JPEG terbaru ke AI tanpa membuka stream"""
    if latest_jpeg is not None:
        return web.Response(body=latest_jpeg, content_type="image/jpeg")
    return web.Response(status=404, text="Kamera belum siap")


# --- 8. PRE-GENERATE COMMON TTS PHRASES ---
async def preload_common_tts() -> None:
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
                cache_filename = tts_cache.hash_text(phrase) + ".mp3"
                cache_filepath = os.path.join(tts_cache.get_cache_dir(), cache_filename)
                
                with open(cache_filepath, 'wb') as f:
                    f.write(fp.getvalue())
                
                tts_cache.add_to_cache(phrase, cache_filepath)
                print(f"[TTS Preload] Cached: {phrase[:40]}...")
        except Exception as e:
            print(f"[TTS Preload] Error with '{phrase}': {e}")
    
    cache_info = tts_cache.get_cache_info()
    print(f"[TTS Preload] Complete! Cache info: {cache_info}")


# --- SHUTDOWN HANDLER ---
async def on_shutdown(app: web.Application) -> None:
    """Bersihkan koneksi WebRTC saat server dimatikan."""
    print("\n[Mematikan server...]")
    for pc in list(pcs):
        await pc.close()
        pcs.discard(pc)
    for ws in list(frontend_clients):
        await ws.close()
    frontend_clients.clear()
    for ws in list(audio_clients):
        await ws.close()
    audio_clients.clear()
    for ws in list(command_clients):
        await ws.close()
    command_clients.clear()
    print("[Server dimatikan.]")


# --- JALANKAN SERVER ---
if __name__ == "__main__":
    app = web.Application()
    
    app.on_shutdown.append(on_shutdown)

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
    async def preload_tts_endpoint(request: web.Request) -> web.Response:
        await preload_common_tts()
        return web.Response(text="TTS preload completed")
    
    app.router.add_post("/preload_tts", preload_tts_endpoint)
    
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
    print(f"Server Utama WebRTC: http://{SERVER_HOST}:{SERVER_PORT}")
    if SERVER_HOST == "0.0.0.0":
        print("  Peringatan: server bind ke semua interface (0.0.0.0).")
        print("  Set env SERVER_HOST=127.0.0.1 untuk localhost saja.")
    web.run_app(app, host=SERVER_HOST, port=SERVER_PORT)
