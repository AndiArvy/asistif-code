import asyncio
import websockets
import requests
import json
import re

import vision_reasoning as vision_reasoning
from vision_reasoning import process_vlm_reasoning

# --- KONFIGURASI ---
COMMAND_URI = "ws://localhost:8080/command_feed"
LOG_URI = "http://localhost:8080/send_log"

def clean_command(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()

async def run_vlm_logic(text, source="Manual"):
    """Algoritma utama pemrosesan teks ke VLM"""
    if not text.strip(): return
    
    print(f"\n[{source}]: {text}")
    
    # Kirim log ke dashboard
    try:
        await asyncio.to_thread(
            lambda: requests.post(LOG_URI, json={"sender": source, "text": text}, timeout=1)
        )
    except: pass

    # Jalankan VLM Reasoning
    await asyncio.to_thread(process_vlm_reasoning, text)
    print(f"\n{'-'*20}\n>> ", end="")

async def listen_to_tap_commands():
    """Mendengarkan sinyal Tap Layar dari server via WebSocket"""
    while True:
        try:
            async with websockets.connect(COMMAND_URI) as websocket:
                print("\n[Sistem] Terhubung ke sinyal Tap Layar.")
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    # Jika ada sinyal tap, jalankan logika VLM
                    asyncio.create_task(run_vlm_logic(data["text"], source="Tap Layar"))
        except:
            await asyncio.sleep(2) # Reconnect jika gagal

async def manual_input_loop():
    """Menerima input ketik dari terminal"""
    loop = asyncio.get_event_loop()
    print(">> ", end="", flush=True)
    while True:
        user_input = await loop.run_in_executor(None, input, "")
        if user_input.lower() in ['exit', 'quit']: break
        asyncio.create_task(run_vlm_logic(user_input, source="Manual"))
        
async def auto_scan_layout():
    """Looping background untuk mendeteksi remote secara otomatis saat baru mulai atau setelah di-reset"""
    print("Pemindai otomatis aktif. Menunggu remote masuk frame...")
    while True:
        # Jika layout belum ada, coba tangkap frame secara diam-diam (silent=True)
        if not vision_reasoning.is_layout_ready:
            sukses = await asyncio.to_thread(vision_reasoning.auto_setup_layout, silent=True)
            if sukses:
                print(">>> Layout remote otomatis tertangkap dan diproses! <<<")
                
        # Polling setiap 2 detik (jangan terlalu cepat agar PC tidak berat)
        await asyncio.sleep(2)

async def main():
    print("--- SISTEM AKTIF (Tap Layar + Input Manual) ---")
    vision_reasoning.start_background_task_monitor()
    # Menjalankan keduanya secara paralel
    await asyncio.gather(
        listen_to_tap_commands(),
        manual_input_loop(),
        auto_scan_layout()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSistem dimatikan.")
