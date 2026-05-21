from __future__ import annotations

import asyncio
import websockets
import requests
import json
import re

import vision_reasoning
from vision_reasoning import process_vlm_reasoning

COMMAND_URI: str = "ws://localhost:8080/command_feed"
LOG_URI: str = "http://localhost:8080/send_log"


def clean_command(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


async def run_vlm_logic(text: str, source: str = "Manual") -> None:
    if not text.strip():
        return
    print(f"\n[{source}]: {text}")
    try:
        await asyncio.to_thread(
            lambda: requests.post(LOG_URI, json={"sender": source, "text": text}, timeout=1)
        )
    except Exception:
        pass
    await asyncio.to_thread(process_vlm_reasoning, text)
    print(f"\n{'-'*20}\n>> ", end="")


async def listen_to_tap_commands() -> None:
    while True:
        try:
            async with websockets.connect(COMMAND_URI) as websocket:
                print("\n[Sistem] Terhubung ke sinyal Tap Layar.")
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if data.get("type") == "hold_action":
                        continue
                    asyncio.create_task(run_vlm_logic(data["text"], source="Tap Layar"))
        except websockets.exceptions.WebSocketException:
            await asyncio.sleep(2)
        except Exception as e:
            print(f"[listen_to_tap_commands] Error: {e}")
            await asyncio.sleep(2)


async def manual_input_loop() -> None:
    loop = asyncio.get_event_loop()
    print(">> ", end="", flush=True)
    while True:
        user_input = await loop.run_in_executor(None, input, "")
        if user_input.lower() in ['exit', 'quit']:
            break
        asyncio.create_task(run_vlm_logic(user_input, source="Manual"))


async def auto_scan_layout() -> None:
    print("Pemindai otomatis aktif. Menunggu remote masuk frame...")
    while True:
        if not vision_reasoning.is_layout_ready:
            sukses = await asyncio.to_thread(vision_reasoning.auto_setup_layout, silent=True)
            if sukses:
                print(">>> Layout remote otomatis tertangkap dan diproses! <<<")
        await asyncio.sleep(2)


async def main() -> None:
    print("--- SISTEM AKTIF (Tap Layar + Input Manual) ---")
    vision_reasoning.start_background_task_monitor()
    await asyncio.gather(
        listen_to_tap_commands(),
        manual_input_loop(),
        auto_scan_layout()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Mematikan hybrid inference...]")
        print("[Hybrid inference dimatikan.]")
