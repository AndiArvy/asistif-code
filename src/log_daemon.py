from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import websockets

SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
WS_FRONTEND_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/frontend_ws"
WS_COMMAND_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/command_feed"
POLL_INTERVAL = int(os.getenv("LOG_POLL_INTERVAL", "1"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs")).resolve()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[log_daemon] psutil tidak tersedia. CPU/memory monitoring dinonaktifkan.")

LOG_DIR.mkdir(parents=True, exist_ok=True)

system_status: dict = {
    "is_layout_ready": "?",
    "system_busy": "?",
    "active_task": "",
    "active_task_intent": "",
}

current_session: dict = {
    "id": None,
    "t_start": None,
    "t_stt_done": None,
    "t_response": None,
    "user_text": "",
    "response_text": "",
    "intent": "",
    "target_button": "",
    "is_layout_ready": "?",
    "system_busy": "?",
    "stt_time_ms": None,
    "reasoning_time_ms": None,
    "total_time_ms": None,
}

session_counter: int = 0

CONVERSATION_FIELDS = [
    "timestamp", "session_id", "event_type", "sender", "message",
    "intent", "target_button", "stt_time_ms", "reasoning_time_ms",
    "total_time_ms", "is_layout_ready", "system_busy",
]

SYSTEM_STATUS_FIELDS = [
    "timestamp", "is_layout_ready", "system_busy", "active_task",
    "active_task_intent", "camera_ready", "server_alive",
    "cpu_percent", "memory_percent",
]

EVENTS_FIELDS = [
    "timestamp", "event_type", "raw_data", "session_id",
]

conv_file = None
conv_writer = None
status_file = None
status_writer = None
events_file = None
events_writer = None


def rotate_files() -> None:
    global conv_file, conv_writer, status_file, status_writer, events_file, events_writer

    for f in [conv_file, status_file, events_file]:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    today = datetime.now().strftime("%Y-%m-%d")

    conv_path = LOG_DIR / f"conversation_{today}.csv"
    exists = conv_path.exists()
    conv_file = open(conv_path, "a", newline="", encoding="utf-8")
    conv_writer = csv.DictWriter(conv_file, fieldnames=CONVERSATION_FIELDS)
    if not exists:
        conv_writer.writeheader()

    status_path = LOG_DIR / f"system_status_{today}.csv"
    exists = status_path.exists()
    status_file = open(status_path, "a", newline="", encoding="utf-8")
    status_writer = csv.DictWriter(status_file, fieldnames=SYSTEM_STATUS_FIELDS)
    if not exists:
        status_writer.writeheader()

    events_path = LOG_DIR / f"events_{today}.csv"
    exists = events_path.exists()
    events_file = open(events_path, "a", newline="", encoding="utf-8")
    events_writer = csv.DictWriter(events_file, fieldnames=EVENTS_FIELDS)
    if not exists:
        events_writer.writeheader()


def write_event(event_type: str, raw_data: str, sess_id: str | None = None) -> None:
    events_writer.writerow({
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "raw_data": str(raw_data)[:1000],
        "session_id": sess_id or "",
    })
    events_file.flush()


def write_conversation(event_type: str, sender: str, message: str, **kwargs) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "session_id": current_session["id"] or "",
        "event_type": event_type,
        "sender": sender,
        "message": str(message)[:1000],
        "intent": kwargs.get("intent", ""),
        "target_button": kwargs.get("target_button", ""),
        "stt_time_ms": kwargs.get("stt_time_ms", ""),
        "reasoning_time_ms": kwargs.get("reasoning_time_ms", ""),
        "total_time_ms": kwargs.get("total_time_ms", ""),
        "is_layout_ready": kwargs.get("is_layout_ready", current_session.get("is_layout_ready", "?")),
        "system_busy": kwargs.get("system_busy", current_session.get("system_busy", "?")),
    }
    conv_writer.writerow(row)
    conv_file.flush()


def write_status(**kwargs) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "is_layout_ready": kwargs.get("is_layout_ready", "?"),
        "system_busy": kwargs.get("system_busy", "?"),
        "active_task": kwargs.get("active_task", ""),
        "active_task_intent": kwargs.get("active_task_intent", ""),
        "camera_ready": kwargs.get("camera_ready", "?"),
        "server_alive": kwargs.get("server_alive", "?"),
        "cpu_percent": kwargs.get("cpu_percent", ""),
        "memory_percent": kwargs.get("memory_percent", ""),
    }
    status_writer.writerow(row)
    status_file.flush()


def start_new_session() -> str:
    global session_counter, current_session

    old = dict(current_session)
    if old["user_text"]:
        stt = old.get("stt_time_ms", "")
        reas = old.get("reasoning_time_ms", "")
        total = old.get("total_time_ms", "")
        write_conversation(
            "session_end", "System",
            f"Session: {old['user_text'][:100]} -> {old['response_text'][:100]}",
            stt_time_ms=stt, reasoning_time_ms=reas, total_time_ms=total,
        )

    session_counter += 1
    session_id = str(uuid.uuid4())
    current_session = {
        "id": session_id,
        "t_start": None,
        "t_stt_done": None,
        "t_response": None,
        "user_text": "",
        "response_text": "",
        "intent": "",
        "target_button": "",
        "is_layout_ready": system_status.get("is_layout_ready", "?"),
        "system_busy": system_status.get("system_busy", "?"),
        "stt_time_ms": None,
        "reasoning_time_ms": None,
        "total_time_ms": None,
    }
    return session_id


def finalize_session() -> None:
    if current_session["t_start"] is None:
        return
    response_text = current_session.get("response_text", "")
    user_text = current_session.get("user_text", "")
    if not response_text and not user_text:
        return
    stt = current_session.get("stt_time_ms", "")
    reas = current_session.get("reasoning_time_ms", "")
    total = current_session.get("total_time_ms", "")
    write_conversation(
        "response", "System",
        current_session["response_text"] or "(no response)",
        stt_time_ms=stt, reasoning_time_ms=reas, total_time_ms=total,
        is_layout_ready=current_session.get("is_layout_ready", "?"),
        system_busy=current_session.get("system_busy", "?"),
    )


def extract_target_button(text: str) -> str:
    m = re.search(r'tombol (\w+)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:tombol )?(\w+)(?: berada| ada di)', text)
    if m:
        return m.group(1)
    return ""


async def handle_frontend_message(data: dict) -> None:
    msg_type = data.get("type", "")
    write_event(f"frontend_{msg_type}", json.dumps(data, ensure_ascii=False), current_session["id"])

    if msg_type == "log":
        sender = data.get("sender", "")
        text = data.get("text", "")

        write_conversation("log", sender, text)

        if sender == "Sistem":
            if "Pemetaan remote berhasil" in text or "Mau saya bantu" in text:
                system_status["is_layout_ready"] = "yes"
            elif "Layout direset" in text or "Letakkan remote" in text or "tidak terlihat" in text:
                system_status["is_layout_ready"] = "no"
            elif "Menganalisis" in text:
                system_status["system_busy"] = "yes"

            if "Geser jempol" in text or "Tombol tujuan" in text:
                system_status["system_busy"] = "yes"
                btn = extract_target_button(text)
                if btn:
                    system_status["active_task"] = btn
            elif "Selesai" in text or "Ini tombol" in text:
                system_status["system_busy"] = "no"

        elif sender == "User" and text:
            now = time.time()
            current_session["user_text"] = text
            current_session["is_layout_ready"] = system_status.get("is_layout_ready", "?")
            current_session["system_busy"] = system_status.get("system_busy", "?")
            if current_session["t_start"] is not None:
                current_session["t_stt_done"] = now
                ms = round((now - current_session["t_start"]) * 1000, 1)
                current_session["stt_time_ms"] = ms
                write_event("stt_complete", f"STT={ms}ms: {text[:100]}", current_session["id"])
            else:
                write_event("stt_orphan", f"STT tanpa T0: {text[:100]}", current_session["id"])

        elif sender == "Error":
            system_status["system_busy"] = "no"

    elif msg_type == "speak":
        text = data.get("text", "")
        current_session["response_text"] = text
        now = time.time()

        if current_session["t_stt_done"] is not None:
            current_session["t_response"] = now
            reas = round((now - current_session["t_stt_done"]) * 1000, 1)
            total = round((now - current_session["t_start"]) * 1000, 1) if current_session["t_start"] else ""
            current_session["reasoning_time_ms"] = reas
            current_session["total_time_ms"] = total
            write_event("response_ready", f"Reasoning={reas}ms Total={total}ms: {text[:100]}", current_session["id"])
        else:
            write_event("response_orphan", f"Response tanpa T1: {text[:100]}", current_session["id"])

        finalize_session()
        system_status["system_busy"] = "no"

    elif msg_type == "audio":
        text = data.get("text", "")
        current_session["response_text"] = text
        now = time.time()

        if current_session["t_stt_done"] is not None:
            current_session["t_response"] = now
            reas = round((now - current_session["t_stt_done"]) * 1000, 1)
            total = round((now - current_session["t_start"]) * 1000, 1) if current_session["t_start"] else ""
            current_session["reasoning_time_ms"] = reas
            current_session["total_time_ms"] = total
            write_event("response_ready", f"Reasoning={reas}ms Total={total}ms (gTTS): {text[:100]}", current_session["id"])
        else:
            write_event("response_orphan", f"Response tanpa T1 (gTTS): {text[:100]}", current_session["id"])

        finalize_session()
        system_status["system_busy"] = "no"


async def handle_command_message(data: dict) -> None:
    write_event("command", json.dumps(data, ensure_ascii=False), current_session["id"])

    msg_type = data.get("type", "")
    action = data.get("action", "")

    if msg_type == "hold_action":
        if action == "start_listening":
            write_event("hold_start", "User mulai bicara", current_session["id"])

        elif action == "stop_listening":
            start_new_session()
            current_session["t_start"] = time.time()
            write_event("hold_stop", "User selesai bicara", current_session["id"])
            write_conversation("voice_command", "User", "[Hold-to-speak]")

    elif data.get("text"):
        text = data["text"]
        if text == "ambil_layout":
            write_event("layout_capture", "Mengambil layout remote", current_session["id"])


async def poll_system_status() -> None:
    last_status = {}

    while True:
        try:
            server_alive = "no"
            camera_ready = "no"

            try:
                r = requests.get(f"{BASE_URL}/", timeout=3)
                server_alive = "yes" if r.status_code == 200 else "no"
            except Exception:
                server_alive = "no"

            try:
                r2 = requests.get(f"{BASE_URL}/snapshot", timeout=3)
                camera_ready = "yes" if r2.status_code == 200 else "no"
            except Exception:
                camera_ready = "no"

            cpu_pct = ""
            mem_pct = ""
            if HAS_PSUTIL:
                try:
                    cpu_pct = psutil.cpu_percent(interval=0.3)
                    mem_pct = psutil.virtual_memory().percent
                except Exception:
                    pass

            current_status = {
                "is_layout_ready": system_status.get("is_layout_ready", "?"),
                "system_busy": system_status.get("system_busy", "?"),
                "active_task": system_status.get("active_task", ""),
                "active_task_intent": system_status.get("active_task_intent", ""),
                "camera_ready": camera_ready,
                "server_alive": server_alive,
                "cpu_percent": cpu_pct,
                "memory_percent": mem_pct,
            }

            if current_status != last_status:
                write_status(**current_status)
                last_status = dict(current_status)

        except Exception as e:
            write_event("poll_error", f"Status poll: {e}")

        await asyncio.sleep(POLL_INTERVAL)


async def connect_frontend_ws() -> None:
    while True:
        try:
            async with websockets.connect(WS_FRONTEND_URL) as ws:
                print("[log_daemon] Terhubung ke /frontend_ws")
                write_event("ws_connect", "frontend_ws: terhubung")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        await handle_frontend_message(data)
                    except json.JSONDecodeError as e:
                        write_event("parse_error", f"frontend_ws JSON: {e}")
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            write_event("ws_disconnect", f"frontend_ws: {e}")
            await asyncio.sleep(3)
        except Exception as e:
            write_event("ws_error", f"frontend_ws: {e}")
            await asyncio.sleep(3)


async def connect_command_ws() -> None:
    while True:
        try:
            async with websockets.connect(WS_COMMAND_URL) as ws:
                print("[log_daemon] Terhubung ke /command_feed")
                write_event("ws_connect", "command_feed: terhubung")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        await handle_command_message(data)
                    except json.JSONDecodeError as e:
                        write_event("parse_error", f"command_feed JSON: {e}")
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            write_event("ws_disconnect", f"command_feed: {e}")
            await asyncio.sleep(3)
        except Exception as e:
            write_event("ws_error", f"command_feed: {e}")
            await asyncio.sleep(3)


async def daily_rotation() -> None:
    last_date = datetime.now().strftime("%Y-%m-%d")
    while True:
        await asyncio.sleep(3600)
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != last_date:
            rotate_files()
            last_date = current_date
            print(f"[log_daemon] Rotasi file: {current_date}")
            write_event("file_rotation", f"Rotasi ke {current_date}")


async def main() -> None:
    print("=" * 60)
    print("  LOG DAEMON — Perekam Data Skripsi")
    print(f"  Server       : {BASE_URL}")
    print(f"  WS Frontend  : {WS_FRONTEND_URL}")
    print(f"  WS Command   : {WS_COMMAND_URL}")
    print(f"  Log dir      : {LOG_DIR}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  psutil       : {'Tersedia' if HAS_PSUTIL else 'TIDAK tersedia (CPU/mem dimatikan)'}")
    print("=" * 60)
    print("  File output:")
    print(f"    - conversation_YYYY-MM-DD.csv (percakapan + response time)")
    print(f"    - system_status_YYYY-MM-DD.csv (status sistem periodik)")
    print(f"    - events_YYYY-MM-DD.csv (semua raw event)")
    print("=" * 60)
    print("  Menunggu koneksi ke server...")
    print("  (Jalankan server_vision.py dan audio_inference.py terlebih dahulu)")
    print("=" * 60)

    rotate_files()

    env_snapshot = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("SERVER_", "LOG_", "WHISPER_", "YOLO_", "OWL_", "VISION_", "TTS_", "JPEG_"))}
    write_event("daemon_start", f"Log daemon started. Env: {json.dumps(env_snapshot, ensure_ascii=False)}")

    try:
        await asyncio.gather(
            connect_frontend_ws(),
            connect_command_ws(),
            poll_system_status(),
            daily_rotation(),
        )
    except KeyboardInterrupt:
        print("\n[log_daemon] Dimatikan oleh user.")
        write_event("daemon_stop", "Log daemon stopped by user")
    finally:
        for f in [conv_file, status_file, events_file]:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        print("[log_daemon] File log ditutup.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[log_daemon] Selesai.")
