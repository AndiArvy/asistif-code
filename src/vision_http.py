from __future__ import annotations

import base64
import cv2
import json
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple
from tts_cache import get_tts_cache

LM_STUDIO_URL: str = "http://localhost:1234/v1/chat/completions"
SNAPSHOT_URL: str = "http://localhost:8080/snapshot"
TTS_TRIGGER_URL: str = "http://localhost:8080/trigger_tts"
LOG_URL: str = "http://localhost:8080/send_log"

_request_session: requests.Session = requests.Session()
_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)


def safe_post(url: str, **kwargs: Any) -> Optional[requests.Response]:
    try:
        timeout = kwargs.pop("timeout", 15)
        return _request_session.post(url, timeout=timeout, **kwargs)
    except Exception as e:
        print(f"[Warning] POST gagal ke {url}: {e}")
        return None


def _background_post_worker(url: str, kwargs: Dict[str, Any]) -> None:
    try:
        timeout = kwargs.pop("timeout", 5)
        _request_session.post(url, timeout=timeout, **kwargs)
    except Exception as e:
        print(f"[Warning] Background POST gagal ke {url}: {e}")


def fire_and_forget_post(url: str, **kwargs: Any) -> None:
    _thread_pool.submit(_background_post_worker, url, kwargs)


def capture_current_frame() -> Optional[np.ndarray]:
    try:
        response = _request_session.get(SNAPSHOT_URL, timeout=2)
        if response.status_code == 200:
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return frame
    except requests.exceptions.RequestException:
        pass
    except Exception as e:
        print(f"[Warning] Gagal decode frame: {e}")
    return None


def cv2_to_base64(image_array: np.ndarray | None, target_size: tuple[int, int] = (480, 854)) -> Optional[str]:
    if image_array is None or image_array.size == 0:
        return None

    target_w, target_h = target_size
    h, w = image_array.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def extract_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx : i + 1])
                except Exception:
                    return None
    return None


def speak(text: str) -> None:
    try:
        cache = get_tts_cache()
        cached_file = cache.get_cached_file(text)
        if cached_file:
            print(f"[TTS] Using cached audio for: {text[:50]}...")
            fire_and_forget_post(
                TTS_TRIGGER_URL,
                json={"text": text, "use_cache": True, "cache_file": cached_file},
            )
        else:
            print(f"[TTS] Generating new audio for: {text[:50]}...")
            fire_and_forget_post(
                TTS_TRIGGER_URL, json={"text": text, "use_cache": False}
            )
    except Exception as e:
        print(f"[TTS Cache Error] {e}, falling back to direct TTS")
        fire_and_forget_post(TTS_TRIGGER_URL, json={"text": text})


def log_system(text: str) -> None:
    fire_and_forget_post(LOG_URL, json={"sender": "Sistem", "text": text})
