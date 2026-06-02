# Code Review — `src/`

## Critical / High Severity

| Issue | File | Status |
|---|---|---|
| **Path traversal** — parameter `cache_file` bisa baca file arbitrary | `server_vision.py` | **FIXED** — validasi `relative_to(cache_dir_resolved)` |
| **Crash bug** — `_safe_post` return `None` lalu dipanggil `.json()` | `vision_reasoning.py` | **FIXED** — ada pengecekan `if not response: return` |
| **Bare `except:` / `except: pass`** — menelan error | `audio_inference.py`, `hybrid_inference.py`, `tts_cache.py`, `vision_http.py` | **SEBAGIAN** — masih ada beberapa `except: pass` |
| **No graceful shutdown** (SIGINT/SIGTERM) | `audio_inference.py` | **SEBAGIAN** — ada `try/except KeyboardInterrupt` |
| **conversation_history membengkak** tanpa batas | `vision_reasoning.py` | **FIXED** — dibatasi `MAX_CONVERSATION_HISTORY` (default 20) |

---

## Medium Severity

| Issue | File | Status |
|---|---|---|
| Thread-per-call tanpa thread pool | `vision_http.py` | **FIXED** — `ThreadPoolExecutor(max_workers=4)` |
| Polling busy-loop video_feed | `server_vision.py` | **FIXED** — event-driven via `asyncio.Event()` |
| `asyncio.ensure_future` deprecated | `server_vision.py` | **FIXED** — sudah pakai `asyncio.create_task()` |
| Tidak ada input validation JSON | `server_vision.py` | **BELUM** — masih belum ada schema validation |
| Double-checked locking tanpa lock | `tts_cache.py` | **FIXED** — sudah pakai `_tts_cache_lock` |
| YOLO confidence threshold hardcode | `rotate_remote.py` | **FIXED** — via env `YOLO_CONF_THRESHOLD` |
| No cache eviction | `tts_cache.py` | **BELUM** — file MP3 menumpuk |
| No `unload_models()` | `vision_models.py` | **BELUM** — GPU memory tidak pernah dilepas |
| WebSocket protocol hardcode ws:// | `index.html` | **FIXED** — dynamic `ws: atau wss:` |
| `http_session` tidak pernah di-close | `audio_inference.py` | **FIXED** — ada di `finally` |

---

## Low / Cosmetic

| Issue | File | Status |
|---|---|---|
| Ada type hints di semua file | Semua `.py` | **FIXED** — `from __future__ import annotations` + type hints |
| `import vision_reasoning as vision_reasoning` redundant | `audio_inference.py`, `hybrid_inference.py` | **MASIH ADA** |
| OBB rotation direction inverted | `rotate_remote.py` | **FIXED** — env `YOLO_INVERT_OBB_ANGLE` |
| Hardcoded model path `best.pt` | `vision_models.py` | **FIXED** — env `YOLO_MODEL_PATH` |
| Variable `normalized_text` di-set dua kali | `vision_reasoning.py` | **MASIH ADA** |
| Dead code branch `exploration` | `vision_reasoning.py` | **REMOVED** |

---

## Per-file Summary

### `audio_inference.py` (257 lines)
**Hold-to-Speak + Whisper STT.** Kode sudah diperbaiki dari code review sebelumnya: redundant import masih ada, `global` variabel dikurangi, background tasks menggunakan `create_task`. Graceful shutdown sudah ada. Hold-to-speak menggantikan continuous VAD.

### `hybrid_inference.py` (87 lines)
**Alternate entry (tap + manual input).** Kode sederhana, minimal. Tidak ada perbaikan signifikan yang diperlukan untuk fungsinya yang sederhana.

### `rotate_remote.py` (183 lines)
**YOLO OBB rotation.** Kualitas kode baik. OBB angle inversion sudah konfigurable via env var. Portrait lock bisa dimatikan. Confidence threshold configurable.

### `server_vision.py` (550 lines)
**Main aiohttp/WebRTC server.** Path traversal sudah di-fix. Event-driven video feed. Web Speech API support. Graceful shutdown. Preload cache. Thread-safe TTS cache. Code quality baik.

### `tts_cache.py` (97 lines)
**TTS file caching.** Thread-safe dengan lock. Get cache info. MD5 hash index. Tidak ada eviction policy. Kode bersih dan efisien.

### `vision_http.py` (121 lines)
**HTTP utility layer.** ThreadPoolExecutor (4 workers). cv2_to_base64 dengan letterbox. extract_json_object robust. Kode baik dengan error handling proper.

### `vision_models.py` (48 lines)
**Lazy model loader.** Thread-safe double-checked locking. Path configurable via env. Kode minimal dan bersih.

### `vision_reasoning.py` (~1206 lines)
**Core VLM reasoning engine.** File terbesar. Banyak perbaikan: NMS untuk OWL, padding adaptif thumb mapping, hardcode tanpa VLM untuk senter/konfirmasi/dimana, ACTION_INTENTS (11 grup perintah AC), location descriptions, background monitor, conversation history cap. Masih ada global state yang banyak.

### `index.html` (~618 lines)
**Frontend UI.** Web Speech API, flashlight toggle, dual TTS mode, hold-to-speak dengan pointer events + touch events, haptic vibration feedback, beep sound effects (Web Audio API), safety timeout 30 detik, visibility change handler. Kode JS terstruktur baik dengan error handling.
