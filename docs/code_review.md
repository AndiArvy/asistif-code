# Code Review — `src/`

## 🔴 Critical / High Severity

| Issue | File | Line |
|---|---|---|
| **Path traversal** — parameter `cache_file` bisa baca file arbitrary (e.g. `/etc/passwd`) | `server_vision.py` | 186 |
| **Crash bug** — `_safe_post` return `None` lalu dipanggil `.json()` tanpa pengecekan | `vision_reasoning.py` | 324-325 |
| **File handle leak** — `open()` tanpa context manager | `server_vision.py` | 27 |
| **Bare `except:` / `except: pass`** — menelan semua error termasuk `KeyboardInterrupt` | `hybrid_inference.py` | 28, 48 |
| | `tts_cache.py` | 33 |
| | `vision_http.py` | 28 |

---

## 🟡 Medium Severity

| Issue | File | Line |
|---|---|---|
| Thread-per-call tanpa thread pool — bisa ratusan thread | `vision_http.py` | 32-35 |
| Redundant YOLO inference di frame yang sama | `vision_reasoning.py` | 388 |
| Coordinate mapping thumb-to-button mengabaikan aspect ratio — akurasi turun 10-30px | `vision_reasoning.py` | 592-593 |
| Dead code branch `if intent == "exploration"` — tidak akan pernah terpanggil | `vision_reasoning.py` | 892 |
| `conversation_history` membengkak tanpa batas — memory leak | `vision_reasoning.py` | 906-907 |
| Polling busy-loop `video_feed` — boros CPU saat tanpa client | `server_vision.py` | 41-57 |
| Server bind `0.0.0.0:8080` — terbuka ke semua interface jaringan | `server_vision.py` | 382 |
| Global state 12+ variabel — tidak testable, rawan side-effect | `vision_reasoning.py` | — |
| `asyncio.ensure_future` deprecated, sebaiknya `asyncio.create_task` | `server_vision.py` | 91-110, 121-140 |
| Tidak ada input validation / schema pada JSON endpoints | `server_vision.py` | 175, 242 |
| `trigger_tts` startup fetch tidak dicek success/failure-nya | `index.html` | 424-428 |

---

## 🟢 Low / Cosmetic

| Issue | File | Line |
|---|---|---|
| Tidak ada type hints di semua file Python | Semua `.py` | — |
| Import redundant `import vision_reasoning as vision_reasoning` | `audio_inference.py` | 19 |
| | `hybrid_inference.py` | 7 |
| `global` deklarasi meaningless di module level (`global system_is_busy`) | `audio_inference.py` | 14 |
| Dead code variabel `whisper_is_inferencing` — dideklarasi tidak pernah dibaca | `audio_inference.py` | 39-40 |
| OBB rotation direction mungkin terbalik (dikomentari "fix me") | `rotate_remote.py` | 38-42 |
| Tidak ada cache eviction — file MP3 menumpuk selamanya | `tts_cache.py` | — |
| Double-checked locking tanpa lock di first check | `tts_cache.py` | 132-133 |
| Hardcoded model path `best.pt` — error tanpa pesan jelas | `vision_models.py` | 24 |
| `extract_json_object` pakai `find`/`rfind` — rawan salah ekstrak saat ada `{`/`}` dalam string | `vision_http.py` | 71-86 |
| WebSocket protocol hardcode `ws://` — broken di halaman HTTPS | `index.html` | 207 |
| YOLO confidence threshold hardcode `0.3` — tidak konfigurabel | `rotate_remote.py` | 129 |
| Portrait lock paksa 90° rotasi — tidak cocok untuk remote landscape | `rotate_remote.py` | 90-91 |
| `crop` dari slicing tidak bisa `None` — pengecekan `is None` dead code | `rotate_remote.py` | 85 |
| `http_session` tidak pernah di-close | `audio_inference.py` | 17 |
| Background tasks via `asyncio.create_task` tidak ditracking | `audio_inference.py` | 152-153 |
| Tidak ada graceful shutdown handler (SIGINT/SIGTERM) | `audio_inference.py` | 275-276 |
| Debug `cv2.imwrite("debug_frame.jpg", ...)` overwrite terus tanpa flag | `vision_reasoning.py` | 363 |
| Variabel `normalized_text` di-set dua kali dengan cara berbeda | `vision_reasoning.py` | 682, 716 |
| `get_tts_cache()` bisa membuat dua instance jika race condition | `tts_cache.py` | 132-133 |
| Tidak ada retry logic untuk transient network failure | `vision_http.py` | — |
| Tidak ada `unload_models()` untuk free GPU memory | `vision_models.py` | — |
| Resolusi video 1920x1080 @ 20fps — boros bandwidth, cukup 720p | `index.html` | 260-262 |

---

## Per-file Summary

### `audio_inference.py` (276 lines)
**Speech-to-text via Whisper + WebSocket.** Banyak dead code (`global` meaningless, variabel tak terpakai), import redundant, background tasks tidak ditracking, tidak ada graceful shutdown.

### `hybrid_inference.py` (87 lines)
**Alternate entry point (tap + manual input).** Dua bare `except: pass`, tidak ada penanganan EOF di `input()`, thread leak via `fire_and_forget`.

### `rotate_remote.py` (138 lines)
**YOLO remote crop rotation.** Kualitas kode paling baik. Ada catatan OBB rotation direction yang belum di-fix, portrait lock kaku, confidence threshold hardcode.

### `server_vision.py` (383 lines)
**Main aiohttp/WebRTC server.** **Path traversal critical**, file handle leak, `ensure_future` deprecated, busy-loop polling, bind `0.0.0.0`, akses private attribute class lain.

### `tts_cache.py` (135 lines)
**TTS file caching.** Bare `except:`, double-checked locking rawan race, tidak ada cache eviction, blocking I/O tanpa versi async.

### `vision_http.py` (110 lines)
**HTTP utility layer.** Thread-per-call tanpa pool, bare `except:`, `extract_json_object` fragile, tidak ada retry logic.

### `vision_models.py` (35 lines)
**Lazy model loader.** Paling kecil. Tidak ada error handling, `best.pt` hardcode, tidak ada `unload_models()`.

### `vision_reasoning.py` (910 lines)
**Core VLM reasoning engine.** File terbesar dan paling problematik: crash bug di `map_functions_with_vlm`, redundant YOLO inference, coordinate mapping tidak akurat, dead code `exploration`, memory leak `conversation_history`, 12+ global variables, debug file overwrite tanpa flag.

### `index.html` (441 lines)
**Frontend UI.** WS protocol hardcode `ws://`, startup TTS fetch tidak dicek, resolusi video berlebihan, kode JS secara umum baik dengan error handling yang memadai.
