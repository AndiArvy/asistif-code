# Development Guide

Panduan untuk pengembang yang ingin berkontribusi atau memodifikasi sistem.

---

## Struktur Proyek

```
code2/
├── src/                          # Kode sumber utama
│   ├── server_vision.py          # WebRTC server + HTTP API + TTS
│   ├── audio_inference.py        # STT Whisper + trigger VLM
│   ├── vision_reasoning.py       # Pipeline vision + VLM reasoning (~950 baris)
│   ├── index.html                # Frontend HP (WebRTC + hold-to-speak)
│   ├── rotate_remote.py          # YOLO OBB rotation & cropping
│   ├── vision_models.py          # Lazy-loading model YOLO + OWL-ViT (singleton thread-safe)
│   ├── vision_http.py            # HTTP utilities (LM Studio, snapshot, TTS)
│   ├── tts_cache.py              # TTS file caching dengan MD5 index
│   ├── hybrid_inference.py       # Entry alternatif (terminal input)
│   └── layout_OWL.py             # [DEPRECATED] Skrip standalone OWL
├── docs/                         # Dokumentasi
│   ├── ARSITEKTUR_SISTEM.md      # Dokumentasi arsitektur
│   ├── API.md                    # Dokumentasi API
│   ├── ADRs.md                   # Architecture Decision Records
│   ├── DEPLOYMENT.md             # Panduan deployment
│   ├── USER_GUIDE.md             # Panduan pengguna
│   ├── DEVELOPMENT.md            # Panduan pengembangan (ini)
│   └── code_review.md            # Laporan code review
├── best.pt                       # Model YOLOv8 (remote + jempol detection)
├── tts_cache/                    # Cache TTS lokal
│   └── cache_index.json          # Index mapping hash → file
├── layout.json                   # Hasil mapping tombol remote
├── debug_*.jpg                   # Debug images (jika VISION_DEBUG=true)
└── README.md                     # Quick-start guide
```

---

## Stack Teknologi

| Komponen | Teknologi | Versi |
|----------|-----------|-------|
| WebRTC Server | aiohttp + aiortc | 3.9+ / 1.5+ |
| Frontend | HTML/JS (vanilla) | ES2020 |
| Speech-to-Text | faster-whisper | large-v3-turbo |
| Object Detection | Ultralytics YOLOv8 | 8.x |
| Zero-shot Detection | HuggingFace OWL-ViT | owlv2 |
| Vision LLM | LM Studio API | OpenAI-compatible |
| Text-to-Speech | Google gTTS | 3.x |
| Audio Resample | PyAV | 10.x |

---

## Setup Development

```bash
# Clone
git clone <repo-url>
cd code2

# Virtual env
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate

# Editable install
pip install -e .
```

Atau install dependensi utama:
```bash
pip install aiohttp aiortc av faster-whisper ultralytics \
            transformers torch torchvision gtts opencv-python \
            numpy websockets requests Pillow websocket-client
```

## Menjalankan Mode Development

```bash
# Terminal 1
python src/server_vision.py

# Terminal 2
python src/audio_inference.py

# Terminal 3 (opsional - mode hybrid tanpa HP)
python src/hybrid_inference.py
```

Set `VISION_DEBUG=true` untuk menyimpan debug images di root folder.

## Alur Data & Cara Kerja

### Pipeline Utama

1. **WebRTC Handshake** — HP → `/offer` (SDP) → server buat `RTCPeerConnection` → return SDP answer
2. **Video Streaming** — HP → WebRTC video track → `latest_jpeg` (global) + `frame_event` (asyncio.Event)
3. **Audio Streaming** — HP → WebRTC audio track → `AudioResampler` (s16, mono, 16kHz) → `/audio_feed` (WebSocket broadcast)
4. **Hold-to-Speak** — HP touchstart → `hold_action:start_listening` via `/frontend_ws` → `audio_inference.py` mulai buffer → touchend → `hold_action:stop_listening` → proses Whisper
5. **VLM Reasoning** — hasil STT → `process_vlm_reasoning()` → ambil snapshot → deteksi thumb → kirim prompt ke LM Studio → parse response JSON → TTS
6. **Background Monitor** — Thread terpisah, setiap 500ms cek thumb position, auto-confirm jika menyentuh target

### Threading Model

```
Main Thread (asyncio):
├── server_vision.py — aiohttp server (WebRTC, HTTP, WebSocket)
└── audio_inference.py — asyncio.gather
    ├── listen_to_mic()       → WebSocket audio → Whisper STT
    ├── listen_to_commands()   → WebSocket command → trigger VLM
    └── auto_scan_layout()    → periodic YOLO scan

Background Thread:
└── background_task_monitor_loop() — 500ms thumb detection + auto-confirm

ThreadPool (vision_http.py):
└── fire_and_forget_post() — 4 workers untuk HTTP non-blocking
```

## Konvensi Kode

### Python

- **Type hints** — Wajib untuk fungsi baru (`def foo(x: int) -> str:`)
- **Imports** — Group: standard library, third-party, local (alphabetical)
- **Docstrings** — Gunakan comment `#` untuk fungsi internal, docstring untuk public API
- **Global state** — Hindari global variables. Jika terpaksa, gunakan uppercase + type hints
- **Error handling** — Jangan `except: pass`. Log error dengan `print(f"[Module] Error: {e}")`
- **Async patterns** — Gunakan `asyncio.create_task()` (bukan `ensure_future`)

### JavaScript (index.html)

- **var** → gunakan `let` / `const`
- **Event listeners** — `passive: false` untuk touch events
- **WebSocket** — Handle `onmessage` dengan switch/case berdasarkan `data.type`
- **Error handling** — Setiap `await` / Promise harus punya `.catch()`

## Testing

Saat ini belum ada test suite formal. Panduan testing manual:

1. **Test WebRTC** — Buka `index.html` via browser, cek koneksi, cek `latest_jpeg` diupdate
2. **Test STT** — Jalankan `audio_inference.py`, kirim audio via `/audio_feed`
3. **Test Layout** — Letakkan remote di kamera, cek `layout.json` terbentuk
4. **Test VLM** — Jalankan `hybrid_inference.py`, ketik perintah, cek response TTS

Untuk menambahkan test:
- Gunakan `pytest` untuk unit test Python
- Mock WebRTC dan HTTP calls dengan `pytest-asyncio` + `aioresponses`
- Test frontend dengan Playwright atau Cypress

## Debugging

### Env vars untuk debug:
```bash
VISION_DEBUG=true    # Simpan debug images ke root (default: true)
```

### Debug images yang dihasilkan:
- `debug_frame.jpg` — Frame mentah dari kamera
- `debug_cropped.jpg` — Crop remote setelah YOLO rotation
- `debug_current_guided.jpg` — Crop remote + overlay tombol + jempol
- `debug_clean_reference.jpg` — Referensi layout bersih
- `debug_indexed_reference.jpg` — Referensi layout dengan indeks b1..bN

### Tips debugging:
- Cek `layout.json` untuk melihat hasil mapping tombol
- Cek `tts_cache/cache_index.json` untuk status cache
- Error di console server biasanya diawali `[Warning]`, `[Error]`, atau `[TTS]`

## Kontribusi

### Pull Request Process

1. Fork repo dan buat branch dari `main`
2. Ikuti konvensi kode di atas
3. Test perubahan secara manual
4. Update dokumentasi jika perlu
5. Buat PR dengan deskripsi jelas

### Area yang Butuh Perbaikan (berdasarkan code review)

**Critical:**
- Perbaiki path traversal di `trigger_tts` (sudah ada validasi, perlu review)
- Fix crash `None.json()` di `map_functions_with_vlm` (sudah ada fallback, perlu review)

**Medium:**
- Ganti `ensure_future` → `create_task`
- Tambah input validation untuk JSON endpoints
- Ganti thread-per-call dengan thread pool (sudah dilakukan di `vision_http.py`)

**Low:**
- Tambah type hints ke semua fungsi
- Setup test suite
- Tambah graceful shutdown handler untuk audio_inference.py
- Implement cache eviction di tts_cache.py
- Tambah `unload_models()` untuk GPU memory management

### Coding Standards

- Python: Ikuti PEP 8 — gunakan `black` untuk formatting, `ruff` untuk linting
- JavaScript: Ikuti standard ES2020
- HTML: Valid HTML5, semantic elements
- Commit messages: Indonesian atau English, prefixed dengan `[module]` jika relevan
