# Development Guide

Panduan untuk pengembang yang ingin berkontribusi atau memodifikasi sistem.

---

## Struktur Proyek

```
code2/
├── src/                          # Kode sumber utama
│   ├── server_vision.py          # WebRTC server + HTTP API + TTS (~550 baris)
│   ├── audio_inference.py        # STT Whisper + Hold-to-Speak + trigger VLM (~257 baris)
│   ├── vision_reasoning.py       # Pipeline vision + VLM reasoning + task monitor (~1206 baris)
│   ├── index.html                # Frontend HP (WebRTC + hold-to-speak + Web Speech)
│   ├── rotate_remote.py          # YOLO OBB rotation & cropping (~183 baris)
│   ├── vision_models.py          # Lazy-loading model YOLO + OWL-ViT (~48 baris)
│   ├── vision_http.py            # HTTP utilities dgn ThreadPool (~121 baris)
│   ├── tts_cache.py              # TTS file caching dengan MD5 index (~97 baris)
│   ├── hybrid_inference.py       # Entry alternatif (terminal input) (~87 baris)
│   └── log_daemon.py             # Perekam data ke CSV, opsional (~476 baris)
├── docs/                         # Dokumentasi
│   ├── ARSITEKTUR_SISTEM.md      # Dokumentasi arsitektur lengkap
│   ├── API.md                    # Dokumentasi API
│   ├── ADRs.md                   # Architecture Decision Records
│   ├── DEPLOYMENT.md             # Panduan deployment
│   ├── USER_GUIDE.md             # Panduan pengguna
│   ├── DEVELOPMENT.md            # Panduan pengembangan (ini)
│   └── code_review.md            # Laporan code review
├── best.pt                       # Model YOLOv26n OBB (remote + jempol detection)
├── tts_cache/                    # Cache TTS lokal
│   └── cache_index.json          # Index mapping hash → file
├── logs/                         # Output log_daemon (CSV, gitignored)
├── requirements.txt              # Dependensi (torch cu126 default)
├── layout.json                   # Hasil mapping tombol remote
├── debug_*.jpg                   # Debug images (jika VISION_DEBUG=true)
└── README.md                     # Quick-start guide
```

---

## Stack Teknologi

| Komponen | Teknologi | Versi |
|----------|-----------|-------|
| WebRTC Server | aiohttp + aiortc | 3.10+ / 1.5+ |
| Frontend | HTML/JS (vanilla) | ES2020 |
| Speech-to-Text | faster-whisper | large-v3-turbo |
| Object Detection | Ultralytics YOLOv26n OBB | custom |
| Zero-shot Detection | HuggingFace OWL-ViT | owlv2-base |
| Vision LLM | LM Studio API | OpenAI-compatible |
| Text-to-Speech | Google gTTS + Web Speech API | 3.x / browser native |
| Audio Resample | PyAV | 10.x |
| Deep Learning | PyTorch (CUDA cu126) | 2.12 |
| Perekam data | psutil (opsional) | 5.9+ |

> **Catatan torch:** `requirements.txt` menyematkan `torch==2.12.0+cu126` via `--extra-index-url https://download.pytorch.org/whl/cu126`. Untuk CUDA lain atau CPU-only, sesuaikan URL & pin di bagian atas file (lihat komentar di sana).

---

## Setup Development

```bash
# Clone
git clone <repo-url>
cd code2

# Virtual env
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate

# Install dependencies (default: CUDA 12.6 build torch)
pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan Mode Development

```bash
# Terminal 1
python src/server_vision.py

# Terminal 2
python src/audio_inference.py

# Terminal 3 (opsional - mode hybrid tanpa HP)
python src/hybrid_inference.py

# Terminal 4 (opsional - rekam data ke logs/*.csv)
python src/log_daemon.py
```

Set `VISION_DEBUG=true` untuk menyimpan debug images di root folder.

## Alur Data & Cara Kerja

### Pipeline Utama

1. **WebRTC Handshake**: HP → `/offer` (SDP) → server buat `RTCPeerConnection` → return SDP answer
2. **Video Streaming**: HP → WebRTC video track → `latest_jpeg` (global) + `frame_event` (asyncio.Event)
3. **Audio Streaming**: HP → WebRTC audio track → `AudioResampler` (s16, mono, 16kHz) → `/audio_feed` (WebSocket broadcast)
4. **Hold-to-Speak**: HP touchstart → `hold_action:start_listening` → `audio_inference.py` mulai buffer → touchend → `hold_action:stop_listening` → drain 0.4s → Whisper STT
5. **VLM Reasoning**: hasil STT → `process_vlm_reasoning()`:
   - Hardcode tanpa VLM: senter, reset layout, "apa tombol ini?", konfirmasi, "dimana tombol X?", ACTION_INTENTS (nyalakan AC, atur suhu, ganti mode, dll)
   - Jika perlu VLM: ambil snapshot → deteksi thumb → kirim 1 gambar ke LM Studio → parse response → TTS
6. **Background Monitor**: Thread daemon terpisah, setiap 500ms cek thumb position, auto-confirm jika menyentuh target

### Threading Model

```
Main Thread (asyncio):
├── server_vision.py: aiohttp server (WebRTC, HTTP, WebSocket)
└── audio_inference.py: asyncio.gather
    ├── listen_to_mic()         → WebSocket audio → Whisper STT
    ├── listen_to_commands()    → WebSocket command → trigger VLM
    └── auto_scan_layout()      → periodic YOLO scan (tiap 3 detik)

Background Thread (daemon):
└── background_task_monitor_loop(): 500ms thumb detection + auto-confirm

ThreadPool (vision_http.py 4 workers):
└── fire_and_forget_post(): HTTP non-blocking
```

### Hold-to-Speak vs VAD

Sistem menggunakan **Hold-to-Speak** (push-to-talk), bukan continuous VAD:

1. **touchstart** (HP) → server kirim `hold_action:start_listening` → audio_inference set `hold_to_speak_active = True`
2. User bicara: audio di-buffer di `hold_audio_buffer`
3. **touchend** (HP) → server kirim `hold_action:stop_listening` → `hold_to_speak_active = False`, `draining = True`
4. Tunggu 0.4s drain → `draining = False`
5. `process_buffered_audio()` → Whisper → filter noise/halusinasi → VLM

Keuntungan: Tidak ada false positive dari percakapan sekitar, segmentasi audio jelas.

---

## Konvensi Kode

### Python

- **Type hints**: Wajib untuk fungsi baru (`def foo(x: int) -> str:`)
- **Imports**: Group: standard library, third-party, local (alphabetical)
- **Global state**: Minimalkan. Jika terpaksa, gunakan uppercase + type hints
- **Error handling**: Jangan `except: pass`. Log error dengan `print(f"[Module] Error: {e}")`
- **Async patterns**: Gunakan `asyncio.create_task()` (bukan `ensure_future`)
- **Threading**: Gunakan `threading.Lock()` untuk shared resource, ThreadPoolExecutor untuk HTTP

### JavaScript (index.html)

- **var** → gunakan `let` / `const`
- **Event listeners**: `passive: false` untuk touch events
- **WebSocket**: Handle `onmessage` dengan switch/case berdasarkan `data.type`
- **Error handling**: Setiap `await` / Promise harus punya `.catch()`
- **Haptic**: `navigator.vibrate()` untuk feedback sentuhan

---

## Environment Variables untuk Development

```bash
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
JPEG_QUALITY=99

# TTS
TTS_PLAYBACK_RATE=1.2

# Whisper
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_CPU_THREADS=8
WHISPER_MODEL_SIZE=deepdml/faster-whisper-large-v3-turbo-ct2
WHISPER_BEAM_SIZE=1
WHISPER_BEST_OF=1
WHISPER_MIN_SILENCE_MS=500

# YOLO
YOLO_MODEL_PATH=best.pt
YOLO_CONF_THRESHOLD=0.3
YOLO_FORCE_PORTRAIT=true
YOLO_INVERT_OBB_ANGLE=false

# OWL-ViT
OWL_MODEL_NAME=google/owlv2-base-patch16-ensemble

# Debug
VISION_DEBUG=true
MAX_CONVERSATION_HISTORY=20

# Log daemon (opsional)
LOG_DIR=logs
LOG_POLL_INTERVAL=1
```

## Testing

Saat ini belum ada test suite formal. Panduan testing manual:

1. **Test WebRTC**: Buka `index.html` via browser, cek koneksi, cek `latest_jpeg` diupdate
2. **Test STT**: Jalankan `audio_inference.py`, hold-to-speak via HP, cek output teks
3. **Test Layout**: Letakkan remote di kamera, cek `layout.json` terbentuk
4. **Test VLM**: Jalankan `hybrid_inference.py`, ketik perintah, cek response TTS
5. **Test Background Monitor**: Setelah navigasi aktif, geser jempol ke tombol target, cek auto-confirm

Untuk menambahkan test formal:
- Gunakan `pytest` untuk unit test Python
- Mock WebRTC dan HTTP calls dengan `pytest-asyncio` + `aioresponses`
- Test frontend dengan Playwright atau Cypress

## Debugging

### Env vars untuk debug:
```bash
VISION_DEBUG=true    # Simpan debug images ke root (default: true)
```

### Debug images yang dihasilkan:
- `debug_frame.jpg`: Frame mentah dari kamera
- `debug_cropped.jpg`: Crop remote setelah YOLO rotation
- `debug_current_guided.jpg`: Crop remote + overlay tombol + jempol
- `debug_clean_reference.jpg`: Referensi layout bersih
- `debug_indexed_reference.jpg`: Referensi layout dengan indeks b1..bN

### Tips debugging:
- Cek `layout.json` untuk melihat hasil mapping tombol
- Cek `tts_cache/cache_index.json` untuk status cache
- Error di console server biasanya diawali `[Warning]`, `[Error]`, atau `[TTS]`
- Lihat output `[DEBUG VISION]` untuk info task aktif dan deteksi jempol
- Hold-to-speak log: `[Hold] MULAI/BERHENTI/memproses audio...`

### Log terstruktur & perekaman data:
- `vision_reasoning.py` memanggil `_log(...)` untuk mencatat keputusan rule/VLM (mis. `Rule: Confirm match`, `VLM: intent=...`) ke frontend. Pesan ini tampil di chat HP **dan** ikut direkam log daemon.
- Jalankan `python src/log_daemon.py` untuk merekam percakapan, waktu respons (STT/reasoning/total), dan status sistem ke `logs/*.csv` — berguna untuk analisis performa & evaluasi. Lihat [ARSITEKTUR_SISTEM.md §4.10](ARSITEKTUR_SISTEM.md#410-log_daemonpy-perekam-data).

## Kontribusi

### Pull Request Process

1. Fork repo dan buat branch dari `main`
2. Ikuti konvensi kode di atas
3. Test perubahan secara manual
4. Update dokumentasi jika perlu
5. Buat PR dengan deskripsi jelas

### Area yang Butuh Perbaikan

**Critical:**
- Path traversal protection (sudah diimplementasi, perlu review)
- Crash `None.json()` di VLM response (sudah ada fallback, perlu review)

**Medium:**
- Tambah input validation untuk JSON endpoints
- Ganti thread-per-call dengan thread pool (sudah dilakukan di `vision_http.py`)
- Cache eviction di `tts_cache.py`
- Setup test suite formal

**Low:**
- Tambah `unload_models()` untuk GPU memory management
- Tambah graceful shutdown handler yang lebih baik
- Reduksi resolusi video (dari 1920x1080 ke 1280x720)

### Coding Standards

- Python: Ikuti PEP 8: gunakan `black` untuk formatting, `ruff` untuk linting
- JavaScript: Ikuti standard ES2020
- HTML: Valid HTML5, semantic elements
- Commit messages: Indonesian atau English, prefixed dengan `[module]` jika relevan
