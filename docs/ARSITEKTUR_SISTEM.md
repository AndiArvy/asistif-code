# Dokumentasi Sistem Asistif Remote AC untuk Tunanetra

## Daftar Isi
1. [Gambaran Umum Sistem](#1-gambaran-umum-sistem)
2. [Arsitektur & Alur Data](#2-arsitektur--alur-data)
3. [Struktur File](#3-struktur-file)
4. [Deskripsi & Alur Setiap Modul](#4-deskripsi--alur-setiap-modul)
   - [4.1 server_vision.py](#41-server_visionpy---server-utama-webrtc)
   - [4.2 index.html](#42-indexhtml---frontend-hp)
   - [4.3 audio_inference.py](#43-audio_inferencepy---pemrosesan-suara)
   - [4.4 vision_reasoning.py](#44-vision_reasoningpy---logika-vlm-dan-navigasi)
   - [4.5 rotate_remote.py](#45-rotate_remotepy---rotasi-gambar-remote)
   - [4.6 vision_models.py](#46-vision_modelspy---manajemen-model-ai)
   - [4.7 vision_http.py](#47-vision_httppy---utilitas-http)
   - [4.8 tts_cache.py](#48-tts_cachepy---sistem-cache-tts)
   - [4.9 hybrid_inference.py](#49-hybrid_inferencepy---alternatif-entri-point)
5. [Alur Proses Lengkap](#5-alur-proses-lengkap)
6. [Diagram Alur](#6-diagram-alur)

---

## 1. Gambaran Umum Sistem

Sistem ini adalah **asisten berbasis AI untuk penyandang tunanetra** yang membantu mengoperasikan **remote AC** melalui:
- **Perintah suara** (didengar via mic HP → Whisper STT)
- **Navigasi taktil** (panduan arah sentuhan jempol ke tombol remote)
- **Umpan balik suara** (gTTS + Web Speech API offline)
- **Auto-konfirmasi** (background monitor mendeteksi sentuhan tanpa perlu bertanya)
- **Kontrol senter HP** via perintah suara

**Komponen Utama:**
| Komponen | Teknologi | Peran |
|----------|-----------|-------|
| Server WebRTC | Python (aiohttp, aiortc) | Menjembatani HP dengan PC |
| Frontend HP | HTML/JS + WebRTC | Streaming video/audio dari HP, Web Speech API |
| Speech-to-Text | Faster-Whisper | Konversi suara ke teks |
| Vision AI | YOLOv26n OBB + OWL-ViT + LLM | Deteksi remote, tombol, dan jempol |
| Text-to-Speech | Google gTTS + Web Speech API | Umpan balik suara ke pengguna |

**Aliran Data Utama:**
```
HP (Kamera + Mic)
     ↓ WebRTC (video + audio)
Server WebRTC (server_vision.py)
     ├── → HTTP Snapshot → vision_reasoning.py (YOLO + OWL + LLM)
     ├── → WebSocket Audio → audio_inference.py (Whisper STT)
     └── → WebSocket / HTTP ← TTS + Log + Perintah
          ↓ WebSocket / HTTP Response
     HP (Speaker + Chat UI)
```

---

## 2. Arsitektur & Alur Data

Sistem menggunakan **arsitektur Client-Server** dengan dua entitas utama:

### Server (PC)
- **Port 8080**: Server WebRTC utama
- **Port 1234**: LM Studio (Qwen3.5 9B Vision lokal)
- Menjalankan: `server_vision.py` + `audio_inference.py`
- Model AI: YOLOv26n OBB (`best.pt`), OWL-ViT, Faster-Whisper, Qwen3.5 9B (LM Studio)

### Client (HP)
- Browser membuka `http://<IP_PC>:8080`
- Mengirim video & audio via WebRTC
- Menerima suara TTS & log chat via WebSocket
- Dapat menggunakan Web Speech API (TTS offline) jika browser mendukung

### Alur Data Lengkap:

```
[HP]
  ├── Kamera → WebRTC Video → server_vision.py → latest_jpeg (global)
  │                                                    ↓
  │                                          /snapshot endpoint
  │                                                    ↓
  │                                          vision_reasoning.py (YOLO + OWL)
  │
  ├── Mic → WebRTC Audio → server_vision.py → AudioResampler → 16kHz mono s16
  │                                                              ↓
  │                                                   /audio_feed WebSocket
  │                                                              ↓
  │                                                   audio_inference.py (Whisper)
  │                                                              ↓
  │                                                   Teks → process_vlm_reasoning()
  │                                                              ↓
  │                                                   vision_reasoning.py (LLM)
  │                                                              ↓
  │                                          /trigger_tts → gTTS → base64
  │                                          /trigger_speak → Web Speech API (teks)
  │                                                              ↓
  └── Speaker ← WebSocket ← /frontend_ws ← {type:"audio"/"speak", text}
```

---

## 3. Struktur File

```
code2/
├── src/
│   ├── server_vision.py        # [INTI] Server WebRTC, HTTP API, TTS, WebSocket
│   ├── audio_inference.py      # [INTI] STT Whisper + Trigger VLM + Hold-to-Speak
│   ├── vision_reasoning.py     # [INTI] Vision pipeline, VLM reasoning, task monitor
│   ├── index.html              # Frontend UI HP (WebRTC, Hold-to-Speak, Web Speech)
│   ├── rotate_remote.py        # Rotasi gambar remote via YOLO OBB
│   ├── vision_models.py        # Lazy-loading model (YOLO + OWL) thread-safe singleton
│   ├── vision_http.py          # HTTP utilities (LM Studio, snapshot, TTS, ThreadPool)
│   ├── tts_cache.py            # Cache TTS lokal dengan MD5 index
│   └── hybrid_inference.py     # [ALTERNATIF] Input terminal + tap layar
├── best.pt                     # Model YOLOv26n OBB (remote + jempol detection)
├── tts_cache/                  # Folder cache suara TTS
│   └── cache_index.json        # Index mapping hash → filename
├── layout.json                 # Hasil mapping tombol remote
└── debug_*.jpg                 # Debug images (jika VISION_DEBUG=true)
```

---

## 4. Deskripsi & Alur Setiap Modul

### 4.1 `server_vision.py`: Server Utama WebRTC

**Tujuan:** Bertindak sebagai jembatan antara HP (WebRTC) dan script AI Python.

**Komponen Endpoint:**

| Fitur | Endpoint | Method | Deskripsi |
|-------|----------|--------|-----------|
| Halaman Utama | `/` | GET | Serve `index.html` |
| WebRTC Signaling | `/offer` | POST | Terima SDP offer dari HP, buat answer |
| Video Stream | `/video_feed` | GET | Streaming MJPEG: event-driven via `asyncio.Event()`, bukan polling busy-loop |
| Audio Stream | `/audio_feed` | WS | WebSocket untuk audio raw ke Python |
| Snapshot | `/snapshot` | GET | Ambil 1 frame JPEG terbaru |
| Chat Log | `/send_log` | POST | Terima log dari script AI, kirim ke HP |
| TTS Trigger | `/trigger_tts` | POST | Generate gTTS audio, kirim base64 ke HP |
| TTS Speak | `/trigger_speak` | POST | Kirim teks TTS via Web Speech API (offline) |
| Frontend WS | `/frontend_ws` | WS | WebSocket komunikasi dengan HP |
| Command Feed | `/command_feed` | WS | WebSocket untuk sinyal hold-to-speak (start/stop) |
| Preload TTS | `/preload_tts` | POST | Pre-generate common TTS phrases |

**Variabel Global:**
- `latest_jpeg`: Frame video terbaru dari HP (byte JPEG)
- `frame_event`: `asyncio.Event()`: sinyal frame baru (event-driven, bukan polling)
- `frontend_clients`: Set koneksi WebSocket ke HP
- `audio_clients`: Set koneksi WebSocket ke audio_inference.py
- `pcs`: Set koneksi RTCPeerConnection aktif
- `command_clients`: Set koneksi WebSocket untuk perintah hold-to-speak
- `webspeech_clients`: Set koneksi WebSocket yang mendukung Web Speech API

**Alur WebRTC:**
1. HP → `/offer` (SDP offer)
2. Server buat `RTCPeerConnection`, tambahkan event handler
3. `@pc.on("track")` → menangkap video & audio dari HP
4. Video → `cv2.imencode` → `latest_jpeg` + trigger `frame_event`
5. Audio → `AudioResampler` (s16, mono, 16kHz) → kirim ke `/audio_feed` clients
6. Server → `/offer` response (SDP answer)

**Alur TTS (Dua Mode):**
1. **gTTS (online):** POST `/trigger_tts` → generate audio via Google → cache lokal → kirim base64 ke HP
2. **Web Speech API (offline):** POST `/trigger_speak` → kirim teks saja ke HP → browser TTS native
3. HP mengirim `capability` saat koneksi untuk memberi tahu mode TTS yang didukung
4. Path traversal dicegah dengan validasi `cache_file` terhadap `cache_dir`

**Startup Preload:**
- `preload_common_tts()` dipanggil otomatis saat startup
- Pre-generate ~20 frasa umum (navigasi, panduan arah, konfirmasi) ke cache

---

### 4.2 `index.html`: Frontend HP

**Tujuan:** UI berbasis web untuk HP yang menampilkan video, kontrol kamera, log chat, dan memutar TTS.

**Fitur:**
- **WebRTC**: Streaming video & audio dari HP ke server
- **Hold-to-Speak**: Tekan & tahan layar untuk bicara, lepas untuk proses:
  - `touchstart` + `pointerdown` → `startHolding()` → mic ON + kirim `hold_action:start_listening`
  - `touchend` + `pointerup` + `touchcancel` → `stopHolding()` → mic OFF + kirim `hold_action:stop_listening`
  - Safety timeout 30 detik jika pointerup tidak pernah sampai
  - `visibilitychange` → stop mic jika pindah app
  - Haptic feedback (vibrate) saat mulai/berhenti
- **Web Speech API**: Deteksi browser support → kirim `capability:{tts_mode:"webspeech"}` ke server
- **Dual TTS Mode:**
  - `{type:"speak"}`: TTS offline via Web Speech API (text-only)
  - `{type:"audio"}`: TTS via gTTS (base64 audio)
- **Kontrol Senter**: Flashlight toggle via tombol atau perintah suara
- **Ganti Kamera**: Switch depan/belakang dengan replaceTrack
- **Log Chat**: Warna: biru (User), hijau (AI), kuning (Sistem)
- **Mic Control**: Otomatis mute saat TTS berbicara
- **Scroll Dicegah**: `touch-action:none`, `position:fixed`

**Fungsi Penting (JavaScript):**
| Fungsi | Peran |
|--------|-------|
| `startHolding()` | Aktifkan mic, kirim start_listening, vibrate |
| `stopHolding()` | Nonaktifkan mic, kirim stop_listening, vibrate |
| `speakOffline()` | TTS via Web Speech API (native browser) |
| `isSpeechSynthesisSupported()` | Deteksi dukungan Web Speech API |
| `canToggleTorch()` | Cek dukungan flashlight |
| `toggleTorch()` | Nyalakan/matikan senter |

---

### 4.3 `audio_inference.py`: Pemrosesan Suara

**Tujuan:** Menerima audio dari mic HP via WebSocket, melakukan Speech-to-Text dengan Faster-Whisper, lalu memicu VLM reasoning.

**Alur Utama (`listen_to_mic`):**
1. Loop utama: konek ke `ws://localhost:8080/audio_feed`
2. Jika `layout_ready` false → break, reconnect loop
3. Menerima chunk audio 16-bit integer
4. Hold-to-Speak: jika `hold_to_speak_active` atau `draining`, buffer audio
5. Saat server kirim `stop_listening`, tunggu 0.4s drain → proses buffer
6. `process_buffered_audio()`:
   - Filter: buffer < 2048 bytes → ignore
   - Konversi int16 → float32
   - Whisper transcribe dengan filter noise (no_speech_prob > 0.5)
   - Filter halusinasi (terima kasih, dll.)
   - Jika valid → `process_vlm_reasoning(text)` di background thread
7. System busy flag mencegah tumpang tindih pemrosesan

**Fungsi Lain:**
| Fungsi | Peran |
|--------|-------|
| `listen_to_commands()` | WebSocket ke `/command_feed` untuk sinyal hold-to-speak + command |
| `auto_scan_layout()` | Background loop deteksi remote otomatis (setiap 3 detik) |
| `ensure_tts_cache_preloaded()` | Preload TTS cache saat startup |
| `wait_until_layout_ready()` | Tahan mic sampai layout siap |
| `send_log_async()` | Kirim log ke server (non-blocking via to_thread) |
| `run_vlm_task()` | Jalankan VLM reasoning di thread terpisah |

**Hold-to-Speak Flow:**
```
touchstart (HP)
  → /frontend_ws → {type:"hold_action", action:"start_listening"}
  → /command_feed → audio_inference.py
  → hold_to_speak_active = True → buffer audio

touchend (HP)
  → /frontend_ws → {type:"hold_action", action:"stop_listening"}
  → /command_feed → audio_inference.py
  → hold_to_speak_active = False, draining = True
  → tunggu 0.4s, draining = False
  → process_buffered_audio() → Whisper STT → VLM
```

---

### 4.4 `vision_reasoning.py`: Logika VLM dan Navigasi

**Tujuan:** Modul paling kompleks. Menangani:
- Setup layout remote (deteksi → krop → indeks tombol → mapping fungsi)
- Deteksi posisi jempol pengguna
- VLM reasoning (LLM Vision) untuk navigasi
- Background monitor untuk auto-konfirmasi sentuhan

#### A. Setup Layout (`auto_setup_layout`)

```
auto_setup_layout(silent=True)
  ├── 1. capture_current_frame() → ambil 1 frame dari server
  ├── 2. process_yolo_rotation() → deteksi & krop remote dengan YOLO OBB
  ├── 3. Stabilisasi: tunggu 1.5 detik, ambil frame lagi
  ├── 4. Validasi ukuran (luas > 60000 piksel)
  ├── 5. Validasi margin (tidak terpotong tepi frame, MARGIN=50px)
  │      ├── Cek 4 sisi: kiri/kanan/atas/bawah
  │      ├── 1 sisi terpotong → "bagian {sisi} terpotong"
  │      └── >1 sisi → "Terlalu dekat, Jauhkan sedikit"
  ├── 6. generate_owl_layout() → deteksi tombol dengan OWL-ViT
  │      ├── Query: "a remote", "number buttons", "individual button", dll.
  │      ├── NMS (IoU threshold 0.5) hapus bounding box duplikat
  │      ├── Filter: tombol dalam area remote, area < 15% total gambar
  │      ├── Sort: kiri-ke-kanan, atas-ke-bawah
  │      └── Output: indexed image + layout dictionary {b1:{coords}}
  ├── 7. map_functions_with_vlm() → LLM mapping fungsi tombol
  │      ├── Kirim 2 gambar (clean + indexed) ke LM Studio
  │      ├── System prompt dengan batasan fungsi dari SYNONYM_GROUPS
  │      ├── Heuristic: tombol besar/oval/rocker → "suhu naik/turun"
  │      └── Output JSON: {b1:"power", b2:"suhu", ...} → simpan layout.json
  ├── 8. _generate_location_descriptions() → grid 3x3 untuk deskripsi posisi
  │      └── Map (col_idx, row_idx) → "pojok kiri atas", "tengah remote", dll.
  └── 9. is_layout_ready = True, speak welcome
```

#### B. Deteksi Jempol (`detect_current_thumb_touch`)

```
detect_current_thumb_touch()
  ├── 1. Capture frame → YOLO rotate → crop remote
  ├── 2. YOLO deteksi jempol (class_id=1) di dalam crop
  ├── 3. Hitung thumb_center (midpoint bounding box)
  ├── 4. Scale koordinat jempol dari crop ke reference (per sumbu x/y)
  ├── 5. Padding adaptif: 15px × max(1.0, aspect_ratio × 0.5)
  ├── 6. Overlay bbox + indeks tombol ke current crop (1 gambar utk LLM)
  ├── 7. Cocokkan dengan bounding box tombol (padding adaptif)
  ├── 8. Return: fungsi disentuh, thumb center, current_drawn, reference_drawn
  └── Debug: simpan ke debug_current_guided.jpg & debug_reference_guided.jpg
```

#### C. VLM Reasoning (`process_vlm_reasoning`)

```
process_vlm_reasoning(user_text)
  ├── 0. Cek kata kunci tanpa VLM:
  │      ├── "senter/flashlight" → TOGGLE_FLASH (via log system)
  │      ├── "reset layout/ulang" → reset is_layout_ready
  ├── 1. Jika layout belum siap → auto_setup_layout(silent=False)
  ├── 2. detect_current_thumb_touch() → fungsi yang disentuh
  ├── 3. Deteksi "apa tombol ini?" (regex patterns):
  │      → "Ini adalah tombol {fungsi}." [tanpa VLM]
  ├── 4. Deteksi konfirmasi (regex patterns + keyword):
  │      "benar/betul/tepat" + "ini/itu" + ada task aktif
  │      → is_target_matched() → jika cocok: konfirmasi & reset task
  │      → jika tidak: "Ini tombol {X}, bukan {Y}, coba raba di {lokasi}"
  ├── 5. Deteksi "dimana tombol X?" (regex patterns):
  │      → Pass 1: exact match nama fungsi di teks
  │      → Pass 2: synonym match
  │      → Pass 3: jika ada task aktif, cari task context
  │      → Jika ditemukan: speak lokasi, set active task [tanpa VLM]
  ├── 6. Deteksi ACTION_INTENTS "nyalakan/atur suhu/ganti mode/dll" (regex):
  │      → Iterasi ACTION_INTENTS: cocokkan pola → target function
  │      → Cari target function di layout via _match_task_texts()
  │      → Jika ditemukan: speak lokasi, set active_task_context, return
  │      → Jika tidak ditemukan: break → fallback ke VLM
  ├── 7. Jika hardcode gagal + ada task aktif → inject petunjuk ke user_text
  ├── 8. Kirim prompt ke LM Studio:
  │      - 1 image: current remote crop (overlay bbox + thumb red dot)
  │      - Available Functions + Previous Task + Current Thumb
  │      - System prompt → 10 aturan (SINGLE IMAGE, LCD reading, dll.)
  ├── 8. Parse response JSON: {intent, updated_task, target_location_desc, instruction}
  ├── 9. Handle intent:
  │      - "question" → baca status layar (suhu, mode, fan)
  │      - "navigation" → set active_task, beri panduan arah
  │      - "unknown" → redirect ke perintah remote AC
  ├── 10. Bersihkan label indeks (b1, b2) dari teks
  ├── 11. Batasi conversation_history (default 20)
  └── 12. Speak response via TTS
```

#### D. Background Task Monitor (`background_task_monitor_loop`)

Thread daemon terpisah yang berjalan terus-menerus (setiap 0.5 detik):
1. Cek jika ada `active_task_context` dengan intent "navigation"
2. Snapshot task & intent (cegah race condition)
3. Deteksi posisi jempol (dengan `vision_processing_lock`)
4. Jika fungsi yang disentuh cocok dengan task → otomatis konfirmasi
5. Tidak perlu menunggu user bertanya "apakah ini tombol yang benar?"

#### E. Matching Fungsi (`is_target_matched` + `_match_task_texts`)

Sistem matching cerdas dengan 3 mekanisme:
1. **Direct match**: sama persis
2. **Synonym groups**: 19 grup sinonim khusus remote AC:
   - power, suhu naik, suhu turun, fan, mode, swing, turbo, eco, sleep, light, timer on/off/naik/turun, set, cancel, clock
3. **Timer guard clause**: mencegah false positive timer vs fitur lain
4. **Index alias**: teks mengandung "b3" → cari fungsi di layout_data

#### F. SYNONYM_GROUPS

19 grup sinonim untuk matching cerdas tanpa VLM:
- power: on/off/nyala/mati/hidup/matikan/nyalakan
- suhu naik: temp up/naikkan/tambah/panas/+/warmer
- suhu turun: temp down/turunkan/kurang/dingin/-/cooler
- fan: kipas/angin/kecepatan/speed/wind
- mode: cool/dry/heat/auto/dingin/kering/otomatis
- swing: a.swing/m.swing/ayun/arah angin/sirip
- turbo: powerful/jet/fast cooling/cepat/max
- eco: economic/hemat/energy saving/irit
- sleep: malam/tidur/quiet/silent/senyap
- light: lampu/display/led/layar
- timer on/off/naik/turun
- set: atur/konfirmasi/ok/simpan
- cancel: batal/batalkan/reset
- clock: jam/waktu sekarang

---

### 4.5 `rotate_remote.py`: Rotasi Gambar Remote

**Tujuan:** Mendeteksi dan merotasi gambar remote agar tegak lurus (portrait) menggunakan YOLO OBB (Oriented Bounding Box).

**Alur:**
1. `process_yolo_rotation(img, model, target_class)` → predict YOLO
2. Jika model support OBB → `_process_with_obb()`:
   - Ekstrak `xywhr` (center, width, height, rotation radian)
   - Hitung diagonal untuk bounding box aman (anti terpotong)
   - Crop area sekitar remote dengan padding jika perlu
   - Rotate ROI berdasarkan sudut deteksi (invert via `YOLO_INVERT_OBB_ANGLE`)
   - **Lock portrait** (via `YOLO_FORCE_PORTRAIT`)
3. Fallback → `_process_with_axis_aligned()`:
   - Crop bounding box axis-aligned
   - Lock portrait jika diaktifkan
4. Return: list of dicts dengan `image`, `angle`, `raw_boxes`, `index`

**Konfigurasi (Environment Variable):**
| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `YOLO_CONF_THRESHOLD` | `0.3` | Confidence threshold deteksi |
| `YOLO_FORCE_PORTRAIT` | `true` | Paksa output portrait |
| `YOLO_INVERT_OBB_ANGLE` | `false` | Balik arah rotasi OBB |

---

### 4.6 `vision_models.py`: Manajemen Model AI

**Tujuan:** Lazy-loading singleton thread-safe untuk model-model AI berat.

**Model yang dimuat:**
| Model | Tipe | Fungsi |
|-------|------|--------|
| YOLOv26n OBB `best.pt` | Ultralytics (custom OBB) | Deteksi remote (class 0) & jempol (class 1) |
| OWL-ViT Processor | `owlv2-base-patch16-ensemble` | Preprocessing untuk OWL |
| OWL-ViT Model | `owlv2-base-patch16-ensemble` | Zero-shot object detection tombol |

**Pattern:** Double-checked locking dengan `threading.Lock()` → model dimuat sekali, aman dari race condition.

**Konfigurasi:**
| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `YOLO_MODEL_PATH` | `best.pt` | Path model YOLO |
| `OWL_MODEL_NAME` | `google/owlv2-base-patch16-ensemble` | Model OWL-ViT dari HuggingFace |

---

### 4.7 `vision_http.py`: Utilitas HTTP

**Tujuan:** Menyediakan fungsi-fungsi HTTP untuk komunikasi antar komponen dengan ThreadPoolExecutor.

**Konstanta URL:**
| URL | Port | Fungsi |
|-----|------|--------|
| `LM_STUDIO_URL` | `1234` | LLM Vision API chat completions |
| `SNAPSHOT_URL` | `8080` | Ambil frame dari server |
| `TTS_TRIGGER_URL` | `8080` | Trigger TTS |
| `LOG_URL` | `8080` | Kirim log ke frontend |

**ThreadPool:** 4 workers untuk `fire_and_forget_post()`: mencegah thread leak.

**Fungsi Utama:**
| Fungsi | Deskripsi |
|--------|-----------|
| `safe_post(url)` | POST dengan error handling, timeout configurable |
| `fire_and_forget_post(url)` | POST di ThreadPool (non-blocking, 4 workers) |
| `capture_current_frame()` | GET /snapshot → decode JPEG → numpy array |
| `cv2_to_base64(image)` | Resize letterbox 360×640 → encode PNG → base64 |
| `extract_json_object(text)` | Parse JSON dari teks (handle markdown, nested braces) |
| `speak(text)` | Cek cache TTS → trigger TTS via HTTP (fire-and-forget) |
| `log_system(text)` | Kirim log sistem ke frontend |

---

### 4.8 `tts_cache.py`: Sistem Cache TTS

**Tujuan:** Menyimpan file audio TTS secara lokal dengan thread-safe singleton pattern.

**Struktur Cache:**
```
tts_cache/
├── cache_index.json     # Index mapping hash → {text, filename, hash}
├── {md5_hash1}.mp3      # File audio cache
├── {md5_hash2}.mp3
└── ...
```

**Metode:**
| Method | Deskripsi |
|--------|-----------|
| `__init__(cache_dir)` | Buat folder cache, load index dari JSON |
| `_text_to_hash(text)` | MD5 hash dari teks (lowercase, stripped) |
| `get_cached_file(text)` | Cek cache, return path atau None |
| `add_to_cache(text, file_path)` | Tambah entry ke cache index + save |
| `get_cache_info()` | Statistik cache (jumlah entry, file, ukuran MB) |
| `hash_text(text)` | Public wrapper untuk hash text |

**Thread Safety:** `threading.Lock` untuk akses cache index.

---

### 4.9 `hybrid_inference.py`: Alternatif Entry Point

**Tujuan:** Entry point alternatif untuk testing tanpa HP (menggunakan input terminal + tap layar).

**Alur:**
1. `listen_to_tap_commands()`: WebSocket ke `/command_feed` (terima tap)
2. `manual_input_loop()`: Input dari terminal (asyncio loop.run_in_executor)
3. `auto_scan_layout()`: Background scan remote (setiap 2 detik)
4. Ketiga fungsi berjalan parallel via `asyncio.gather()`

**Cocok untuk:** Debugging, testing tanpa HP, atau situasi di mana mic tidak tersedia.

---

## 5. Alur Proses Lengkap

### Siklus Hidup Sistem

```
STARTUP
  ├── server_vision.py (Port 8080)
  │     └── Preload common TTS phrases (~20 frasa)
  └── audio_inference.py
        ├── Load Whisper model (CUDA/CPU)
        ├── ensure_tts_cache_preloaded() → trigger preload
        ├── start_background_task_monitor() → start thread daemon
        └── Mulai 3 async loops:
              ├── listen_to_mic()          → Hold-to-Speak + STT
              ├── listen_to_commands()     → Hold signals + Tap
              └── auto_scan_layout()       → Deteksi otomatis (3s)

HP TERKONEKSI
  ├── Buka browser → http://<PC_IP>:8080
  ├── WebRTC negotiation → video & audio streaming mulai
  ├── Kirim capability (Web Speech API?)
  ├── TTS Welcome: "Letakkan remote di depan kamera..."
  └── server_vision.py menerima video frames → latest_jpeg update

AUTO LAYOUT DETECTION
  └── auto_scan_layout() → setiap 3 detik:
        ├── capture_current_frame()
        ├── YOLO detect remote?
        │     ├── Ya → auto_setup_layout(silent=True)
        │     │        ├── Stabilisasi 1.5 detik
        │     │        ├── Validasi ukuran (>60000 px)
        │     │        ├── Validasi margin (MARGIN=50px)
        │     │        ├── generate_owl_layout() → NMS → indeks tombol
        │     │        └── map_functions_with_vlm() → fungsi + lokasi
        │     └── Tidak → tunggu 3 detik lagi
        └── is_layout_ready = True

USER HOLDS TO SPEAK (Contoh: "Nyalakan AC")
  ├── 1. HP touchstart → vibrate 100ms + beep 400→800Hz → hold_action:start_listening → audio buffer ON
  ├── 2. User bicara "nyalakan AC"
  ├── 3. HP touchend → vibrate [30,30,30] + beep 600→300Hz → hold_action:stop_listening → buffer OFF
  ├── 4. audio_inference.py:
  │      ├── Tunggu 0.4s drain
  │      ├── Whisper STT → "nyalakan AC"
  │      └── Filter noise & halusinasi → valid
  ├── 5. system_is_busy = True
  ├── 6. process_vlm_reasoning("nyalakan AC"):
  │      ├── Cek "senter" / "reset layout" → handle tanpa VLM
  │      ├── Cek "apa tombol ini?" → handle tanpa VLM
  │      ├── Cek konfirmasi → handle tanpa VLM
  │      ├── Cek "dimana tombol" → hardcode lookup (3 pass)
  │      ├── Cek ACTION_INTENTS → regex match "nyalakan AC" → target="power"
  │      │     └── Cari "power" di layout_data → ketemu → speak lokasi + set task + return
  │      ├── (Jika tombol tidak ada di layout) detect_current_thumb_touch() → deteksi jempol
  │      ├── Kirim prompt ke LM Studio (1 gambar + user teks)
  │      ├── VLM response: {intent, updated_task, instruction}
  │      └── Speak: "Tombol power di kanan atas. Geser ke kanan..."
  └── 7. system_is_busy = False

USER MENGGERAKKAN JEMPOL
  ├── Background task monitor (setiap 0.5 detik):
  │      ├── Snapshot task & intent (thread-safe)
  │      ├── Deteksi jempol → fungsi "power"
  │      ├── is_target_matched("power", active_task="power") → TRUE
  │      └── Speak: "Nah, yang itu tombolnya." + reset task
  └── Mic aktif kembali → siap perintah baru

USER TEKAN TOMBOL & BERTANYA "Apakah ini tombol power?"
  ├── 1. Hold-to-speak → Whisper STT → "apakah ini tombol power"
  ├── 2. process_vlm_reasoning():
  │      ├── Deteksi konfirmasi → regex match
  │      ├── is_target_matched("power", active_task="power") → TRUE
  │      └── Speak: "Iya, benar. Ini tombol yang tepat. Silakan tekan."
  └── Reset active_task → DEFAULT

USER MINTA RESET
  └── "reset layout" atau "ganti remote"
        ├── is_layout_ready = False
        ├── active_task_context = DEFAULT
        └── Speak: "Layout direset. Letakkan remote di depan kamera."
```

---

## 6. Diagram Alur

### 6.1 Diagram Alur Data

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HANDHPHONE (Browser)                            │
│                                                                          │
│  ┌──────────┐    WebRTC     ┌────────────────────────────────────────┐  │
│  │  Camera  │─────────────▶ │  server_vision.py (Port 8080)         │  │
│  │  + Mic   │               │                                        │  │
│  └──────────┘               │  latest_jpeg (global)                  │  │
│       │                      │  AudioResampler → 16kHz mono          │  │
│       │                      │  frame_event (asyncio.Event)          │  │
│       │                      │                                        │  │
│       │              ┌───────┴──────────────┐                        │  │
│       │              │                      │                        │  │
│       │              ▼                      ▼                        │  │
│       │    ┌─────────────────┐   ┌──────────────────┐                │  │
│       │    │  /snapshot GET  │   │ /audio_feed WS   │                │  │
│       │    └────────┬────────┘   └────────┬─────────┘                │  │
│       │             │                     │                          │  │
│       │             ▼                     ▼                          │  │
│       │    ┌─────────────────┐   ┌──────────────────┐                │  │
│       │    │ vision_reasoning│   │ audio_inference  │                │  │
│       │    │ .py             │   │ .py (Whisper)    │                │  │
│       │    │                 │   │                  │                │  │
│       │    │ YOLO detect     │   │ Hold-to-Speak    │                │  │
│       │    │ OWL-ViT index   │   │ VAD filter       │                │  │
│       │    │ VLM (LM Studio) │   │ Trigger VLM      │                │  │
│       │    │ Background Mon  │   │                  │                │  │
│       │    └────────┬────────┘   └──────────────────┘                │  │
│       │             │                                                │  │
│       │             ▼                                                │  │
│       │    ┌──────────────────────────┐                              │  │
│       │    │  /trigger_tts (gTTS)     │                              │  │
│       │    │  /trigger_speak (WebSpch)│                              │  │
│       │    └────────┬─────────────────┘                              │  │
│       │             │                                                │  │
│       │             ▼                                                │  │
│       │    ┌─────────────────────────────────────┐                    │  │
│       │    │  WebSocket /frontend_ws             │                    │  │
│       │    │  {type:"audio", audio:base64}       │                    │  │
│       │    │  {type:"speak", text, lang}         │                    │  │
│       │    │  {type:"log", sender, text}         │                    │  │
│       │    │  {type:"Control", text:"TOGGLE_FLSH"}│                   │  │
│       │    └─────────────────────────────────────┘                    │  │
│       │                     │                                         │  │
│       │                     ▼                                         │  │
│  ┌────┴────────────┐                                                  │  │
│  │  Speaker + Log  │                                                  │  │
│  │  Chat UI        │                                                  │  │
│  └─────────────────┘                                                  │  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Diagram State Machine

```
                ┌──────────┐
                │ STARTUP  │
                └────┬─────┘
                     │
                     ▼
              ┌──────────────┐
         ┌───▶│ WAITING FOR  │
         │    │ CONNECTION   │
         │    └──────┬───────┘
         │           │ HP connects
         │           ▼
         │    ┌──────────────┐
         │    │  STREAMING   │◀──────────────┐
         │    │  (video/audio)               │
         │    └──────┬───────┘               │
         │           │                       │
         │           ▼                       │
         │    ┌──────────────┐               │
         │    │  LAYOUT NOT  │───auto────┐   │
         │    │  READY       │  scan 3s  │   │
         │    └──────┬───────┘           │   │
         │           │                   ▼   │
         │           │           ┌──────────────────┐
         │           │           │ AUTO SETUP       │
         │           │           │ LAYOUT           │
         │           │           │ (YOLO+OWL+VLM)   │
         │           │           └────────┬─────────┘
         │           │                    │ success
         │           ▼                    ▼
         │    ┌──────────────────────────────┐
         │    │        LAYOUT READY          │
         │    │  ┌──────────────────────┐    │
         │    │  │ Background Monitor   │    │
         │    │  │ (0.5s check thumb)   │    │
         │    │  └──────────────────────┘    │
         │    └──────────────┬───────────────┘
         │                   │
         │           ┌───────┴────────────────┐
         │           │                        │
         │           ▼                        ▼
         │    ┌──────────────┐    ┌──────────────────┐
         │    │ HOLD-TO-SPEAK│    │ TAP LAYAR /      │
         │    │ (touchstart) │    │ MANUAL INPUT     │
         │    └──────┬───────┘    └────────┬─────────┘
         │           │                     │
         │           ▼                     │
         │    ┌──────────────┐             │
         │    │ Mic ON       │             │
         │    │ + audio buf  │             │
         │    └──────┬───────┘             │
         │           │ touchend            │
         │           ▼                     │
         │    ┌──────────────┐             │
         │    │ Whisper STT  │             │
         │    └──────┬───────┘             │
         │           │ text                │ text
         │           ▼                     ▼
         │    ┌──────────────────────────────────────┐
         │    │   PROCESS VLM REASONING              │
         │    │                                      │
         │    │  ┌──────────────────────────┐        │
         │    │  │ Tanpa VLM (Hardcode):    │        │
         │    │  │  Senter / Reset Layout   │        │
         │    │  │  "Apa tombol ini?"       │        │
         │    │  │  Konfirmasi task         │        │
         │    │  │  "Dimana tombol X?" / "Nyalakan AC"      │        │
         │    │  └──────────────────────────┘        │
         │    │                                      │
         │    │  ┌──────────────────────────┐        │
         │    │  │ Dengan VLM (LM Studio):  │        │
         │    │  │  intent: navigation      │        │
         │    │  │  → set active_task       │───┼───▶│ Bg Monitor
         │    │  │  → speak guidance        │        │
         │    │  ├──────────────────────────┤        │
         │    │  │  intent: confirmation    │        │
         │    │  │  → match dengan task     │        │
         │    │  │  → confirm/deny          │        │
         │    │  ├──────────────────────────┤        │
         │    │  │  intent: question        │        │
         │    │  │  → baca status layar     │        │
         │    │  └──────────────────────────┘        │
         │    └──────────────┬───────────────────────┘
         │                   │
         │                   ▼
         │    ┌───────────────────────────┐
         │    │   SPEAK (TTS)             │
         │    │   gTTS / Web Speech API   │
         │    │   + Log ke Frontend       │
         │    └──────────┬────────────────┘
         │               │
         └───────────────┘
                         │ "reset layout"
                         ▼
                ┌──────────────────┐
                │   LAYOUT RESET   │
                │   is_layout_ready│
                │   = False        │
                └──────┬───────────┘
                       │
                       └──────────────────→ WAITING FOR CONNECTION
```

---

## 7. Threading & Concurrency Model

```
Main Thread (asyncio event loop):
├── server_vision.py: aiohttp server
│   ├── WebRTC signaling & streaming
│   ├── HTTP endpoints (/snapshot, /trigger_tts, dll.)
│   └── WebSocket endpoints (/frontend_ws, /audio_feed, /command_feed)
│
└── audio_inference.py: 3 async loops via asyncio.gather()
    ├── listen_to_mic()         → WebSocket audio → Whisper STT
    ├── listen_to_commands()    → WebSocket command → VLM trigger
    └── auto_scan_layout()      → periodic YOLO scan (tiap 3 detik)

Background Thread (daemon):
└── background_task_monitor_loop()
    └── Setiap 500ms: deteksi jempol → auto-confirm (dengan vision_processing_lock)

ThreadPool (vision_http.py):
└── ThreadPoolExecutor(max_workers=4)
    └── fire_and_forget_post(): HTTP non-blocking untuk TTS, log, dll.

Thread Safety:
- vision_processing_lock (threading.Lock) → protect model inference
- _MODEL_LOCK (threading.Lock) → singleton model loading
- _tts_cache_lock (threading.Lock) → singleton TTS cache
- tts_cache.lock (threading.Lock) → cache index read/write
```

---

## Ringkasan Komponen

| Modul | Bahasa | Baris | Fungsi Utama |
|-------|--------|-------|--------------|
| `server_vision.py` | Python | ~550 | WebRTC server, HTTP API, TTS delivery (gTTS + Web Speech), graceful shutdown, preload cache |
| `index.html` | HTML/JS | ~618 | Frontend HP WebRTC, hold-to-speak (touch+pointer), Web Speech API, flashlight, haptic feedback, beep sound effects |
| `audio_inference.py` | Python | ~257 | STT Whisper, hold-to-speak buffer, filter noise/halusinasi, VLM trigger, background monitor start |
| `vision_reasoning.py` | Python | ~1206 | Layout setup (YOLO+OWL+VLM), thumb detection, VLM reasoning, background monitor, synonym matching, hardcode action intents |
| `rotate_remote.py` | Python | ~183 | YOLO OBB rotation & cropping (OBB + axis-aligned fallback), configurable portrait lock |
| `vision_models.py` | Python | ~48 | Lazy-loading singleton YOLO + OWL-ViT (thread-safe double-checked locking) |
| `vision_http.py` | Python | ~121 | HTTP utilities, ThreadPool (4 workers), cv2_to_base64 letterbox, extract_json_object |
| `tts_cache.py` | Python | ~97 | TTS audio caching, MD5 hash index, thread-safe, cache info/statistics |
| `hybrid_inference.py` | Python | ~87 | Alternative entry (terminal + tap), no mic needed |

---

## Cara Menjalankan

1. **Jalankan LM Studio** di port 1234 dengan Qwen3.5 9B (model vision)
2. **Jalankan server utama:**
   ```bash
   python src/server_vision.py
   ```
3. **Jalankan audio inference:**
   ```bash
   python src/audio_inference.py
   ```
4. **Buka HP ke:** `http://<IP_PC>:8080`
5. Atau **mode hybrid (tanpa HP):**
   ```bash
   python src/hybrid_inference.py
   ```
