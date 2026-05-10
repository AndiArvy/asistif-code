# Dokumentasi Sistem Asistif Remote AC untuk Tunanetra

## 📋 Daftar Isi
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
   - [4.10 layout_OWL.py](#410-layout_owlpy---skrip-standalone-deprecated)
5. [Alur Proses Lengkap](#5-alur-proses-lengkap)
6. [Diagram Alur](#6-diagram-alur)

---

## 1. Gambaran Umum Sistem

Sistem ini adalah **asisten berbasis AI untuk penyandang tunanetra** yang membantu mengoperasikan **remote AC** melalui:
- **Perintah suara** (didengar via mic HP → Whisper STT)
- **Navigasi taktil** (panduan arah sentuhan jempol ke tombol remote)
- **Umpan balik suara** (TTS via Google gTTS)

**Komponen Utama:**
| Komponen | Teknologi | Peran |
|----------|-----------|-------|
| Server WebRTC | Python (aiohttp, aiortc) | Menjembatani HP dengan PC |
| Frontend HP | HTML/JS + WebRTC | Streaming video/audio dari HP |
| Speech-to-Text | Faster-Whisper | Konversi suara ke teks |
| Vision AI | YOLOv8 + OWL-ViT + LLM | Deteksi remote, tombol, dan jempol |
| Text-to-Speech | Google gTTS | Umpan balik suara ke pengguna |

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

### 🔵 Server (PC)
- **Port 8080** - Server WebRTC utama
- **Port 1234** - LM Studio (LLM Vision lokal)
- Menjalankan: `server_vision.py` + `audio_inference.py`
- Model AI: YOLOv8 (`best.pt`), OWL-ViT, Faster-Whisper, LLM

### 🟢 Client (HP)
- Browser membuka `http://<IP_PC>:8080`
- Mengirim video & audio via WebRTC
- Menerima suara TTS & log chat via WebSocket

### 🔄 Alur Data Lengkap:

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
  │                                                              ↓
  └── Speaker ← WebSocket ← /frontend_ws ← {type:"audio", audio:"base64..."}
```

---

## 3. Struktur File

```
code2/
├── src/
│   ├── server_vision.py        # [INTI] Server WebRTC, HTTP API, TTS
│   ├── audio_inference.py      # [INTI] STT Whisper + Trigger VLM
│   ├── vision_reasoning.py     # [INTI] Vision pipeline + VLM reasoning
│   ├── index.html              # Frontend UI HP
│   ├── rotate_remote.py        # Rotasi gambar remote via YOLO OBB
│   ├── vision_models.py        # Lazy-loading model (YOLO + OWL)
│   ├── vision_http.py          # HTTP utilities (LM Studio, snapshot, TTS)
│   ├── tts_cache.py            # Cache TTS lokal
│   ├── hybrid_inference.py     # [ALTERNATIF] Input terminal + tap
│   └── layout_OWL.py           # [DEPRECATED] Skrip standalone OWL
├── best.pt                     # Model YOLOv8 (remote + jempol detection)
├── tts_cache/                  # Folder cache suara TTS
├── layout.json                 # Hasil mapping tombol remote
└── debug_*.jpg                 # Debug images
```

---

## 4. Deskripsi & Alur Setiap Modul

### 4.1 `server_vision.py` - Server Utama WebRTC

**Tujuan:** Bertindak sebagai jembatan antara HP (WebRTC) dan script AI Python.

**Komponen:**

| Fitur | Endpoint | Method | Deskripsi |
|-------|----------|--------|-----------|
| Halaman Utama | `/` | GET | Serve `index.html` |
| WebRTC Signaling | `/offer` | POST | Terima SDP offer dari HP, buat answer |
| Video Stream | `/video_feed` | GET | Streaming MJPEG dari frame terbaru |
| Audio Stream | `/audio_feed` | WS | WebSocket untuk audio raw ke Python |
| Snapshot | `/snapshot` | GET | Ambil 1 frame JPEG terbaru |
| Chat Log | `/send_log` | POST | Terima log dari script AI, kirim ke HP |
| TTS Trigger | `/trigger_tts` | POST | Generate TTS, kirim audio ke HP |
| Frontend WS | `/frontend_ws` | WS | WebSocket komunikasi dengan HP |
| Command Feed | `/command_feed` | WS | WebSocket untuk perintah tap layar |
| Preload TTS | `/preload_tts` | POST | Pre-generate common TTS phrases |

**Variabel Global:**
- `latest_jpeg` - Frame video terbaru dari HP (byte JPEG)
- `frontend_clients` - Set koneksi WebSocket ke HP
- `audio_clients` - Set koneksi WebSocket ke audio_inference.py
- `pcs` - Set koneksi RTCPeerConnection aktif
- `command_clients` - Set koneksi WebSocket untuk perintah tap

**Alur WebRTC:**
1. HP → `/offer` (SDP offer)
2. Server buat `RTCPeerConnection`, tambahkan event handler
3. `@pc.on("track")` → menangkap video & audio dari HP
4. Video → `cv2.imencode` → `latest_jpeg`
5. Audio → `AudioResampler` (s16, mono, 16kHz) → kirim ke `/audio_feed` clients
6. Server → `/offer` response (SDP answer)

**Alur TTS:**
1. Request POST ke `/trigger_tts` dengan `{text, use_cache, cache_file}`
2. Cek cache TTS (jika `use_cache=True` dan `cache_file` ada)
3. Jika tidak ada cache → generate via `gTTS` (Google Text-to-Speech)
4. Simpan hasil ke cache lokal
5. Kirim `{type: "audio", audio: base64, text, playback_rate}` ke semua `frontend_clients`
6. HP memutar audio tersebut

---

### 4.2 `index.html` - Frontend HP

**Tujuan:** UI berbasis web untuk HP yang menampilkan video, kontrol kamera, log chat, dan memutar TTS.

**Fitur:**
- **WebRTC** - Streaming video & audio dari HP ke server
- **UI Taktil** - Tombol besar, warna kontras untuk tunanetra sebagian
- **Log Chat** - Menampilkan percakapan User ↔ AI dalam kode warna:
  - 🔵 Biru: User message
  - 🟢 Hijau: AI/TTS response
  - 🟡 Kuning: System messages
- **Mic Control** - Otomatis mute saat TTS berbicara, unmute setelah selesai
- **Ganti Kamera** - Switch antara kamera depan/belakang
- **Flashlight** - Kontrol senter HP

**Alur Frontend:**
1. `window.onload` → `start()` → minta izin kamera & mic
2. Buat `RTCPeerConnection`, add tracks audio & video
3. `createOffer()` → POST `/offer` → dapat `answer`
4. Set `remoteDescription` dengan answer
5. Buka WebSocket ke `/frontend_ws`
6. Terima pesan:
   - `{type:"audio"}` → decode base64 → mainkan via `<audio>` element
   - `{type:"log"}` → tampilkan di chat log
7. Saat TTS player `onended`/`onerror` → `handleTtsEnd()` → unmute mic

**Fungsi Penting (JavaScript):**
| Fungsi | Peran |
|--------|-------|
| `setMicEnabled()` | Mute/unmute mic track |
| `handleTtsStart/End()` | Track jumlah TTS yang diputar, kontrol mic |
| `canToggleTorch()` | Cek apakah perangkat mendukung torch |
| `toggleTorch()` | Nyalakan/matikan senter |
| `switchCamera()` | Ganti kamera depan/belakang |

---

### 4.3 `audio_inference.py` - Pemrosesan Suara

**Tujuan:** Menerima audio dari mic HP via WebSocket, melakukan Speech-to-Text dengan Faster-Whisper, lalu memicu VLM reasoning.

**Alur Utama (`listen_to_mic`):**
1. Tunggu `is_layout_ready == True` (layout remote sudah dipetakan)
2. Konek ke `ws://localhost:8080/audio_feed` (stream audio dari HP)
3. **Voice Activity Detection (VAD):**
   - Buffer audio 16-bit integer
   - Deteksi suara: jika `volume > SILENCE_THRESHOLD` (default 3000)
   - Jika diam lebih dari `SILENCE_CHUNKS_LIMIT` (20 chunk → ~silence threshold)
   - Kirim buffer ke Whisper
4. **Whisper Inference:**
   - Model: `faster-whisper-large-v3-turbo-ct2`
   - Bahasa: Indonesia (`language="id"`)
   - Filter noise: `no_speech_prob > 0.6` → abaikan
   - Filter hallucination: daftar kata seperti "terima kasih", dll.
5. **Trigger VLM:** Jika perintah valid, jalankan `process_vlm_reasoning(text_result)` di background thread
6. **System Busy Flag:** `system_is_busy` mencegah tumpang tindih pemrosesan

**Fungsi Lain:**
| Fungsi | Peran |
|--------|-------|
| `listen_to_commands()` | WebSocket ke `/command_feed` untuk sinyal tap layar |
| `auto_scan_layout()` | Background loop deteksi remote otomatis (setiap 2 detik) |
| `ensure_tts_cache_preloaded()` | Preload TTS cache saat startup |
| `wait_until_layout_ready()` | Tahan mic sampai layout siap |
| `send_log_async()` | Kirim log ke server (non-blocking) |

---

### 4.4 `vision_reasoning.py` - Logika VLM dan Navigasi

**Tujuan:** Modul paling kompleks. Menangani:
- Setup layout remote (deteksi → krop → indeks tombol → mapping fungsi)
- Deteksi posisi jempol pengguna
- VLM reasoning (LLM Vision) untuk navigasi
- Background monitor untuk auto-deteksi sentuhan

#### A. Setup Layout (`auto_setup_layout`)

```
auto_setup_layout(silent=True)
  ├── 1. capture_current_frame() → ambil 1 frame dari server
  ├── 2. process_yolo_rotation() → deteksi & krop remote dengan YOLO OBB
  ├── 3. Stabilisasi: tunggu 2 detik, ambil frame lagi
  ├── 4. Validasi ukuran (luas > 40000 piksel)
  ├── 5. Validasi margin (tidak terpotong tepi frame)
  ├── 6. generate_owl_layout() → deteksi tombol dengan OWL-ViT
  │      ├── Query teks: "a remote", "number buttons", dll.
  │      ├── Filter: tombol dalam area remote, area < 15% total gambar
  │      └── Output: indexed image + layout dictionary {b1:{coords}}
  ├── 7. map_functions_with_vlm() → LLM mapping fungsi tombol
  │      ├── Kirim 2 gambar (clean + indexed) ke LM Studio
  │      ├── Output JSON: {b1:"power", b2:"suhu", b3:"tidak diketahui"}
  │      └── Simpan ke layout.json
  └── 8. is_layout_ready = True
```

#### B. Deteksi Jempol (`detect_current_thumb_touch`)

```
detect_current_thumb_touch()
  ├── 1. Capture frame → YOLO rotate → crop remote
  ├── 2. YOLO detect jempol (class_id=1) di dalam crop
  ├── 3. Hitung thumb_center (midpoint bounding box)
  ├── 4. Map koordinat jempol ke layout (relative position)
  ├── 5. Cocokkan dengan bounding box tombol (padding 15px)
  ├── 6. Return: fungsi yang disentuh, thumb center, debug images
  └── Debug: simpan ke debug_current_guided.jpg & debug_reference_guided.jpg
```

#### C. VLM Reasoning (`process_vlm_reasoning`)

```
process_vlm_reasoning(user_text)
  ├── 1. Cek reset layout keywords → reset is_layout_ready
  ├── 2. Jika layout belum siap → auto_setup_layout()
  ├── 3. detect_current_thumb_touch() → dapatkan fungsi yang disentuh
  ├── 4. Deteksi pertanyaan konfirmasi:
  │      "benar/betul/tepat" + "ini/itu/tombol ini" + ada task aktif
  │      → is_target_matched() → jika cocok: konfirmasi & reset task
  ├── 5. Kirim prompt ke LM Studio:
  │      - Image 1: Reference layout (posisi & fungsi tombol)
  │      - Image 2: Current state (posisi jempol real-time)
  │      - Available Functions (dari layout_data)
  │      - Previous Task & Current Thumb Location
  │      System prompt → intent classification + instruction
  ├── 6. Parse response JSON:
  │      {intent, updated_task, target_location_desc, instruction}
  ├── 7. Handle berdasarkan intent:
  │      - "question" → baca status layar/AC
  │      - "navigation" → set active_task, beri panduan arah
  │      - "confirmation" → cocokkan dengan task aktif
  │      - "exploration" → informasikan tombol yang disentuh
  │      - "unknown" → redirect ke perintah remote AC
  └── 8. Speak response via TTS
```

#### D. Background Task Monitor (`background_task_monitor_loop`)

Thread terpisah yang berjalan terus-menerus (setiap 0.5 detik):
1. Cek jika ada `active_task_context` dengan intent "navigation"
2. Deteksi posisi jempol
3. Jika fungsi yang disentuh cocok dengan task → otomatis konfirmasi
4. Tidak perlu menunggu user bertanya "apakah ini tombol yang benar?"

#### E. Matching Fungsi (`is_target_matched`)

Sistem matching yang cerdas untuk mencocokkan fungsi tombol:
1. **Direct match** - sama persis
2. **Synonym groups** - grup sinonim khusus remote AC:
   - power: on/off/nyala/mati
   - suhu naik: temp up/naikkan/tambah
   - suhu turun: temp down/turunkan/kurang
   - fan: kipas/angin/kecepatan
   - mode: cool/dry/heat/auto
   - swing: ayun/arah angin
   - timer: waktu
3. **Timer guard clause** - mencegah false positive antara timer dan fitur lain
4. **Index alias** - jika teks mengandung "b3", cari fungsi di layout_data

---

### 4.5 `rotate_remote.py` - Rotasi Gambar Remote

**Tujuan:** Mendeteksi dan merotasi gambar remote agar tegak lurus (portrait) menggunakan YOLO OBB (Oriented Bounding Box).

**Alur:**
1. `process_yolo_rotation(img, model, target_class)` → predict YOLO
2. Jika model support OBB → `_process_with_obb()`:
   - Ekstrak `xywhr` (center, width, height, rotation radian)
   - Hitung diagonal untuk bounding box aman (anti terpotong)
   - Crop area sekitar remote dengan padding jika perlu
   - Rotate ROI berdasarkan sudut deteksi
   - **Lock portrait**: jika width > height, rotate 90°
3. Fallback → `_process_with_axis_aligned()`:
   - Crop bounding box axis-aligned
   - Lock portrait jika landscape

**Rumus Rotasi:**
```
angle_deg = math.degrees(angle_rad)  # dari YOLO OBB
M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
rotated = cv2.warpAffine(roi_padded, M, (w, h))
```

---

### 4.6 `vision_models.py` - Manajemen Model AI

**Tujuan:** Lazy-loading singleton untuk model-model AI berat.

**Model yang dimuat:**
| Model | Tipe | Fungsi |
|-------|------|--------|
| YOLO `best.pt` | Ultralytics YOLOv8 | Deteksi remote (class 0) & jempol (class 1) |
| OWL-ViT Processor | `owlv2-base-patch16-ensemble` | Preprocessing untuk OWL |
| OWL-ViT Model | `owlv2-base-patch16-ensemble` | Zero-shot object detection tombol |

**Pattern:** Thread-safe singleton dengan `threading.Lock()` → model dimuat sekali, digunakan bersama.

---

### 4.7 `vision_http.py` - Utilitas HTTP

**Tujuan:** Menyediakan fungsi-fungsi HTTP untuk komunikasi antar komponen.

**Konstanta URL:**
| URL | Port | Fungsi |
|-----|------|--------|
| `LM_STUDIO_URL` | `1234` | LLM Vision API |
| `SNAPSHOT_URL` | `8080` | Ambil frame dari server |
| `TTS_TRIGGER_URL` | `8080` | Trigger TTS |
| `LOG_URL` | `8080` | Kirim log ke frontend |

**Fungsi Utama:**
| Fungsi | Deskripsi |
|--------|-----------|
| `safe_post(url)` | POST dengan error handling, timeout 15s |
| `fire_and_forget_post(url)` | POST di thread terpisah (non-blocking) |
| `capture_current_frame()` | GET /snapshot → decode JPEG → numpy array |
| `cv2_to_base64(image)` | Resize ke 360×640 → encode PNG → base64 |
| `extract_json_object(text)` | Parse JSON dari teks (handle markdown) |
| `speak(text)` | Cek cache TTS → trigger TTS via HTTP |
| `log_system(text)` | Kirim log sistem ke frontend |

---

### 4.8 `tts_cache.py` - Sistem Cache TTS

**Tujuan:** Menyimpan file audio TTS secara lokal untuk mengurangi ketergantungan internet dan mempercepat respon.

**Struktur Cache:**
```
tts_cache/
├── cache_index.json     # Index mapping hash → filename
├── {md5_hash1}.mp3      # File audio cache
├── {md5_hash2}.mp3
└── ...
```

**Metode:**
| Method | Deskripsi |
|--------|-----------|
| `__init__(cache_dir)` | Buat folder cache, load index |
| `_text_to_hash(text)` | MD5 hash dari teks (lowercase, stripped) |
| `get_cached_file(text)` | Cek cache, return path atau None |
| `add_to_cache(text, file_path)` | Tambah entry ke cache index |
| `clear_cache()` | Hapus semua file `.mp3`, `.wav`, `.ogg` |
| `get_cache_info()` | Statistik cache (jumlah file, ukuran) |

---

### 4.9 `hybrid_inference.py` - Alternatif Entry Point

**Tujuan:** Entry point alternatif untuk testing tanpa mic (menggunakan input terminal + tap layar).

**Alur:**
1. `listen_to_tap_commands()` - WebSocket ke `/command_feed`
2. `manual_input_loop()` - Input dari terminal
3. `auto_scan_layout()` - Background scan remote
4. Ketiga fungsi berjalan parallel via `asyncio.gather()`

**Cocok untuk:** Debugging, testing tanpa HP, atau situasi di mana mic tidak tersedia.

---

### 4.10 `layout_OWL.py` - Skrip Standalone (Deprecated)

**Tujuan:** Skrip eksperimental/standalone untuk testing deteksi tombol dengan OWL-ViT.

**Keterbatasan:**
- Static image dari `debug_cropped.jpg`
- Tidak terintegrasi dengan pipeline utama
- Hanya mendeteksi tombol "number buttons" (terbatas)
- Tidak ada mapping fungsi via LLM

> **Status:** Digantikan oleh fungsi `generate_owl_layout()` di `vision_reasoning.py`

---

## 5. Alur Proses Lengkap

### 🔄 Siklus Hidup Sistem

```
STARTUP
  ├── server_vision.py (Port 8080)
  │     └── Preload common TTS phrases
  │           └── Generate & cache ~20 phrases umum
  └── audio_inference.py
        ├── Load Whisper model (CUDA)
        ├── ensure_tts_cache_preloaded() → trigger preload
        ├── start_background_task_monitor() → start thread monitor jempol
        └── Mulai 3 async loops:
              ├── listen_to_mic()          → STT
              ├── listen_to_commands()     → Tap layar
              └── auto_scan_layout()       → Deteksi otomatis

HP TERKONEKSI
  ├── Buka browser → http://<PC_IP>:8080
  ├── WebRTC negotiation → video & audio streaming mulai
  ├── TTS Welcome: "Letakkan remote di depan kamera..."
  └── server_vision.py menerima video frames → latest_jpeg update

AUTO LAYOUT DETECTION
  └── auto_scan_layout() → setiap 2 detik:
        ├── capture_current_frame()
        ├── YOLO detect remote?
        │     ├── Ya → auto_setup_layout(silent=True)
        │     │        ├── Stabilisasi 2 detik
        │     │        ├── Validasi ukuran & margin
        │     │        ├── generate_owl_layout() → indeks tombol
        │     │        └── map_functions_with_vlm() → fungsi tombol
        │     └── Tidak → tunggu 2 detik lagi
        └── is_layout_ready = True

USER BICARA (Contoh: "Cari tombol power")
  ├── 1. Mic HP → WebRTC Audio → server_vision.py → resample 16kHz
  ├── 2. audio_inference.py (listen_to_mic):
  │      ├── VAD → deteksi akhir kalimat
  │      ├── Whisper STT → "cari tombol power"
  │      └── Filter noise & hallucination → valid
  ├── 3. system_is_busy = True
  ├── 4. process_vlm_reasoning("cari tombol power"):
  │      ├── detect_current_thumb_touch() → deteksi jempol
  │      ├── Kirim prompt ke LM Studio (2 gambar + user teks)
  │      ├── LLM response: {intent:"navigation", target:"power", ...}
  │      ├── set active_task_context = "power"
  │      └── Speak: "Tombol power di kanan atas. Geser ke kanan..."
  └── 5. system_is_busy = False

USER MENGGERAKKAN JEMPOL
  ├── Background task monitor (setiap 0.5 detik):
  │      ├── Deteksi jempol → fungsi "power"
  │      ├── is_target_matched("power", active_task="power") → TRUE
  │      └── Speak: "Nah, yang itu tombolnya." + reset task
  └── Mic aktif kembali → siap perintah baru

USER TEKAN TOMBOL & BERTANYA "Apakah ini tombol power?"
  ├── 1. Whisper STT → "apakah ini tombol power"
  ├── 2. process_vlm_reasoning():
  │      ├── Deteksi konfirmasi → "benar/ini/itu" + "tombol?"
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
┌─────────────────────────────────────────────────────────────────────┐
│                           HANDHPHONE (Browser)                      │
│                                                                     │
│  ┌──────────┐    WebRTC     ┌────────────────────────────────────┐  │
│  │  Camera  │─────────────▶ │  server_vision.py (Port 8080)     │  │
│  │  + Mic   │               │                                    │  │
│  └──────────┘               │  latest_jpeg (global)              │  │
│       │                      │  AudioResampler → 16kHz mono      │  │
│       │                      │                                   │  │
│       │              ┌───────┴──────────────┐                    │  │
│       │              │                      │                    │  │
│       │              ▼                      ▼                    │  │
│       │    ┌─────────────────┐   ┌──────────────────┐            │  │
│       │    │  /snapshot GET  │   │ /audio_feed WS   │            │  │
│       │    └────────┬────────┘   └────────┬─────────┘            │  │
│       │             │                     │                      │  │
│       │             ▼                     ▼                      │  │
│       │    ┌─────────────────┐   ┌──────────────────┐            │  │
│       │    │ vision_reasoning│   │ audio_inference  │            │  │
│       │    │ .py             │   │ .py (Whisper)    │            │  │
│       │    │                 │   │                  │            │  │
│       │    │ YOLO detect     │   │ VAD → STT        │            │  │
│       │    │ OWL-ViT index   │   │ Filter noise     │            │  │
│       │    │ VLM (LM Studio) │   │ Trigger VLM      │            │  │
│       │    └────────┬────────┘   └──────────────────┘            │  │
│       │             │                                            │  │
│       │             ▼                                            │  │
│       │    ┌──────────────────┐                                  │  │
│       │    │  /trigger_tts    │                                  │  │
│       │    │  gTTS → base64   │                                  │  │
│       │    └────────┬─────────┘                                  │  │
│       │             │                                            │  │
│       │             ▼                                            │  │
│       │    ┌─────────────────────────────────┐                   │  │
│       │    │  WebSocket /frontend_ws         │                   │  │
│       │    │  {type:"audio", audio:base64}   │                   │  │
│       │    │  {type:"log", sender, text}     │                   │  │
│       │    │  {type:"Control", text:"MIC_ON"}│                   │  │
│       │    └─────────────────────────────────┘                   │  │
│       │                     │                                    │  │
│       │                     ▼                                    │  │
│  ┌────┴────────────┐                                             │  │
│  │  Speaker + Log  │                                             │  │
│  │  Chat UI        │                                             │  │
│  └─────────────────┘                                             │  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Diagram State Machine

```
                ┌──────────┐
                │ STARTUP  │
                └────┬─────┘
                     │
                     ▼
              ┌──────────────┐
         ┌───▶│ WAITING FOR │
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
         │    │  READY       │  scan     │   │
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
         │           ┌───────┴────────┐
         │           │                │
         │           ▼                ▼
         │    ┌──────────┐    ┌──────────────┐
         │    │ USER     │    │ TAP LAYAR    │
         │    │ SPEAKS   │    │ (manual)     │
         │    └────┬─────┘    └──────┬───────┘
         │         │                 │
         │         ▼                 │
         │    ┌──────────┐           │
         │    │ WHISPER  │           │
         │    │ STT      │           │
         │    └────┬─────┘           │
         │         │ text            │ text
         │         ▼                 ▼
         │    ┌──────────────────────────────┐
         │    │   PROCESS VLM REASONING      │
         │    │                              │
         │    │  ┌────────────────────┐      │
         │    │  │ intent: navigation │      │
         │    │  │  → set task       │──────┼───▶ Background Monitor Aktif
         │    │  │  → speak guidance │      │
         │    │  └────────────────────┘      │
         │    │  ┌────────────────────┐      │
         │    │  │ intent: confirmation│     │
         │    │  │  → match task      │      │
         │    │  │  → confirm/deny    │      │
         │    │  │  → reset task      │      │
         │    │  └────────────────────┘      │
         │    │  ┌────────────────────┐      │
         │    │  │ intent: question   │      │
         │    │  │  → answer status   │      │
         │    │  └────────────────────┘      │
         │    │  ┌────────────────────┐      │
         │    │  │ intent: exploration│      │
         │    │  │  → inform fungsi   │      │
         │    │  └────────────────────┘      │
         │    └──────────────┬───────────────┘
         │                   │
         │                   ▼
         │    ┌──────────────────────┐
         │    │   SPEAK (TTS)        │
         │    │   + Log ke Frontend  │
         │    └──────────┬───────────┘
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

## Ringkasan Komponen

| Modul | Bahasa | Baris | Fungsi Utama |
|-------|--------|-------|--------------|
| `server_vision.py` | Python | 376 | WebRTC server, HTTP API, TTS delivery |
| `index.html` | HTML/JS | 400 | Frontend HP WebRTC client, TTS player |
| `audio_inference.py` | Python | 284 | STT Whisper, VAD, VLM trigger |
| `vision_reasoning.py` | Python | 839 | Layout setup, thumb detection, VLM reasoning, task monitor |
| `rotate_remote.py` | Python | 138 | YOLO OBB rotation & cropping |
| `vision_models.py` | Python | 35 | Lazy-loading YOLO + OWL-ViT |
| `vision_http.py` | Python | 110 | HTTP utilities (LM Studio, snapshot, TTS) |
| `tts_cache.py` | Python | 135 | TTS audio caching with MD5 index |
| `hybrid_inference.py` | Python | 84 | Alternative entry (terminal + tap) |
| `layout_OWL.py` | Python | 117 | Deprecated standalone OWL script |

---

## Cara Menjalankan

1. **Jalankan LM Studio** di port 1234 dengan model vision
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