# Remote AC Assistive System for the Blind

> Sistem asisten berbasis AI yang membantu penyandang tunanetra mengoperasikan **remote AC fisik** melalui perintah suara, navigasi taktil (panduan jempol), dan umpan balik suara — cukup dengan sebuah HP dan sebuah PC.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg">
  <img alt="CUDA" src="https://img.shields.io/badge/GPU-CUDA%2012.x-76B900.svg">
</p>

---

## Daftar Isi

- [Ringkasan](#ringkasan)
- [Cara Kerja](#cara-kerja)
- [Fitur Utama](#fitur-utama)
- [Arsitektur](#arsitektur)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Menjalankan Sistem](#menjalankan-sistem)
- [Panduan Cepat Penggunaan](#panduan-cepat-penggunaan)
- [Konfigurasi (Environment Variables)](#konfigurasi-environment-variables)
- [Perekaman Data (Log Daemon)](#perekaman-data-log-daemon)
- [Struktur Proyek](#struktur-proyek)
- [Dokumentasi Lengkap](#dokumentasi-lengkap)
- [Troubleshooting](#troubleshooting)
- [Lisensi](#lisensi)

---

## Ringkasan

Penyandang tunanetra kesulitan mengoperasikan remote AC karena tombol fisiknya seragam dan tidak berlabel Braille. Sistem ini menjembatani hal tersebut: pengguna cukup mengarahkan kamera HP ke remote, lalu **berbicara** untuk meminta bantuan. Sistem akan memetakan tombol secara otomatis, memandu jempol pengguna ke tombol yang tepat lewat instruksi suara, dan mengonfirmasi secara otomatis saat jempol menyentuh tombol yang benar.

Semua pemrosesan AI (deteksi, STT, reasoning, TTS) berjalan **lokal di PC** — privasi terjaga dan tidak perlu langganan API cloud.

## Cara Kerja

```
1. Pengguna arahkan kamera HP ke remote AC
2. Sistem otomatis mendeteksi & memetakan tombol (YOLO OBB + OWL-ViT + VLM)
3. Pengguna tekan-tahan layar HP → ucapkan perintah → lepas ("Cari tombol power")
4. Whisper mengubah suara menjadi teks → VLM menyusun panduan arah
5. Sistem memandu jempol via suara ("Geser ke kanan atas...")
6. Background monitor mendeteksi sentuhan → auto-konfirmasi ("Nah, itu tombolnya")
7. Pengguna menekan tombol fisik
```

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🎙️ **Perintah Suara** | Tekan & tahan layar, bicara, lepas untuk proses (Hold-to-Speak + Faster-Whisper STT) |
| 🧭 **Navigasi Taktil** | Panduan arah jempol ke tombol dengan deteksi real-time (YOLO OBB) |
| 🔍 **Deteksi Tombol Otomatis** | Zero-shot detection (OWL-ViT) + pemetaan fungsi via VLM, tanpa training per-remote |
| 🔊 **Umpan Balik Suara** | TTS ganda: gTTS (online) + Web Speech API (offline, zero server load) |
| ⚡ **Auto-Confirm** | Background monitor mengonfirmasi otomatis saat jempol menyentuh target |
| 🗣️ **Perintah Cepat AC** | Intent hardcode ("nyalakan AC", "atur suhu", "ganti mode") tanpa memanggil VLM |
| 🔦 **Kontrol Senter** | Perintah suara "senter" untuk menyalakan/mematikan lampu HP |
| 📳 **Haptic & Beep** | Getaran + suara sintetis saat mic aktif/mati dan konfirmasi tombol |
| 🖥️ **Mode Hybrid** | Input terminal untuk testing tanpa HP |
| 📊 **Log Daemon** | Perekam data (percakapan, waktu respons, status sistem) ke CSV untuk riset/evaluasi |

## Arsitektur

```
┌─────────────────────┐         ┌──────────────────────────────────────────┐
│   HP (Browser)      │ WebRTC  │            Server PC (port 8080)           │
│                     │◄───────►│                                            │
│  Kamera + Mic       │         │  ├── YOLO OBB      (deteksi remote+jempol) │
│  Speaker            │         │  ├── OWL-ViT       (deteksi tombol)        │
│  Web Speech API     │         │  ├── Faster-Whisper (speech-to-text)       │
│  Hold-to-Speak UI   │         │  └── gTTS          (text-to-speech)        │
└─────────────────────┘         │                                            │
                                │        │ HTTP (OpenAI-compatible)          │
                                │        ▼                                   │
                                │  LM Studio / VLM (port 1234)               │
                                └──────────────────────────────────────────┘
```

Detail lengkap: [`docs/ARSITEKTUR_SISTEM.md`](docs/ARSITEKTUR_SISTEM.md).

## Persyaratan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| **Python** | 3.10 | 3.10 – 3.12 |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060 12GB+ |
| **RAM** | 16 GB | 32 GB |
| **CUDA** | 12.x (atau CPU-only) | 12.6 |
| **LM Studio** | v0.3+ dengan model vision (mis. Qwen VL) di port 1234 | — |
| **HP** | Android/iOS, browser Chrome 120+ / Safari 17+ | — |
| **Jaringan** | HP & PC di Wi-Fi/LAN yang sama | — |

> Total kebutuhan VRAM ~14–18 GB saat semua model aktif. Lihat [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) untuk optimasi pada GPU lebih kecil.

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/AndiArvy/asistif-code.git
cd asistif-code
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Instal dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Catatan GPU/PyTorch:** `requirements.txt` sudah dikonfigurasi untuk **CUDA 12.6** (`torch==2.12.0+cu126`) melalui `--extra-index-url`. Untuk versi CUDA lain atau CPU-only, ubah URL dan pin torch di bagian atas `requirements.txt` sesuai petunjuk komentar di sana (`cu124`, `cu128`, atau `cpu`).

### 4. Siapkan model AI

| Model | Cara Menyiapkan |
|-------|-----------------|
| **YOLO OBB** (`best.pt`) | Sudah disertakan di root project (atau set env `YOLO_MODEL_PATH`). Harus mendukung Oriented Bounding Box. |
| **LM Studio (VLM)** | Unduh model vision (mis. Qwen VL), jalankan API server di port **1234**, aktifkan CORS. |
| **OWL-ViT** | Diunduh otomatis dari HuggingFace saat pertama dijalankan. |
| **Faster-Whisper** | Diunduh otomatis saat pertama dijalankan. |

## Menjalankan Sistem

Jalankan setiap perintah di terminal terpisah:

```bash
# Terminal 1 — Server WebRTC + HTTP + TTS
python src/server_vision.py

# Terminal 2 — Speech-to-Text + VLM reasoning
python src/audio_inference.py

# Terminal 3 (opsional) — Perekam data untuk riset/evaluasi
python src/log_daemon.py
```

Pastikan **LM Studio** sudah menjalankan API server di port 1234 dengan model vision termuat.

Di **HP**, buka browser ke `http://<IP_PC>:8080`, lalu izinkan akses kamera & mikrofon.

### Mode Hybrid (tanpa HP)

Untuk testing/debugging dari terminal PC tanpa HP:

```bash
python src/hybrid_inference.py
```

## Panduan Cepat Penggunaan

1. Letakkan remote AC di depan kamera HP dengan posisi jelas (isi 40–80% frame).
2. Tunggu sistem memetakan tombol otomatis (~5–15 detik) — akan ada konfirmasi suara.
3. Tekan & tahan layar HP → ucapkan perintah (mis. *"Cari tombol power"*) → lepas.
4. Ikuti panduan arah suara sambil menggeser jempol perlahan di atas remote.
5. Sistem otomatis mengonfirmasi saat jempol menyentuh tombol yang benar.
6. Tekan tombol fisik untuk mengeksekusi.

Panduan lengkap untuk pengguna akhir: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Konfigurasi (Environment Variables)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `SERVER_HOST` | `0.0.0.0` | Host binding server |
| `SERVER_PORT` | `8080` | Port server |
| `JPEG_QUALITY` | `99` | Kualitas JPEG frame video |
| `TTS_PLAYBACK_RATE` | `1.2` | Kecepatan playback TTS |
| `WHISPER_MODEL_SIZE` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Model Whisper |
| `WHISPER_DEVICE` | `cuda` | Device Whisper (`cuda`/`cpu`) |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` | Compute type Whisper |
| `WHISPER_CPU_THREADS` | `8` | CPU threads untuk Whisper |
| `WHISPER_BEAM_SIZE` | `1` | Beam size Whisper |
| `WHISPER_BEST_OF` | `1` | Best-of candidates Whisper |
| `WHISPER_MIN_SILENCE_MS` | `500` | Min silence duration untuk VAD |
| `YOLO_MODEL_PATH` | `best.pt` | Path model YOLO |
| `YOLO_CONF_THRESHOLD` | `0.3` | Confidence threshold YOLO |
| `YOLO_FORCE_PORTRAIT` | `true` | Paksa output portrait |
| `YOLO_INVERT_OBB_ANGLE` | `false` | Balik arah rotasi OBB |
| `OWL_MODEL_NAME` | `google/owlv2-base-patch16-ensemble` | Model OWL-ViT |
| `VISION_DEBUG` | `true` | Simpan debug images |
| `MAX_CONVERSATION_HISTORY` | `20` | Maks riwayat percakapan |
| `LOG_DIR` | `logs` | Folder output log daemon |
| `LOG_POLL_INTERVAL` | `1` | Interval poll status sistem (detik) |

## Perekaman Data (Log Daemon)

`src/log_daemon.py` adalah proses opsional yang merekam jalannya sistem untuk keperluan riset/evaluasi (mis. skripsi). Ia terhubung ke WebSocket server (`/frontend_ws`, `/command_feed`) sebagai pengamat pasif dan menulis tiga file CSV per hari ke folder `logs/`:

| File | Isi |
|------|-----|
| `conversation_YYYY-MM-DD.csv` | Percakapan per sesi + waktu STT, reasoning, dan total respons |
| `system_status_YYYY-MM-DD.csv` | Snapshot status sistem periodik (layout ready, busy, CPU/memori) |
| `events_YYYY-MM-DD.csv` | Semua event mentah (hold-to-speak, STT, response, error) |

Monitoring CPU/memori memerlukan `psutil` (opsional — dinonaktifkan otomatis jika tidak terpasang). Log daemon tidak memengaruhi jalannya sistem utama; jalankan hanya bila Anda perlu mengumpulkan data.

## Struktur Proyek

```
code2/
├── src/
│   ├── server_vision.py      # Server WebRTC, HTTP API, TTS, WebSocket
│   ├── audio_inference.py    # STT Whisper + Hold-to-Speak + trigger VLM
│   ├── vision_reasoning.py   # Pipeline vision, VLM reasoning, task monitor
│   ├── index.html            # Frontend HP (WebRTC, Hold-to-Speak, Web Speech)
│   ├── rotate_remote.py      # Rotasi & crop remote via YOLO OBB
│   ├── vision_models.py      # Lazy-loading model YOLO + OWL-ViT (singleton)
│   ├── vision_http.py        # Utilitas HTTP (LM Studio, snapshot, TTS)
│   ├── tts_cache.py          # Cache TTS lokal (MD5 index)
│   ├── hybrid_inference.py   # Entry alternatif (terminal + tap)
│   └── log_daemon.py         # Perekam data ke CSV (opsional)
├── docs/                     # Dokumentasi lengkap
├── best.pt                   # Model YOLO OBB (remote + jempol)
├── requirements.txt          # Dependensi (torch cu126 by default)
├── LICENSE                   # MIT
└── README.md
```

## Dokumentasi Lengkap

| Dokumen | Deskripsi |
|---------|-----------|
| [`docs/ARSITEKTUR_SISTEM.md`](docs/ARSITEKTUR_SISTEM.md) | Arsitektur, alur data, dan deskripsi tiap modul |
| [`docs/API.md`](docs/API.md) | Referensi endpoint HTTP, WebSocket, dan WebRTC |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Panduan deployment produksi & optimasi |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Panduan pengembangan & kontribusi |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Panduan penggunaan untuk tunanetra |
| [`docs/ADRs.md`](docs/ADRs.md) | Architecture Decision Records |
| [`docs/code_review.md`](docs/code_review.md) | Laporan code review |

## Troubleshooting

| Masalah | Solusi Cepat |
|---------|--------------|
| **WebRTC gagal konek** | Pastikan HP & PC di jaringan sama; buka port 8080 di firewall; nonaktifkan VPN di HP |
| **LM Studio connection refused** | Pastikan API server aktif di port 1234; cek `http://localhost:1234/v1/models` |
| **Tidak ada audio dari mic** | Cek izin mic browser; pastikan mic tidak di-mute OS |
| **Layout gagal terus** | Pastikan remote fokus & tidak blur; cukup cahaya; dekatkan (isi ≥40% frame) |
| **GPU out of memory** | Set `WHISPER_COMPUTE_TYPE=int8`; gunakan model VLM lebih kecil; tutup app lain |

Detail lengkap: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#troubleshooting).

## Lisensi

Dirilis di bawah lisensi [MIT](LICENSE).
