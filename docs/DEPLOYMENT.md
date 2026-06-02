# Deployment Guide

Panduan untuk menjalankan sistem dalam mode produksi.

---

## Arsitektur Deployment

```
┌──────────────────────┐        ┌─────────────────────────┐
│     HP (Browser)     │        │      Server PC           │
│                      │ WebRTC │                          │
│  Chrome / Safari     │◄──────►│  server_vision.py:8080   │
│  Kamera + Mic + Spkr │        │  audio_inference.py      │
│  Web Speech API      │        │  vision_reasoning.py     │
└──────────────────────┘        │  LM Studio (port 1234)   │
                                │  GPU (CUDA)              │
                                └─────────────────────────┘
```

### Persyaratan Minimum Hardware

| Komponen | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 core, 2.5 GHz | 8 core, 3.0+ GHz |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060 12GB+ |
| Storage | 20 GB free | 50 GB SSD |
| Jaringan | Wi-Fi 5 (AC) | Wi-Fi 6 (AX) / LAN |

### Software Dependencies

- **OS:** Windows 10/11, Linux (Ubuntu 22.04+), macOS 14+
- **Python:** 3.10 – 3.12
- **CUDA:** 11.8+ (jika GPU NVIDIA)
- **LM Studio:** v0.3+ (unduh dari [lmstudio.ai](https://lmstudio.ai))
- **Browser HP:** Chrome 120+ / Safari 17+

---

## Instalasi Produksi

### 1. Setup Python Environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Instal Dependensi

```bash
pip install -r requirements.txt
```

### 3. Siapkan Model AI

**YOLO Model (`best.pt`)**
- Letakkan di root project
- Atau set environment variable: `YOLO_MODEL_PATH=/path/to/best.pt`
- Model harus support OBB (Oriented Bounding Box) untuk rotasi remote

**LM Studio**
1. Download model vision (`qwen3.5-9b-vl` atau model vision lainnya)
2. Buka LM Studio → Load model
3. Start API server di `localhost:1234`
4. Pastikan `CORS` diaktifkan (Settings → CORS → Allow all origins)

**OWL-ViT**
- Didownload otomatis dari HuggingFace saat pertama run
- Set `OWL_MODEL_NAME` untuk mengganti varian (default: `google/owlv2-base-patch16-ensemble`)

**Faster-Whisper**
- Model didownload otomatis saat pertama run
- Set `WHISPER_MODEL_SIZE` untuk mengganti ukuran model

### 4. Environment Variables untuk Produksi

```bash
# Server
SERVER_HOST=0.0.0.0         # Bind ke semua interface (default)
SERVER_PORT=8080             # Port HTTP + WebRTC
JPEG_QUALITY=85              # Kualitas JPEG (lebih rendah = lebih cepat)

# TTS
TTS_PLAYBACK_RATE=1.2

# Whisper
WHISPER_DEVICE=cuda          # atau "cpu" untuk CPU-only
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_MODEL_SIZE=deepdml/faster-whisper-large-v3-turbo-ct2
WHISPER_CPU_THREADS=8

# YOLO
YOLO_MODEL_PATH=best.pt
YOLO_CONF_THRESHOLD=0.3
YOLO_FORCE_PORTRAIT=true
YOLO_INVERT_OBB_ANGLE=false

# Debug (matikan di produksi)
VISION_DEBUG=false
```

### 5. Menjalankan Service

**Menggunakan Terminal Multiplexer (Linux/macOS):**

```bash
# Terminal 1: Server utama
python src/server_vision.py

# Terminal 2: Audio inference
python src/audio_inference.py
```

**Menggunakan systemd (Linux):**

Buat file service `/etc/systemd/system/remote-ac-assistive@.service`:

```ini
[Unit]
Description=Remote AC Assistive Service (%i)
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/code2
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/path/to/code2/venv/bin/python src/%i.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable remote-ac-assistive@server_vision
sudo systemctl enable remote-ac-assistive@audio_inference
sudo systemctl start remote-ac-assistive@server_vision
sudo systemctl start remote-ac-assistive@audio_inference
```

---

## Optimasi Performa

### GPU Memory Management

- YOLOv26n OBB ~1-2 GB VRAM
- OWL-ViT ~2-3 GB VRAM (hanya saat setup layout)
- Faster-Whisper ~2-3 GB VRAM
- LM Studio (Qwen3.5 9B) ~8-10 GB VRAM

**Total: ~14-18 GB VRAM**: pastikan GPU 16GB+ mencukupi.

Tips:
- Set `WHISPER_COMPUTE_TYPE=int8` untuk mengurangi VRAM Whisper
- Set `YOLO_CONF_THRESHOLD=0.4` untuk mengurangi FP
- Nonaktifkan `VISION_DEBUG` di produksi
- Turunkan `JPEG_QUALITY` ke 85 untuk mengurangi bandwidth

### Network Optimization

- Gunakan LAN (Ethernet) untuk koneksi stabil antara PC dan router
- Pastikan HP di jaringan Wi-Fi yang sama
- Untuk stabilitas maksimal, gunakan dedicated Wi-Fi Access Point untuk HP
- Latency <50ms untuk WebRTC ideal

---

## Troubleshooting

### WebRTC Connection Failed
- Pastikan HP dan PC di jaringan yang sama
- Cek firewall: port 8080 harus terbuka
- Coba nonaktifkan VPN di HP

### LM Studio Connection Refused
- Pastikan LM Studio API server berjalan di port 1234
- Cek `http://localhost:1234/v1/models` dari browser PC
- Restart LM Studio jika perlu

### No Audio from Mic
- Pastikan mic tidak di-mute oleh OS
- Cek izin browser: Settings → Mic → Allow
- Coba restart browser

### Hold-to-Speak Tidak Berfungsi
- Pastikan browser mendukung pointer events
- Coba sentuh layar di area bukan tombol
- Cek console server untuk log `[Hold] MULAI mendengarkan...`
- Safety timeout 30 detik: jika tidak, coba reload halaman

### Layout Setup Fails Repeatedly
- Pastikan remote dalam fokus dan tidak blur
- Cek pencahayaan: minimal cahaya ruangan cukup
- Coba dekatkan remote ke kamera (min 40% frame)
- Cek `debug_frame.jpg` dan `debug_cropped.jpg` untuk debugging

### GPU Out of Memory
- Turunkan `WHISPER_COMPUTE_TYPE=int8`
- Gunakan model Whisper lebih kecil: `base` atau `small`
- Tutup aplikasi lain yang menggunakan GPU
- Restart LM Studio dengan model lebih kecil

---

## Monitoring & Logging

Sistem menyediakan:

1. **Console logs**: semua service mencetak log ke stdout
2. **Debug images**: disimpan ke root folder (jika `VISION_DEBUG=true`):
   - `debug_frame.jpg`: frame mentah
   - `debug_cropped.jpg`: crop remote
   - `debug_current_guided.jpg`: overlay tombol + jempol
   - `debug_*_reference.jpg`: reference layout images
3. **TTSCache**: `tts_cache/cache_index.json` untuk audit TTS
4. **Layout JSON**: `layout.json` untuk hasil mapping tombol

Untuk logging production, redirect stdout:
```bash
python src/server_vision.py > logs/server.log 2>&1
python src/audio_inference.py > logs/audio.log 2>&1
```

---

## Keamanan

**Peringatan:** Sistem ini didesain untuk penggunaan personal di jaringan lokal.

- **Bind ke LAN saja**: Set `SERVER_HOST=192.168.1.10` (IP lokal) untuk mencegah akses dari jaringan luar
- **Path traversal**: Endpoint `/trigger_tts` sudah memiliki validasi path traversal
- **LM Studio**: API lokal, tidak terekspos ke network
- **Tidak ada autentikasi**: Cocok untuk lingkungan terpercaya. Untuk akses publik, tambahkan reverse proxy (nginx) dengan autentikasi
