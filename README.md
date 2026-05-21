# Remote AC Assistive System for the Blind

Sistem asisten berbasis AI untuk penyandang tunanetra yang membantu mengoperasikan remote AC fisik melalui perintah suara, navigasi taktil (panduan jempol), dan umpan balik suara.

## Fitur Utama

- **Perintah Suara** — Tekan & tahan layar HP, bicara, lepas untuk proses (Whisper STT)
- **Navigasi Taktil** — Panduan arah jempol ke tombol remote dengan deteksi real-time (YOLO)
- **Deteksi Tombol Otomatis** — Zero-shot object detection (OWL-ViT) + mapping fungsi via VLM
- **Umpan Balik Suara** — TTS otomatis (gTTS) untuk setiap respon
- **Navigasi Layar Real-time** — Background monitor mendeteksi sentuhan jempol tanpa perlu bertanya
- **Mode Hybrid** — Input terminal untuk testing tanpa HP

## Arsitektur Singkat

```
HP (Kamera + Mic) ←→ WebRTC ←→ Server PC (port 8080)
                              ├── YOLOv8 (deteksi remote + jempol)
                              ├── OWL-ViT (deteksi tombol zero-shot)
                              ├── LM Studio / VLM (reasoning & navigasi)
                              └── Faster-Whisper (speech-to-text)
```

## Persyaratan Sistem

- **Python** 3.10+
- **CUDA** (recommended untuk GPU) atau CPU
- **LM Studio** (port 1234) dengan model vision
- **HP Android/iOS** dengan browser modern (Chrome/Safari)

## Instalasi & Persiapan

### 1. Clone repository

```bash
git clone git@github.com:AndiArvy/asistif-code.git
cd asistif-code
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

> Catatan: Jika `requirements.txt` belum tersedia, instal manual: `aiohttp`, `aiortc`, `av`, `faster-whisper`, `ultralytics`, `transformers`, `torch`, `torchvision`, `gtts`, `opencv-python`, `numpy`, `websockets`, `requests`, `Pillow`, `websocket-client`.

### 4. Siapkan model AI

- **YOLO** — Letakkan `best.pt` di root project (atau set env `YOLO_MODEL_PATH`)
- **LM Studio** — Download model vision (contoh: LLaVA, CogVLM), jalankan di port 1234, aktifkan API server
- **OWL-ViT** — Didownload otomatis dari HuggingFace saat pertama kali dijalankan

## Menjalankan Sistem

### Terminal 1: Server WebRTC

```bash
python src/server_vision.py
```

### Terminal 2: Audio Inference

```bash
python src/audio_inference.py
```

### Terminal 3 (opsional): LM Studio

Jalankan LM Studio GUI, load model vision, start server di port 1234.

### HP

Buka browser → `http://<IP_PC>:8080` → izinkan kamera & mic.

### Mode Hybrid (tanpa HP)

```bash
python src/hybrid_inference.py
```

## Panduan Cepat

1. Letakkan remote AC di depan kamera HP dengan posisi jelas
2. Tunggu sistem memetakan tombol secara otomatis (~5-15 detik)
3. Tekan & tahan layar HP → ucapkan perintah (contoh: "Cari tombol power")
4. Lepas layar → sistem memproses dan memberikan panduan suara
5. Geser jempol di atas remote mengikuti panduan
6. Sistem akan otomatis mengkonfirmasi ketika jempol menyentuh tombol yang benar
7. Tekan tombol fisik untuk mengeksekusi

## Variabel Lingkungan

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `SERVER_HOST` | `0.0.0.0` | Host binding server |
| `SERVER_PORT` | `8080` | Port server |
| `TTS_PLAYBACK_RATE` | `1.2` | Kecepatan playback TTS |
| `WHISPER_MODEL_SIZE` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Model Whisper |
| `WHISPER_DEVICE` | `cuda` | Device Whisper |
| `YOLO_MODEL_PATH` | `best.pt` | Path model YOLO |
| `YOLO_CONF_THRESHOLD` | `0.3` | Confidence threshold YOLO |
| `YOLO_FORCE_PORTRAIT` | `true` | Paksa output portrait |
| `VISION_DEBUG` | `true` | Simpan debug images |
| `MAX_CONVERSATION_HISTORY` | `20` | Maks riwayat percakapan |

## Dokumentasi Lainnya

| File | Deskripsi |
|------|-----------|
| `docs/ARSITEKTUR_SISTEM.md` | Dokumentasi arsitektur lengkap |
| `docs/API.md` | Dokumentasi API endpoint |
| `docs/DEPLOYMENT.md` | Panduan deployment produksi |
| `docs/DEVELOPMENT.md` | Panduan pengembangan & kontribusi |
| `docs/USER_GUIDE.md` | Panduan penggunaan untuk tunanetra |
| `docs/code_review.md` | Laporan code review |

## Lisensi

MIT
