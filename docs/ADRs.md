# Architecture Decision Records (ADR)

Dokumen ini mencatat keputusan arsitektur utama yang diambil selama pengembangan sistem.

---

## ADR-001: WebRTC untuk Streaming Video & Audio Real-time

**Status:** Accepted

**Konteks:** Sistem perlu menerima video dan audio real-time dari HP ke PC untuk diproses oleh model AI. Opsi yang dipertimbangkan: WebRTC, HTTP chunked streaming, WebSocket binary streaming.

**Keputusan:** Menggunakan WebRTC (`aiortc` + browser native `RTCPeerConnection`).

**Alasan:**
- Native di browser: tidak perlu install app tambahan di HP
- Adaptive bitrate: otomatis menyesuaikan kualitas dengan koneksi
- Dukungan audio-video sinkron dalam satu koneksi
- Firewall-friendly (menggunakan port 8080 yang sama dengan HTTP)

**Konsekuensi:**
- Kompleksitas signaling (SDP offer/answer)
- Bergantung pada browser: fitur torch tidak konsisten antar browser
- Perlu `av` library untuk resample audio

---

## ADR-002: Faster-Whisper untuk Speech-to-Text Lokal

**Status:** Accepted

**Konteks:** Perlu STT real-time dengan latency rendah. Opsi: cloud API (Google/Azure), Whisper lokal, Faster-Whisper.

**Keputusan:** Faster-Whisper dengan model `large-v3-turbo`.

**Alasan:**
- 4-6x lebih cepat dari OpenAI Whisper (CTranslate2 optimization)
- Berjalan lokal: tidak perlu internet, privasi terjaga
- Model turbo memberikan keseimbangan akurasi-kecepatan
- Dukungan bahasa Indonesia yang baik

**Konsekuensi:**
- Perlu GPU CUDA untuk real-time performance
- Model ~3GB, perlu download awal

---

## ADR-003: OWL-ViT untuk Zero-Shot Object Detection Tombol

**Status:** Accepted

**Konteks:** Tombol remote memiliki layout yang sangat bervariasi antar merek AC. Opsi: train custom detector (butuh dataset besar), template matching, OWL-ViT zero-shot.

**Keputusan:** OWL-ViT (`google/owlv2-base-patch16-ensemble`) untuk deteksi tombol tanpa training.

**Alasan:**
- Zero-shot: tidak perlu dataset tombol remote
- Generalisasi ke remote manapun tanpa retraining
- Akurasi cukup baik untuk tombol dengan text query yang tepat

**Konsekuensi:**
- ~2GB model, loading lambat
- Threshold 0.09 untuk recall tinggi, kadang over-detect (difilter area 15% + NMS)
- Hanya digunakan saat setup layout, tidak untuk runtime

---

## ADR-004: YOLOv26n OBB untuk Deteksi & Rotasi Remote

**Status:** Accepted

**Konteks:** Remote bisa dalam posisi miring di kamera. Deteksi butuh rotasi agar tombol bisa diindeks secara konsisten.

**Keputusan:** YOLOv26n OBB untuk deteksi + rotasi (model custom, `best.pt`).

**Alasan:**
- OBB memberikan sudut rotasi langsung dari model
- YOLOv26n OBB sudah terintegrasi di Ultralytics: tanpa preprocessing tambahan
- Lock portrait memastikan orientasi konsisten untuk mapping layout
- Fallback ke axis-aligned boxes jika OBB tidak tersedia

**Konsekuensi:**
- Model `best.pt` harus support OBB
- Perlu validasi margin agar remote tidak terpotong
- Rotasi kadang perlu invert angle (env `YOLO_INVERT_OBB_ANGLE`)

---

## ADR-005: LM Studio untuk VLM Reasoning Lokal

**Status:** Accepted

**Konteks:** Perlu model Vision-Language untuk memahami layout remote dan memberikan panduan navigasi. Opsi: cloud API (GPT-4V, Claude), LM Studio lokal, Ollama.

**Keputusan:** LM Studio (API server lokal di port 1234) dengan Qwen3.5 9B.

**Alasan:**
- Berjalan lokal: privasi, tanpa biaya API
- Qwen3.5 9B memberikan keseimbangan akurasi-kecepatan untuk vision reasoning
- API kompatibel dengan OpenAI format: mudah diganti ke cloud jika perlu
- Bisa dijalankan di PC yang sama dengan server

**Konsekuensi:**
- Perlu GPU dengan VRAM 8-10GB untuk Qwen3.5 9B
- Kualitas reasoning tergantung model yang digunakan (Qwen3.5 9B cukup baik)
- Startup lambat (loading model ~30-60 detik)

---

## ADR-006: gTTS + Web Speech API untuk Text-to-Speech

**Status:** Accepted

**Konteks:** Perlu TTS bahasa Indonesia. Opsi: gTTS (online), Web Speech API (offline browser native), pyttsx3 (offline), Coqui (offline ML).

**Keputusan:** Hybrid: gTTS sebagai default + Web Speech API sebagai opsi offline.

**Alasan:**
- Web Speech API: TTS offline tanpa internet, zero server load, instant
- gTTS: suara natural, aksen Indonesia baik, fallback jika Web Speech tidak tersedia
- TTS cache (`tts_cache.py`) mengurangi latency untuk frasa umum
- Browser mendeteksi kapabilitas dan mengirim `capability` ke server

**Konsekuensi:**
- Dua mode TTS perlu di-handle di frontend dan server
- Web Speech API kualitas suara tergantung browser/OS
- gTTS masih perlu internet untuk generate baru (cache mengurangi ini)

---

## ADR-007: Hold-to-Speak (Push-to-Talk) untuk Input Suara

**Status:** Accepted

**Konteks:** Sistem perlu tahu kapan user mulai dan selesai bicara. Opsi: voice activity detection (VAD) kontinu, push-to-talk / hold-to-speak.

**Keputusan:** Hold-to-speak: user menekan & menahan layar untuk bicara, melepas untuk proses.

**Alasan:**
- Menghindari false positive dari percakapan di sekitar
- Lebih intuitif untuk tunanetra (tactile feedback dari layar)
- Memudahkan segmentasi audio (mulai dan akhir jelas)
- Dual event: touch events (mobile) + pointer events (desktop fallback)
- Safety timeout 30 detik jika pointerup tidak pernah sampai

**Konsekuensi:**
- User harus selalu menyentuh layar: tidak bisa hands-free
- Perlu penanganan event touch + pointer untuk kompatibilitas browser
- Perlu drain time 0.4s setelah touchend untuk menangkap sisa audio

---

## ADR-008: Background Task Monitor untuk Auto-Confirm

**Status:** Accepted

**Konteks:** Setelah memberi panduan arah, user menggeser jempol. Sistem perlu mendeteksi kapan jempol sampai di tombol yang benar tanpa menunggu pertanyaan konfirmasi.

**Keputusan:** Thread daemon terpisah (`background_task_monitor_loop`) yang mengecek posisi jempol setiap 500ms.

**Alasan:**
- Mengurangi jumlah interaksi: user tidak perlu bertanya "apakah ini tombol yang benar?"
- Real-time: konfirmasi otomatis saat jempol menyentuh target
- Thread terpisah agar tidak memblokir pipeline utama
- Snapshot task & intent untuk mencegah race condition

**Konsekuensi:**
- Thread concurrency: perlu `vision_processing_lock`
- Konsumsi CPU tambahan (setiap 500ms YOLO inference small crop)
- False positive jika jempol melewati tombol target

---

## ADR-009: Auto-Scan Layout untuk Deteksi Remote Otomatis

**Status:** Accepted

**Konteks:** User tunanetra tidak bisa melihat apakah remote sudah terdeteksi. Sistem perlu mendeteksi secara otomatis tanpa perintah eksplisit.

**Keputusan:** Background loop `auto_scan_layout()` setiap 3 detik mengecek keberadaan remote.

**Alasan:**
- User tidak perlu mengucapkan "setup layout" secara manual
- Deteksi dini: sistem langsung memproses saat remote masuk frame
- Stabilisasi 1.5 detik mencegah blur/autofocus
- Validasi ukuran (>60000 px) dan margin (50px) cegah false positive

**Konsekuensi:**
- YOLO inference setiap 3 detik (boros GPU jika tidak ada remote)
- Layout reset manual tetap diperlukan untuk ganti remote

---

## ADR-010: Architecture Satu Server (Monolith) dengan Port Tunggal

**Status:** Accepted

**Konteks:** Sistem terdiri dari WebRTC server, HTTP API, dan WebSocket endpoints.

**Keputusan:** Semua service dalam satu proses Python (`server_vision.py`) di port 8080.

**Alasan:**
- Sederhana: tidak perlu orkestrasi multi-service
- Semua data (video, audio, log, TTS) lewat satu koneksi
- Mudah di-deploy dan di-debug
- Cocok untuk penggunaan personal/single-user

**Konsekuensi:**
- Tidak ada isolasi: crash satu komponen menghentikan semua
- Kurang cocok untuk multi-user
- Tidak scalable horizontal

---

## ADR-011: Audio Buffer Hold-to-Speak (Bukan Continuous VAD)

**Status:** Accepted

**Konteks:** Awalnya sistem menggunakan continuous Voice Activity Detection (VAD). Masalah: false positive dari suara sekitar, segmentasi tidak akurat.

**Keputusan:** Hold-to-speak dengan audio buffer (`hold_audio_buffer`) yang dikumpulkan saat `hold_to_speak_active=True` atau `draining=True`.

**Alasan:**
- Segmentasi audio sempurna: mulai dan akhir ditentukan user
- `draining` flag 0.4s menangkap audio setelah touchend
- Filter buffer < 2048 bytes untuk mencegah false positive
- `system_is_busy` flag mencegah tumpang tindih pemrosesan

**Konsekuensi:**
- Frontend perlu handle touch + pointer events
- Safety timeout 30 detik untuk mencegah buffer tak terbatas

---

## ADR-012: Hardcode Tanpa VLM untuk Perintah Umum

**Status:** Accepted

**Konteks:** Beberapa perintah user tidak perlu VLM (lambat, boros GPU, bisa error). Contoh: "senter", "apa tombol ini?", "dimana tombol X?", konfirmasi.

**Keputusan:** Handle perintah umum dengan regex + hardcode lookup, tanpa memanggil LM Studio.

**Alasan:**
- Lebih cepat (ms vs detik)
- Lebih reliable (tidak tergantung kualitas LLM)
- Mengurangi beban GPU
- Pattern matching cukup akurat untuk perintah sederhana

**Perkembangan Terbaru (ADR-012b):** ACTION_INTENTS ditambahkan: 11 grup regex untuk perintah AC umum (power, suhu naik/turun, mode, fan, swing, turbo, eco, sleep, timer on/off). Jika tombol ditemukan di layout, sistem memberikan panduan arah langsung; jika tidak, fallback ke VLM.

**Konsekuensi:**
- Perlu maintenance SYNONYM_GROUPS untuk akurasi matching
- Regex patterns perlu diupdate jika ada variasi bahasa baru
- Tidak bisa handle pertanyaan kompleks tanpa VLM
- ACTION_INTENTS menambah ~90 baris kode di `vision_reasoning.py`
