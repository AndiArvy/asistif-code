# API Documentation

Server WebRTC berjalan di port `8080` (default) dan menyediakan endpoint HTTP, WebSocket, serta WebRTC signaling.

## Daftar Isi

- [WebRTC Signaling](#webrtc-signaling)
- [HTTP Endpoints](#http-endpoints)
- [WebSocket Endpoints](#websocket-endpoints)
- [LM Studio API (Eksternal)](#lm-studio-api-eksternal)

---

## WebRTC Signaling

### `POST /offer`

Menerima SDP offer dari browser HP dan mengembalikan SDP answer.

**Request:**
```json
{
  "sdp": "string (SDP offer dari RTCPeerConnection)",
  "type": "offer"
}
```

**Response:**
```json
{
  "sdp": "string (SDP answer)",
  "type": "answer"
}
```

**Catatan:** Server akan membuat `RTCPeerConnection` baru, menangani track video (→ `latest_jpeg` + frame event) dan audio (→ resample 16kHz mono s16 → kirim ke `/audio_feed`).

---

## HTTP Endpoints

### `GET /`

Menyajikan halaman frontend `index.html`.

**Response:** `text/html` — Halaman UI HP dengan WebRTC client, hold-to-speak, log chat, dan kontrol kamera.

---

### `GET /snapshot`

Mengembalikan 1 frame JPEG terbaru dari video stream HP.

**Response:**
- `200 OK` — `image/jpeg` (frame terbaru)
- `404 Not Found` — Kamera belum siap

---

### `GET /video_feed`

Streaming MJPEG (multipart/x-mixed-replace) dari video HP.

**Response:** `multipart/x-mixed-replace;boundary=frame-boundary`

Menggunakan event-driven `asyncio.Event()` — tidak melakukan polling busy-loop.

---

### `POST /trigger_tts`

Generate dan kirim TTS ke semua frontend clients.

**Request:**
```json
{
  "text": "string (teks yang akan diucapkan)",
  "use_cache": true,
  "cache_file": "string (path file cache, opsional)"
}
```

**Response:** `text/plain` — "Suara berhasil dikirim ke HP" atau error.

**Alur:**
1. Cek cache TTS jika `use_cache=true` dan `cache_file` disediakan
2. Path traversal dicegah — file harus di dalam `cache_dir`
3. Jika tidak ada cache → generate via gTTS
4. Simpan ke cache lokal
5. Kirim `{type:"audio", audio:base64, text, playback_rate}` ke semua frontend WebSocket

**Catatan Keamanan:** `cache_file` divalidasi agar tidak keluar dari `cache_dir` (path traversal protection).

---

### `POST /send_log`

Menerima log dari script Python dan meneruskannya ke frontend HP.

**Request:**
```json
{
  "sender": "User | Sistem | Error",
  "text": "string (isi pesan)"
}
```

**Response:** `text/plain` — "Log terkirim ke HP"

**Broadcast:** Pesan dikirim ke semua koneksi `/frontend_ws` sebagai `{type:"log", sender, text}`.

---

### `POST /preload_tts`

Memicu pre-generation semua frasa TTS umum (digunakan saat startup).

**Response:** `text/plain` — "TTS preload completed"

**Endpoint internal** — biasanya dipanggil otomatis saat startup, bisa dipanggil manual untuk reload cache.

---

## WebSocket Endpoints

### `WS /frontend_ws`

WebSocket utama untuk komunikasi dengan frontend HP.

**Pesan dari Server (HP menerima):**
```json
// Audio TTS
{"type": "audio", "audio": "base64...", "text": "string", "playback_rate": 1.2}

// Log chat
{"type": "log", "sender": "User | Sistem | Error", "text": "string"}

// Kontrol khusus
{"type": "Control", "text": "TOGGLE_FLASH"}
```

**Pesan dari Client (server menerima):**
```json
{"type": "hold_action", "action": "start_listening"}
{"type": "hold_action", "action": "stop_listening"}
```

Pesan `hold_action` dari HP diteruskan ke `/command_feed` untuk diproses oleh `audio_inference.py`.

---

### `WS /audio_feed`

Menerima audio dari server dan mengirimkannya ke client Python (audio_inference.py).

**Data:** Binary raw audio — 16-bit signed integer, mono, 16kHz.

**Alur:**
1. `server_vision.py` menerima audio track dari HP via WebRTC
2. Audio di-resample ke format s16, mono, 16000 Hz
3. Byte stream dikirim ke semua client `/audio_feed`

---

### `WS /command_feed`

Meneruskan perintah tap layar / hold-to-speak dari HP ke `audio_inference.py`.

**Pesan:**
```json
{"type": "hold_action", "action": "start_listening"}
{"type": "hold_action", "action": "stop_listening"}
{"text": "ambil_layout"}
```

---

## LM Studio API (Eksternal)

Sistem menggunakan LM Studio sebagai VLM backend di port `1234`. API kompatibel dengan OpenAI chat completions format.

### `POST http://localhost:1234/v1/chat/completions`

**Request:**
```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": [
      {"type": "text", "text": "user query"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
  ],
  "temperature": 0.2,
  "max_tokens": 2000,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}
}
```

**Response:**
```json
{
  "choices": [{"message": {"content": "{\"intent\": \"navigation\", \"updated_task\": \"power\", ...}"}}]
}
```

**Dua tipe request VLM:**
1. **Mapping layout** (`map_functions_with_vlm`) — 2 gambar (clean + indexed) → output JSON mapping b1..bN ke fungsi
2. **VLM Reasoning** (`process_vlm_reasoning`) — 1 gambar (current crop + overlay) + user text → output JSON intent + instruksi

---

## Alur Data Lengkap

```
HP Browser                     Server PC (port 8080)               AI Backend (port 1234)
├── / → index.html             │                                    │
├── /offer (WebRTC)            │                                    │
├── WebRTC video → latest_jpeg │← snapshot GET ← vision_reasoning  │
│                              │← audio_feed WS ← audio_inference  │
├── WebRTC audio → resample 16kHz                                  │
├── /frontend_ws ← TTS audio   │                                    │
│              ← log chat      │                                    │
└── /frontend_ws → hold_action → command_feed → audio_inference    │
                                                                   │
                                         vision_reasoning → LM Studio (VLM)
                                                            ↑ layout mapping
                                                            ↑ VLM reasoning
```
