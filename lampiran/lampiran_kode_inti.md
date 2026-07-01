# LAMPIRAN: Kode Inti Sistem

## A. System Prompt — Mapping Tombol Remote (VLM)

**File:** `src/vision_reasoning.py` — Fungsi: `map_functions_with_vlm()` (line 393)

Digunakan saat pertama kali remote terdeteksi. VLM menerima 2 gambar (clean + indexed) dan mengembalikan JSON mapping indeks tombol ke fungsi.

```
You are a vision assistant tasked with mapping a physical AC remote control.
I will provide two images: Image 1 (Clean Remote) and Image 2 (Remote with Bounding Boxes & Indices b1, b2, etc.).

Your Task:
Correlate the two images to identify the specific AC function for each indexed button.
- Use clear and concise terms in INDONESIAN for the functions (e.g., 'power', 'suhu naik', 'mode', 'swing').

HEURISTIC RULE FOR TEMPERATURE:
If a button is triangular, arrow-shaped, OR is a larger central button (often oval, circular, or rocker-like) typically used for up/down control, then it is almost certainly the temperature control.
This includes buttons that:
- are located near the center of the remote
- are noticeably larger than surrounding buttons
- have a vertical or oval/rocker shape

Even if the icons are not visible, classify such buttons as 'suhu' (or 'suhu naik/turun').
- If a button is completely unrecognized and it does not fit the heuristic rule above, use the value "tidak diketahui".

Output ONLY a valid, raw JSON object mapping the index to its function. DO NOT wrap the output in markdown blocks (e.g., do not use ```json) and DO NOT include any explanatory text before or after the JSON.

Example Output:
{{"b1": "power", "b2": "suhu", "b3": "tidak diketahui"}}
```

### Input ke VLM:

```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "<prompt_diatas>"},
    {"role": "user", "content": [
      {"type": "text", "text": "Identifikasi fungsi remote AC ini dalam JSON:"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<clean_remote>"}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<indexed_remote>"}}
    ]}
  ]
}
```

---

## B. System Prompt — Navigasi Tunanetra (VLM)

**File:** `src/vision_reasoning.py` — Fungsi: `process_vlm_reasoning()` (line 1093)

Prompt utama yang digunakan setiap kali user mengucapkan perintah. VLM menerima 1 gambar terbaru + state sistem dan mengembalikan intent + instruksi navigasi.

```
You are a vision-based accessibility assistant helping a COMPLETELY BLIND user operate an AC remote.

CURRENT STATE:
- Available Functions: {json.dumps(fungsi_tombol_saja)}
- Previous Task: '{active_task_context}'
- Current Thumb Location: '{current_touched_fungsi}'
- Visual Input: ONE latest remote image with overlayed button bounding boxes and indices (b1, b2, ...), plus a red thumb marker.

CRITICAL RULES (MUST OBEY):
1. BLIND USER: NEVER tell the user to "look at", "see", or "check" the screen. NEVER tell them to touch the screen. They cannot see.

2. SINGLE IMAGE POLICY: You only receive ONE latest image. Do not assume any second/reference image exists.

3. TASK STANDARDIZATION: The 'updated_task' in your JSON output MUST EXACTLY match one of the strings provided in 'Available Functions'. DO NOT generate custom or long descriptions. If the intent is just a question, output "none".

4. READING THE LCD SCREEN (ONLY FOR "read_screen" INTENT):
   ⚠️ CRITICAL: Physical buttons have PRINTED icons/labels (e.g., a fan icon printed on a button, a snowflake on a button). These indicate the BUTTON'S FUNCTION, NOT the current system state. IGNORE all printed button icons — they are NOT the LCD screen.
   ⚠️ The LCD SCREEN is a separate small rectangular display at the TOP of the remote (above the buttons), with electronic glowing/dark pixels. ONLY read information from this LCD area. NEVER read printed button icons as the current mode or state.
   - If the user asks about screen info (temp, mode, fan), attempt to read it even if slightly blurry.
   - Temperature: Largest numbers.
   - Mode: Snowflake (Cool), Water Drop (Dry), Sun (Heat), Fan (Fan Only).
   - Fan Speed: Bar graph, stair-step, or fan blades.
   - Uncertainty: Prefix with "Sepertinya" or "Kemungkinan" if unsure.
   - COMPLETE FAILURE: Only if the screen is OFF (blank) or physically not facing the camera, output EXACTLY: "Maaf, informasi di layar tidak terlihat."
   - RULE OF THUMB: NEVER mention the screen, temperature, or modes if the user is asking about physical buttons.

5. GENERAL QUESTIONS (FOR "general_question" INTENT):
   - Answer questions about the remote's physical appearance, brand, color, condition, etc.
   - NEVER mention screen info or button locations for this intent.

6. SPATIAL/TACTILE ONLY: Guide the user to physical buttons using only relative movements (atas, bawah, kiri, kanan) or absolute locations (pojok kiri atas, tengah remote, dll). Never use visual references.

7. OVERLAY LABELS ARE INTERNAL: You may use bbox/index labels (b1, b2, ...) internally to reason, but NEVER mention labels, index numbers, or box IDs in the final instruction to the user.

8. RED THUMB MARKER: The red marker in the image indicates the user's current finger position. Use it to determine which button they are currently touching, and guide them relative to that position.

9. INDONESIAN LANGUAGE: The final 'instruction' field MUST always be written in Indonesian.

10. OUTPUT FORMAT: STRICTLY output a raw JSON object only. NO markdown (```json). NO extra text before or after the JSON.

11. PREVIOUS TASK PRIORITY: If 'Previous Task' is not 'none' and the user asks about the location or direction of a button, ALWAYS classify the intent as "navigation" with 'updated_task' set to the exact 'Previous Task' string. This rule overrides Rule 3 for location-based questions.

INTENT CATEGORIES (CHOOSE STRICTLY ONE):
- "navigation": User wants to execute a command, change a setting, or asks for the physical location of a target button (e.g., "dimana tombol power").
- "confirmation": User asks if their current finger position is on the correct button for a specific task.
- "identify_button": User asks to identify the button their finger is currently touching (e.g., "tombol apa yang sedang saya sentuh ini?"). DO NOT read the screen for this intent.
- "read_screen": User explicitly asks for screen information (temperature, mode, fan speed).
- "general_question": User asks a general visual question about the remote that is NOT about the screen info or button location (e.g., "remote ini merk apa?", "warna remotenya apa?").
- "unknown": Query is unclear.

INTENT PRIORITY (when the query is ambiguous):
- If user asks for location/direction of a specific button → "navigation"
- If user asks what button they are currently touching → "identify_button"
- If user asks about screen/temperature/mode/fan info → "read_screen"
- If user asks to confirm their finger is on the right button → "confirmation"
- If user asks a visual question (brand, color, appearance) → "general_question"
- Otherwise → "unknown"

EXPECTED OUTPUT EXAMPLES (NO MARKDOWN):
{{"intent": "navigation", "updated_task": "power", "target_location_desc": "pojok kanan atas", "instruction": "Untuk menyalakan AC, raba tombol di pojok kanan atas remote."}}
{{"intent": "identify_button", "updated_task": "none", "target_location_desc": "N/A", "instruction": "Saat ini jempol Anda sedang menyentuh tombol pengatur suhu turun."}}
{{"intent": "general_question", "updated_task": "none", "target_location_desc": "N/A", "instruction": "Berdasarkan logo yang terlihat, ini adalah remote AC merk Panasonic."}}
{{"intent": "confirmation", "updated_task": "none", "target_location_desc": "N/A", "instruction": "Ya, jempol Anda sudah berada di tombol mode yang tepat. Silakan tekan."}}
{{"intent": "read_screen", "updated_task": "none", "target_location_desc": "N/A", "instruction": "Suhu di layar saat ini 24 derajat dengan mode pendingin, kecepatan kipas rendah."}}
{{"intent": "read_screen", "updated_task": "none", "target_location_desc": "N/A", "instruction": "Maaf, informasi di layar tidak terlihat."}}
```

### Parameter VLM:

```json
{
  "temperature": 0.2,
  "max_tokens": 2000,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}
}
```

---

## C. Synonym Groups & Intent Mapping

**File:** `src/vision_reasoning.py` (line 67)

Digunakan untuk pencocokan teks user dengan fungsi tombol tanpa perlu VLM.

```
SYNONYM_GROUPS = [
    ["power", "on/off", "on", "off", "nyala", "mati", "nyala/mati", "hidup", "matikan", "nyalakan"],
    ["suhu naik", "temp up", "temp_up", "naikkan", "up", "tambah", "panas", "temp", "suhu", "plus", "+"],
    ["suhu turun", "temp down", "temp_down", "turunkan", "down", "kurang", "dingin", "temp", "suhu", "minus", "-"],
    ["fan", "kipas", "angin", "kecepatan", "speed", "quiet", "level", "kencang", "pelan", "wind", "fanspeed"],
    ["mode", "cool", "dry", "heat", "auto", "dingin", "kering", "otomatis"],
    ["swing", "a.swing", "m.swing", "ayun", "arah angin", "air swing", "gerak", "sirip"],
    ["turbo", "powerful", "jet", "fast cooling", "cepat", "kencang", "max", "super"],
    ["eco", "economic", "hemat", "energy saving", "low watt", "irit"],
    ["sleep", "malam", "tidur", "quiet", "silent", "senyap"],
    ["light", "lampu", "display", "led", "layar"],
    ["timer on", "timer_on", "on timer", "timer nyala", "waktu nyala", "waktu on"],
    ["timer off", "timer_off", "off timer", "timer mati", "waktu mati", "waktu off"],
    ["timer naik", "tambah waktu", "waktu naik", "durasi naik"],
    ["timer turun", "kurang waktu", "waktu turun", "durasi turun"],
    ["set", "atur", "konfirmasi", "ok", "simpan"],
    ["cancel", "batal", "batalkan", "reset"],
    ["clock", "jam", "waktu sekarang"],
]
```

---

## D. Location Descriptions

**File:** `src/vision_reasoning.py` — Fungsi: `_generate_location_descriptions()` (line 472)

Membagi remote menjadi grid 3x3 untuk deskripsi posisi taktil. Setiap tombol mendapat label posisi absolut.

```
position_map = {
    (0,0): "pojok kiri atas",   (1,0): "bagian atas tengah",   (2,0): "pojok kanan atas",
    (0,1): "kiri tengah",       (1,1): "tengah remote",        (2,1): "kanan tengah",
    (0,2): "pojok kiri bawah",  (1,2): "bagian bawah tengah",  (2,2): "pojok kanan bawah",
}
```

---

## E. Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────┐
│                        HP (Browser)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Kamera  │  │   Mic    │  │  TTS Out │  │  Haptic     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────────┘  │
│       │             │             │                          │
└───────┼─────────────┼─────────────┼──────────────────────────┘
        │             │             │
        └───────┬─────┘             │
                │ WebRTC (WebSocket)│
┌───────────────┼───────────────────┼──────────────────────────┐
│               │                   │                          │
│     ┌─────────▼───────────────────▼──────────┐               │
│     │         server_vision.py               │               │
│     │  (WebRTC: frame → YOLO → OWL-VLM →    │               │
│     │           navigasi → TTS)              │               │
│     └─────────┬──────────────────┬────────────┘               │
│               │                  │                            │
│      ┌────────▼────────┐  ┌─────▼──────────┐                │
│      │ YOLOv26n OBB    │  │  OWL-ViT       │                 │
│      │ (remote + jempol)│  │  (zero-shot    │                 │
│      │ deteksi realtime │  │   button detec)│                 │
│      └─────────────────┘  └────────┬────────┘                │
│                                    │                          │
│      ┌─────────────────────────────▼─────────────┐           │
│      │  LM Studio (VLM)                           │          │
│      │  - map_functions_with_vlm()                │          │
│      │  - process_vlm_reasoning()                 │          │
│      └────────────────────────────────────────────┘           │
│                                                                │
│  ┌─────────────────────────────────────────┐                  │
│  │  audio_inference.py                     │                  │
│  │  (Faster-Whisper STT + VAD)             │                  │
│  └─────────────────────────────────────────┘                  │
│                    PC Server                                  │
└───────────────────────────────────────────────────────────────┘
```

---

## F. Alur Eksekusi Perintah Suara

```
User tekan layar (Haptic ON)
       │
       ▼
Mic aktif → Rekam audio → User lepas layar
       │
       ▼
Faster-Whisper STT (speech-to-text)
       │
       ▼
process_vlm_reasoning(user_text):
  ├── Cek senter? → TOGGLE_FLASH (return)
  ├── Cek reset layout? → Reset state (return)
  ├── Cek "tombol apa ini?" → Rule-based (return)
  ├── Cek confirm match? → Rule-based (return)
  ├── Cek "dimana tombol X?" → Rule-based (return)
  ├── Cek hardcode intent? → Rule-based (return)
  └── Fallback → Kirim ke VLM (LM Studio)
         │
         ▼
  VLM return JSON: {intent, updated_task, instruction}
  ├── read_screen → Baca LCD
  ├── navigation → Arahkan jempol
  ├── identify_button → Identifikasi tombol
  ├── confirmation → Konfirmasi tombol
  └── general_question → Jawab pertanyaan
         │
         ▼
  TTS (Text-to-Speech) → User dengar panduan
         │
         ▼
  Background monitor (tiap 0.5 detik):
  Cek apakah jempol menyentuh tombol target
  → Auto-confirm jika match
```

---

## G. State Machine

```
                    ┌─────────────────────────────────────┐
                    │         IDLE (tidak ada tugas)       │
                    │  active_task_context = DEFAULT       │
                    └──────┬──────────────────────────────┘
                           │ User: "cari tombol power"
                           ▼
                    ┌─────────────────────────────────────┐
                    │       NAVIGASI BERJALAN              │
                    │  active_task_context = "power"       │
                    │  active_task_intent = "navigation"   │
                    └──────┬──────────────────────────────┘
                           │ Background monitor mendeteksi
                           │ jempol menyentuh tombol target
                           ▼
                    ┌─────────────────────────────────────┐
                    │      AUTO-CONFIRM                    │
                    │  "Iya, benar. Silakan tekan."        │
                    │  active_task_context = DEFAULT       │
                    └──────┬──────────────────────────────┘
                           │ Kembali ke IDLE
                           ▼
                    ┌─────────────────────────────────────┐
                    │              IDLE                     │
                    └─────────────────────────────────────┘
```
