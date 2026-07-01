"""
LAMPIRAN KODE INTI
Remote AC Assistive System for the Blind
==========================================
Berisi potongan kode paling esensial:
  1. System Prompt Mapping Tombol
  2. System Prompt Navigasi Tunanetra
  3. Synonym Groups & Intent Mapping
  4. Deteksi Tombol OWL-ViT
  5. Deteksi Posisi Jempol
  6. Pipeline Navigasi Utama
  7. Layout Setup Pipeline
"""

# ============================================================
# IMPORT & KONFIGURASI GLOBAL
# ============================================================
import json, re, cv2, numpy as np, torch, threading, time, os
from typing import Any
from PIL import Image, ImageDraw, ImageFont

CLASS_ID_REMOTE = 0
CLASS_ID_JEMPOL = 1
WRITE_DEBUG = os.getenv("VISION_DEBUG", "true").lower() == "true"
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
DEFAULT_TASK_CONTEXT = "Belum ada tugas aktif."

conversation_history = []
active_task_context = DEFAULT_TASK_CONTEXT
active_task_intent = None
is_layout_ready = False
layout_data = {}
vision_processing_lock = threading.Lock()


# ============================================================
# 1. SYNONYM GROUPS — Pencocokan fungsi tombol
# ============================================================
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


# ============================================================
# 2. ACTION INTENTS — Hardcode perintah cepat (tanpa VLM)
# ============================================================
ACTION_INTENTS = [
    ([r'(nyalakan|hidupkan|turn on).*(ac|pendingin|dingin)',
      r'(matikan|turn off|shut ?off).*(ac|pendingin|dingin)',
      r'\b(ac|pendingin|dingin)\b.*\b(nyala|mati|hidup)\b'], "power"),

    ([r'(naikkan|nambah|tambah|increase|up|panaskan).*(suhu|temp)',
      r'suhu.*(naik|tambah|up|panas)'], "suhu naik"),

    ([r'(nurunkan|turunkan|kurang|decrease|down|dinginkan).*(suhu|temp)',
      r'suhu.*(turun|kurang|down|dingin)'], "suhu turun"),

    ([r'(ganti|ubah|switch|set).*(mode)',
      r'(set).*(mode).*(cool|dry|heat|auto|fan)'], "mode"),

    ([r'(atur|naikkan|turunkan|set|ganti).*(kipas|fan|angin|kecepatan)',
      r'kecepatan.*(kipas|fan|angin)'], "fan"),

    ([r'(aktifkan|matikan|nyalakan|ganti|set).*(swing|ayun|sirip)',
      r'(arah).*(angin)'], "swing"),

    ([r'(nyalakan|aktifkan|set|pakai).*(turbo|powerful|jet|fast)',
      r'mode.*(turbo|cepat|kencang)'], "turbo"),

    ([r'(nyalakan|aktifkan|set|pakai).*(eco|hemat|energy|irit)',
      r'mode.*(eco|hemat|irit)'], "eco"),

    ([r'(nyalakan|aktifkan|mode).*(sleep|tidur|malam|senyap|quiet)',
      r'\b(tidur|sleep)\b'], "sleep"),

    ([r'(nyalakan|aktifkan|set|atur|pasang).*(timer|waktu)'], "timer on"),

    ([r'(matikan|nonaktifkan|hapus|cancel|batal).*(timer|waktu)'], "timer off"),
]


# ============================================================
# 3. DETEKSI TOMBOL OWL-ViT
# ============================================================
def generate_owl_layout(cv2_image: np.ndarray) -> tuple:
    """Mendeteksi tombol dengan OWL-ViT zero-shot + NMS."""
    color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_coverted)
    img_width, img_height = pil_img.size
    total_area = img_width * img_height

    teks_pencarian = [[
        "a remote", "number buttons", "individual button on a remote control",
        "small buttons", "control buttons", "keypad buttons"
    ]]

    inputs = owl_processor(text=teks_pencarian, images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = owl_processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.09
    )[0]

    remotes, buttons = [], []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [round(i, 2) for i in box.tolist()]
        kata = teks_pencarian[0][label.item()]
        if kata == "a remote":
            remotes.append({"box": box_coords, "score": score.item()})
        else:
            box_w = box_coords[2] - box_coords[0]
            box_h = box_coords[3] - box_coords[1]
            if (box_w * box_h / total_area) > 0.15:
                continue
            buttons.append({"box": box_coords, "score": score.item()})

    buttons = apply_nms(buttons, iou_threshold=0.5)
    remotes = apply_nms(remotes, iou_threshold=0.5)

    # Filter tombol di dalam area remote
    tombol_valid = []
    for btn in buttons:
        bx1, by1, bx2, by2 = btn["box"]
        tx, ty = (bx1 + bx2) / 2, (by1 + by2) / 2
        for rmt in remotes:
            rx1, ry1, rx2, ry2 = rmt["box"]
            if (rx1 <= tx <= rx2) and (ry1 <= ty <= ry2):
                tombol_valid.append(btn)
                break
    if not remotes:
        tombol_valid = buttons

    # Urut kiri-ke-kanan, atas-ke-bawah
    if tombol_valid:
        heights = [b["box"][3] - b["box"][1] for b in tombol_valid]
        row_h = sorted(heights)[len(heights) // 2] * 1.2 if heights else 50
        tombol_valid.sort(key=lambda b: (
            int(((b["box"][1] + b["box"][3]) / 2) // row_h),
            (b["box"][0] + b["box"][2]) / 2,
        ))

    # Gambar bounding box + indeks
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", 30) if os.name == "nt" else ImageFont.load_default()
    local_layout = {}
    for i, btn in enumerate(tombol_valid, 1):
        indeks = f"b{i}"
        x, y, w, h = btn["box"][0], btn["box"][1], btn["box"][2] - btn["box"][0], btn["box"][3] - btn["box"][1]
        local_layout[indeks] = {"coords": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]}
        draw.rectangle(btn["box"], outline="lime", width=4)
        draw.text((btn["box"][0] + 4, btn["box"][1] + 4), indeks, fill="lime", font=font)

    return np.array(pil_img), local_layout


# ============================================================
# 4. VLM MAPPING — Identifikasi fungsi tiap tombol
# ============================================================
def map_functions_with_vlm(clean_b64, indexed_b64, layout_dict, synonym_groups=None):
    """Mengirim 2 gambar (clean + indexed) ke VLM untuk mapping fungsi tombol."""
    system_prompt = """You are a vision assistant tasked with mapping a physical AC remote control.
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
{{"b1": "power", "b2": "suhu", "b3": "tidak diketahui"}}"""

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Identifikasi fungsi remote AC ini dalam JSON:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{indexed_b64}"}},
            ]},
        ],
    }
    # Response diparse dan digabung dengan synonym matching...


# ============================================================
# 5. DETEKSI POSISI JEMPOL + OVERLAY
# ============================================================
def detect_current_thumb_touch(write_debug=False):
    """Ambil frame → YOLO detect jempol → map ke layout tombol."""
    frame = capture_current_frame()
    if frame is None:
        return {"current_touched_fungsi": "tidak ada", "thumb_center": None,
                "current_drawn_cv": None, "reference_drawn_cv": None}

    outputs = process_yolo_rotation(frame, yolo_model, target_class=CLASS_ID_REMOTE)
    current_remote_crop = outputs[0]["image"] if outputs else None
    if current_remote_crop is None:
        return {"current_touched_fungsi": "tidak ada", "thumb_center": None,
                "current_drawn_cv": None, "reference_drawn_cv": None}

    # Deteksi jempol di atas crop remote
    jempol_results = yolo_model(current_remote_crop, classes=[CLASS_ID_JEMPOL], verbose=False)
    thumb_center = None
    thumb_boxes = _result_xyxy_list(jempol_results[0])
    if thumb_boxes:
        bx1, by1, bx2, by2 = thumb_boxes[0]
        thumb_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

    current_drawn_cv = current_remote_crop.copy()
    reference_drawn_cv = reference_indexed_image_cv.copy()
    current_touched_fungsi = "tidak ada"
    h_curr, w_curr = current_drawn_cv.shape[:2]
    h_ref, w_ref = reference_drawn_cv.shape[:2]

    # Overlay bbox tombol pada frame terkini
    for indeks, data in layout_data.items():
        bx, by, bw, bh = data["coords"]
        sx, sy = w_curr / max(w_ref, 1), h_curr / max(h_ref, 1)
        x1, y1, x2, y2 = int(bx * sx), int(by * sy), int((bx + bw) * sx), int((by + bh) * sy)
        cv2.rectangle(current_drawn_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(current_drawn_cv, indeks, (max(0, x1 + 4), max(0, y1 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Gambar titik merah jempol + cek tabrakan dengan tombol
    if thumb_center:
        cv2.circle(current_drawn_cv, thumb_center, radius=30, color=(0, 0, 255), thickness=-1)
        scale_x, scale_y = w_ref / max(w_curr, 1), h_ref / max(h_curr, 1)
        rel_x, rel_y = int(thumb_center[0] * scale_x), int(thumb_center[1] * scale_y)
        cv2.circle(reference_drawn_cv, (rel_x, rel_y), radius=30, color=(0, 0, 255), thickness=-1)

        for indeks, data in layout_data.items():
            bx, by, bw, bh = data["coords"]
            if (bx - 15 <= rel_x <= bx + bw + 15) and (by - 15 <= rel_y <= by + bh + 15):
                current_touched_fungsi = data.get("fungsi", "tidak diketahui")
                break

    return {"current_touched_fungsi": current_touched_fungsi, "thumb_center": thumb_center,
            "current_drawn_cv": current_drawn_cv, "reference_drawn_cv": reference_drawn_cv}


# ============================================================
# 6. LOCATION DESCRIPTIONS — Grid 3×3
# ============================================================
def _generate_location_descriptions():
    """Membagi remote menjadi grid 3×3 dan label posisi tiap tombol."""
    h, w = reference_indexed_image_cv.shape[:2]
    position_map = {
        (0, 0): "pojok kiri atas",       (1, 0): "bagian atas tengah",  (2, 0): "pojok kanan atas",
        (0, 1): "kiri tengah",           (1, 1): "tengah remote",       (2, 1): "kanan tengah",
        (0, 2): "pojok kiri bawah",      (1, 2): "bagian bawah tengah", (2, 2): "pojok kanan bawah",
    }
    row_h = (2 * h / 3) / 3
    row_offset = h / 3
    for indeks, data in layout_data.items():
        bx, by, bw, bh = data["coords"]
        cx, cy = bx + bw / 2, by + bh / 2
        col_idx = min(2, int(cx / (max(w, 1) / 3)))
        row_idx = min(2, int(max(0, cy - row_offset) / max(row_h, 1)))
        data["location_desc"] = position_map.get((col_idx, row_idx), "tidak diketahui")


# ============================================================
# 7. SYSTEM PROMPT NAVIGASI — Prompt utama VLM
# ============================================================
SYSTEM_PROMPT_NAVIGASI_TEMPLATE = """
You are a vision-based accessibility assistant helping a COMPLETELY BLIND user operate an AC remote.

CURRENT STATE:
- Available Functions: {available_functions}
- Previous Task: '{previous_task}'
- Current Thumb Location: '{thumb_location}'
- Visual Input: ONE latest remote image with overlayed button bounding boxes and indices (b1, b2, ...), plus a red thumb marker.

CRITICAL RULES (MUST OBEY):
[1] BLIND USER: NEVER tell the user to "look at", "see", or "check" the screen.
[2] SINGLE IMAGE POLICY: You only receive ONE latest image.
[3] TASK STANDARDIZATION: 'updated_task' MUST match one of Available Functions.
[4] READING LCD: Only for read_screen intent. IGNORE printed button icons.
[5] GENERAL QUESTIONS: Answer visual appearance questions only.
[6] SPATIAL/TACTILE ONLY: Use relative (atas/bawah/kiri/kanan) or absolute positions.
[7] OVERLAY LABELS ARE INTERNAL: Never mention b1/b2 indices to user.
[8] RED THUMB MARKER: Guide based on current thumb position.
[9] INDONESIAN LANGUAGE: Final instruction MUST be in Indonesian.
[10] OUTPUT FORMAT: Raw JSON only. No markdown.
[11] PREVIOUS TASK PRIORITY: If Previous Task exists and user asks location → navigation.

INTENT CATEGORIES:
- "navigation": User wants button location or to execute a command
- "confirmation": User asks if finger is on correct button
- "identify_button": "tombol apa yang saya sentuh?"
- "read_screen": User asks screen info (temp, mode, fan)
- "general_question": Brand, color, appearance
- "unknown": Unclear query

OUTPUT EXAMPLES:
{{"intent":"navigation","updated_task":"power","target_location_desc":"pojok kanan atas","instruction":"...di pojok kanan atas remote."}}
{{"intent":"identify_button","updated_task":"none","target_location_desc":"N/A","instruction":"...tombol pengatur suhu turun."}}
{{"intent":"read_screen","updated_task":"none","target_location_desc":"N/A","instruction":"Suhu 24 derajat, mode dingin."}}
"""


# ============================================================
# 8. PIPELINE UTAMA: process_vlm_reasoning()
# ============================================================
def process_vlm_reasoning(user_text):
    """Pipeline utama: atur perintah → deteksi jempol → VLM → TTS."""
    global conversation_history, active_task_context, active_task_intent, is_layout_ready

    normalized_text = re.sub(r"\s+", " ", user_text.lower()).strip()

    # --- SENTER (tanpa VLM) ---
    if any(k in normalized_text for k in ["senter", "flash", "flashlight", "lampu senter"]):
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "TOGGLE_FLASH"})
        _speak("Senter")
        return

    # --- RESET LAYOUT ---
    if any(p in normalized_text for p in ["reset layout", "ulang layout", "ganti remote"]):
        is_layout_ready = False
        active_task_context = DEFAULT_TASK_CONTEXT
        _speak("Layout direset. Letakkan remote di depan kamera.")
        return

    # --- AUTO-SETUP ---
    if not is_layout_ready:
        auto_setup_layout(silent=False)
        return

    # --- DETEKSI POSISI JEMPOL ---
    with vision_processing_lock:
        touch_info = detect_current_thumb_touch(write_debug=WRITE_DEBUG)
    current_touched_fungsi = touch_info["current_touched_fungsi"]
    thumb_center = touch_info["thumb_center"]
    current_drawn_cv = touch_info["current_drawn_cv"]

    if current_drawn_cv is None:
        _speak("Remote tidak terlihat di kamera. Arahkan kembali.")
        return

    has_active_task = active_task_context != DEFAULT_TASK_CONTEXT
    is_touch_valid = thumb_center and current_touched_fungsi != "tidak ada"

    # --- RULE: "TOMBOL APA INI?" ---
    if re.search(r'(tombol apa|apa ini|jempol.*sentuh apa)', normalized_text):
        if is_touch_valid:
            teks = f"Ini adalah tombol {current_touched_fungsi}."
        else:
            teks = "Jempol anda belum menyentuh tombol manapun."
        _speak(teks)
        return

    # --- RULE: KONFIRMASI ---
    if is_touch_valid and has_active_task and \
       any(re.search(p, normalized_text) for p in [
           r'(udah|sudah|apakah).*(benar|betul|tepat)',
           r'(ini|itu).*(benar|betul|tepat|sesuai)',
           r'sudah (tepat|benar|betul)',
       ]):
        if is_target_matched(current_touched_fungsi, active_task_context):
            _speak("Iya, benar. Ini tombol yang tepat. Silakan tekan sekarang.")
            active_task_context = DEFAULT_TASK_CONTEXT
        else:
            _speak(f"Maaf, ini tombol {current_touched_fungsi}, bukan {active_task_context}.")
        return

    # --- RULE: "DIMANA TOMBOL X?" ---
    where_match = None
    for indeks, data in layout_data.items():
        fungsi = data.get("fungsi", "")
        if fungsi and fungsi not in ["", "tidak diketahui"] and fungsi in normalized_text:
            where_match = indeks
            break
    if where_match:
        lokasi = layout_data[where_match].get("location_desc", "")
        fungsi = layout_data[where_match]["fungsi"]
        if lokasi:
            _speak(f"Tombol {fungsi} berada di {lokasi}.")
            active_task_context = fungsi
            return

    # --- RULE: HARDCODE INTENT PERINTAH ---
    for patterns, target_function in ACTION_INTENTS:
        if any(re.search(p, normalized_text) for p in patterns):
            for indeks, data in layout_data.items():
                fungsi = data.get("fungsi", "")
                if fungsi and fungsi not in ["", "tidak diketahui"] and _match_task_texts(target_function, fungsi):
                    lokasi = data.get("location_desc", "")
                    teks = f"Untuk {fungsi}, tombolnya berada di {lokasi}." if lokasi else f"Tombol {fungsi} sudah ditentukan."
                    _speak(teks)
                    active_task_context = fungsi
                    return
            break

    # --- FALLBACK: KIRIM KE VLM ---
    fungsi_tombol_saja = {i: d.get("fungsi", "?") for i, d in layout_data.items()}
    system_prompt = SYSTEM_PROMPT_NAVIGASI_TEMPLATE.format(
        available_functions=json.dumps(fungsi_tombol_saja),
        previous_task=active_task_context,
        thumb_location=current_touched_fungsi,
    )

    current_guided_b64 = cv2_to_base64(current_drawn_cv)
    messages_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_guided_b64}"}},
            {"type": "text", "text": f"User's current utterance: {user_text}"},
        ]},
    ]

    payload = {
        "model": "local-model",
        "messages": messages_payload,
        "temperature": 0.2, "max_tokens": 2000, "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    }

    response = _safe_post(LM_STUDIO_URL, json=payload, timeout=60)
    if not response:
        return

    content = response.json()["choices"][0]["message"].get("content", "")
    parsed = _extract_json_object(content)

    if isinstance(parsed, dict):
        intent = (parsed.get("intent") or "navigation").strip().lower()
        new_task = (parsed.get("updated_task") or "").strip()
        instruction = (parsed.get("instruction") or "").strip()

        if intent == "read_screen":
            teks = instruction if instruction else "Maaf, informasi di layar tidak terlihat."
        else:
            if new_task and new_task.lower() != DEFAULT_TASK_CONTEXT.lower():
                active_task_context = new_task
            teks = instruction
    else:
        teks = "Maaf, panduan terputus. Bisa ulangi?"

    if teks:
        teks = re.sub(r'\bb\d+\b', '', teks).strip()
        _speak(teks)
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": teks})
        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
            conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]


# ============================================================
# 9. AUTO SETUP LAYOUT — Pipeline lengkap
# ============================================================
def auto_setup_layout(silent=True):
    """Pipeline: deteksi remote → stabilisasi → crop → OWL-ViT → VLM mapping."""
    frame = capture_current_frame()
    if frame is None:
        return False

    outputs = process_yolo_rotation(frame, yolo_model, target_class=CLASS_ID_REMOTE)
    if outputs and outputs[0]["image"] is not None:
        _speak("Remote terlihat. Tahan sebentar...")
        time.sleep(1.5)
        frame_stabil = capture_current_frame()
        outputs_stabil = process_yolo_rotation(frame_stabil, yolo_model, target_class=CLASS_ID_REMOTE)
        remote_crop = outputs_stabil[0]["image"] if outputs_stabil else None
        if remote_crop is None:
            return False

        # Validasi ukuran & margin
        h, w = remote_crop.shape[:2]
        if h * w < 60000:
            _speak("Terlalu jauh. Dekatkan sedikit.")
            return False

        # Ekstrak layout OWL-ViT
        reference_clean_b64 = cv2_to_base64(remote_crop)
        indexed_cv, local_layout = generate_owl_layout(remote_crop)
        reference_indexed_image_cv = indexed_cv.copy()
        reference_indexed_b64 = cv2_to_base64(indexed_cv)

        # Mapping fungsi dengan VLM
        layout_data = map_functions_with_vlm(reference_clean_b64, reference_indexed_b64, local_layout)

        is_layout_ready = True
        _generate_location_descriptions()
        _speak("Pemetaan remote berhasil. Mau saya bantu apa?")
        return True

    if not silent:
        _speak("Remote belum terlihat. Coba arahkan kamera lagi.")
    return False
