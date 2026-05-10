import cv2
import re
import json
import numpy as np
import torch
import threading
import time
from PIL import Image, ImageDraw, ImageFont
from rotate_remote import process_yolo_rotation
from vision_models import get_vision_models
from vision_http import (
    LM_STUDIO_URL,
    LOG_URL,
    safe_post as _safe_post,
    fire_and_forget_post as _fire_and_forget_post,
    capture_current_frame,
    cv2_to_base64,
    extract_json_object as _extract_json_object,
    speak as _speak,
)

CLASS_ID_REMOTE = 0
CLASS_ID_JEMPOL = 1

yolo_model = None
owl_processor = None
owl_model = None


def ensure_models_loaded():
    global yolo_model, owl_processor, owl_model
    if yolo_model is None or owl_processor is None or owl_model is None:
        yolo_model, owl_processor, owl_model = get_vision_models()
        print("Model sudah siap digunakan.")

# =========================
# VARIABEL GLOBAL (STATE)
# =========================
conversation_history = []
DEFAULT_TASK_CONTEXT = "Belum ada tugas aktif."
active_task_context = DEFAULT_TASK_CONTEXT
active_task_intent = None
background_task_monitor_running = False
background_task_monitor_thread = None
vision_processing_lock = threading.Lock()

# State Layout
is_layout_ready = False
reference_clean_b64 = None
reference_indexed_b64 = None
reference_indexed_image_cv = None  # Simpan versi CV2 untuk digambar titik merah nanti
layout_data = {}


def _result_xyxy_list(result):
    """Normalize YOLO detection outputs into xyxy integer boxes."""
    boxes = []
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polys = obb.xyxyxyxy.cpu().numpy()
        for poly in polys:
            pts = np.array(poly).reshape(-1, 2)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            boxes.append(
                (
                    int(np.min(x_coords)),
                    int(np.min(y_coords)),
                    int(np.max(x_coords)),
                    int(np.max(y_coords)),
                )
            )
        return boxes

    axis_boxes = getattr(result, "boxes", None)
    if axis_boxes is not None and len(axis_boxes) > 0:
        for box in axis_boxes:
            boxes.append(tuple(map(int, box.xyxy[0].cpu().numpy())))
    return boxes


def is_similar(word, keywords):
    return any(k in word or word in k for k in keywords)

def _log(text):
    _fire_and_forget_post(LOG_URL, json={"sender": "Sistem", "text": text})


# =========================
# LOGIKA SETUP LAYOUT OTOMATIS
# =========================
def generate_owl_layout(cv2_image):
    """Mendeteksi tombol dengan OWL-ViT dan mengembalikan gambar berindeks & data kotak."""
    ensure_models_loaded()
    # Convert CV2 (BGR) to PIL (RGB)
    color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_coverted)
    
    # --- TAMBAHAN: Hitung total luas gambar untuk referensi filter ---
    img_width, img_height = pil_img.size
    total_area = img_width * img_height

    teks_pencarian = [[
    "a remote",
    "number buttons",
    "individual button on a remote control",
    "small buttons",
    "control buttons",
    "keypad buttons"
    ]]
    inputs = owl_processor(text=teks_pencarian, images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = owl_processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.09
    )[0]

    remotes, buttons = [], []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box_coords = [round(i, 2) for i in box.tolist()]
        kata = teks_pencarian[0][label.item()]
        
        if kata == "a remote":
            remotes.append({"box": box_coords})
        else:
            # --- TAMBAHAN: LOGIKA FILTER AREA UNTUK TOMBOL ---
            # Kalkulasi luas box (x2-x1) * (y2-y1)
            box_w = box_coords[2] - box_coords[0]
            box_h = box_coords[3] - box_coords[1]
            box_area = box_w * box_h
            
            # Jika box tombol > 15% dari total luas gambar, abaikan (anggap over-detection)
            if (box_area / total_area) > 0.15:
                continue # Loncat ke deteksi berikutnya, jangan masukkan ke 'buttons'
                
            buttons.append({"box": box_coords})

    tombol_valid = []
    for btn in buttons:
        bx1, by1, bx2, by2 = btn["box"]
        tx, ty = (bx1 + bx2) / 2, (by1 + by2) / 2
        for rmt in remotes:
            rx1, ry1, rx2, ry2 = rmt["box"]
            if (rx1 <= tx <= rx2) and (ry1 <= ty <= ry2):
                tombol_valid.append(btn)
                break

    # Jika tidak ada remote utuh terdeteksi OWL, anggap semua tombol valid
    if not remotes:
        tombol_valid = buttons

    # --- LOGIKA DRAWING ASLI ANDA (TIDAK DIUBAH) ---
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    local_layout = {}
    button_counter = 1

    for btn in tombol_valid:
        box = btn["box"]
        indeks = f"b{button_counter}"
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        local_layout[indeks] = {
            "coords": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
        }

        # Menggambar kotak lime
        draw.rectangle(box, outline="lime", width=2)
        
        # Logika teks asli Anda: Teks di posisi (x+2, y+2), dengan background hitam rectangle
        text_pos = (box[0] + 2, box[1] + 2)
        # Menghitung perkiraan area background hitam agar teks arial tidak tumpang tindih
        draw.rectangle([text_pos, (text_pos[0] + 25, text_pos[1] + 20)], fill="black")
        draw.text(text_pos, indeks, fill="lime", font=font)
        button_counter += 1

    # Convert back to CV2
    indexed_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return indexed_cv_image, local_layout

def map_functions_with_vlm(clean_b64, indexed_b64, layout_dict):
    """Meminta VLM mengenali fungsi spesifik remote AC untuk tiap indeks."""
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
- If a button are completely unrecognized and it does not fit the heuristic rule above, use the value "tidak diketahui".

Output ONLY a valid, raw JSON object mapping the index to its function. DO NOT wrap the output in markdown blocks (e.g., do not use ```json) and DO NOT include any explanatory text before or after the JSON.

Example Output:
{
  "b1": "power",
  "b2": "suhu",
  "b3": "tidak diketahui"
}"""

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identifikasi fungsi remote AC ini dalam JSON:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{indexed_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }

    try:
        response = _safe_post(LM_STUDIO_URL, json=payload, timeout=60)
        content = response.json()["choices"][0]["message"].get("content", "")
        mapped_functions = _extract_json_object(content) or {}

        # Gabungkan koordinat dan fungsi ke layout_dict
        for key in layout_dict:
            layout_dict[key]["fungsi"] = mapped_functions.get(key, "tidak diketahui")

        with open("layout.json", "w") as f:
            json.dump(layout_dict, f, indent=4)

        return layout_dict
    except Exception as e:
        print(f"Error mapping functions: {e}")
        return layout_dict


def auto_setup_layout(silent=True):
    """Mencari remote, merotasi, mengindeks, dan memetakan fungsinya."""
    global is_layout_ready, reference_clean_b64, reference_indexed_b64, reference_indexed_image_cv, layout_data
    ensure_models_loaded()

    # --- 1. PEMINDAIAN AWAL (Mendeteksi kehadiran remote) ---
    frame = capture_current_frame()
    if frame is None:
        return False

    outputs = process_yolo_rotation(frame, yolo_model, target_class=CLASS_ID_REMOTE)
    
    if outputs and outputs[0]["image"] is not None:
        # --- 2. FASE STABILISASI (Mencegah Blur & Menunggu Autofocus) ---
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})
        _speak("Remote terlihat. Tahan sebentar...")
        
        # Jeda 4 detik. Karena dipanggil via asyncio.to_thread, ini tidak akan membuat server WebRTC crash.
        time.sleep(4.0) 
        
        # --- 3. PENGAMBILAN GAMBAR UTAMA (Setelah Stabil) ---
        _fire_and_forget_post(LOG_URL, json={"sender": "Sistem", "text": "Mengambil gambar jernih..."})
        frame_stabil = capture_current_frame()
        cv2.imwrite("debug_frame.jpg", frame_stabil)
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})
        
            
        outputs_stabil = process_yolo_rotation(frame_stabil, yolo_model, target_class=CLASS_ID_REMOTE)
        remote_crop = outputs_stabil[0]["image"] if outputs_stabil else None
        
        if remote_crop is None:
            _speak("Remote hilang dari pandangan. Silakan arahkan lagi.")
            _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_ON"})
            return False
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        # --- 4. VALIDASI UKURAN GAMBAR (Jarak Remote) ---
        tinggi, lebar = remote_crop.shape[:2]
        luas_area = tinggi * lebar
        
        if luas_area < 40000: # Threshold luas (bisa disesuaikan nanti)
            _speak("Terlalu jauh. Dekatkan sedikit.")
            _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_ON"})
            return False
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        # --- 4b. VALIDASI MARGIN REMOTE (Tidak Terlalu Dekat Tepi Frame) ---
        MARGIN = 3  # Margin dalam pixel dari tepi frame
        frame_height, frame_width = frame_stabil.shape[:2]
        
        # Cek bounding box remote dari YOLO di frame asli
        results_margin_check = yolo_model(frame_stabil, classes=[CLASS_ID_REMOTE], verbose=False)
        margin_boxes = _result_xyxy_list(results_margin_check[0])
        if margin_boxes:
            for x1, y1, x2, y2 in margin_boxes:
                
                # Cek tiap sisi frame dan kumpulkan sisi mana yang terpotong
                clipped_sides = []
                if x1 < MARGIN:
                    clipped_sides.append("kiri")
                if x2 > frame_width - MARGIN:
                    clipped_sides.append("kanan")
                if y1 < MARGIN:
                    clipped_sides.append("atas")
                if y2 > frame_height - MARGIN:
                    clipped_sides.append("bawah")
                
                # Jika ada bagian yang terpotong, berikan pesan
                if clipped_sides:
                    if len(clipped_sides) == 1:
                        # Hanya satu sisi terpotong, bilang sisi mana
                        sisi = clipped_sides[0]
                        message = f"bagian {sisi} terpotong."
                    else:
                        # Lebih dari satu sisi terpotong, bilang generic
                        message = "Terlalu dekat, Jauhkan sedikit."
                    
                    _speak(message)
                    _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_ON"})
                    return False
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        # --- 5. EKSTRAKSI LAYOUT & MAPPING VLM ---
        _fire_and_forget_post(LOG_URL, json={"sender": "Sistem", "text": "Kamera stabil! Mengunci gambar dan mengekstrak layout..."})
        _speak("Memetakan tombol remote...")
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        reference_clean_b64 = cv2_to_base64(remote_crop)
        cv2.imwrite("debug_clean_reference.jpg", remote_crop)
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        indexed_cv, local_layout = generate_owl_layout(remote_crop)
        reference_indexed_image_cv = indexed_cv.copy()
        reference_indexed_b64 = cv2_to_base64(indexed_cv)
        cv2.imwrite("debug_indexed_reference.jpg", indexed_cv)

        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})
        layout_data = map_functions_with_vlm(
            reference_clean_b64, reference_indexed_b64, local_layout
        )

        is_layout_ready = True

        _speak("Pemetaan remote berhasil. Mau saya bantu apa?")
        _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

        return True

    if not silent:
        _speak("Remote belum terlihat. Coba arahkan kamera lagi.")
    return False


import re

def _task_aliases_with_button_index(text):
    """Kembalikan kandidat teks task, termasuk fungsi jika teks berisi indeks tombol."""
    raw_text = (text or "").lower().strip()
    aliases = [raw_text]

    # VLM kadang masih mengembalikan indeks seperti "b3" atau "tombol b3".
    # Treat indeks itu sebagai alias dari fungsi di layout_data.
    index_matches = re.findall(r"\bb\d+\b", raw_text)
    for indeks in index_matches:
        fungsi = layout_data.get(indeks, {}).get("fungsi", "")
        fungsi = fungsi.lower().strip()
        if fungsi and fungsi not in aliases:
            aliases.append(fungsi)

    return aliases


def _match_task_texts(touched_fungsi, current_task):
    """Mengecek apakah dua teks fungsi sama, termasuk sinonimnya."""
    t1 = (touched_fungsi or "").lower().strip()
    t2 = (current_task or "").lower().strip()
    
    # 1. Cek langsung sama persis
    if t1 == t2:
        return True
        
    # --- GUARD CLAUSE KHUSUS TIMER & WAKTU ---
    # Kumpulkan semua kata kunci yang bisa mengubah makna "on/off/nyala/mati" menjadi fitur timer
    timer_keywords = ["timer", "waktu"]
    
    # Cek apakah masing-masing teks mengandung unsur kata timer/waktu
    t1_is_timer = any(kw in t1 for kw in timer_keywords)
    t2_is_timer = any(kw in t2 for kw in timer_keywords)
    
    # Jika salah satu adalah tombol timer/waktu, tapi yang satunya BUKAN, langsung tolak!
    if t1_is_timer != t2_is_timer:
        return False

    # 2. Cek Sinonim / Alias khusus Remote AC
    synonym_groups = [
        ["power", "on/off", "on", "off", "nyala", "mati", "nyala/mati"],
        ["temp up", "suhu naik", "naikkan", "up", "tambah", "panas"],
        ["temp down", "suhu turun", "turunkan", "down", "kurang", "dingin"],
        ["fan", "kipas", "angin", "kecepatan", "speed"],
        ["mode", "cool", "dry", "heat", "auto"],
        ["swing", "a.swing", "m.swing", "ayun", "arah angin", "swing otomatis"],
        # Jangan lupa tambahkan variasi "waktu" ke dalam grup timer agar mereka bisa saling mengenali
        ["timer on", "timer nyala", "waktu nyala", "waktu on"],
        ["timer off", "timer mati", "waktu mati", "waktu off"]
    ]
    
    def contains_word(text, word):
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text))

    for group in synonym_groups:
        is_touched_in_group = any(contains_word(t1, alias) for alias in group)
        is_task_in_group = any(contains_word(t2, alias) for alias in group)
        
        if is_touched_in_group and is_task_in_group:
            return True
            
    return False


def is_target_matched(touched_fungsi, current_task):
    """Mengecek apakah tombol yang disentuh sama dengan tugas, termasuk indeks dan sinonimnya."""
    touched_aliases = _task_aliases_with_button_index(touched_fungsi)
    task_aliases = _task_aliases_with_button_index(current_task)

    for touched_alias in touched_aliases:
        for task_alias in task_aliases:
            if _match_task_texts(touched_alias, task_alias):
                return True

    return False


def detect_current_thumb_touch(write_debug=False):
    """Ambil frame terbaru, rotate remote, deteksi jempol, lalu map ke fungsi tombol."""
    ensure_models_loaded()
    if not is_layout_ready or reference_indexed_image_cv is None:
        return {
            "current_touched_fungsi": "tidak ada",
            "thumb_center": None,
            "current_drawn_cv": None,
            "reference_drawn_cv": None,
        }

    frame = capture_current_frame()
    if frame is None:
        return {
            "current_touched_fungsi": "tidak ada",
            "thumb_center": None,
            "current_drawn_cv": None,
            "reference_drawn_cv": None,
        }

    outputs = process_yolo_rotation(frame, yolo_model, target_class=CLASS_ID_REMOTE)
    current_remote_crop = outputs[0]["image"] if outputs else None

    if current_remote_crop is None:
        return {
            "current_touched_fungsi": "tidak ada",
            "thumb_center": None,
            "current_drawn_cv": None,
            "reference_drawn_cv": None,
        }

    jempol_results = yolo_model(
        current_remote_crop, classes=[CLASS_ID_JEMPOL], verbose=False
    )
    thumb_center = None
    thumb_boxes = _result_xyxy_list(jempol_results[0])
    if thumb_boxes:
        bx1, by1, bx2, by2 = thumb_boxes[0]
        thumb_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

    current_drawn_cv = current_remote_crop.copy()
    reference_drawn_cv = reference_indexed_image_cv.copy()
    current_touched_fungsi = "tidak ada"

    if thumb_center:
        cv2.circle(current_drawn_cv, thumb_center, radius=8, color=(0, 0, 255), thickness=-1)
        h_curr, w_curr = current_drawn_cv.shape[:2]
        h_ref, w_ref = reference_drawn_cv.shape[:2]

        rel_x = int((thumb_center[0] / w_curr) * w_ref)
        rel_y = int((thumb_center[1] / h_curr) * h_ref)
        cv2.circle(reference_drawn_cv, (rel_x, rel_y), radius=8, color=(0, 0, 255), thickness=-1)

        padding = 15
        for indeks, data in layout_data.items():
            bx, by, bw, bh = data["coords"]
            if (bx - padding <= rel_x <= bx + bw + padding) and (by - padding <= rel_y <= by + bh + padding):
                current_touched_fungsi = data.get("fungsi", "tidak diketahui")
                break

    if write_debug:
        cv2.imwrite("debug_frame.jpg", frame)
        cv2.imwrite("debug_cropped.jpg", current_remote_crop)
        cv2.imwrite("debug_current_guided.jpg", current_drawn_cv)
        cv2.imwrite("debug_reference_guided.jpg", reference_drawn_cv)

    return {
        "current_touched_fungsi": current_touched_fungsi,
        "thumb_center": thumb_center,
        "current_drawn_cv": current_drawn_cv,
        "reference_drawn_cv": reference_drawn_cv,
    }


def background_task_monitor_loop():
    """Pantau posisi jempol saat ada task aktif tanpa menunggu pertanyaan konfirmasi."""
    global active_task_context, active_task_intent

    print("[Background Monitor] Aktif. Mengecek posisi jempol tiap 0.5 detik saat intent navigasi.")
    while background_task_monitor_running:
        time.sleep(0.5)

        if (
            not is_layout_ready
            or active_task_context == DEFAULT_TASK_CONTEXT
            or active_task_intent != "navigation"
        ):
            continue

        task_snapshot = active_task_context
        intent_snapshot = active_task_intent
        try:
            with vision_processing_lock:
                touch_info = detect_current_thumb_touch(write_debug=False)
        except Exception as e:
            print(f"[Background Monitor] Gagal cek posisi jempol: {e}")
            continue

        current_touched_fungsi = touch_info["current_touched_fungsi"]
        if current_touched_fungsi in ["tidak ada", "tidak diketahui"]:
            continue

        print(
            f"[Background Monitor] Task: {task_snapshot} | "
            f"Jempol: {current_touched_fungsi}"
        )

        if (
            active_task_context == task_snapshot
            and active_task_intent == intent_snapshot
            and is_target_matched(current_touched_fungsi, task_snapshot)
        ):
            teks = "Nah, yang itu tombolnya."
            _speak(teks)
            active_task_context = DEFAULT_TASK_CONTEXT
            active_task_intent = None
            conversation_history.append({"role": "assistant", "content": teks})
            _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_ON"})


def start_background_task_monitor():
    """Mulai background monitor sekali saja."""
    global background_task_monitor_running, background_task_monitor_thread

    if background_task_monitor_running:
        return

    background_task_monitor_running = True
    background_task_monitor_thread = threading.Thread(
        target=background_task_monitor_loop,
        daemon=True,
    )
    background_task_monitor_thread.start()


def process_vlm_reasoning(user_text):
    global conversation_history, active_task_context, active_task_intent, is_layout_ready
    global reference_clean_b64, reference_indexed_b64, reference_indexed_image_cv, layout_data

    _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_OFF"})

    normalized_text = re.sub(r"\s+", " ", user_text.lower()).strip()
    if any(phrase in normalized_text for phrase in ["reset layout", "ulang layout", "ganti remote", "remote baru"]):
        is_layout_ready = False
        active_task_context = DEFAULT_TASK_CONTEXT
        active_task_intent = None
        _speak("Layout direset. Letakkan remote di depan kamera.")
        return

    if not is_layout_ready:
        sukses = auto_setup_layout(silent=False)
        if not sukses:
            return

    # --- 1-3. PREPROCESSING VISUAL: frame -> rotate -> deteksi jempol -> map ke layout ---
    with vision_processing_lock:
        touch_info = detect_current_thumb_touch(write_debug=True)

    current_touched_fungsi = touch_info["current_touched_fungsi"]
    thumb_center = touch_info["thumb_center"]
    current_drawn_cv = touch_info["current_drawn_cv"]
    reference_drawn_cv = touch_info["reference_drawn_cv"]

    if current_drawn_cv is None or reference_drawn_cv is None:
        _speak("Remote tidak terlihat di kamera. Arahkan kembali.")
        return

    #pengecekan cepat berdasarkan kata kunci untuk menghindari pemanggilan LLM yang tidak perlu.
    normalized_text = user_text.lower().strip()
    words = normalized_text.split()

    # --- TAMBAHKAN LOG INI UNTUK DEBUGGING ---
    print(f"\n[DEBUG VISION] Task Aktif: {active_task_context}")
    print(f"[DEBUG VISION] Python mendeteksi jempol menyentuh: {current_touched_fungsi}\n")

    current_guided_b64 = cv2_to_base64(current_drawn_cv)
    reference_guided_b64 = cv2_to_base64(reference_drawn_cv)

    # --- DETEKSI PERTANYAAN KONFIRMASI TOMBOL ---
    CONFIRM_KEYWORDS = {"benar", "betul", "tepat", "sesuai", "cocok"}
    OBJECT_KEYWORDS = {"ini", "itu", "yang ini", "yang itu", "tombol ini", "tombol itu"}
    
    
    def contains_phrase(text, phrases):
        return any(p in text for p in phrases)

    is_confirm = any(w in CONFIRM_KEYWORDS for w in words)
    is_object = contains_phrase(normalized_text, OBJECT_KEYWORDS)

    has_active_task = active_task_context != DEFAULT_TASK_CONTEXT
    is_touch_valid = thumb_center and current_touched_fungsi != "tidak ada"

    is_asking_confirmation = (
        has_active_task and
        is_touch_valid and
        (is_confirm or is_object)
    )
    
    # Jika user bertanya "apakah benar tombol ini?" dan semua kondisi terpenuhi
    has_active_task = active_task_context != DEFAULT_TASK_CONTEXT
    is_touch_valid = thumb_center and current_touched_fungsi != "tidak ada"

    if is_asking_confirmation and has_active_task and is_touch_valid:
        # Cek apakah tombol yang disentuh cocok dengan task menggunakan sinonim
        if is_target_matched(current_touched_fungsi, active_task_context):
            teks = "Iya, benar. Ini tombol yang tepat. Silakan tekan sekarang."
            _speak(teks)
            active_task_context = DEFAULT_TASK_CONTEXT  # Reset setelah konfirmasi
            active_task_intent = None
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": teks})
            _fire_and_forget_post(LOG_URL, json={"sender": "Control", "text": "MIC_ON"})
            return
        # else:
            # teks = f"Tidak, tombol ini salah. Geser jempol Anda ke tombol yang lain."

    # --- PERSIAPAN DATA PROMPT ---
    fungsi_tombol_saja = {
        indeks: data.get("fungsi", "tidak diketahui")
        for indeks, data in layout_data.items()
    }

    _fire_and_forget_post(LOG_URL, json={"sender": "Sistem", "text": "Menganalisis posisi dan perintah..."})

    # --- 4. PROMPT VLM BARU ---
    system_prompt = f"""You are a vision-based accessibility assistant helping a COMPLETELY BLIND user operate an AC remote.
    
Inputs:
- Image 1: Reference Layout
- Image 2: Actual Condition (with bounding boxes)
- Available Functions: {json.dumps(fungsi_tombol_saja)}
- Previous Task: '{active_task_context}'
- Current Thumb Location: '{current_touched_fungsi}'

CRITICAL ACCESSIBILITY RULES (MUST OBEY):
1. NO VISUAL DESCRIPTIONS: The user is blind. NEVER mention icons, colors, or text.
2. STRICTLY NO IMAGE INDICES: NEVER output bounding box labels like "b1", "b2", etc. 
3. TACTILE & SPATIAL ONLY: Use only directions (atas, bawah, kiri, kanan), relative distances (e.g., "satu tombol ke kanan"), and absolute positions (pojok kiri atas, dll). Keep instructions SHORT.
4. ANSWER WITH INDONESIAN LANGUAGE ONLY.

==================================================
STEP 1: DETERMINE INTENT (STRICT CLASSIFICATION)

- "exploration": User asks what their finger is currently touching. (e.g., "ini tombol apa?")
- "confirmation": User asks if their current position is correct based on a previous task. (e.g., "apakah ini sudah benar?")
- "navigation": User clearly wants to perform an action. (e.g., "nyalakan AC", "turunkan suhu 2 derajat")
- "question": User asks about AC status or display. (e.g., "berapa suhu sekarang?")
- "unknown": Query does not clearly match any category.

CRITICAL: INTENT LOCK
Once intent is determined, strictly follow the rules for that intent. Do not mix logic.

==================================================
STEP 2: TASK & LOCATION EVALUATION

- If "exploration" or "question" or "unknown":
  updated_task = "N/A"
  target_location_desc = "N/A"

- If "navigation":
  updated_task = Map request to EXACTLY one string from 'Available Functions'.
  target_location_desc = Describe its absolute position.

- If "confirmation":
  If '{active_task_context}' is empty, "None", or "N/A":
    Change intent to "unknown" (user has no active task to confirm).
    updated_task = "N/A"
    target_location_desc = "N/A"
  Else:
    updated_task MUST STRICTLY EQUAL '{active_task_context}'.
    target_location_desc = Describe location of '{active_task_context}'.

==================================================
STEP 3: STATUS & INSTRUCTION

- If "exploration":
  If '{current_touched_fungsi}' contains exactly 1 function:
    instruction = "Jempol Anda berada di tombol [Current Thumb Location]."
  If '{current_touched_fungsi}' contains >1 function (finger between buttons):
    instruction = "Jempol Anda berada di antara tombol [List functions]. Geser sedikit agar tepat di satu tombol."
  If '{current_touched_fungsi}' is empty:
    instruction = "Jempol Anda tidak terdeteksi menyentuh tombol apapun."

- If "question":
  If the AC screen is clearly readable: Answer briefly in Indonesian based ONLY on the display.
  If the AC screen is off, blank, occluded, or unreadable: 
    instruction = "Layar remote tidak terbaca atau dalam keadaan mati."

- If "navigation" or "confirmation":
  Compare '{current_touched_fungsi}' with updated_task:
  
  IF MATCH:
    If User Query implies multiple presses (e.g., "turunkan 3 derajat"):
      instruction = "Ya, tepat. Silakan tekan [jumlah] kali."
    Else:
      instruction = "Ya, tepat sekali. Silakan ditekan."
      
  IF NOT MATCH:
    If '{current_touched_fungsi}' is not empty:
      Determine relative path from '{current_touched_fungsi}' to updated_task.
      instruction = "Geser jempol Anda ke [arah relatif, misal: kanan/atas] sejauh [perkiraan jumlah tombol/jarak] untuk menuju tombol [updated_task]." (Example: "Geser ke atas satu tombol...")
    If '{current_touched_fungsi}' is empty:
      instruction = "Mulai raba dari [target_location_desc] untuk mencari tombol [updated_task]."

- If "unknown":
  instruction = "Perintah tidak dikenali. Silakan sebutkan tujuan seperti 'nyalakan AC' atau 'turunkan suhu'."

NEVER MENTION VISUAL ELEMENTS, INDICES, OR USE NON-TACTILE DESCRIPTIONS IN ANY INTENT. NEVER MENTION "b1", "b2", etc. or colors or shapes. ALWAYS USE FUNCTION NAMES AND TACTILE/SPATIAL LANGUAGE ONLY. AND KEEP INSTRUCTION SHORT.

==================================================
OUTPUT FORMAT:
Output ONLY a valid JSON string. 
CRITICAL: DO NOT wrap the output in markdown code blocks (like ```json ... ```). DO NOT include any conversational text outside the JSON structure.

{{
  "intent": "question" | "navigation" | "confirmation" | "exploration" | "unknown",
  "updated_task": "...",
  "target_location_desc": "...",
  "instruction": "..."
}}
"""

    messages_payload = [
        {"role": "system", "content": system_prompt},
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image 1 (Reference Layout - IGNORE FOR CURRENT STATUS):"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{reference_guided_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": "Image 2 (Actual Current Condition - USE THIS TO ANSWER STATUS/DISPLAY QUESTIONS):"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{current_guided_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": f"User's current utterance: {user_text}", 
                },
            ],
        },
    ]

    payload = {
        "model": "local-model",
        "messages": messages_payload,
        "temperature": 0.2,
        "max_tokens": 2000,
    }

    try:
        response = _safe_post(LM_STUDIO_URL, json=payload, timeout=60)
        if not response: return
        
        content_text = response.json()["choices"][0]["message"].get("content", "")
        parsed = _extract_json_object(content_text)

        if isinstance(parsed, dict):
            # --- AMBIL INTENT ---
            intent = (parsed.get("intent") or "navigation").strip().lower()
            
            new_task = (parsed.get("updated_task") or "").strip()
            target_location = (parsed.get("target_location_desc") or "").strip()
            instruction = (parsed.get("instruction") or "").strip()

            print(f"[DEBUG VISION] Intent: {intent}")

            # ========================================================
            # --- JIKA NIATNYA BERTANYA (MEMBACA STATUS LAYAR) ---
            # ========================================================
            if intent == "question":
                teks = instruction if instruction else "Maaf, layar remote kurang jelas."
                active_task_intent = None
                # Tidak perlu update active_task_context, biarkan user bisa lanjut navigasi
                
            # ========================================================
            # --- JIKA NIATNYA NAVIGASI (CARI TOMBOL) ---
            # ========================================================
            else:
                active_task_intent = intent if intent == "navigation" else None
                is_new_task = False
                if new_task and new_task.lower() != DEFAULT_TASK_CONTEXT.lower():
                    if new_task.lower() != active_task_context.lower():
                        is_new_task = True
                        active_task_context = new_task

                # Penyusunan kalimat navigasi
                if intent == "exploration":
                    teks = instruction
                    active_task_context = DEFAULT_TASK_CONTEXT
                elif is_new_task and target_location and intent == "navigation":
                    teks = f"Tombol tujuan berada di {target_location}. {instruction}"
                else:
                    teks = f"{instruction}"
        else:
            teks = "Maaf, panduan terputus. Bisa ulangi?"

        if teks:
            _speak(teks)
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": teks})

    except Exception as e:
        _fire_and_forget_post(LOG_URL, json={"sender": "Error", "text": f"Koneksi LLM gagal: {e}"})
