import cv2
import numpy as np
import math
import os

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(0, 0, 0))


def _process_with_obb(img, result):
    results_list = []
    obb = getattr(result, "obb", None)
    if obb is None:
        return results_list

    # Ambil nilai xywhr (x_center, y_center, width, height, rotation_radian)
    xywhr_array = getattr(obb, "xywhr", None)
    if xywhr_array is None:
        return results_list

    xywhr_array = xywhr_array.cpu().numpy()

    for i, box in enumerate(xywhr_array):
        x_c, y_c, w, h, angle_rad = box
        
        # 1. Konversi sudut ke derajat
        angle_deg = math.degrees(angle_rad)

        # Jika YOLO OBB dan OpenCV berlawanan arah, balik sudutnya
        if INVERT_OBB_ANGLE:
            angle_deg = -angle_deg
        
        # 2. Hitung garis diagonal (Sisi miring)
        # Ini adalah batas ruang paling aman agar objek bisa berputar bebas tanpa terpotong
        diagonal = int(math.ceil(math.hypot(w, h)))
        
        # 3. Hitung batas pemotongan kasar (Region of Interest)
        x1 = int(x_c - diagonal / 2)
        y1 = int(y_c - diagonal / 2)
        x2 = int(x_c + diagonal / 2)
        y2 = int(y_c + diagonal / 2)
        
        # 4. Hitung apakah objek menabrak batas luar layar, siapkan padding (bingkai hitam)
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img.shape[1])
        pad_bottom = max(0, y2 - img.shape[0])
        
        # Koordinat aman di dalam gambar asli
        x1_safe = max(0, x1)
        y1_safe = max(0, y1)
        x2_safe = min(img.shape[1], x2)
        y2_safe = min(img.shape[0], y2)
        
        # Crop area yang agak luas di sekitar objek
        roi = img[y1_safe:y2_safe, x1_safe:x2_safe]
        
        # Terapkan padding jika objek melewati batas tepi kamera
        roi_padded = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # 5. Putar HANYA potongan kecil (ROI) tersebut agar komputasi sangat ringan
        roi_center = (roi_padded.shape[1] / 2.0, roi_padded.shape[0] / 2.0)
        M = cv2.getRotationMatrix2D(roi_center, angle_deg, 1.0)
        rotated_roi = cv2.warpAffine(roi_padded, M, (roi_padded.shape[1], roi_padded.shape[0]))
        
        # 6. Setelah tegak lurus, crop pas sesuai ukuran `w` dan `h`
        crop_x_start = max(0, int(roi_center[0] - w / 2))
        crop_y_start = max(0, int(roi_center[1] - h / 2))
        crop_x_end = min(rotated_roi.shape[1], int(roi_center[0] + w / 2))
        crop_y_end = min(rotated_roi.shape[0], int(roi_center[1] + h / 2))
        
        crop = rotated_roi[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        if crop is None or crop.size == 0:
            continue
            
        # 7. Lock selalu Portrait (Jika objek melebar / landscape, putar ke portrait)
        if FORCE_PORTRAIT:
            ch, cw = crop.shape[:2]
            if cw > ch:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            
        results_list.append({"image": crop, "angle": angle_deg, "index": i})
        
    return results_list


def _process_with_axis_aligned(img, result):
    """Fallback for non-OBB models using axis-aligned boxes."""
    results_list = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return results_list

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
        if FORCE_PORTRAIT:
            h, w = crop.shape[:2]
            if w > h:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        results_list.append({"image": crop, "angle": 0, "index": i})
    return results_list


def process_yolo_rotation(img, model, target_class=1):
    """
    Return portrait remote crops.
    Uses YOLO OBB output when available; falls back to axis-aligned boxes.
    Returns: list of dicts with keys: image, angle, index, raw_boxes
    """
    if img is None:
        return []

    results = model.predict(img, classes=[target_class], verbose=False, conf=CONF_THRESHOLD)
    if not results:
        return []

    result = results[0]
    raw_boxes = _result_xyxy_list(result)

    obb_results = _process_with_obb(img, result)
    if obb_results:
        for item in obb_results:
            item["raw_boxes"] = raw_boxes
        return obb_results

    axis_results = _process_with_axis_aligned(img, result)
    for item in axis_results:
        item["raw_boxes"] = raw_boxes
    return axis_results


# --- KONFIGURASI ---
CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.3"))
FORCE_PORTRAIT = os.getenv("YOLO_FORCE_PORTRAIT", "true").lower() == "true"
INVERT_OBB_ANGLE = os.getenv("YOLO_INVERT_OBB_ANGLE", "false").lower() == "true"


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
            boxes.append((
                int(np.min(x_coords)),
                int(np.min(y_coords)),
                int(np.max(x_coords)),
                int(np.max(y_coords)),
            ))
        return boxes

    axis_boxes = getattr(result, "boxes", None)
    if axis_boxes is not None and len(axis_boxes) > 0:
        for box in axis_boxes:
            boxes.append(tuple(map(int, box.xyxy[0].cpu().numpy())))
    return boxes