import cv2
import numpy as np
import math

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
        
        # 1. Konversi sudut dari radian ke derajat
        angle_deg = math.degrees(angle_rad)
        
        # 2. Putar keseluruhan gambar berpusat pada objek agar objeknya "tegak"
        M = cv2.getRotationMatrix2D((x_c, y_c), angle_deg, 1.0)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # 3. Crop kotak secara lurus (karena gambar sudah diputar)
        x_start = max(0, int(x_c - w / 2))
        y_start = max(0, int(y_c - h / 2))
        x_end = min(img.shape[1], int(x_c + w / 2))
        y_end = min(img.shape[0], int(y_c + h / 2))
        
        crop = rotated_img[y_start:y_end, x_start:x_end]
        
        if crop is None or crop.size == 0:
            continue
            
        # 4. Lock selalu Portrait (Rotasi 90 derajat jika landscape)
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
        h, w = crop.shape[:2]
        if w > h:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        results_list.append({"image": crop, "angle": 0, "index": i})
    return results_list


def process_yolo_rotation(img, model, target_class=1):
    """
    Return portrait remote crops.
    Uses YOLO OBB output when available; falls back to axis-aligned boxes.
    """
    if img is None:
        return []

    results = model.predict(img, classes=[target_class], verbose=False, conf=0.3)
    if not results:
        return []

    result = results[0]
    obb_results = _process_with_obb(img, result)
    if obb_results:
        return obb_results

    return _process_with_axis_aligned(img, result)