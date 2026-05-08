import cv2
import numpy as np
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

def process_yolo_rotation(img, model, target_class=1):
    results_list = []
    if img is None: return results_list

    results = model(img, classes=[target_class], verbose=False)

    for i, box in enumerate(results[0].boxes):
        # --- Pre-processing & Padding ---
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w_orig, h_orig = x2 - x1, y2 - y1
        diag = int(np.sqrt(w_orig**2 + h_orig**2))
        pad = diag // 2 
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, ny1 = max(0, cx - pad), max(0, cy - pad)
        nx2, ny2 = min(img.shape[1], cx + pad), min(img.shape[0], cy + pad)
        crop_img = img[ny1:ny2, nx1:nx2]

        def get_area_at_angle(angle):
            rot_img = rotate_image(crop_img, angle)
            res = model(rot_img, classes=[target_class], verbose=False, conf=0.4)
            if len(res[0].boxes) > 0:
                b = res[0].boxes[0]
                rx1, ry1, rx2, ry2 = map(int, b.xyxy[0].cpu().numpy())
                return (rx2 - rx1) * (ry2 - ry1), rot_img[ry1:ry2, rx1:rx2]
            return float('inf'), None

        # --- Efficient 1-Degree Step Search Logic ---
        # Cek titik awal (0 derajat)
        min_area, best_crop = get_area_at_angle(0)
        best_angle = 0
        direction = 0

        # 1. Tentukan Arah: Coba +1 derajat
        area_pos1, crop_pos1 = get_area_at_angle(1)
        
        if area_pos1 < min_area:
            # Jika +1 lebih kecil, arah kita ke kanan (positif)
            direction = 1
            min_area, best_crop, best_angle = area_pos1, crop_pos1, 1
        else:
            # Jika +1 tidak lebih kecil, coba -1 derajat
            area_neg1, crop_neg1 = get_area_at_angle(-1)
            if area_neg1 < min_area:
                # Jika -1 lebih kecil, arah kita ke kiri (negatif)
                direction = -1
                min_area, best_crop, best_angle = area_neg1, crop_neg1, -1
            # Jika keduanya lebih besar dari 0, berarti 0 sudah titik terkecil (direction tetap 0)

        # 2. Tracking: Terus maju 1 derajat sampai area mulai membesar lagi
        if direction != 0:
            while True:
                next_angle = best_angle + direction
                if abs(next_angle) > 90: 
                    break # Limit rotasi manusia
                
                new_area, new_crop = get_area_at_angle(next_angle)
                
                if new_area < min_area:
                    # Masih mengecil, perbarui data terbaik dan terus jalan
                    min_area = new_area
                    best_crop = new_crop
                    best_angle = next_angle
                else:
                    # Area mulai membesar, berhenti di sini! 
                    # best_angle saat ini adalah nilai minimum lokalnya.
                    break

        # --- Finalisasi Portrait ---
        if best_crop is not None:
            fh, fw = best_crop.shape[:2]
            if fw > fh:
                best_crop = cv2.rotate(best_crop, cv2.ROTATE_90_CLOCKWISE)
            
            results_list.append({"image": best_crop, "angle": best_angle, "index": i})

    return results_list