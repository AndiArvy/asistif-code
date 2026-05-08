import torch
import json # DITAMBAHKAN: Import modul json
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# --- 1. Memuat Model dan Processor ---
print("Sedang memuat model...")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", use_fast=True)
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# --- 2. Siapkan Gambar Lokal ---
nama_file_gambar = "debug_cropped.jpg" 

try:
    image = Image.open(nama_file_gambar).convert("RGB")
except FileNotFoundError:
    print(f"Error: Tidak bisa menemukan '{nama_file_gambar}'")
    exit()

# --- 3. Masukkan Kueri Teks ---
teks_pencarian = [["a remote", "number buttons"]]

# --- 4. Proses & 5. Jalankan Model ---
inputs = processor(text=teks_pencarian, images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# --- 6. Penyaringan ---
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.08)[0]

# --- 7. LOGIKA FILTERING SPASIAL ---
remotes = []
buttons = []

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box_coords = [round(i, 2) for i in box.tolist()]
    kata = teks_pencarian[0][label.item()]
    if kata == "a remote":
        remotes.append({"kata": kata, "box": box_coords, "yakin": round(score.item() * 100, 2)})
    else:
        buttons.append({"kata": kata, "box": box_coords, "yakin": round(score.item() * 100, 2)})

# Validasi tombol harus di dalam remote
tombol_valid = []
for btn in buttons:
    bx1, by1, bx2, by2 = btn["box"]
    tx, ty = (bx1 + bx2) / 2, (by1 + by2) / 2
    for rmt in remotes:
        rx1, ry1, rx2, ry2 = rmt["box"]
        if (rx1 <= tx <= rx2) and (ry1 <= ty <= ry2):
            tombol_valid.append(btn)
            break

# --- 8. VISUALISASI DENGAN INDEKS ---
draw = ImageDraw.Draw(image)
# Cobalah memuat font default yang lebih besar jika tersedia, jika tidak pakai default
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

print("\n--- HASIL DETEKSI DENGAN INDEKS ---")

# --- 8. VISUALISASI DENGAN INDEKS ---
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

print("\n--- HASIL DETEKSI DENGAN INDEKS ---")

# 1. Gambar Remote terlebih dahulu (Warna Biru)
# for rmt in remotes:
#     # Menggunakan kunci "box" sesuai kode asli Anda
#     draw.rectangle(rmt["box"], outline="blue", width=4)
#     draw.text((rmt["box"][0], rmt["box"][1]-20), "REMOTE", fill="blue", font=font)


# DITAMBAHKAN: Dictionary untuk menyimpan data layout json
layout_data = {}


# 2. Gambar Tombol dengan Warna Lime dan Indeks (b1, b2, dst)
button_counter = 1
for btn in tombol_valid:
    box = btn["box"] # Menggunakan kunci "box" sesuai kode asli Anda
    indeks = f"b{button_counter}"
    
    # DITAMBAHKAN: Hitung x, y, w, h dari box (x1, y1, x2, y2)
    x, y = box[0], box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    layout_data[indeks] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
    
    warna_tombol = "lime"
    
    # Gambar Kotak Tombol
    draw.rectangle(box, outline=warna_tombol, width=2)
    
    # Tambahkan background hitam kecil agar teks indeks terbaca jelas
    text_pos = (box[0] + 2, box[1] + 2)
    draw.rectangle([text_pos, (text_pos[0]+25, text_pos[1]+20)], fill="black")
    draw.text(text_pos, indeks, fill=warna_tombol, font=font)
    
    print(f"Indeks: {indeks} | Fungsi Prediksi: {btn['kata']} | Box: {box}")
    button_counter += 1

image.show()
image.save("hasil_indeks.jpg")

# DITAMBAHKAN: Menyimpan hasil ke dalam file layout.json
with open("layout.json", "w") as json_file:
    json.dump(layout_data, json_file, indent=4)
print("\n[INFO] Data layout berhasil disimpan ke 'layout.json'.")