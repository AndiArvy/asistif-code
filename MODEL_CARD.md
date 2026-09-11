# Model Card: AC Remote and Thumb Detector

## Model details

- **File:** `best.pt`
- **Architecture:** Ultralytics YOLO26n Oriented Bounding Box (OBB)
- **Classes:** remote AC dan ibu jari
- **Developer:** M. Andi Abdillah
- **Purpose:** menormalkan orientasi remote dan memperkirakan posisi ibu jari untuk panduan suara adaptif
- **License:** MIT, mengikuti [LICENSE](LICENSE)

Model ini dilatih sebagai bagian dari tugas akhir *Pengembangan Sistem Instruksi Suara Adaptif Berbasis LLM-Vision Reasoning untuk Membantu Tunanetra Mengoperasikan Remote AC yang Belum Dikenal*.

## Reported evaluation

Pada data validasi penelitian, model mencapai mAP@0.5 sebesar 0,995. Angka ini menggambarkan dataset dan kondisi pengujian tugas akhir dan tidak menjamin performa yang sama pada perangkat, pencahayaan, tangan, atau bentuk remote lain.

## Intended use

Model ditujukan untuk riset dan prototipe teknologi asistif pada remote AC. Model bukan perangkat medis atau sistem keselamatan. Selalu sediakan cara bagi pengguna untuk menghentikan interaksi dan memverifikasi hasil secara mandiri.

## Limitations

Performa dapat menurun akibat blur, oklusi tangan, pencahayaan rendah, remote terlalu kecil dalam frame, atau domain visual yang berbeda dari data pelatihan. Evaluasi pengguna masih terbatas dan belum membuktikan generalisasi pada populasi luas atau penggunaan longitudinal.

## Loading safety

Checkpoint PyTorch dapat menggunakan serialisasi berbasis pickle. Muat hanya file `best.pt` dari release atau commit repository yang dipercaya dan verifikasi integritasnya sebelum deployment.
