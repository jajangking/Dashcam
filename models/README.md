# Bobot model marka

Folder ini untuk berkas model `.onnx` (UFLD/UFLDv2). **Tidak ada bobot yang
di-commit** karena berukuran puluhan MB dan lisensinya milik proyek asalnya.

## Cara memasang

Letakkan satu berkas `.onnx` di folder ini, lalu aplikasi akan mendeteksinya
otomatis (`lane_model.find_model()`):

```bash
# muat modelnya saja, tanpa menjalankan deteksi
python -c "from lane_model import OnnxLaneDetector, find_model; \
d=OnnxLaneDetector(find_model('models') or 'tidak_ada.onnx'); print(d.summary())"
```

Model yang cocok: **UFLD / UFLDv2** (row-anchor) dengan bentuk masukan
`(1, 3, 288, 800)` dan keluaran `(1, G*L, R)` — mis. ekspor dari
`cfzd/Ultra-Fast-Lane-Detection` (CULane) atau `cfzd/Ultra-Fast-Lane-Detection-v2`.

## Kalau bentuknya berbeda

`OnnxLaneDetector` membaca bentuk masukan/keluaran dari model dan menyesuaikan
jumlah baris anchor serta panjang griding sendiri. Kalau modelnya tipe lain
(segmentation seperti LaneNet, atau anchor-based seperti CLRNet/LaneATT),
pasca-prosesnya **berbeda** dan modul ini akan menolak dengan pesan jelas —
bukan menghasilkan angka ngawur. Untuk tipe itu, tulis pasca-proses baru di
`lane_model.py` bagian `postprocess`.

## Target kualitas (jangan berharap lebih)

F1 per kategori CULane dari literatur — bukan janji, tapi patokan:

| Metode | Crowded | Curve | Night | No line |
|---|---|---|---|---|
| UFLD | 70,2 | 69,7 | 66,7 | 44,4 |
| UFLDv2 | 74,9 | 69,7 | 70,6 | 49,0 |

Artinya di jalan Jakarta yang ramai, sekitar **1 dari 4 deteksi bisa salah**.
Yang penting bukan "sempurna", tapi berhenti membuat garis yang tidak ada.
