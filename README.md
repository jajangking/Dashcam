# Prototipe Deteksi Marka Jalan (Lane Detection)

Prototipe untuk mendeteksi **marka/jalur jalan** dari video atau gambar dashcam.
Mesin deteksi: `lane_v2.py` (OpenCV klasik — ambang warna, window search,
fit polinomial yang distabilkan antar-frame). Tidak butuh model AI besar,
jadi bisa jalan di Termux/Android.

> Catatan: ini deteksi *garis marka*, bukan pemahaman jalan penuh
> (drivable area/segmentation). Akurasi dipengaruhi pencahayaan, cuaca,
> marka pudar, dan posisi kamera.

## Struktur
- `lane_v2.py` — mesin deteksi jalur (dipakai aplikasi web & CLI)
- `lane_detect.py` — CLI pemrosesan video/gambar + transcode H.264
- `web_app.py` — antarmuka web (upload, progres, hasil, halaman uji)
- `make_tests.py` — pembuat contoh uji sintetis (`samples/`)
- `requirements.txt` — dependensi Python

## Cara pakai

### Web (disarankan)
```bash
PORT=5001 python -u web_app.py
```
Buka `http://127.0.0.1:5001` (atau `http://<ip-hp>:5001` dari perangkat lain
di jaringan yang sama). Halaman hasil menampilkan:
- video/gambar hasil overlay,
- **goyangan garis** (median geser posisi per frame — makin kecil makin stabil),
- **cakupan** tiap sisi (berapa persen frame terdeteksi, berapa yang estimasi),
- **kualitas** tiap sisi (kuat/sedang/lemah),
- untuk gambar: **masker deteksi** (area yang dianggap marka).

Halaman `/tests` menampilkan galeri contoh uji (lurus, tikungan, marka
putus-putus, malam, satu sisi, tanpa marka).

### CLI
```bash
python lane_detect.py --input input.mp4 --output output.mp4
python lane_detect.py --image frame.jpg --output out.jpg
```

## Cara kerja (ringkas)
1. **Masker warna** putih/kuning terang di dalam ROI trapesium.
2. **Window search**: histogram di bagian bawah -> jendela pencarian ke atas
   (tahan terhadap marka putus-putus dan satu sisi saja).
3. **Fit polinomial derajat <= 2** pada koordinat ternormalisasi
   (`u = y/h`), dengan regularisasi ke bentuk frame sebelumnya.
   Derajat 2 adalah batas: derajat 3 terbukti membuat garis "berbelok-belok".
4. **Robust IRLS** (bobot Cauchy) supaya potongan yang salah (bayangan,
   mobil lain) tidak menarik garis.
5. **Gerbang kestabilan**: lompatan posisi mendadak ditolak, penyaringan
   bentuk (tidak boleh keluar layar / terlalu miring), dan saringan
   **dukungan piksel** (garis harus benar-benar di atas marka).
6. **Smoothing antar-frame** dilakukan pada *nilai kurva* di beberapa titik
   (bukan koefisien satu-satu), jadi bentuk tidak berubah aneh dan tidak
   terjadi pergeseran bertahap.
7. Sisi yang tidak terlihat **diperkirakan** dari lebar jalur yang dipelajari
   (hanya setelah ≥5 frame melihat kedua sisi kuat), dan ditandai "estimasi".
   Kalau marka hilang >8 frame, garis sisi itu tidak digambar (bukan
   dipertahankan di tempat acak).

## Penyesuaian
Semua ada di `lane_v2.py` (`LaneDetector.__init__`):
- `max_jump_pct` — toleransi lompatan posisi per frame (persen lebar).
- `max_missing` — berapa frame marka boleh hilang sebelum garis disembunyikan.
- `estimate_missing` — boleh/tidak menebak sisi yang tidak terlihat.
- `roi_vertices()` — bentuk area jalan yang diproses.
- `color_mask()` — ambang warna putih/kuning.

## Hasil uji
- Contoh sintetis (`samples/case_*.jpg`): galat < 1 px terhadap geometri asli.
- Video sintetis (lurus, kamera goyang, marka putus, tol): goyangan
  median ≈ 0.01–2.3 px/frame.

## Pengembangan lanjut
- Deteksi area jalan (drivable area) dengan model segmentasi ringan (ONNX).
- Kalibrasi kamera -> estimasi jarak/kecepatan.
- Deteksi kendaraan/plat.
