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
- `live_app.py` — **mode LIVE**: kamera real-time + mode file (Flask)
- `templates/live.html`, `static/app.css`, `static/live.js` — UI mode live
- `web_app.py` — antarmuka web lama (upload, progres, hasil, halaman uji)
- `make_tests.py` — pembuat contoh uji sintetis (`samples/`)
- `road_geometry.py` — piksel -> meter: jarak, posisi lateral, path jalur, deteksi pindah lajur
- `objects.py` — kontrak objek (mobil/motor/orang) + estimasi jarak per objek
- `tests/` — test pytest (`pytest`, 64 test)
- `AUDIT.md` — hasil audit kodebase (temuan + prioritas perbaikan)
- `requirements.txt` — dependensi Python

> **Syarat penting:** output video H.264 butuh `ffmpeg` terpasang
> (`apt install ffmpeg` / `pkg install ffmpeg` di Termux). Tanpa itu server tetap
> "selesai" tapi file memakai codec `mp4v` yang **tidak bisa diputar di
> Chrome/Firefox**. Mode live menampilkan peringatan bila `ffmpeg` tidak ada.

## Cara pakai

### Mode LIVE — kamera real-time (disarankan untuk mencoba)
```bash
PORT=5001 python -u live_app.py
```
Buka halamannya, tekan **Nyalakan kamera**. Browser mengambil frame dari kamera,
mengirim JPEG kecil (lebar 640 px, ±13 KB) ke `/live/frame`, lalu server
mengembalikan **koordinat kurva + angka** — gambar overlay digambar di browser,
jadi lebih hemat kuota daripada streaming video.

Tersedia juga **Mode file** untuk gambar/video lewat pipeline yang sama (berguna
kalau kamera diblokir), plus pengaturan fps kirim, kualitas JPEG, dan pilihan kamera.

**Kamera butuh HTTPS.** `getUserMedia` diblokir browser pada `http://` biasa.
Kalau Anda membuka `http://<ip-hp>:5001` dari HP lain, izin kamera **tidak akan
muncul** — pakai Mode file, atau taruh di belakang HTTPS (mis. reverse proxy/Caddy,
atau `ngrok`/`cloudflared`). Ini batasan browser, bukan batasan aplikasi.

Biaya kuota mode live pada 8 fps: ±100 KB/detik (±360 MB/jam). Turunkan fps/kualitas
di panel kontrol kalau memakai data seluler.

### Web (mode lama: upload file)
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

Halaman `/tests` menampilkan galeri contoh uji, tapi **kosong pada clone baru** —
jalankan `python make_tests.py --run` lebih dulu untuk mengisi `results/`.

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

Jalankan sendiri: `pip install -r requirements-dev.txt && pytest`

Diukur terhadap geometri asli (`samples/` + `make_tests.road()`):

| Kondisi | Hasil |
|---|---|
| Contoh sintetis normal (lurus, tikungan landai, putus, malam, pudar) | galat **< 1 px** (median 0.50 px) |
| Estimasi sisi yang hilang | **galat maks < 1.2 px** (sebelumnya sampai 754 px) |
| Frame noise / dinding terang | **ditolak** dengan alasan jelas (sebelumnya "jalur terdeteksi, kualitas kuat") |
| Goyangan video | 1.06–1.54 px/frame |
| Tikungan kanan tajam (`case_belok_240/360/480`) | **masih meleset 24–52 px** — belum diperbaiki, lihat di bawah |

### Perbaikan yang sudah masuk
- **Estimasi sisi hilang** sekarang memakai model perspektif: lebar jalur dipelajari
  sebagai fungsi ketinggian (`W(u) = a·u + b`), bukan satu angka konstan. Dulu sisi
  yang tak terlihat digeser sejajar sehingga makin ke atas makin melenceng.
- **Penjaga "bukan gambar jalan"**: kalau >35 % area jalan berwarna putih/kuning, atau
  piksel terang menyebar >14 px di sekitar garis, frame ditolak dan `info["rejected"]
  berisi alasannya. UI menampilkan alasan itu, tidak lagi mengaku "kualitas: kuat".

### Geometri jalan & pindah lajur (baru)
`road_geometry.py` mengubah deteksi marka menjadi besaran metrik. Kuncinya lebar
jalur di layar tumbuh linier terhadap baris gambar: `W(u) = a·u + b`. Detektor
sudah mempelajarinya, jadi:

- cakrawala = `-b·h/a` (terukur **galat 1.1 px** terhadap geometri sintetis),
- skala `m/px` per baris → posisi lateral dalam meter,
- jarak ke depan dari model pinhole: `Z = f_y·H/(y−y_horizon)`,
- **path tengah jalur ke depan** digambar di UI (oranye putus-putus),
- **pindah lajur**: offset lateral terhadap tengah jalur dibandingkan antar frame.

Uji pindah lajur (5 skenario sintetis, 90 frame masing-masing):

| Skenario | Perpindahan terukur | Klasifikasi |
|---|---|---|
| lurus | +0.00 m | di tengah jalur |
| marka bergeser (kendaraan ke kiri) | −1.42 m | bergeser ke kiri ✓ |
| marka bergeser (kendaraan ke kanan) | +1.42 m | bergeser ke kanan ✓ |
| goyangan kamera | +0.14 m | di tengah jalur ✓ |
| goyangan kuat | +0.20 m | di tengah jalur ✓ |

**Penting:** percobaan pertama saya memakai flow optik dan **gagal** — goyangan
kamera justru mendapat skor tertinggi (3.9) dibanding jalan yang benar-benar
belok (1.1). Metode itu saya buang, bukan dipertahankan.

Batasan estimasi jarak: jalan dianggap **datar**, lebar lajur diasumsikan 3.5 m,
dan parameter kamera (tinggi 1.3 m, sudut pandang 55°) masih nilai tipikal —
belum dikalibrasi ke kamera Anda. Angka jarak adalah **estimasi**, bukan sensor.

### Objek (mobil/motor/orang)
`objects.py` sudah berisi kontrak + geometri estimasi jarak per objek (diuji
dengan kotak buatan: objek lebih tinggi di layar = lebih jauh, objek di lajur
sebelah tidak dianggap "di depan kita"). **Belum ada model AI-nya**, karena
bobot YOLO tidak bisa diunduh di lingkungan pengembangan ini. Lihat catatan di
modul tersebut.

### Masih terbuka
- **Tikungan kanan tajam meleset 24–52 px**: `lane_v2._fit()` menurunkan derajat
  polinomial ke garis lurus saat rentang vertikal marka < 15 % tinggi gambar. Di
  `case_belok_480` rentangnya tepat 15.7 %. Perbaikannya: pilih derajat berdasarkan
  penurunan sisa galat, bukan ambang span.
- 22 dari 30 fixture di `samples/` belum punya generator di `make_tests.py`.

## Pengembangan lanjut
- Deteksi area jalan (drivable area) dengan model segmentasi ringan (ONNX).
- Kalibrasi kamera -> estimasi jarak/kecepatan.
- Deteksi kendaraan/plat.
