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

### Jalur model ONNX (`lane_model.py`)
Supaya bisa langsung dipakai di mesin yang punya akses internet penuh, sudah
disiapkan pemuat model **UFLD/UFLDv2** (row-anchor, CULane/TuSimple):

```bash
# 1. taruh bobotnya
cp ufld_culane.onnx models/
# 2. cek termuat
python -c "from lane_model import OnnxLaneDetector, find_model; \
print(OnnxLaneDetector(find_model('models')).summary())"
```

Yang sudah diuji di sini: pemuatan, praproses (288x800, normalisasi ImageNet),
pembacaan bentuk masukan/keluaran, pasca-proses `(G*L, R) -> 18 titik x,y`, dan
penolakan yang jujur saat bobot tidak ada. Diuji memakai model uji berbentuk
identik UFLD (`tests/test_lane_model.py`).

Bug yang tertangkap justru dari uji itu: pasca-proses versi pertama membaca sumbu
keluaran terbalik sehingga menghasilkan **800 "baris" alih-alih 18** — angka yang
terlihat masuk akal tapi salah total. Sekarang jumlah baris diperiksa (5-200).

Status model selalu dilaporkan di payload `/live/frame` (`"model": ...`), jadi
tidak ada kesan "sudah pakai AI" kalau bobotnya belum ada.

### Jalur model LiteRT/TFLite di Termux (`lane_model_tflite.py`)
Bobot model ternyata **bisa** diunduh sekarang (REFERENSI.md ditulis saat host
masih tertutup). Memakai UFLD ResNet18/CULane versi LiteRT — 178 MB, asli Android,
bukan ONNX 385 MB.

```bash
pkg install -y python-tflite-runtime
mkdir -p models
curl -L -o models/ufld.tflite \
  https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT/resolve/main/ufld.tflite

python lane_model_tflite.py --input samples/case_clear.jpg --output hasil_model.jpg
python lane_model_tflite.py --input samples/mentah/mentah_1001183916.mp4 --output hasil_model.mp4
```

Bobot **tidak** di-commit (`.gitignore`). Web upload sekarang memakai pilihan
mesin: Klasik kuning (default, sekarang **juga memakai masker area jalan**),
**Klasik + cyan (ROAD)** (disarankan, paling stabil), atau Model UFLD (MODEL).
Mode LIVE tetap memakai engine klasik dan belum di-wire ke model.

Hasil ukur di perangkat ini (jangan dipercaya tanpa data berlabel sendiri):

| Uji | Model UFLD | Engine klasik |
|---|---|---|
| `case_clear`, garis bawah | meleset 3,4 px / 11,4 px | 0,5 px |
| `case_belok_240/360/480` | meleset 50 / 103 / 115 px | < 1,1 px |
| footage asli, 30 frame pertama | 22/30 frame dua marka, selisih 7,6 / 10,3 px | — |
| goyangan lebar jalur, footage asli | 5,1 px/frame | 0,3 px/frame |
| kecepatan (4 thread, CPU) | **660 ms/frame** | ±1 ms/frame |

Kesimpulan jujurnya: model ini **lebih lambat dan lebih goyah di rekaman kita**
(dilatih untuk empat lajur jalan Tiongkok, 800×288 — domain shift). Disimpan
sebagai jalur percobaan dan perbandingan, bukan pengganti default.

#### Mode klasik kuning (default) — rewrite BEV
Mode "Klasik" sekarang memakai detektor baru `lane_bev.py`, bukan sliding-window
`lane_v2`. Alurannya: **IPM ke bird's-eye view → Hough segmen vertikal →
RANSAC → Kalman/EMA → gerbang akhir di gambar asli**. Di BEV, marka yang sejajar
jadi hampir vertikal dan paralel, jadi jauh lebih sulit melintir.

Contoh perintah:

```bash
python lane_detect.py --input video.mp4 --output hasil.mp4
#_engine lama masih bisa:
python lane_detect.py --engine lane_v2 --input video.mp4 --output lama.mp4
```

Dibanding klasik lama + masker area jalan, pada footage 444 frame:

| Metrik | Klasik lama | BEV (baru) |
|---|---|---|
| Sisi kiri tergambar | 385/444 | 376/444 |
| Sisi kanan tergambar | 377/444 | **438/444** |
| Frame dua sisi hilang | 59 | **6** |
| Loncatan > 15 px/frame | 11 | **0** |
| Perubahan slope > 0,15 | 68 | **0** |
| Goyakan kiri / kanan | 0,19 / 0,00 | **0,00 / 0,00** |

Di clip4 120 frame: lost 39 → **11**, loncatan > 15 px 12 → **0**.
Catatan jujur: metrik ini mengukur **konsistensi geometri**, bukan akurasi
terhadap ground truth (dataset berlabel tidak ada). Tapi dua gejala yang kamu
laporkan — lost dan melintir — memang turun drastis. BEV juga belum diuji ulang
pada fixture sintetis dan footage low-light; mode `lane_v2` lama masih tersedia
lewat `--engine lane_v2` kalau diperlukan. Contoh hasil:
`samples/hasil/klasik_bev_1001183916.mp4`.

#### Mode ROAD — klasik + cyan (disarankan)
Deteksi tetap memakai `lane_v2.py` (cepat, tidak butuh bobot 178 MB), tapi
digambar seperti garis model: cyan + halo dengan label **ROAD**. Sekarang ada
**masker area jalan** (diadaptasi dari Ozgunalp dkk. 2019): pencarian marka
dibatasi ke koridor dua kurva terpercaya dari frame sebelumnya, jadi piksel
terang di pagar, bodi mobil, dan gedung tidak lagi menarik sliding-window.
Satu marka yang tertutup orang/benda disintesis memakai profil lebar jalur
klasik (garis putus-putus + **EST**). Kalau dua sisi hilang sebentar, kurva
terakhir dilanjutkan dengan kecepatan geser terakhir (maksimal 45 frame) dan
ditandai **HOLD** — jadi tidak membeku di satu posisi.

```bash
python hybrid_video.py --mode road \
  --input samples/mentah/mentah_1001183916.mp4 \
  --output samples/hasil/road_cyan_1001183916.mp4
```

Di web upload, pilih **"Klasik + cyan — garis ROAD, paling stabil"**.

Hasil footage penuh 444 frame: **377/444** frame dua marka asli, 432 frame sisi
kiri dan 424 frame sisi kanan tergambar (termasuk estimasi/HOLD), **12 frame
kosong**, koridor area jalan aktif di 383 frame, lebar jalur 442 px, goyakan
lebar **0,00 px/frame** dan goyakan sisi bawah **1,00 px/frame**. Artinya lebar
jalur tidak berubah-ubah, tapi garis masih ikut bergeser halus mengikuti jalan.
Bandingkan dengan mode MODEL: 438/444 dan 1,74 px/frame. File contoh:
`samples/hasil/road_cyan_1001183916.mp4`.

#### Mode model saja / cyan (`hybrid_video.py`, opsional)
Mode ini tetap ada untuk perbandingan. Garisnya cyan dengan label **MODEL**,
semua frame diproses UFLD TFLite (lebih lambat, ±660 ms/frame di perangkat ini):

```bash
python hybrid_video.py --mode model \
  --input samples/mentah/mentah_1001183916.mp4 \
  --output samples/hasil/model_saja_1001183916.mp4
```

CLI tanpa `--mode` masih memakai MODEL (kompatibilitas dengan perintah lama);
untuk mode baru wajib ditulis `--mode road`.

Mode lama masih ada untuk perbandingan:
`python hybrid_video.py --mode hybrid --input ... --output ...`
(klasik di semua frame + cyan hanya saat dua marka hilang).

Kurva di-rata-rata antarframe (`alpha=0.35`). Kalau orang lewat dan marka
tertutup, model sempat memilih pasangan lajur yang salah (lebar 208 → 395 px,
tepi kanan bolak-balik 494 ↔ 681 px). Sekarang **setiap kandidat diuji dulu
terhadap kurva terakhir yang dipercaya**: pergeseran tepi < 15% gambar, lebar
maksimal berubah 35%/frame, dan di antara yang lolos dipilih yang paling sedikit
bergeser. Kalau tidak ada yang lolos, kurva lama ditahan sampai 12 frame.

Confidence absolut model ini ternyata rendah di footage asli (median 0,15),
jadi ambang 0,35 akan membekukan semua deteksi. Dipakai ambang 0,05 hanya untuk
membuang kandidat yang benar-benar nol; penahan garis lebih banyak bekerja lewat
konsistensi geometri daripada confidence.

Kalau **satu marka** yang tertutup, jalur diasumsikan tetap muat satu mobil:
marka yang hilang dibayangkan sejajar dengan yang masih terlihat, memakai profil
lebar jalur terakhir yang dipercaya (aling-aling dengan perspektif). Sisi marka
yang terlihat ditentukan dari posisinya terhadap kurva terakhir, jadi bukan
tebakan pasangan lajur sebelah.

Diukur pada clip 120 frame dari footage asli:

| | Model mentah | Smoothing saja | + penahan | + satu sisi disintesis |
|---|---|---|---|---|
| goyakan lebar jalur | 20,21 px/frame | 7,04 px/frame | 4,27 px/frame | **4,95 px/frame** |
| dua marka | 92/120 | 92/120 | 95/120 | **115/120** |

Sintesis dipakai 9 frame; sisanya frame kosong turun dari 25 menjadi 5.
Goyakan naik tipis karena kurva hasil sintesis ikut sedikit goyakan marka yang
masih terlihat, tapi jauh lebih rendah daripada tanpa penahan.

Fixture sintetis tetap mengikuti tikungan, tidak membeku: `clear` 60/60 dengan
goyakan 0,45 px/frame, `wobble` 60/60 dengan 1,44 px/frame, `belok` 60/60 dengan
0,28 px/frame.

Pada footage penuh 444 frame: **438/444 (99%)** dua marka, goyakan lebar **1,74
px/frame**, 122 frame marka disintesis dari satu sisi, 26 frame kurva ditahan,
**6 frame kosong**. Bandingkan sebelum logika satu sisi: 372/444 dan 2,36
px/frame. Jadi cakupan naik sekaligus goyakan turun — tebakan marka sebelah
ternyata lebih stabil daripada membekukan dua garis.

Catatan kalibrasi lama tetap berlaku: `max_stale` 20/30 frame membuat kurva
terlalu lama membeku (goyakan 0,00), jadi dipertahankan 12 frame.

Lewat web UI: buka `http://127.0.0.1:5001`, unggah video, lalu pilih mesin
**"Model UFLD — garis MODEL, lebih lambat"**. Halaman job menampilkan baris
"Model (cyan)", "Goyakan lebar jalur", "Kurva ditahan", dan "Marka disintesis".
Kalau `models/ufld.tflite` tidak ada, proses otomatis jatuh ke engine klasik.
Untuk hasil yang lebih stabil dan jauh lebih cepat, pilih **ROAD** di dropdown
yang sama.

Contoh hasil: `samples/hasil/model_saja_1001183916.mp4`. Catatan jujur: model
masih **lebih lambat** (~660 ms/frame) dan goyakannya masih di atas engine klasik
(0,3 px/frame), tapi tidak ada lagi loncatan warna di tengah video.

### Referensi & open-source
Riset lengkap ada di **[REFERENSI.md](REFERENSI.md)**. Ringkasnya:

- Target realistis dari benchmark CULane: model terbaik hanya **±70 % F1 di
  kategori Curve** dan **±75 % di Crowded**. Jadi tikungan sulit bukan tanda
  kode kita buruk; yang tidak boleh terjadi adalah hasil 0.
- CULane **menganotasi marka yang terhalang kendaraan berdasarkan konteks**,
  bukan dikosongkan. Ini melawan keputusan awal saya, dan sejalan dengan usul
  pengguna.
- Semua jalur memasang model siap-pakai sudah diuji dan gagal **karena tidak ada
  akses ke bobot** (bukan karena belum dicari): aset rilis 404/TLS ditutup,
  4 repo hanya berisi kode, `model_zoo` hanya berisi `url.txt`, HuggingFace &
  Google Drive tidak bisa diakses.

### Satu ukuran lebar jalur (permintaan pengguna)
Sebelumnya sisi yang tidak terdeteksi ditebak dari model lebar dua parameter
`W(u) = a*u + b`. Itu terlalu rapuh. Sekarang:

- **Satu angka** `lane_w_m` diukur dari frame yang kedua marka terlihat, di baris
  acuan `u = 0.95`, lalu dihaluskan kuat (alpha 0.06, lompatan >35% ditolak).
- Kalau satu garis tidak terdeteksi, sisi yang hilang dibangun dari garis yang
  terlihat + lebar itu. Bentuk perspektif tetap dipakai; angka tunggal hanya
  **menormalkan skala**-nya.
- Kalau **kedua** garis tidak terdeteksi, path jalur **tetap tidak digambar**.
  Ini saya coba dan hasilnya buruk: menarik garis bantu ke tepi jalan salah
  (terlihat di footage, garis kiri jatuh ke gerbang). Lebih jujur kosong.

Terukur di footage asli: lebar terukur di **84% frame** (375/444), median 478 px
dengan goyang std 49 px. Sisi kanan yang ditebak: galat median **89.0 -> 90.4 px**,
porsi tebakan 33% -> **30%**. Jadi ini sedikit lebih baik dan lebih sederhana,
bukan lompatan besar. Memakai angka datar (tanpa penskalaan perspektif) justru
lebih buruk: galat kanan **98.7 px**.

### Marka tertutup benda & tikungan tajam (footage mentah asli)
Diuji pada `samples/mentah/mentah_1001183916.mp4` (864x544, 444 frame, dashcam
DDPAI Mola N3 Pro, Jakarta siang, hujan).

**Bug: kurva diekstrapolasi jauh di luar data teramati.** Pada tikungan tajam,
marka kanan hanya terlihat sampai `u = 0.74` (terhalang angkot), tapi kurva
derajat 2 tetap dipakai sampai `u = 1.0`. Diukur pada `case_belok_480` dengan
ground truth generator: galat **0.4 px di dalam data, 223 px di luar data**.
Sekarang rentang teramati (`*_obs_u`) dicatat tiap frame, dan bagian bawah yang
tidak terlihat dibangun dari sisi yang terlihat + lebar jalur (kedua batas jalan
nyata hampir sejajar). Pada video tikungan sintetis: **172 px -> 25 px**.

**Bug: lebar jalur dipelajari dari dua titik ekstrapolasi.** Versi lama memakai
`u=0.86` dan `u=1.0`, yang pada footage asli justru di luar data (titik nol lebar
jalur terhitung di `u=-0.58`, di luar gambar) sehingga geometri metrik mati.
Sekarang lebar dipelajari dengan least-squares pada **irisan rentang teramati
kedua sisi**, dan ditolak kalau cakrawala hasilnya di luar `u 0.25-0.80`.

**Yang TIDAK berhasil dan saya batalkan.** Percobaan memberi fallback cermin
cakrawala `u=0.45` terlihat bagus di sintetis (geometri valid 0% -> 36%), tapi di
footage asli galat sisi kanan **memburuk: median 89 -> 123 px**, lalu 193 px saat
perluasan ikut memakai lebar asumsi itu. Perubahannya dibatalkan, dan perluasan
sekarang hanya boleh memakai lebar jalur terukur (>=3 sampel, bukan asumsi).
Hasil akhir footage asli: **89.0 -> 89.8 px** (praktis tidak berubah).

**Bug "orang melintas di detik ke-2": penyebabnya ketemu, perbaikannya GAGAL.**
Terukur pada footage asli: detik ke-2 = frame 52-67, **15 frame berturut-turut tanpa
deteksi apa pun** (orang menutup marka). Setelah itu penjaga bentuk `_sanity`
tidak punya pembanding, sehingga frame 68 lolos dengan kurva derajat 2 yang
melengkung keluar layar — garis kiri tampak seperti kail menyentuh tepi gambar,
garis kanan menempel di badan angkot. Terukur: **17 dari 444 frame** punya kurva
keluar dari batas gambar.

Dua perbaikan dicoba, keduanya **dibatalkan** karena hasil ukurnya lebih buruk:

| Percobaan | Hasil |
|---|---|
| Penjaga kurva di dalam gambar + pita piksel ramping | 17 -> 17 frame (tidak berubah) |
| Batas pergeseran jendela 0,07w | 17 -> 1 frame, TAPI sisi kanan hilang 14% -> 52% |
| Batas pergeseran jendela 0,10w | 0 frame, TAPI sisi kanan hilang **88%** |

Kesimpulan jujur: jendela pencarian **memang** melompat ke pagar/tembok putih,
tapi membatasinya mematikan deteksi sisi kanan yang markanya sudah tipis.
Membuat ini benar butuh pemisahan objek (segmentasi), bukan tambalan di pencarian
jendela. Itu pekerjaan model, dan modelnya belum bisa diunduh di sini.

**Bug arah belok: saya tidak berhasil membuktikannya.**
Anda melaporkan "jalan belok kanan tapi deteksi belok kiri". Yang bisa saya ukur:
engine lama menggambar kurva dengan arah lengkung **berlawanan dengan marka yang
terlihat** pada 29 dari 360 frame (8%). Saya lacak penyebabnya ke tahap
"ambil ulang piksel di sekitar kurva lalu fit lagi". Tapi tiga penjaga yang saya
coba (arah sama dengan fit awal, arah sama dengan data terlihat, perbandingan
sisa galat) semuanya gagal: ada yang memblokir perbaikan yang benar, ada yang
meloloskan pembalikan arah. Setelah pemeriksaan visual frame 277 dan 380, kedua
"pembalikan" itu ternyata **bacaan yang benar** pada jalan berbelok-S.

Kesimpulan jujur: **saya tidak punya bukti pasti bug arah belok**, dan perbaikan
saya untuk itu **tidak saya masukkan**. Kalau Anda masih melihat arah salah,
kirim timestamp atau screenshot-nya — tanpa itu saya hanya akan menebak lagi.

**Batas yang diakui:**
- Sisi kanan di footage asli masih meleset **~90 px** median. Marka kanan di sana
  memang sebagian besar tidak terlihat; yang diperbaiki adalah sistem berhenti
  mengarang di area yang tidak pernah dilihatnya.
- Pada **satu gambar** tikungan tanpa riwayat frame, manfaat perbaikan belum ada
  (lebar jalur belum pernah terukur). Perbaikan bekerja pada **urutan frame**.
- Video malam potret (`mentah_1001183936.mp4`): marka terdeteksi **45%** frame.
  Engine memang hanya bekerja kalau marka masih terlihat kamera.
- `samples/hasil/asli_v2.mp4` **bukan bukti valid**: overlay-nya berwarna kuning,
  dan kuning diterima `color_mask`. Diuji: frame ber-overlay terdeteksi dengan
  kualitas **1.00/1.00**, padahal setelah overlay dibersihkan sisi kanan tidak
  terdeteksi sama sekali. Video itu mendeteksi gambarnya sendiri.

### Bug yang ditemukan dari keluhan pengguna (path loncat, estimasi melebar)
Dua bug nyata, keduanya sudah diperbaiki:

1. **Geometri metrik praktis tidak pernah aktif.** Lebar jalur hanya dipelajari kalau
   *skor kualitas* kedua sisi >= 0.55, padahal pada video tol/dashgap skor itu wajar
   hanya ~0.33 walaupun kedua marka terdeteksi benar. Terukur: **0 % frame** pada 3
   dari 6 video contoh. Akibatnya path jarak kosong dan angka melompat-lompat.
   Validitas sekarang dijaga oleh pemeriksaan geometri (lebar wajar + menyempit ke
   atas), bukan oleh skor bentuk. Hasil: **0 % -> 100 %**.

2. **"Drift bersama"** — kedua marka bergeser bersama padahal jalannya lurus.
   Terukur di video tol: korelasi geser kiri vs kanan **0.994**, keduanya bergeser
   ~59 px, sementara lebar jalur hanya berubah std 1.1 %. Itu deteksi yang ikut
   mengunci objek bergeser, bukan kendaraan pindah lajur — dan posisi lateral
   dilaporkan mengembara sampai **2.3 m** tanpa alasan.
   Sekarang `LaneDrift` mendeteksinya; saat aktif, path **disembunyikan** dan UI
   menulis terus terang *"estimasi lateral ditahan (garis bergeser bersama)"*,
   alih-alih menampilkan angka yang salah.

Batasan yang tersisa dan tidak saya sembunyikan: pada video tol, 24 % frame masih
ditandai drift, dan pada frame sisanya titik path 4 m masih bergoyang ~37 px. Artinya
**di kondisi marka pudar/jalan tol, estimasi lateral video ini belum layak dipakai** —
yang saya perbaiki adalah agar sistem berhenti menampilkan angka palsu, bukan
membuat deteksinya sempurna.

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
