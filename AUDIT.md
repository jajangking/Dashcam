# Audit Kodebase Dashcam

Tanggal: 2026-11-17 · Branch: `arena/b9715baf-dashcam` (base `84d4335`)

Semua angka di bawah **hasil eksekusi nyata** di sandbox (Python 3.11, OpenCV 5.0.0,
`.venv`), bukan perkiraan dari membaca kode.

---

> **Pembaruan (branch `arena/9850f1b8-dashcam`)** — dua temuan di bawah sudah diperbaiki,
> plus satu bug baru yang justru lebih parah dari semuanya:
>
> | Temuan | Status |
> |---|---|
> | §2.2 deteksi palsu pada frame noise | **Diperbaiki** — penjaga `_sanity()` menolak frame dengan alasan |
> | *(baru)* estimasi sisi hilang meleset hingga **754 px** | **Diperbaiki** — lebar jalur kini model perspektif `W(u)=a·u+b`, galat < 1.2 px |
> | §2.1 tikungan kanan tajam 24–52 px | **Belum** — butuh pemilihan derajat berbasis sisa galat |
>
> Bug estimasi tidak tertangkap audit pertama karena saya dulu hanya menguji gambar
> statis, tidak pernah menyembunyikan satu marka untuk memaksa jalur estimasi dipakai.
> Sekarang ada 44 test pytest yang mengunci keduanya (`tests/test_lane.py`).

## 1. Ringkasan

Kabar baiknya dulu: **mesin deteksinya jauh lebih baik daripada kesan yang diberikan
kodebase ini.** `lane_v2.py` adalah pekerjaan yang serius dan dipikirkan — IRLS dengan
bobot Cauchy, regularisasi ke kurva frame sebelumnya, gerbang dukungan piksel, smoothing
pada nilai kurva (bukan koefisien). Itu semua keputusan yang benar, dan hasilnya terbukti.
Pada fixture sintetis, **9 dari 11 kasus galat < 1 px**.

Yang membuat repo ini terasa "jauh dari bagus" bukan mesinnya, tapi **kulit di sekelilingnya**:

| Area | Kondisi |
|---|---|
| Akurasi deteksi | Bagus (median 0.50 px) **kecuali tikungan kanan tajam** → sampai 52 px |
| Stabilitas antar-frame | Bagus (1.06–1.54 px/frame) |
| Kecepatan | Cukup (42 fps @540p, 14 fps @1080p) — tapi 1 menit video 1080p = ±130 detik |
| Deteksi palsu | **Buruk**: frame noise total dilaporkan "jalur terdeteksi, kualitas: kuat" |
| Aplikasi web | Prototipe, tapi nyata dipakai → banyak lubang |
| Kemampuan reproduksi | **Buruk**: 22 dari 30 fixture uji tidak bisa dibuat ulang |
| Struktur kode | Klasik "satu file jadi-jadian", tanpa test, tanpa CI, tanpa template |

Satu file `web_app.py` 504 baris berisi 3 halaman HTML sebagai string, thread manager,
dan renderer. `lane_v2.py` 725 baris satu kelas tanpa satu pun unit test. README menjanjikan
lebih dari yang benar-benar ada (lihat §5).

**Prioritas #1 bukan menambah fitur, tapi memasang jaring pengaman** (test + fixture yang
bisa digenerate) supaya perbaikan tikungan kanan dan deteksi palsu itu tidak merusak
yang sudah berjalan.

---

## 2. Temuan kritis (P0)

### 2.1 Tikungan ke kanan tajam: galat 24–52 px, dan mesin tidak menyadarinya

Ini bug paling mahal, karena justru di kondisi yang paling penting untuk dashcam (belokan).

Galat RMS terhadap geometri marka asli (`make_tests.road`), hanya pada baris yang markanya
terlihat di layar:

| Kasus | Kiri | Kanan |
|---|---|---|
| `case_clear`, `case_dashed`, `case_night`, `case_faded`, `case_belok_0/120` | 0.25–0.48 px | 0.51–0.81 px |
| `case_belok_240` | 0.44 px | **24.37 px** |
| `case_belok_360` | 0.42 px | **33.62 px** |
| `case_belok_480` | 0.41 px | **52.04 px** |

Bukti akar masalah (lihat `lane_v2.py:183`):

```
def _fit(...):
    deg = 2 if (span >= self.h * 0.15 and len(points) >= 150) else 1
```

Pada tikungan kanan tajam, marka kanan hanya memotong sebagian layar → rentang vertikalnya
menyusut. Terukur: span kanan **15.7–16.9 %** pada `case_belok_480` vs **41.9 %** di kiri.
Ambang `0.15 * h` tepat berada di zona itu, sehingga **marka kanan diturunkan ke derajat 1
(garis lurus)** sementara jalannya melengkung keras.

Yang membuatnya berbahaya: tidak ada sinyal apa pun ke pengguna. `info` melaporkan
`right_span_pct: 15.7` dan `right_quality: 0.67`, dan overlay tetap menulis
**"Jalur terdeteksi (kiri+kanan) · kualitas: kuat"**. Pengguna tidak punya cara tahu
bahwa sisi kanan sedang menggambar garis lurus di jalan yang menikung.

**Perbaikan:** jangan pakai derajat sebagai keputusan biner berdasarkan span. Pakai
seleksi model dengan penalti (AIC/BIC) atau `max(abs(sisa kurva derajat-1))`: kalau kurva
derajat 2 menurunkan sisa galat secara signifikan, terima derajat 2 walau span pendek.
Dan apa pun hasilnya, laporkan ketidakpastiannya.

### 2.2 Deteksi palsu: frame yang isinya cuma noise dilaporkan "kualitas: kuat"

```
noise acak 640×480 → left=True right=True, quality 1.0 / 1.0,
                     dukungan piksel 0.96, overlay "kualitas: kuat"
```

Penyebabnya dua, dan keduanya soal **skala**:

1. `lane_v2.py:278` — blob diterima sejak `area < 25` piksel. Fitur noise 3×3 = 9 px
   ditolak, tapi noise yang saling menempel lolos.
2. Skor keyakinan menjenuh terlalu cepat: `quality = 0.7 * min(1, span/(0.30*h)) + 0.3 * min(1, len(pts)/1200)`
   (`lane_v2.py:400`). Window search mengembalikan ribuan titik, jadi `len(pts)/1200`
   langsung 1.0. Noise mana pun yang membentuk dua gumpalan akan mengunci skor ~1.0.

**Perbaikan:** normalisasi jumlah titik terhadap luas ROI (bukan konstanta 1200), naikkan
`min area` relatif terhadap resolusi, dan tambah uji "apakah masker punya struktur garis
memanjang" (rasio `area / bbox`, atau uji konsistensi arah antar-blob) sebelum menerima.

### 2.3 Hasil video tidak bisa diputar di browser kalau `ffmpeg` tidak terpasang

Terbukti: di sandbox ini `ffmpeg` tidak ada, job selesai `status: done`, dan FOURCC file
hasil = **`FMP4`** (`mp4v`). Chrome/Firefox tidak memutar `mp4v`; hasilnya video hitam.

Lebih buruk lagi `lane_detect.py:270` menelan kegagalan secara diam-diam:

```python
try:
    from lane_v2 import LaneDetector
    det = LaneDetector()
except Exception:
    det = None          # → silently pakai lane_detect Hough yang jauh lebih buruk
```

`process_video` juga sama sekali tidak memeriksa `writer.isOpened()`. README tidak pernah
menyebut `ffmpeg` sebagai syarat. **Perbaikan:** cek `shutil.which("ffmpeg")` di startup,
tampilkan banner jelas + perintah instalasi, dan gagal keras (bukan diam-diam) kalau
transcode tidak mungkin.

### 2.4 Nama file non-Latin ditolak, pesan galatnya kosong

```
secure_filename("路.mp4")     -> "mp4"      -> ekstensi hilang -> "Format tidak didukung: "
secure_filename("видео.mp4")  -> "mp4"      -> idem
```

Pengguna yang memakai nama file Mandarin/Arab/Rusia **tidak bisa upload sama sekali**, dan
yang mereka lihat adalah galat dengan format kosong di ujungnya. Untuk aplikasi yang
menyasar pengguna HP Indonesia ini akan muncul cepat. **Perbaikan:** ambil ekstensi dari
nama asli (`.suffix.lower()`), slug-kan hanya bagian batang nama, fallback ke `job_id`.

### 2.5 Aplikasi berbohong ke pengguna setelah server restart

`JOBS` hanya di memori. Restart → semua job hilang, tapi:

* halaman `/job/<id>` muncul sebagai *"Job tidak ditemukan (status bisa hilang setelah
  server restart...)"* — pesan yang menyalahkan pengguna;
* `serve_result` tetap menyajikan `results/<id>_result.mp4` yang **masih ada di disk**.

Jadi setelah restart, "hasil dikatakan hilang padahal tidak". Minimal: pindai folder
`results/` dan pulihkan job dari nama file, atau perjelas pesannya.

---

## 3. Penting (P1)

### 3.1 22 dari 30 fixture uji tidak bisa dibuat ulang
`make_tests.py` hanya membuat 8 gambar + 2 video. `case_belok_*` (5), `case_dash_*` (9),
`case_only_left/right`, `case_dash_big_gap`, dan 4 video tol/belok **tidak punya generator**.
Akibatnya tidak ada yang bisa meregresi kasus tikungan kanan (§2.1) secara otomatis.
`/tests` juga selalu kosong di clone baru karena membaca `results/test_*.jpg` yang
git-ignored dan tidak pernah dibuat otomatis.

### 3.2 Tidak ada satu pun test
`pytest` tidak ada, CI tidak ada. Padahal infrastrukturnya sudah 90 % siap: `make_tests.road()`
menghasilkan geometri yang diketahui persis, jadi ground-truth tinggal dipanggil. Saya sudah
membuktikan ini bisa dipakai sebagai metrik otomatis (itulah cara saya menemukan §2.1).
**Ini investasi paling murah dengan hasil terbesar di seluruh daftar ini.**

### 3.3 `/status/<id>` membocorkan path server dan seluruh log internal
Respons JSON memuat `"input_path": "/home/user/Dashcam/uploads/..."` plus array `logs`.
Tidak perlu ada. Selain itu endpoint ini tidak punya rate limit sementara `/tests` dan
`/results/` terbuka tanpa autentikasi.

### 3.4 Memori: `JOBS` tak terbatas + upload 300 MB di-buffer penuh
`JOBS` tidak pernah dibersihkan; tiap entri menyimpan log. Upload 300 MB ditahan di memori
(Flask menyimpan ke `SpooledTemporaryFile` lalu `f.save()`), artinya beberapa upload
paralel bisa menghabiskan RAM server kecil. Di Termux/HP ini akan langsung OOM.
Saran: batas 300 MB → turunkan (atau streaming), dan pakai antrean berkapasitas 1–2 worker
bukan `threading.Thread` tanpa batas.

### 3.5 Penanganan galat 413 dan galat upload masih HTML telanjang
Upload > 300 MB menghasilkan halaman 413 bawaan Werkzeug (tanpa CSS, tanpa jalan kembali).
Begitu juga 404. Tambahkan `@app.errorhandler`.

### 3.6 Latihan/test beban 1080p terlalu lambat untuk dipakai
```
960×540 : 41.7 fps
1920×1080: 13.9 fps   → 1 menit video 30 fps butuh ±130 detik
```
Kalau targetnya "upload video dashcam HP", ini akan terasa lama. Jalur perbaikan termurah:
deteksi pada frame yang di-resize ke lebar ~640 px, lalu skalakan koefisien balik ke resolusi
asli (galat yang diperkenalkan biasanya < 1 px, perlu diverifikasi dengan fixture §3.1).

### 3.7 `web_app.py:13` mengimpor tiga simbol yang tidak dipakai
```python
from lane_detect import detect_lanes, draw_overlay, process_video   # hanya process_video dipakai
```
`detect_lanes`/`draw_overlay` adalah fallback Hough di `lane_detect.py` yang praktis mati
(`draw_overlay` hanya dipanggil dari jalur `det is None`). Ini menyisakan **dua mesin deteksi
di satu repo** dan mengaburkan "mesin mana yang sebenarnya dipakai".

### 3.8 Analitik "turn" dan "offset" tidak punya arti terkalibrasi
`turn_label()` mengklasifikasi tikungan dari sepersekian persen lebar gambar — tidak ada
hubungan dengan radius tikungan nyata. `center_offset()` hanya berarti kalau lebar jalur
sudah dipelajari (`lane_w_bottom`), tapi tetap ditampilkan walau nilainya dari estimasi
cermin. Untuk dashcam, nilai seperti ini lama-lama dianggap kebenaran oleh pengguna.
Beri label "(estimasi kasar)" atau sembunyikan sampai kalibrasi ada.

---

## 4. Kebersihan kode (P2)

* **Dead code:** `make_line_points()` dan `_smooth_line()` di `lane_detect.py` — 0 pemanggil.
* **3 blok `try/except: pass` identik** untuk `progress_cb` di `lane_detect.py` — jadikan helper.
* **CSS diduplikasi 3×** di `INDEX_HTML`, `JOB_HTML`, `TESTS_HTML`. Pindahkan ke
  `templates/` + `static/app.css` dan pakai `render_template`. Ini juga menghapus kebutuhan
  `render_template_string` dan mematikan auto-refresh via `<meta>` (ganti `fetch` ke
  `/status/<id>` — **endpoint itu sudah ada dan tidak terpakai oleh halaman mana pun**).
* **`stats_out.clear(); stats_out.update(stats)`** dipanggil tiap frame padahal hanya dibaca
  setelah selesai (`lane_detect.py:305`). Tidak berbahaya, tapi menandakan kode ini ditulis
  sambil berjalan.
* **`except Exception` terlalu luas** di mana-mana (`process_video`, `_log`, `finally`),
  termasuk yang menyembunyikan kegagalan writer di §2.3.
* **Nilai ajaib tanpa nama:** `0.62`, `0.58`, `0.15`, `1200`, `25`, `300` tersebar di
  `lane_v2.py`. Kumpulkan sebagai konstanta kelas dengan komentar alasan.
* **Tidak ada type hint konsisten / docstring fungsi publik** di `web_app.py`.
* **`requirements.txt` tanpa versi** → tidak reproducible. Tambahkan batas atas/bawah.

---

## 5. Ketidaksesuaian README vs kenyataan

| README bilang | Kenyataan |
|---|---|
| "galat < 1 px terhadap geometri asli" | Benar untuk 9 kasus; **tidak benar** untuk tikungan kanan tajam (24–52 px) |
| "Video sintetis: goyangan 0.01–2.3 px/frame" | Terukur 1.06–1.54 px/frame (konsisten), tapi **hanya 3 dari 6 video** yang diuji |
| Halaman `/tests` menampilkan galeri input vs hasil | Kosong pada clone baru — butuh `python make_tests.py --run` lebih dulu, tidak disebut |
| "bisa jalan di Termux/Android" | Belum diuji; tanpa `ffmpeg` hasil video tidak bisa diputar di browser (§2.3) |
| Tidak menyebut | `ffmpeg` wajib untuk output H.264, padahal itu justru syarat utama |

---

## 6. Catatan khusus Vercel

Anda mengaktifkan konektor Vercel. Sebaiknya jangan langsung deploy: **arsitektur ini tidak
akan jalan di Vercel Functions**, dan gagalnya akan membingungkan.

* Batas body request Vercel Functions ± **4.5 MB**. Video dashcam 1 menit = 30–100 MB.
  Upload akan ditolak sebelum kode Anda dijalankan.
* Batas durasi eksekusi: 10–60 detik (tergantung paket). Satu video 1080p butuh ±130 detik
  hanya untuk deteksi, belum transcode H.264.
* Tidak ada disk persisten → `uploads/`, `results/`, dan `JOBS` di memori hilang tiap
  invocation. Model thread + job polling tidak punya tempat tinggal.
* `opencv-python-headless` + `ffmpeg` mudah melewati batas ukuran bundel.

**Yang realistis:** pakai Vercel hanya untuk frontend statis (upload langsung ke object
storage via presigned URL), dan taruh worker Python ber-`ffmpeg` di layanan yang punya
disk + proses panjang (Fly.io, Railway, Cloud Run, atau VPS). Kalau tujuannya demo cepat,
deploy ke VPS/Fly dan lewati Vercel dulu.

### 6.1 Apa yang sudah dicoba untuk mode LIVE

Mode live (`live_app.py`) saya buat dan uji end-to-end di sandbox ini: 30–80 klien
berurutan, frame grayscale, frame 1280×720, dan payload rusak — semua tertangani, tanpa
traceback. Latensi server **median 36 ms** per frame 640×360, throughput ±88 fps.

Hasil uji klien (DOM tiruan, jsdom): 13 frame terkirim dalam 1,6 detik, HUD terisi,
overlay digambar, garis estimasi memakai pola putus-putus, tombol start/stop bekerja,
tanpa error JS.

**Tapi live mode di Vercel tetap tidak cocok**, dan alasannya lebih halus daripada
mode file:

* Vercel Functions bersifat stateless. `LaneDetector` menyimpan
  `left_coeff/right_coeff/left_quality/lane_w_bottom` **antar frame** — itulah yang
  membuat garis tidak bergetar. Di serverless, tiap invocation bisa mendarat di instance
  berbeda, jadi state itu hilang dan garis akan **bergetar jauh lebih parah** daripada
  di lokal, padahal kelihatannya "jalan".
* Flask tidak bisa dipakai apa adanya (`app.run`). Harus dibungkus `api/index.py`
  yang mengekspos `app` sebagai WSGI callable + `vercel.json` dengan rewrite.
* `app.py` akan butuh `ffmpeg` sebagai binary di dalam bundel function (ratusan MB) untuk
  mode file — praktis tidak realistis.

Kesimpulan praktis: **mode live paling pas di jalankan di VPS/Fly/Railway** (proses
panjang, state di memori), dan Vercel hanya untuk halaman statis kalau nanti UI-nya
dipisah.

---

## 7. Urutan perbaikan yang saya sarankan

**Sprint 1 — jaring pengaman (½ hari).** Tanpa ini, perbaikan berikutnya berbahaya.
1. `make_tests.py` digeneralisasi supaya **semua 30 fixture** bisa dibuat ulang
   (spesifikasi dash/belok/tol dipindah ke satu tabel `CASES`).
2. `tests/test_lane.py` + pytest: galat RMS terhadap ground-truth untuk setiap fixture,
   plus test untuk frame grayscale, 1×1, portrait, dan frame noise (harus **tidak**
   melaporkan deteksi). Target: test ini **gagal** pada kondisi sekarang untuk
   `case_belok_240/360/480` dan noise — itu memang tujuannya.
3. CI GitHub Actions: `pytest` + `ruff`.

**Sprint 2 — akurasi & kejujuran (1–2 hari).**
4. Ganti pemilihan derajat polinomial dengan seleksi model berbasis sisa galat (§2.1).
5. Perbaiki skor kualitas + `min area` relatif resolusi (§2.2).
6. Laporkan ketidakpastian per sisi di overlay ("estimasi", "data tipis") alih-alih
   "kualitas: kuat" yang menyesatkan.

**Sprint 3 — aplikasi web (1–2 hari).**
7. Ekstraksi ekstensi dari nama asli (§2.4), pemulihan job dari disk (§2.5),
   `errorhandler` 404/413, hapus `input_path` & log dari `/status`.
8. Template + static dipisah; ganti auto-refresh `<meta>` dengan polling `fetch` ke
   `/status/<id>` yang sudah ada.
9. Cek `ffmpeg` saat startup dengan pesan jelas (§2.3); worker queue berkapasitas.

**Sprint 4 — performa (opsional).**
10. Deteksi pada frame ter-resize + skala balik; ukur dampak akurasinya dengan test Sprint 1.

---

## 8. Kalau saya lanjut

Saya sarankan mulai dari **Sprint 1**, karena:
* hasilnya terlihat langsung (test merah yang mendokumentasikan bug §2.1 & §2.2),
* tidak mengubah perilaku apa pun yang sudah bekerja,
* membuat dua bug terbesar itu aman diperbaiki sesudahnya.

Bilang saja "kerjakan Sprint 1" (atau pilih nomor lain), dan saya kerjakan di branch ini.
