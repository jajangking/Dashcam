# Video hasil deteksi (untuk ditindak lanjuti)

Kumpulan video **output** dari engine deteksi, supaya kekurangannya bisa dilihat
langsung tanpa harus memproses ulang. Semua H.264 + `yuv420p` (bisa diputar di browser).

> Penting: lihat kolom "Dibuat dengan". Dua video asli dan empat video sintetis
> masih dari engine sebelum commit `06a4750` / `2300cfe`; `model_saja_...` dari
> kode model sebelum penambahan mode ROAD. Hanya `road_cyan_1001183916.mp4`
> yang dibuat dengan kode terbaru. Kalau mau yang lain disegarkan, regenerasi
> seperti di bagian bawah.

| File | Sumber | Durasi / frame | Dibuat dengan |
|---|---|---|---|
| `asli_v1_lama.mp4` | video dashcam pengguna (444 frame) | 14,8 s / 444 | engine v1 (sebelum v2) |
| `asli_v2.mp4` | video dashcam pengguna yang sama | 14,8 s / 444 | engine v2.2 |
| `sintetis_clear.mp4` | `samples/case_video_clear.mp4` | 4,0 s / 120 | engine v2.2 |
| `sintetis_wobble.mp4` | `samples/case_video_wobble.mp4` | 4,0 s / 120 | engine v2.2 |
| `sintetis_belok.mp4` | `samples/case_video_belok.mp4` | 4,0 s / 120 | engine v2.2 |
| `sintetis_toll.mp4` | `samples/case_video_toll.mp4` | 5,0 s / 150 | engine v2.2 |
| `hybrid_1001183916.mp4` | `samples/mentah/mentah_1001183916.mp4` | 14,8 s / 444 | klasik + fallback model UFLD TFLite (commit `e04a642` ke atas) |
| `model_saja_1001183916.mp4` | `samples/mentah/mentah_1001183916.mp4` | 14,8 s / 444 | model UFLD di semua frame, kurva di-smoothing (cyan saja) |
| `road_cyan_1001183916.mp4` | `samples/mentah/mentah_1001183916.mp4` | 14,8 s / 444 | klasik `lane_v2` + overlay cyan `ROAD` (commit baru) |

`asli_v1_lama.mp4` vs `asli_v2.mp4` sengaja berpasangan: footage dan frame-nya sama,
hanya engine-nya beda, jadi perbaikan bisa dibandingkan langsung.

`hybrid_1001183916.mp4` memakai footage yang sama dengan `asli_v2.mp4`; garis
**cyan** adalah hasil model UFLD yang dipanggil saat engine klasik kehilangan
kedua marka. Label "MODEL" muncul di frame tersebut. Ini bukti bahwa model
menolong 65 dari 70 frame yang kosong, bukan pengganti seluruh deteksi.

`model_saja_1001183916.mp4` memakai footage yang sama, tapi **garis kuning klasik
dihilangkan** — semua frame memakai model cyan. Kurva di-smoothing; kalau satu
marka tertutup orang/benda, marka sebelahnya disintesis sejajar memakai profil
lebar terakhir (jalur diasumsikan muat satu mobil). Hasil 444 frame: **438/444**
dua marka, goyakan lebar **1,74 px/frame**, 122 frame disintesis, 6 frame kosong.
Sebelum logika satu sisi: 372/444 dan 2,36 px/frame.

`road_cyan_1001183916.mp4` memakai footage dan frame yang sama, tapi deteksinya
kembali ke `lane_v2` klasik. Garisnya cyan dengan label **ROAD**; marka yang
hilang disintesis dari profil lebar klasik (**EST**), dan saat dua sisi hilang
kurva dilanjutkan dengan kecepatan terakhir maksimal 45 frame (**HOLD**).
Hasil 444 frame: 374/444 dua marka asli, 429/444 sisi kiri dan 421/444 sisi
kanan tergambar, 15 frame kosong, lebar 434 px dengan goyakan **0,00 px/frame**
(dan goyakan sisi bawah 1,00 px/frame). Ini yang paling stabil dan tidak butuh
`models/ufld.tflite`.

## Yang sudah bagus (bisa diverifikasi di video)

- **`sintetis_clear` / `sintetis_wobble`**: garis kiri ikut marka, tidak gemetar.
  Diukur: galat < 1 px, goyangan 1,06–1,54 px/frame.
- **`asli_v2`**: garis kiri menempel marka. Saat garis tergambar, jarak ke marka
  median **0,35 px** dan goyangan median **0,69 px/frame** (diukur pada 444 frame
  hasil rekonstruksi). Cakupan pada file mentah: kiri tergambar 390/444 frame
  (15 tebakan), kanan 390/444 (137 tebakan), 9 frame ditolak penjaga "bukan jalan".
  Bandingkan dengan `asli_v1_lama` yang p90-nya 14,6 px dan sempat "membalik arah"
  sampai 339 px.

## Yang masih kurang (ini yang perlu ditindak lanjuti)

1. **Tikungan kanan tajam masih meleset 24–52 px** — terlihat di `sintetis_belok`
   pada bagian tikungan. Fixture ujinya: `samples/case_belok_240/360/480.jpg`.
   Belum diperbaiki (lihat `README.md` utama).
2. **Sisi kanan jalur hampir tidak terlihat di footage asli.** Di `asli_v2`,
   ±150 dari 444 frame sisi kanan hanya **tebakan** (`right_estimated`), bukan
   deteksi. Garis kanan yang tergambar di sana jangan dipercaya sebagai ukuran.
3. **Video tol / marka pudar** (`sintetis_toll`): estimasi lateral belum layak
   dipakai. Diukur pada video sejenis: 24% frame ditandai "garis bergeser bersama",
   dan titik path 4 m masih goyang ±37 px.
4. **Geometri metrik (jarak/pindah lajur) gagal di footage asli pengguna**:
   `RoadModel.is_valid()` = False, cakrawala terhitung **−514 px** (seharusnya
   ±228 px). Penyebab: lebar jalur dipelajari hanya dari dua titik di
   `u = 0,86–1,0` (14% tinggi gambar), sehingga galat kecil di baris bawah
   meledak saat diekstrapolasi ke cakrawala. Kalau diukur dengan least-squares
   sepanjang `u = 0,62–1,0`, hasilnya 241 px — masuk akal.
5. **Sebagian besar fixture tidak punya generator.** 22 dari 30 file di `samples/`
   tidak bisa dibuat ulang, jadi regresi sulit direproduksi (lihat `AUDIT.md`).
6. **Rekaman malam yang sangat gelap tidak terdeteksi sama sekali.**
   `samples/mentah/mentah_1001183936.mp4` (1080×1920, 315 frame): median piksel 0,
   engine mengembalikan 0 deteksi di semua frame yang disampel. Belum ada mode
   low-light / peningkatan kontras sebelum deteksi.

## Cara membuat ulang

```bash
# video sintetis (pakai kode yang sedang aktif di repo)
python lane_detect.py samples/case_video_clear.mp4  -o out_clear.mp4  --no-show
python lane_detect.py samples/case_video_wobble.mp4 -o out_wobble.mp4 --no-show
python lane_detect.py samples/case_video_belok.mp4  -o out_belok.mp4  --no-show
python lane_detect.py samples/case_video_toll.mp4   -o out_toll.mp4   --no-show
```

Footage aslinya ada di `samples/mentah/` (lihat README di sana):

```bash
python lane_detect.py samples/mentah/mentah_1001183916.mp4 -o asli_v2_ulang.mp4 --no-show
```
