# Footage mentah (video sumber)

Dua rekaman asli yang dipakai untuk menguji engine di luar data sintetis. Di-commit
atas permintaan pemilik rekaman, supaya hasil di `samples/hasil/` bisa diverifikasi
dan dibuat ulang oleh siapa pun.

| File | Resolusi | Durasi / frame | Isi | Hasil deteksi (kode di HEAD, `2300cfe`) |
|---|---|---|---|---|
| `mentah_1001183916.mp4` | 864×544 | 14,8 s / 444 | jalan siang, marka kiri jelas, marka kanan sebagian hilang | kiri tergambar **390/444** frame (15 tebakan), kanan **390/444** (137 tebakan), **9 frame** ditolak penjaga "piksel terang menyebar terlalu lebar" |
| `mentah_1001183936.mp4` | 1080×1920 (potret) | 10,5 s / 315 | malam, hampir seluruh frame hitam (median piksel = 0, p99 = 211) | **0 frame terdeteksi** (sampel 12 frame) — engine belum punya mode gelap |

## Kenapa dua video ini yang dipakai

`mentah_1001183916.mp4` adalah footage yang menghasilkan `samples/hasil/asli_v1_lama.mp4`
(engine lama) dan `asli_v2.mp4` (engine v2.2). Frame dan durasinya sama, jadi
kedua output itu bisa dibandingkan langsung tanpa variabel lain.

`mentah_1001183936.mp4` sengaja disertakan sebagai bukti batas: rekaman malam yang
terlalu gelap membuat deteksi kosong total. Ini bukan bug yang disembunyikan —
engine memang hanya bekerja kalau marka masih terlihat kamera.

## Cara membuat ulang hasilnya

```bash
# output video (butuh beberapa menit di HP)
python lane_detect.py samples/mentah/mentah_1001183916.mp4 -o ulang_864.mp4 --no-show
python lane_detect.py samples/mentah/mentah_1001183936.mp4 -o ulang_1080.mp4 --no-show
```

Atau lewat web UI: `PORT=5001 python web_app.py` lalu unggah filenya.

## Catatan privasi

Kedua file adalah rekaman pribadi pemilik repo. Jangan dipakai ulang di luar
keperluan pengujian proyek ini.
