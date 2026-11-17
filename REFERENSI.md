# Referensi & proyek open-source untuk deteksi marka

Ditulis setelah keluhan pengguna soal arah belok dan marka terhalang. Tujuan
dokumen ini: memisahkan **apa yang sudah terbukti di lapangan** dari **apa yang
belum kita punya**, supaya tidak ada klaim berlebihan di README.

## 1. Benchmark yang dipakai seluruh bidang ini

| Benchmark | Isi | Relevansi untuk kita |
|---|---|---|
| **CULane** | 133.235 frame, 55 jam, Beijing. 9 kategori uji: Normal, **Crowded**, Dazzle, Shadow, **No line**, Arrow, **Curve**, Cross, **Night** | Desain kategorinya **sama dengan kota kita**: ramai dan berbelok |
| **TuSimple** | jalan tol relatif bersih | paling tidak mirip rekaman kita |
| **CurveLanes** | khusus tikungan sulit | untuk mengukur tikungan |

Catatan penting dari halaman resmi CULane: untuk marka yang **terhalang kendaraan
atau tidak terlihat, anotasi tetap dibuat berdasarkan konteks** — bukan
dikosongkan. Artinya pendekatan "kalau tidak terlihat, jangan gambar apa-apa"
yang sekarang kita pakai **melawan praktik standar bidang ini**.

## 2. Angka yang harus jadi target, bukan harapan

F1 per kategori di CULane (dari tabel pembanding metode; UFLDv2 = ResNet34):

| Metode | Total | Normal | **Crowded** | No line | **Curve** | **Night** |
|---|---|---|---|---|---|---|
| UFLD (2019) | 72,3 | 90,7 | 70,2 | 44,4 | 69,7 | 66,7 |
| UFLDv2 (2023) | 75,9 | 92,5 | 74,9 | 49,0 | 69,7 | 70,6 |
| CLRNet (DLA34) | 80,5 | — | — | — | — | — |

Pesan yang tidak nyaman: **bahkan model terbaik 2026 hanya ±70 % F1 di kategori
Curve dan ±75 % di Crowded.** Jadi "estimasinya salah mulu di tikungan" bukan
tanda kita bodoh — itu batas atas bidang ini. Tapi bedanya: mereka masih
menghasilkan ~70 % benar; kita kadang menghasilkan 0.

## 3. Proyek open-source yang nyata (bukan tiruan)

| Proyek | Bintang | Isi | Bisa dipakai di sini? |
|---|---|---|---|
| [`voldemortX/pytorch-auto-drive`](https://github.com/voldemortX/pytorch-auto-drive) | 953 | benchmark lengkap: UFLD, LSTR, LaneATT, SCNN, ENet | butuh PyTorch + bobot; di sandbox ini tidak bisa |
| [`cfzd/Ultra-Fast-Lane-Detection`](https://github.com/cfzd/Ultra-Fast-Lane-Detection) | 2.072 | UFLD rujukan (row-anchor) | kode saja, tanpa bobot |
| [`cfzd/Ultra-Fast-Lane-Detection-v2`](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) | 862 | UFLDv2, F1 75,9 | kode saja, tanpa bobot |
| [`Turoad/CLRNet`](https://github.com/Turoad/CLRNet) | 600 | CLRNet, F1 80,5 | kode saja, tanpa bobot |
| [`XingangPan/SCNN`](https://github.com/XingangPan/SCNN) | 863 | SCNN (AAAI 2018), pembuat CULane | Lua/torch lama |
| [`amusi/awesome-lane-detection`](https://github.com/amusi/awesome-lane-detection) | 3.065 | daftar lengkap 100+ proyek & paper | referensi, bukan kode |

## 4. Kenapa belum dipakai di repo ini — dan saya sudah menguji, bukan menebak

Semua jalur sudah dicoba pada 2026-11-17:

| Jalur | Hasil uji |
|---|---|
| `pip install ultralytics` | **berhasil** (8.4.215), tapi bobot tidak bisa diunduh |
| `yolov8n.onnx` dari aset rilis | **404** — aset itu tidak ada |
| aset rilis (`release-assets.githubusercontent.com`) | 302 lalu koneksi TLS ditutup |
| repo GitHub yang menyertakan `.onnx` | 4 repo diuji, **semuanya hanya kode** (`models/.gitkeep`) |
| `PINTO_model_zoo` | 441 file `.onnx` di dalam repo, tapi model lane hanya `url.txt` menuju host luar |
| `huggingface.co`, `drive.google.com`, `media.githubusercontent.com` | **tidak bisa diakses** dari sandbox |
| paket PyPI (`pylane`, `lane-detection`, dll.) | kosong/tidak ada |

Kesimpulan: **bukan kurang usaha mencari.** Lingkungan ini tidak punya akses ke
host tempat bobot model disimpan. Tanpa bobot, mengganti algoritme berarti
menulis ulang tanpa bisa menguji — persis kesalahan yang sudah saya hindari
sebelumnya (dekoder tanpa model).

## 5. Yang bisa diuji siapa pun di mesin biasa

Dengan akses internet penuh, ini jalur paling wajar:

```bash
pip install onnxruntime opencv-python
# UFLDv2 di CULane: Crowded 74.9 / Curve 69.7 / Night 70.6
# bobot ONNX dari repo cfzd/Ultra-Fast-Lane-Detection-v2 atau rilis PINTO_model_zoo
```

Prasyarat yang harus diperiksa: model ini dilatih untuk **pemandangan jalan Tiongkok
dengan 4 lajur**. Kota kita (Jakarta, marka pudar, angkot menutup marka) adalah
*domain shift* — dan ada bukti nyata hal ini gagal: proyek
[`Shashank1024/LaneSight-AI`](https://github.com/Shashank1024/LaneSight-AI)
melaporkan UFLD mendapat F1 0.72 di Crowded CULane, tapi **tidak menemukan satu
lajur pun di 2 dari 4 jalan India**. Jadi memasang bobot bukan jaminan langsung
membaik; perlu uji di rekaman kita sendiri.

## 6. Yang bisa kita ambil **sekarang** tanpa model (dan itu tidak sedikit)

1. **Anotasi berbasis konteks, seperti CULane.** Jangan kosongkan sisi yang
   terhalang; teruskan dengan penanda eksplisit. Ini justru kebalikan dari
   keputusan saya sebelumnya — dan pengguna menyarankan hal yang sama.
2. **Kategori kondisi di UI**, persis kategori CULane (Normal / Ramai / Malam /
   Tanpa marka / Tikungan), supaya angka akurasi selalu diberi konteks.
3. **Uji dengan data berlabel**, bukan tebakan visual. Kita belum punya satu pun
   frame berlabel manual — itu pekerjaan paling bernilai berikutnya.
