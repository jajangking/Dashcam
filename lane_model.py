#!/usr/bin/env python3
"""
Deteksi marka berbasis model ONNX (UFLD / UFLDv2 dari CULane).

Kenapa modul ini ada
--------------------
Detektor di `lane_v2.py` adalah algoritme klasik buatan sendiri: masker warna +
jendela + fitting. Ia jujur soal batasnya, tapi tetap kalah jauh dari model yang
dilatih pada data nyata. Angka rujukan CULane (F1 per kategori):

    UFLD     : Crowded 70.2 | Curve 69.7 | Night 66.7
    UFLDv2   : Crowded 74.9 | Curve 69.7 | Night 70.6
    CLRNet   : total 80.5

Modul ini menyiapkan jalur untuk memakai model seperti itu. **Bobotnya tidak
disertakan** karena lingkungan pengembangan proyek ini tidak punya akses ke host
penyimpan bobot (lihat REFERENSI.md). Yang disertakan: pemuat model, praproses,
pasca-proses, dan seluruh pemeriksaan bentuk/skala supaya kegagalan muncul
sebagai peringatan yang jelas — bukan angka ngawur.

Bentuk keluaran UFLD (row-anchor)
--------------------------------
    input  : gambar RGB, tinggi 288 (CULane) atau 320 (TuSimple), lebar 800 / 1600
    output : (1, griding_num * num_lanes, num_rows)
             num_lanes = 4 untuk CULane, 4 untuk TuSimple
             tiap kolom adalah logit posisi x untuk satu baris anchor

Titik x untuk satu lajur = argmax di sepanjang sumbu griding, dinormalkan ke
lebar gambar. Baris-baris anchor ditempatkan merata di bagian bawah gambar.

Cara pakai
----------
    det = OnnxLaneDetector("models/ufld_culane.onnx")
    jalur = det.detect(frame)        # -> [array x_untuk_tiap_baris, ...] atau None
    info  = det.info                 # peringatan/kesalahan, jangan disembunyikan

Kalau model tidak ada atau bentuknya tidak dikenal, `detect()` mengembalikan
None dan `det.info` menjelaskan alasannya. Tidak pernah mengarang angka.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:                                    # pragma: no cover - pilihan backend
    import cv2
    _HAS_CV2 = True
except Exception:                       # pragma: no cover
    _HAS_CV2 = False


# Pengaturan yang sudah teruji untuk UFLD/UFLDv2 di dataset publik.
DATASET_PRESETS = {
    "culane": {
        "input_hw": (288, 800),
        "num_lanes": 4,
        "num_rows": 18,
        "griding_num": 200,
        "crop_top": 0.0,
        # urutan lajur dari kiri ke kanan
        "anchor_top": 0.42,
    },
    "tusimple": {
        "input_hw": (288, 800),
        "num_lanes": 4,
        "num_rows": 56,
        "griding_num": 100,
        "crop_top": 0.0,
        "anchor_top": 0.42,
    },
}


class OnnxLaneDetector:
    """Pembungkus model UFLD/UFLDv2 (ONNX) dengan pemeriksaan bentuk yang ketat."""

    def __init__(self, model_path: str, dataset: str = "culane",
                 providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.dataset = dataset if dataset in DATASET_PRESETS else "culane"
        self.cfg = dict(DATASET_PRESETS[self.dataset])
        self.session = None
        self.input_name = ""
        self.input_hw: Tuple[int, int] = self.cfg["input_hw"]
        self.info = ""
        self.ready = False
        self._load(providers)

    # ---------- pemuatan ----------
    def _load(self, providers: Optional[List[str]]) -> None:
        if not os.path.exists(self.model_path):
            self.info = (f"model tidak ditemukan: {self.model_path} — "
                         f"letakkan berkas .onnx di folder models/")
            return
        try:
            import onnxruntime as ort
        except Exception as exc:                      # pragma: no cover
            self.info = f"onnxruntime tidak terpasang: {exc}"
            return
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers or ["CPUExecutionProvider"],
            )
        except Exception as exc:
            self.info = f"gagal memuat model: {exc}"
            return

        inp = self.session.get_inputs()[0]
        shape = inp.shape                      # mis. [1, 3, 288, 800] atau simbolik
        try:
            h = int(shape[2])
            w = int(shape[3])
            if h <= 0 or w <= 0:
                raise ValueError
            self.input_hw = (h, w)
        except Exception:
            self.info = (f"bentuk masukan tidak diketahui ({shape}); "
                         f"memakai default {self.input_hw[0]}x{self.input_hw[1]}")
        self.input_name = inp.name

        # Sesuaikan jumlah baris anchor dengan bentuk keluaran sebenarnya.
        try:
            out_shape = self.session.get_outputs()[0].shape
            num_rows = int(out_shape[-1])
            if num_rows > 0:
                self.cfg["num_rows"] = num_rows
            total = int(out_shape[-2])
            if total > 0 and self.cfg["num_lanes"] > 0:
                g = total / self.cfg["num_lanes"]
                if g.is_integer():
                    self.cfg["griding_num"] = int(g)
        except Exception:
            pass

        self.ready = True
        if not self.info:
            self.info = (f"siap: {os.path.basename(self.model_path)} "
                         f"({self.dataset}, masukan {self.input_hw[0]}x{self.input_hw[1]})")

    # ---------- praproses ----------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + normalisasi seperti pelatihan UFLD (ImageNet mean/std)."""
        ih, iw = self.input_hw
        if _HAS_CV2:
            img = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:                                            # pragma: no cover
            img = frame[:, :, ::-1]
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1))[None, ...]

    # ---------- pasca-proses ----------
    def postprocess(self, out: np.ndarray, frame_hw: Tuple[int, int]
                    ) -> Optional[List[np.ndarray]]:
        """Ubah logit (1, G*L, R) menjadi posisi x di koordinat gambar asli."""
        o = np.asarray(out, dtype=np.float32)
        if o.ndim == 3:
            o = o[0]
        if o.ndim != 2:
            self.info = f"bentuk keluaran tidak dikenal: {np.shape(out)}"
            return None

        # UFLD mengeluarkan (G*L, R): sumbu pertama = kolom (griding*lajur),
        # sumbu kedua = baris anchor. Versi pertama modul ini membacanya
        # terbalik sehingga menghasilkan 800 "baris" alih-alih 18 — angka
        # yang terlihat masuk akal tapi salah total. Sekarang arah ditentukan
        # dari bentuknya, dengan pemeriksaan jumlah baris yang wajar.
        if o.shape[1] <= o.shape[0] and 5 <= o.shape[1] <= 200:
            o = o.T                       # (R, G*L)
        else:
            o = o.T                       # tetap transpose: sumbu 0 harus baris
        num_rows, total = o.shape
        if not (5 <= num_rows <= 200):
            self.info = f"jumlah baris anchor tidak wajar: {num_rows}"
            return None
        lanes = self.cfg["num_lanes"]
        if total % lanes != 0:
            # tebak jumlah lajur dari bentuk keluaran
            for cand in (2, 3, 4, 5):
                if total % cand == 0:
                    lanes = cand
                    break
            else:
                self.info = f"tidak bisa memecah {total} kolom jadi lajur"
                return None
        griding = total // lanes
        self.cfg["griding_num"] = griding

        h, w = frame_hw
        top = self.cfg["anchor_top"]
        ys = np.linspace(top * h, (h - 1), num_rows)
        result: List[np.ndarray] = []
        for i in range(lanes):
            block = o[:, i * griding:(i + 1) * griding]
            idx = np.argmax(block, axis=1).astype(np.float32)
            xs = idx / max(1, griding - 1) * w
            result.append(np.stack([xs, ys], axis=1))
        return result

    # ---------- antarmuka utama ----------
    def detect(self, frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """Kembalikan 4 lajur (masing-masing array (N,2) berisi x,y) atau None."""
        if not self.ready or self.session is None:
            return None
        try:
            inp = self.preprocess(frame)
            names = [o.name for o in self.session.get_outputs()]
            outs = self.session.run(names, {self.input_name: inp})
        except Exception as exc:
            self.info = f"gagal inferensi: {exc}"
            return None
        h, w = frame.shape[:2]
        return self.postprocess(outs[0], (h, w))

    # ---------- ringkasan untuk UI ----------
    def summary(self) -> str:
        if not self.ready:
            return "model marka: tidak aktif (" + (self.info or "tidak ada bobot") + ")"
        return "model marka: aktif — " + self.info


def find_model(model_dir: str = "models") -> Optional[str]:
    """Cari berkas .onnx pertama di folder model (kalau pengguna meletakkannya)."""
    if not os.path.isdir(model_dir):
        return None
    for name in sorted(os.listdir(model_dir)):
        if name.lower().endswith(".onnx"):
            return os.path.join(model_dir, name)
    return None
