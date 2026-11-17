#!/usr/bin/env python3
"""Deteksi marka memakai model UFLD (LiteRT/TFLite), jalur alternatif terhadap lane_v2.

Berkas ini sengaja terpisah dari `lane_model.py` (jalur ONNX yang dipakai
`live_app.py`) supaya tidak mengubah perilaku modul yang sudah ada.

Model: Ultra-Fast-Lane-Detection (ResNet18, CULane) — 178 MB, input [1,3,288,800]
RGB NCHW, output [1,201,18,4] = (grid+1, row anchors, 4 lajur).
Sumber & lisensi: https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT

Bobot TIDAK ikut di-commit (lihat .gitignore). Unduh sendiri:

    mkdir -p models
    curl -L -o models/ufld.tflite \\
      https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT/resolve/main/ufld.tflite

Runtime: `pkg install -y python-tflite-runtime` di Termux.

Contoh:

    python lane_model_tflite.py --input samples/case_clear.jpg --output hasil_model.jpg
    python lane_model_tflite.py --input samples/mentah/mentah_1001183916.mp4 --output hasil_model.mp4

Catatan jujur: model ini dilatih di CULane (empat lajur, jalan Tiongkok).
Marka pudar dan angkot di rekaman kita adalah domain shift — hasilnya wajib
diukur pada footage sendiri, tidak boleh langsung dipercaya.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

MODEL_DEFAULT = Path(__file__).resolve().parent / "models" / "ufld.tflite"

# CULane row anchors untuk input tinggi 288 (UFLD resmi).
ROW_ANCHORS = np.array(
    [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
    dtype=np.float32,
)
GRID = 200
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Kurva selalu di-sampling pada grid u=y/h yang tetap supaya bisa di-rata-rata
# antarframe. Tanpa grid tetap, panjang/posisi titik berubah tiap frame.
CURVE_US = np.linspace(0.45, 1.0, 24, dtype=np.float32)


class ModelLaneDetector:
    """Pembungkus inference TFLite yang menghasilkan dua kurva tepi jalur."""

    def __init__(
        self,
        model_path: Path = MODEL_DEFAULT,
        threads: int = 2,
        smooth: bool = True,
        alpha: float = 0.35,
        max_stale: int = 8,
    ):
        """`smooth` meratakan kurva antarframe; `alpha` = bobot frame baru.

        alpha kecil = lebih stabil tapi lebih lambat bereaksi. max_stale = berapa
        frame kurva lama dipakai saat model gagal sebelum dianggap hilang.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"bobot {model_path} tidak ada. Unduh dulu (lihat docstring lane_model_tflite.py)."
            )
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError("butuh `pkg install -y python-tflite-runtime`") from exc

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_detail = self.interpreter.get_output_details()[0]
        self.smooth = smooth
        self.alpha = float(alpha)
        self.max_stale = int(max_stale)
        self._prev_x: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.prev: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._stale = 0

    def reset(self) -> None:
        """Lupakan riwayat (pakai saat memulai video/segmen baru)."""
        self._prev_x = None
        self.prev = None
        self._stale = 0

    def describe(self) -> str:
        return (
            f"input={self.in_detail['shape'].tolist()} {self.in_detail['dtype']} | "
            f"output={self.out_detail['shape'].tolist()} {self.out_detail['dtype']}"
        )

    def _decode(self, out: np.ndarray, w: int, h: int) -> List[np.ndarray]:
        """Ubah logits row-anchor menjadi empat daftar titik (x, y) dalam piksel."""
        logits = np.asarray(out)
        # bentuk umum: [1, 201, 18, 4] -> [grid+1, anchor, lajur]
        while logits.ndim > 3 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim == 4:  # [1, 201, 18, 1, 4] atau sejenisnya
            logits = logits.reshape(logits.shape[0], logits.shape[1], -1)
        if logits.shape[0] == 4 and logits.shape[-1] == GRID + 1:
            logits = np.transpose(logits, (2, 1, 0))
        if logits.shape[0] != GRID + 1:
            raise ValueError(f"bentuk output tidak dikenal: {logits.shape}")

        ys = ROW_ANCHORS / 288.0 * h
        xs_col = np.linspace(0, 799, GRID) / 800.0 * w
        lanes: List[np.ndarray] = []
        for li in range(logits.shape[2]):
            score = logits[:, :, li]                   # [201, 18]
            cell = score[:GRID, :]
            prob = np.exp(cell - cell.max(axis=0, keepdims=True))
            prob /= prob.sum(axis=0, keepdims=True)
            col_f = (prob * np.arange(GRID)[:, None]).sum(axis=0)
            no_lane = score.argmax(axis=0) == GRID
            pts = np.stack([xs_col[np.clip(np.round(col_f), 0, GRID - 1).astype(int)], ys], axis=1)
            lanes.append(pts[~no_lane])
        return lanes

    @staticmethod
    def _pick_pair(lanes: List[np.ndarray], w: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Pilih dua marka tepi jalur yang paling dekat dengan tengah gambar."""
        cands = []
        for pts in lanes:
            if len(pts) >= 3:
                cands.append((float(np.median(pts[-4:, 0])), pts))
        if len(cands) < 2:
            return None
        cands.sort(key=lambda t: t[0])
        pairs = [(abs((a[0] + b[0]) / 2.0 - w / 2.0), a[1], b[1]) for a, b in zip(cands, cands[1:])]
        _, left, right = min(pairs, key=lambda t: t[0])
        return left, right

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (800, 288), interpolation=cv2.INTER_LINEAR)
        x = (small.astype(np.float32) / 255.0 - MEAN) / STD
        # model ini memakai NCHW, bukan NHWC seperti kebanyakan TFLite
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        self.interpreter.set_tensor(self.in_detail["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_detail["index"])

        pair = self._pick_pair(self._decode(out, w, h), w)
        if pair is None:
            if self.prev is None:
                return (None, None)
            self._stale += 1
            if self._stale > self.max_stale:
                self.reset()
                return (None, None)
            return self.prev

        raw_l, raw_r = self._fit(pair[0], h), self._fit(pair[1], h)
        if (
            self.smooth
            and self._prev_x is not None
            and len(raw_l) == len(self._prev_x[0])
            and len(raw_r) == len(self._prev_x[1])
        ):
            # Lonjakan besar di bawah gambar biasanya salah deteksi, bukan belokan.
            # Jangan tarik kurva lama ke posisi itu; mulai dari awal lagi.
            jump = max(
                abs(raw_l[-1, 0] - self._prev_x[0][-1]),
                abs(raw_r[-1, 0] - self._prev_x[1][-1]),
            )
            if jump < 0.15 * w:
                a = self.alpha
                raw_l = raw_l.copy()
                raw_r = raw_r.copy()
                raw_l[:, 0] = a * raw_l[:, 0] + (1 - a) * self._prev_x[0]
                raw_r[:, 0] = a * raw_r[:, 0] + (1 - a) * self._prev_x[1]
        self._prev_x = (raw_l[:, 0].copy(), raw_r[:, 0].copy())
        self.prev = (raw_l, raw_r)
        self._stale = 0
        return self.prev

    @staticmethod
    def _fit(pts: np.ndarray, h: int) -> np.ndarray:
        """Rapikan titik hasil model jadi 24 titik kurva (derajat 2)."""
        u = pts[:, 1] / h
        keep = np.isfinite(u) & np.isfinite(pts[:, 0]) & (u > 0.25)
        u, x = u[keep], pts[keep, 0]
        if len(u) < 3:
            return pts
        deg = 2 if len(u) >= 6 else 1
        c = np.polyfit(u, x, deg)
        # Jangan extrapolasi polinomial di luar anchor: bisa meledak (x > lebar
        # gambar). Di tepi, pakai nilai anchor terdekat.
        us = np.clip(CURVE_US, u.min(), u.max())
        return np.stack([np.polyval(c, us), CURVE_US * h], axis=1)


def draw(frame: np.ndarray, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> np.ndarray:
    """Satu-satunya gaya garis model: cyan, anti-alias, tanpa garis klasik."""
    out = frame.copy()
    for pts in (left, right):
        if pts is None or len(pts) < 2:
            continue
        p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [p], False, (140, 190, 190), 7, cv2.LINE_AA)   # halo
        cv2.polylines(out, [p], False, (255, 255, 0), 3, cv2.LINE_AA)     # cyan
    if left is not None and right is not None and len(left) and len(right):
        cv2.putText(
            out, "MODEL", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 0), 2, cv2.LINE_AA,
        )
    return out


def transcode(src: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def run(args: argparse.Namespace) -> None:
    det = ModelLaneDetector(Path(args.model))
    print("Model:", det.describe(), flush=True)

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    is_video = cap.isOpened()
    writer = None
    tmp = None
    if is_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tmp = dst.with_name(dst.stem + "_raw.mp4")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    else:
        cap.release()
        frame = cv2.imread(str(src))
        if frame is None:
            raise RuntimeError(f"gambar tidak bisa dibaca: {src}")
        left, right = det.detect(frame)
        if not cv2.imwrite(str(dst), draw(frame, left, right)):
            raise RuntimeError("gagal menulis gambar")
        print(f"Tersimpan: {dst}", flush=True)
        return

    frames = 0
    both = 0
    bots: List[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        left, right = det.detect(frame)
        if left is not None and right is not None:
            both += 1
            bots.append(float(right[-1, 0] - left[-1, 0]))
        writer.write(draw(frame, left, right))
        frames += 1
        if frames % 30 == 0:
            print(f"frame {frames} · dua marka {both}/{frames}", flush=True)
    cap.release()
    writer.release()

    if not transcode(tmp, dst):
        shutil.move(str(tmp), str(dst))
    else:
        tmp.unlink(missing_ok=True)
    print(f"Tersimpan: {dst} ({frames} frame, dua marka {both}/{frames})", flush=True)
    if len(bots) > 1:
        d = np.abs(np.diff(bots))
        print(
            f"lebar jalur bawah: median {np.median(bots):.1f} px · "
            f"goyakan median {np.median(d):.2f} px/frame",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Deteksi marka memakai model UFLD LiteRT")
    p.add_argument("--input", required=True, help="gambar atau video")
    p.add_argument("--output", required=True, help="file hasil")
    p.add_argument("--model", default=str(MODEL_DEFAULT), help="path ke ufld.tflite")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
