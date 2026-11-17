#!/usr/bin/env python3
"""Video dengan fallback model UFLD saat engine klasik kehilangan kedua marka.

Engine klasik (`lane_v2`) tetap dipakai untuk setiap frame. Hanya kalau kedua sisi
hilang, model TFLite dipanggil; garis hasil model digambar warna CYAN dan diberi
label "MODEL" supaya tidak disangka deteksi klasik.

Ini jalur percobaan: model dilatih di CULane (domain berbeda), jadi hasilnya
harus diukur di rekaman sendiri sebelum dipakai untuk keputusan apa pun.

    python hybrid_video.py --input samples/mentah/mentah_1001183916.mp4 \
                           --output results/hybrid_1001183916.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lane_model_tflite import MODEL_DEFAULT, ModelLaneDetector, transcode
from lane_v2 import LaneDetector


def run(args: argparse.Namespace) -> None:
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"tidak bisa membuka {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp = dst.with_name(dst.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("gagal membuat VideoWriter")

    cd = LaneDetector()
    md = ModelLaneDetector(Path(args.model), threads=args.threads)
    total = both = model_ok = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        _out, info = cd.process(frame)

        if cd.left_draw is not None and cd.right_draw is not None:
            both += 1
            out = cd.draw(frame, cd.roi_vertices(h, w))
        else:
            # panggil model dengan keadaan segar, jangan pakai cache frame lama
            md.prev = None
            left, right = md.detect(frame)
            out = frame.copy()
            if left is not None and right is not None:
                model_ok += 1
                for pts in (left, right):
                    cv2.polylines(
                        out, [np_round(pts)], False, (255, 255, 0), 3, cv2.LINE_AA
                    )
                cv2.putText(
                    out, "MODEL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    out, "TIDAK TERDETEKSI", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )
        writer.write(out)
        if total % 50 == 0:
            print(f"frame {total} · klasik {both}/{total} · model dipakai {model_ok}", flush=True)

    cap.release()
    writer.release()
    if not transcode(tmp, dst):
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)

    print(f"Tersimpan: {dst} ({total} frame)", flush=True)
    print(
        f"klasik dua sisi: {both}/{total} | fallback model berhasil: "
        f"{model_ok}/{total - both} | tetap kosong: {total - both - model_ok}",
        flush=True,
    )


def np_round(pts):
    import numpy as np

    return pts.round().astype(int).reshape(-1, 1, 2)


def main() -> None:
    p = argparse.ArgumentParser(description="Fallback model saat marka hilang")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
