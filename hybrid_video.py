#!/usr/bin/env python3
"""Video dengan fallback model UFLD saat engine klasik kehilangan kedua marka.

Engine klasik (`lane_v2`) tetap dipakai untuk setiap frame. Hanya kalau kedua sisi
hilang, model TFLite dipanggil; garis hasil model digambar warna CYAN dan diberi
label "MODEL" supaya tidak disangka deteksi klasik.

Bisa dipakai dua cara:

    # CLI
    python hybrid_video.py --input video.mp4 --output hasil.mp4

    # dari web_app.py (centang "Pakai fallback model" saat upload)
    from hybrid_video import process_video_hybrid
    process_video_hybrid(in_path, out_path, progress_cb=..., status_cb=..., stats_out={})

Ini jalur percobaan: model dilatih di CULane (domain berbeda), jadi hasilnya
harus diukur di rekaman sendiri sebelum dipakai untuk keputusan apa pun.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from lane_model_tflite import MODEL_DEFAULT, ModelLaneDetector, transcode
from lane_v2 import LaneDetector


def _round_pts(pts: np.ndarray) -> np.ndarray:
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _draw_model(frame: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for pts in (left, right):
        cv2.polylines(out, [_round_pts(pts)], False, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(
        out, "MODEL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 0), 2, cv2.LINE_AA,
    )
    return out


def _empty_stats() -> Dict:
    return {
        "frames": 0,
        "left_frames": 0,
        "right_frames": 0,
        "left_est": 0,
        "right_est": 0,
        "left_quality_sum": 0.0,
        "right_quality_sum": 0.0,
        "left_jitter": [],
        "right_jitter": [],
        "model_rescued": 0,
        "model_failed": 0,
    }


def process_video_hybrid(
    input_path,
    output_path,
    progress_cb: Optional[Callable] = None,
    status_cb: Optional[Callable] = None,
    stats_out: Optional[dict] = None,
    model_path=MODEL_DEFAULT,
    threads: int = 4,
) -> None:
    """Proses video: klasik di semua frame + model saat dua marka hilang."""
    src, dst = Path(input_path), Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"tidak bisa membuka {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp = dst.with_name(dst.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("gagal membuat VideoWriter")

    cd = LaneDetector()
    md = ModelLaneDetector(Path(model_path), threads=threads)
    stats = _empty_stats()
    prev_x = {"left": None, "right": None}

    def publish() -> None:
        if stats_out is not None:
            stats_out.clear()
            stats_out.update(stats)

    publish()
    if status_cb:
        status_cb("Fallback model UFLD aktif untuk frame tanpa deteksi")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _out, info = cd.process(frame)
            frames = stats["frames"] + 1
            used_model = False

            if cd.left_draw is not None and cd.right_draw is not None:
                out = cd.draw(frame, cd.roi_vertices(h, w))
            else:
                md.prev = None                 # jangan pakai cache frame lama
                left, right = md.detect(frame)
                if left is not None and right is not None:
                    out = _draw_model(frame, left, right)
                    stats["model_rescued"] += 1
                    used_model = True
                else:
                    out = frame.copy()
                    cv2.putText(
                        out, "TIDAK TERDETEKSI", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                    stats["model_failed"] += 1

            stats["frames"] = frames
            for key, coeff in (
                ("left", cd.left_draw if not used_model else None),
                ("right", cd.right_draw if not used_model else None),
            ):
                detected = bool(info.get(key)) and coeff is not None
                if detected:
                    stats[key + "_frames"] += 1
                    if info.get(key + "_estimated"):
                        stats[key + "_est"] += 1
                    stats[key + "_quality_sum"] += float(info.get(key + "_quality") or 0.0)
                    xb = cd._x(coeff, h - 1)
                    if prev_x[key] is not None and np.isfinite(xb):
                        d = abs(xb - prev_x[key])
                        if d < 0.10 * w:
                            stats[key + "_jitter"].append(float(d))
                    prev_x[key] = xb
                else:
                    prev_x[key] = None
            writer.write(out)

            if frames == 1 or frames % 10 == 0 or (total and frames == total):
                publish()
                if progress_cb:
                    try:
                        progress_cb(frames, total)
                    except Exception:
                        pass
    finally:
        cap.release()
        writer.release()

    if status_cb:
        status_cb("Mengonversi video ke H.264 agar bisa diputar di browser")
    if not transcode(tmp, dst):
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)

    def _med(vals):
        return float(np.median(vals)) if vals else None

    n = max(1, stats["frames"])
    stats["left_jitter_px"] = _med(stats["left_jitter"])
    stats["right_jitter_px"] = _med(stats["right_jitter"])
    stats["left_quality"] = stats["left_quality_sum"] / n
    stats["right_quality"] = stats["right_quality_sum"] / n
    stats["left_jitter"] = len(stats["left_jitter"])
    stats["right_jitter"] = len(stats["right_jitter"])
    publish()


def run(args: argparse.Namespace) -> None:
    stats: Dict = {}
    process_video_hybrid(
        args.input, args.output,
        stats_out=stats,
        model_path=args.model,
        threads=args.threads,
    )
    print(f"Tersimpan: {args.output} ({stats['frames']} frame)", flush=True)
    print(
        f"klasik dua sisi: {stats['frames'] - stats['model_rescued'] - stats['model_failed']}"
        f"/{stats['frames']} | model menolong {stats['model_rescued']} | "
        f"tetap kosong {stats['model_failed']}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Fallback model saat marka hilang")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
