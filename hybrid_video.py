#!/usr/bin/env python3
"""Video deteksi marka memakai model UFLD (garis CYAN saja).

Default: **semua frame** memakai model, engine klasik (`lane_v2`, garis kuning)
tidak dipanggil sama sekali. Kurva diratakan antarframe di `ModelLaneDetector`
supaya tidak bergetar, tapi belokan lambat tetap ikut.

Mode lama masih bisa dipanggil lewat `--mode hybrid`: klasik di semua frame,
model hanya saat dua marka hilang. Dipakai untuk perbandingan, bukan default.

    # model saja (cyan)
    python hybrid_video.py --input video.mp4 --output hasil.mp4

    # perbandingan lama
    python hybrid_video.py --mode hybrid --input video.mp4 --output hybrid.mp4

    # dari web_app.py
    from hybrid_video import process_video_model
    process_video_model(in_path, out_path, progress_cb=..., status_cb=..., stats_out={})
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from lane_model_tflite import (
    MODEL_DEFAULT,
    ModelLaneDetector,
    draw as draw_model,
    transcode,
)


def _empty_stats() -> Dict:
    return {
        "frames": 0,
        "left_frames": 0,
        "right_frames": 0,
        "model_frames": 0,
        "model_failed": 0,
        "model_held": 0,         # kurva ditahan karena marka tertutup/ragukan
        "model_synthesized": 0,  # marka sebelah dibayangkan dari satu sisi
        "model_gate_missing": 0,
        "model_rescued": 0,      # alias untuk kompatibilitas halaman lama
        "left_jitter": [],
        "right_jitter": [],
        "widths": [],
    }


def _open_video(src: Path):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"tidak bisa membuka {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, w, h, total, fps


def _make_writer(dst: Path, fps: float, w: int, h: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("gagal membuat VideoWriter")
    return tmp, writer


def _finish(tmp: Path, dst: Path, status_cb: Optional[Callable]) -> None:
    if status_cb:
        status_cb("Mengonversi video ke H.264 agar bisa diputar di browser")
    if not transcode(tmp, dst):
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)


def _publish(stats: Dict, stats_out: Optional[dict]) -> None:
    if stats_out is not None:
        stats_out.clear()
        stats_out.update(stats)


def _finalize_stats(stats: Dict) -> None:
    def med(vals):
        return float(np.median(vals)) if vals else None

    stats["left_jitter_px"] = med(stats["left_jitter"])
    stats["right_jitter_px"] = med(stats["right_jitter"])
    stats["model_width_jitter_px"] = med(np.abs(np.diff(stats["widths"])).tolist())
    stats["model_width_px"] = med(stats["widths"])
    stats["left_jitter"] = len(stats["left_jitter"])
    stats["right_jitter"] = len(stats["right_jitter"])
    stats["model_rescued"] = stats["model_frames"]


def process_video_model(
    input_path,
    output_path,
    progress_cb: Optional[Callable] = None,
    status_cb: Optional[Callable] = None,
    stats_out: Optional[dict] = None,
    model_path=MODEL_DEFAULT,
    threads: int = 4,
) -> None:
    """Setiap frame memakai model; tidak ada garis kuning dari engine klasik."""
    src, dst = Path(input_path), Path(output_path)
    cap, w, h, total, fps = _open_video(src)
    tmp, writer = _make_writer(dst, fps, w, h)

    md = ModelLaneDetector(Path(model_path), threads=threads, smooth=True)
    stats = _empty_stats()
    prev_x = {"left": None, "right": None}
    _publish(stats, stats_out)
    if status_cb:
        status_cb("Model UFLD aktif di semua frame (garis cyan)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            left, right = md.detect(frame)
            frames = stats["frames"] + 1
            stats["frames"] = frames
            if md.last_gate == "missing":
                stats["model_gate_missing"] += 1
            if left is not None and right is not None:
                if md.last_gate != "accept":
                    stats["model_held"] += 1
                if md.last_synth:
                    stats["model_synthesized"] += 1
                out = draw_model(frame, left, right)
                stats["model_frames"] += 1
                stats["left_frames"] += 1
                stats["right_frames"] += 1
                stats["widths"].append(float(right[-1, 0] - left[-1, 0]))
                for key, pts in (("left", left), ("right", right)):
                    xb = float(pts[-1, 0])
                    if prev_x[key] is not None and np.isfinite(xb):
                        d = abs(xb - prev_x[key])
                        if d < 0.15 * w:
                            stats[key + "_jitter"].append(float(d))
                    prev_x[key] = xb
            else:
                out = frame.copy()
                cv2.putText(
                    out, "TIDAK TERDETEKSI", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA,
                )
                stats["model_failed"] += 1
                prev_x["left"] = prev_x["right"] = None
            writer.write(out)

            if frames == 1 or frames % 10 == 0 or (total and frames == total):
                _publish(stats, stats_out)
                if progress_cb:
                    try:
                        progress_cb(frames, total)
                    except Exception:
                        pass
    finally:
        cap.release()
        writer.release()

    _finish(tmp, dst, status_cb)
    _finalize_stats(stats)
    _publish(stats, stats_out)


def process_video_hybrid(
    input_path,
    output_path,
    progress_cb: Optional[Callable] = None,
    status_cb: Optional[Callable] = None,
    stats_out: Optional[dict] = None,
    model_path=MODEL_DEFAULT,
    threads: int = 4,
) -> None:
    """Mode lama: klasik di semua frame, model hanya saat keduanya hilang."""
    from lane_v2 import LaneDetector

    src, dst = Path(input_path), Path(output_path)
    cap, w, h, total, fps = _open_video(src)
    tmp, writer = _make_writer(dst, fps, w, h)

    cd = LaneDetector()
    md = ModelLaneDetector(Path(model_path), threads=threads, smooth=True)
    stats = _empty_stats()
    prev_x = {"left": None, "right": None}
    _publish(stats, stats_out)
    if status_cb:
        status_cb("Fallback model UFLD aktif untuk frame tanpa deteksi")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _out, info = cd.process(frame)
            frames = stats["frames"] + 1
            stats["frames"] = frames
            used_model = False

            if cd.left_draw is not None and cd.right_draw is not None:
                out = cd.draw(frame, cd.roi_vertices(h, w))
            else:
                md.reset()                     # jangan pakai cache frame lama
                left, right = md.detect(frame)
                if left is not None and right is not None:
                    out = draw_model(frame, left, right)
                    stats["model_frames"] += 1
                    used_model = True
                else:
                    out = frame.copy()
                    cv2.putText(
                        out, "TIDAK TERDETEKSI", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA,
                    )

            for key in ("left", "right"):
                coeff = None if used_model else getattr(cd, key + "_draw")
                detected = bool(info.get(key)) and coeff is not None
                if detected:
                    stats[key + "_frames"] += 1
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
                _publish(stats, stats_out)
                if progress_cb:
                    try:
                        progress_cb(frames, total)
                    except Exception:
                        pass
    finally:
        cap.release()
        writer.release()

    _finish(tmp, dst, status_cb)
    _finalize_stats(stats)
    _publish(stats, stats_out)


def run(args: argparse.Namespace) -> None:
    stats: Dict = {}
    fn = process_video_hybrid if args.mode == "hybrid" else process_video_model
    fn(
        args.input, args.output,
        stats_out=stats,
        model_path=args.model,
        threads=args.threads,
    )
    print(f"Tersimpan: {args.output} ({stats['frames']} frame)", flush=True)
    print(
        f"model dua marka: {stats['model_frames']}/{stats['frames']} | "
        f"tidak terdeteksi: {stats['model_failed']}",
        flush=True,
    )
    if stats.get("model_width_px") is not None:
        print(
            f"lebar jalur: {stats['model_width_px']:.1f} px · "
            f"goyakan {stats['model_width_jitter_px']:.2f} px/frame",
            flush=True,
        )
    print(
        f"kurva ditahan: {stats['model_held']} frame · "
        f"marka disintesis dari satu sisi: {stats['model_synthesized']} frame · "
        f"model tidak menemukan pasangan: {stats['model_gate_missing']} frame",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deteksi marka memakai model UFLD (cyan saja; default)"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--mode", choices=("model", "hybrid"), default="model",
        help="model = cyan saja (default); hybrid = perbandingan dengan klasik",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
