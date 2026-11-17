#!/usr/bin/env python3
"""Video deteksi marka: tiga mode engine.

- ``road``   : engine klasik `lane_v2`, tapi digambar CYAN + halo (label ROAD).
               Sisi yang hilang disintesis memakai profil lebar jalur klasik.
               Paling stabil di footage asli dan tidak butuh weights model.
- ``model``  : semua frame memakai UFLD TFLite (cyan, label MODEL). Default CLI
               lama, lebih halus tapi lebih lambat.
- ``hybrid`` : klasik di semua frame, model hanya saat dua marka hilang
               (perbandingan historis).

    # klasik + cyan (disarankan untuk deteksi jalan)
    python hybrid_video.py --mode road --input video.mp4 --output hasil.mp4

    # model UFLD saja
    python hybrid_video.py --input video.mp4 --output hasil.mp4

    # dari web_app.py
    from hybrid_video import process_video_road, process_video_model
    process_video_road(in_path, out_path, progress_cb=..., status_cb=..., stats_out={})
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
from road_cyan import bottom_pair, draw_points


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
        "road_frames": 0,
        "road_estimated": 0,     # marka sebelah disintesis dari profil klasik
        "road_held": 0,          # dua sisi hilang, kurva terakhir diprediksi
        "road_area_dynamic": 0,  # masker area jalan aktif dari kurva terpercaya
        "road_failed": 0,
        "left_est": 0,
        "right_est": 0,
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
    for key in ("left", "right"):
        total = stats.pop(key + "_quality_sum", None)
        if total is not None:
            stats[key + "_quality"] = total / max(1, stats["frames"])


def process_video_road(
    input_path,
    output_path,
    progress_cb: Optional[Callable] = None,
    status_cb: Optional[Callable] = None,
    stats_out: Optional[dict] = None,
) -> None:
    """Deteksi klasik `lane_v2`, overlay cyan. Tidak memuat model UFLD."""
    from lane_v2 import LaneDetector

    src, dst = Path(input_path), Path(output_path)
    cap, w, h, total, fps = _open_video(src)
    tmp, writer = _make_writer(dst, fps, w, h)

    cd = LaneDetector(use_road_area=True)
    stats = _empty_stats()
    # kualitas ini hanya dihitung mode ROAD; jangan tampilkan angka 0.00
    # di mode MODEL/HYBRID yang tidak mengisi metrik klasik.
    stats["left_quality_sum"] = 0.0
    stats["right_quality_sum"] = 0.0
    prev_x = {"left": None, "right": None}
    hold = {"left": None, "right": None, "vx": 0.0, "last": None, "age": 0}
    max_hold = 45
    _publish(stats, stats_out)
    if status_cb:
        status_cb("Engine klasik aktif (garis cyan, label ROAD)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _unused, info = cd.process(frame)
            stats["road_area_dynamic"] += int(bool(info.get("road_area_dynamic")))
            left, right = bottom_pair(cd, h, w)
            frames = stats["frames"] + 1
            stats["frames"] = frames
            held = False

            if left is None and right is None and hold["left"] is not None and hold["age"] < max_hold:
                # Dua sisi hilang sementara: lanjutkan kurva terakhir dengan
                # kecepatan geser terakhir. Masih ditandai HOLD, bukan deteksi.
                hold["age"] += 1
                shift = hold["vx"] * hold["age"]
                left = hold["left"].copy()
                right = hold["right"].copy()
                left[:, 0] = np.clip(left[:, 0] + shift, -w, 2 * w)
                right[:, 0] = np.clip(right[:, 0] + shift, -w, 2 * w)
                held = True
                stats["road_held"] += 1

            if left is not None:
                stats["left_frames"] += 1
                stats["left_est"] += int(bool(cd.left_estimated) or held)
                stats["left_quality_sum"] += float(info.get("left_quality") or 0.0)
            if right is not None:
                stats["right_frames"] += 1
                stats["right_est"] += int(bool(cd.right_estimated) or held)
                stats["right_quality_sum"] += float(info.get("right_quality") or 0.0)
            if left is not None and right is not None:
                if not held:
                    stats["road_frames"] += 1
                    stats["widths"].append(float(right[-1, 0] - left[-1, 0]))
                if not held and (cd.left_estimated or cd.right_estimated):
                    stats["road_estimated"] += 1

                xl, xr = float(left[-1, 0]), float(right[-1, 0])
                if not held and hold["last"] is not None:
                    dx = ((xl - hold["last"][0]) + (xr - hold["last"][1])) / 2.0
                    if abs(dx) < 0.10 * w and np.isfinite(dx):
                        hold["vx"] = 0.75 * hold["vx"] + 0.25 * dx
                if not held:
                    hold["last"] = (xl, xr)
                    hold["left"] = left.copy()
                    hold["right"] = right.copy()
                    hold["age"] = 0

            if left is None and right is None:
                stats["road_failed"] += 1
                hold["left"] = hold["right"] = None
                hold["last"] = None
                hold["vx"] = 0.0
                hold["age"] = 0

            for key, pts in (("left", left), ("right", right)):
                xb = float(pts[-1, 0]) if pts is not None else None
                if xb is not None and prev_x[key] is not None and np.isfinite(xb):
                    d = abs(xb - prev_x[key])
                    if d < 0.10 * w:
                        stats[key + "_jitter"].append(float(d))
                prev_x[key] = xb

            writer.write(draw_points(
                frame, left, right,
                estimated=(held or bool(cd.left_estimated), held or bool(cd.right_estimated)),
                label="ROAD + HOLD" if held else (
                    "ROAD + EST" if (cd.left_estimated or cd.right_estimated) else "ROAD"
                ),
            ))

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
    stats["road_width_jitter_px"] = stats.get("model_width_jitter_px")
    stats["road_width_px"] = stats.get("model_width_px")
    for key in [k for k in stats if k.startswith("model_")]:
        stats.pop(key, None)   # jangan tampilkan blok MODEL di halaman ROAD
    _publish(stats, stats_out)


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
    if args.mode == "road":
        fn = process_video_road
        kwargs = {}
    elif args.mode == "hybrid":
        fn = process_video_hybrid
        kwargs = {"model_path": args.model, "threads": args.threads}
    else:
        fn = process_video_model
        kwargs = {"model_path": args.model, "threads": args.threads}
    fn(args.input, args.output, stats_out=stats, **kwargs)

    print(f"Tersimpan: {args.output} ({stats['frames']} frame)", flush=True)
    if args.mode == "road":
        print(
            f"dua marka: {stats['road_frames']}/{stats['frames']} | "
            f"kosong: {stats['road_failed']} | "
            f"estimasi satu sisi: {stats['road_estimated']} | "
            f"ditahan: {stats['road_held']} | "
            f"area jalan dinamis: {stats['road_area_dynamic']}",
            flush=True,
        )
        if stats.get("road_width_px") is not None:
            print(
                f"lebar jalur: {stats['road_width_px']:.1f} px · "
                f"goyakan {stats['road_width_jitter_px']:.2f} px/frame",
                flush=True,
            )
        print(
            f"kiri {stats['left_frames']} frame ({stats['left_est']} estimasi) · "
            f"kanan {stats['right_frames']} frame ({stats['right_est']} estimasi)",
            flush=True,
        )
        return

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
        description="Deteksi marka: klasik+cyan (road), UFLD (model), atau hybrid"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--mode", choices=("road", "model", "hybrid"), default="model",
        help="road = klasik + cyan (stabil); model = UFLD; hybrid = perbandingan",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
