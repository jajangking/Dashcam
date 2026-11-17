#!/usr/bin/env python3
"""
Prototipe deteksi jalan/marka jalur (lane detection) untuk video dashcam.

Metode: deteksi tepi (Canny) + ROI + Hough transform untuk garis marka.
Cocok sebagai baseline; hasil sangat bergantung pada pencahayaan, cuaca,
dan kejelasan marka jalan.

Contoh pakai:
  python lane_detect.py --input input.mp4 --output output.mp4
  python lane_detect.py --image frame.jpg --output out.jpg
"""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_line_points(
    y1: int, y2: int, line: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    x1, y_1, x2, y_2 = line.reshape(4)
    if x2 == x1:
        return None
    slope = (y_2 - y_1) / (x2 - x1)
    intercept = y_1 - slope * x1
    if abs(slope) < 1e-3:
        return None
    x1_out = int((y1 - intercept) / slope)
    x2_out = int((y2 - intercept) / slope)
    return x1_out, int(y1), x2_out, int(y2)


def average_lane_lines(
    lines: Optional[np.ndarray], width: int, height: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    left, right = [], []
    left_weights, right_weights = [], []
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        length = math.hypot(x2 - x1, y2 - y1)
        # Filter garis yang terlalu horizontal
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left.append((slope, y1 - slope * x1))
            left_weights.append(length)
        else:
            right.append((slope, y1 - slope * x1))
            right_weights.append(length)

    y_bottom = height
    y_top = int(height * 0.62)

    def avg(points, weights):
        if not points:
            return None
        slope = np.average([p[0] for p in points], weights=weights)
        intercept = np.average([p[1] for p in points], weights=weights)
        if abs(slope) < 1e-3:
            return None
        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        # Batasi agar tidak liar keluar frame
        x_bottom = max(-width, min(2 * width, x_bottom))
        x_top = max(-width, min(2 * width, x_top))
        return x_bottom, y_bottom, x_top, y_top

    return avg(left, left_weights), avg(right, right_weights)


def detect_lanes(frame: np.ndarray):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI berbentuk trapesium menghadap ke depan (tune sesuai kamera dashcam)
    roi_vertices = np.array(
        [
            [
                (int(width * 0.05), height),
                (int(width * 0.45), int(height * 0.6)),
                (int(width * 0.55), int(height * 0.6)),
                (int(width * 0.95), height),
            ]
        ],
        dtype=np.int32,
    )
    masked = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(20, width // 12),
        maxLineGap=80,
    )

    left, right = average_lane_lines(lines, width, height)
    return left, right, roi_vertices


def _smooth_line(prev, curr, alpha=0.25):
    if curr is None:
        if prev is None:
            return None
        # peluruhkan perlahan agar tidak hilang mendadak
        return prev
    if prev is None:
        return curr
    return tuple(int((1 - alpha) * p + alpha * c) for p, c in zip(prev, curr))


def _transcode_h264(in_path: str, out_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-i", in_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", out_path,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def draw_overlay(
    frame: np.ndarray,
    left: Optional[Tuple[int, int, int, int]],
    right: Optional[Tuple[int, int, int, int]],
    roi_vertices: np.ndarray,
) -> np.ndarray:
    overlay = frame.copy()
    height, width = frame.shape[:2]

    # Gambar area ROI (opsional, transparan)
    cv2.polylines(overlay, roi_vertices, True, (80, 80, 80), 1)

    # Area jalan di antara dua marka
    if left and right:
        pts = np.array(
            [[left[0], left[1]], [left[2], left[3]], [right[2], right[3]], [right[0], right[1]]],
            dtype=np.int32,
        )
        road = overlay.copy()
        cv2.fillPoly(road, [pts], (0, 180, 0))
        overlay = cv2.addWeighted(road, 0.25, overlay, 0.75, 0)

    if left:
        cv2.line(overlay, (left[0], left[1]), (left[2], left[3]), (255, 200, 0), 8)
    if right:
        cv2.line(overlay, (right[0], right[1]), (right[2], right[3]), (255, 200, 0), 8)

    status = "Jalur terdeteksi" if (left or right) else "Jalur tidak jelas"
    color = (0, 220, 0) if (left and right) else (0, 180, 255)
    cv2.putText(
        overlay, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
    )
    return overlay


def process_image(input_path: str, output_path: str, progress_cb=None, mask_path: Optional[str] = None) -> None:
    from lane_v2 import LaneDetector

    if progress_cb:
        try:
            progress_cb(0, 1, "Membaca gambar")
        except Exception:
            pass
    frame = cv2.imread(input_path)
    if frame is None:
        raise RuntimeError(f"Gagal membaca gambar: {input_path}")
    det = LaneDetector(use_road_area=True)
    if progress_cb:
        try:
            progress_cb(0, 1, "Mendeteksi jalur")
        except Exception:
            pass
    out, info = det.process(frame)
    if mask_path:
        from road_area import road_area_mask

        roi = det.roi_vertices(frame.shape[0], frame.shape[1])
        area = road_area_mask(det, frame.shape[0], frame.shape[1])
        dbg = det.draw(frame, roi, det.color_mask(frame, roi, area))
        cv2.imwrite(mask_path, dbg)
    if progress_cb:
        try:
            progress_cb(1, 1, "Menyimpan hasil")
        except Exception:
            pass
    if not cv2.imwrite(output_path, out):
        raise RuntimeError(f"Gagal menulis output: {output_path}")
    print(f"Tersimpan: {output_path} (kiri={info['left']}, kanan={info['right']})")


def process_video(
    input_path: str,
    output_path: str,
    show: bool = False,
    progress_cb=None,
    status_cb=None,
    stats_out: Optional[dict] = None,
) -> int:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Tulis ke file sementara lalu transcode ke H.264 (lebih kompatibel di browser)
    use_transcode = bool(shutil.which("ffmpeg")) and output_path.lower().endswith(".mp4")
    work_output = output_path
    tmp_path = None
    if use_transcode:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(output_path) or ".")
        os.close(fd)
        work_output = tmp_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(work_output, fourcc, fps, (width, height))

    frames = 0
    try:
        from lane_v2 import LaneDetector

        det = LaneDetector(use_road_area=True)
    except Exception:
        det = None

    stats = None
    if stats_out is not None:
        stats = {
            "frames": 0,
            "left_frames": 0,
            "right_frames": 0,
            "left_est": 0,
            "right_est": 0,
            "road_area_dynamic": 0,
            "left_quality_sum": 0.0,
            "right_quality_sum": 0.0,
            "left_jitter": [],
            "right_jitter": [],
        }
        stats_out.clear()
        stats_out.update(stats)
    prev_x = {"L": None, "R": None}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            info = None
            if det is not None:
                out, info = det.process(frame)
            else:
                left, right, roi = detect_lanes(frame)
                out = draw_overlay(frame, left, right, roi)
            writer.write(out)
            frames += 1

            if stats is not None and info is not None:
                stats["frames"] += 1
                stats["road_area_dynamic"] += int(bool(info.get("road_area_dynamic")))
                for key, side in (("left", "L"), ("right", "R")):
                    if info.get(key):
                        stats[key + "_frames"] += 1
                        if info.get(key + "_estimated"):
                            stats[key + "_est"] += 1
                        stats[key + "_quality_sum"] += float(info.get(key + "_quality") or 0.0)
                        coeff = det.left_coeff if side == "L" else det.right_coeff
                        xb = det._x(coeff, height - 1) if coeff is not None else float("nan")
                        if prev_x[side] is not None and np.isfinite(xb):
                            d = abs(xb - prev_x[side])
                            if d < 0.10 * width:  # lompatan besar = pindah objek, bukan goyangan
                                stats[key + "_jitter"].append(float(d))
                        prev_x[side] = xb
                    else:
                        prev_x[side] = None
                stats_out.clear()
                stats_out.update(stats)

            if progress_cb and (frames == 1 or frames % 10 == 0 or (total and frames == total)):
                try:
                    progress_cb(frames, total)
                except Exception:
                    pass
            if show:
                cv2.imshow("lane_detect", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    if stats is not None:
        def _med(vals):
            return float(np.median(vals)) if vals else None

        n = max(1, stats["frames"])
        stats["left_jitter_px"] = _med(stats["left_jitter"])
        stats["right_jitter_px"] = _med(stats["right_jitter"])
        stats["left_quality"] = stats["left_quality_sum"] / n
        stats["right_quality"] = stats["right_quality_sum"] / n
        stats["left_jitter"] = len(stats["left_jitter"])
        stats["right_jitter"] = len(stats["right_jitter"])
        stats_out.clear()
        stats_out.update(stats)

    if use_transcode and tmp_path:
        if status_cb:
            try:
                status_cb("Mengonversi video ke H.264 agar bisa diputar di browser")
            except Exception:
                pass
        ok = _transcode_h264(tmp_path, output_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        if not ok:
            raise RuntimeError("Gagal mengonversi video ke H.264 untuk kompatibilitas browser")

    if progress_cb:
        try:
            progress_cb(frames, total or frames)
        except Exception:
            pass
    print(f"Selesai: {frames} frame -> {output_path}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Prototipe deteksi jalan/marka jalur")
    ap.add_argument("--input", help="Path video input (mp4/avi, dll)")
    ap.add_argument("--image", help="Path gambar input (jpg/png)")
    ap.add_argument("--output", required=True, help="Path file output")
    ap.add_argument("--show", action="store_true", help="Tampilkan preview (jika GUI tersedia)")
    args = ap.parse_args()

    if bool(args.input) == bool(args.image):
        raise SystemExit("Pilih salah satu: --input <video> atau --image <gambar>")

    if args.image:
        process_image(args.image, args.output)
    else:
        process_video(args.input, args.output, show=args.show)


if __name__ == "__main__":
    main()
