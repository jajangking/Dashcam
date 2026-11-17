#!/usr/bin/env python3
"""
Generator contoh uji untuk deteksi jalur.
Menghasilkan beberapa gambar & video sintetis dengan kondisi berbeda,
lalu (opsional) memprosesnya dengan lane_v2.

Pakai:
  python make_tests.py            # buat contoh saja
  python make_tests.py --run      # buat + proses, hasil di results/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples"
RESULTS = BASE / "results"
SAMPLES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def road(w, h, curve=0.0, marking_v=255, asphalt=85, night=False, lane_w=0.38, dash=False):
    """Bikin frame jalan sintetis dengan perspektif sederhana (curve = geser tengah jalan)."""
    img = np.zeros((h, w, 3), np.uint8)
    sky_top = int(h * 0.42)
    img[:sky_top] = (90, 70, 60) if night else (200, 190, 170)
    img[sky_top:] = (asphalt, asphalt, asphalt)

    vp_x = w * 0.5
    vp_y = sky_top

    def edge(y, side):
        t = (y - vp_y) / max(1, (h - vp_y))
        center = vp_x + curve * (t ** 2)
        spread = w * lane_w * t
        x = center + side * spread
        return int(np.clip(x, -w, 2 * w))

    yy = np.arange(vp_y, h)
    left = np.array([edge(y, -1) for y in yy])
    right = np.array([edge(y, +1) for y in yy])

    # isi aspal lebih terang di antara marka
    road_pts = np.vstack([
        np.stack([left, yy], axis=1),
        np.stack([right, yy], axis=1)[::-1],
    ]).astype(np.int32)
    cv2.fillPoly(img, [road_pts], (asphalt + 18, asphalt + 18, asphalt + 18))

    thick = max(2, int(w * 0.012))
    col = (marking_v, marking_v, marking_v)
    if dash:
        for i in range(0, len(yy) - 1, 18):
            seg = slice(i, min(i + 10, len(yy)))
            cv2.polylines(img, [np.stack([left[seg], yy[seg]], axis=1).astype(np.int32)],
                          False, col, thick, cv2.LINE_AA)
            cv2.polylines(img, [np.stack([right[seg], yy[seg]], axis=1).astype(np.int32)],
                          False, col, thick, cv2.LINE_AA)
    else:
        cv2.polylines(img, [np.stack([left, yy], axis=1).astype(np.int32)], False, col, thick, cv2.LINE_AA)
        cv2.polylines(img, [np.stack([right, yy], axis=1).astype(np.int32)], False, col, thick, cv2.LINE_AA)

    return img


def add_noise(img, sigma=8):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_images():
    cases = {
        "case_clear": dict(curve=0.0, marking_v=245, asphalt=80),
        "case_faded": dict(curve=0.0, marking_v=140, asphalt=110),
        "case_night": dict(curve=0.0, marking_v=235, asphalt=25, night=True),
        "case_curve_left": dict(curve=-60.0, marking_v=245, asphalt=80),
        "case_curve_right": dict(curve=60.0, marking_v=245, asphalt=80),
        "case_dashed": dict(curve=0.0, marking_v=245, asphalt=80, dash=True),
        "case_nomark": dict(curve=0.0, marking_v=85, asphalt=95),
    }
    out = []
    for name, kw in cases.items():
        img = road(1280, 720, **kw)
        img = add_noise(img)
        p = SAMPLES / f"{name}.jpg"
        cv2.imwrite(str(p), img)
        out.append(p)

    # portrait (seperti video HP)
    img = road(720, 1280, marking_v=245, asphalt=80)
    p = SAMPLES / "case_portrait.jpg"
    cv2.imwrite(str(p), img)
    out.append(p)
    return out


def make_videos(frames=120, w=960, h=540):
    specs = {
        "case_video_clear": dict(curve=0.0),
        "case_video_wobble": dict(curve=0.0),  # kamera goyang
    }
    paths = []
    for name, kw in specs.items():
        p = SAMPLES / f"{name}.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for i in range(frames):
            img = road(w, h, marking_v=240, asphalt=80, **kw)
            img = add_noise(img, sigma=5)
            if name == "case_video_wobble":
                dx = int(12 * np.sin(i * 0.35))
                dy = int(6 * np.sin(i * 0.21))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h))
            vw.write(img)
        vw.release()
        paths.append(p)
    return paths


def run_all(image_paths, video_paths):
    from lane_v2 import LaneDetector
    from lane_detect import process_video

    report = []
    for p in image_paths:
        frame = cv2.imread(str(p))
        det = LaneDetector()
        out, info = det.process(frame)
        dst = RESULTS / f"test_{p.stem}.jpg"
        cv2.imwrite(str(dst), out)
        report.append(
            f"{p.stem}: kiri={info['left']} kanan={info['right']} "
            f"pts={info['left_points']}/{info['right_points']} offset={info['offset_pct']}"
        )

    for p in video_paths:
        dst = RESULTS / f"test_{p.stem}.mp4"
        process_video(str(p), str(dst))
        report.append(f"{p.stem}: video -> {dst.name}")

    print("\n".join(report))
    (RESULTS / "test_report.txt").write_text("\n".join(report), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="langsung proses dengan lane_v2")
    args = ap.parse_args()

    imgs = make_images()
    vids = make_videos()
    print(f"Contoh dibuat: {len(imgs)} gambar, {len(vids)} video di {SAMPLES}")
    if args.run:
        run_all(imgs, vids)


if __name__ == "__main__":
    main()
