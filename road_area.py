"""Masker area jalan untuk detektor klasik (`lane_v2`).

Ide ini diadaptasi dari Ozgunalp dkk. (2019, arXiv:1911.09054): sebelum mencari
marka, batasi pencarian ke **area jalan yang bisa dikendarai**. Di sini versinya
sengaja ringan dan tanpa model — area jalan dibentuk dari ROI geometris, lalu
disempitkan ke koridor dua kurva terpercaya dari frame sebelumnya. Piksel terang
di luar koridor (dinding, pagar, bodi mobil, langit) jadi tidak bisa menarik
sliding-window.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from lane_v2 import LaneDetector


def _static_roi(det: LaneDetector, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, det.roi_vertices(h, w), 255)
    return mask


def road_area_mask(
    det: LaneDetector,
    h: int,
    w: int,
    side_margin: float = 0.28,
    min_margin_px: int = 36,
    max_margin_px: int = 240,
    top_u: float = 0.55,
    n: int = 24,
) -> np.ndarray:
    """Area jalan: koridor kurva terpercaya, atau ROI statis jika belum ada.

    Mengembalikan mask uint8. Flag ``det.__road_area_dynamic`` dipakai pemanggil
    hanya untuk pelaporan statistik, bukan logika deteksi.
    """
    det._road_area_dynamic = False
    left, right = det.left_coeff, det.right_coeff
    if left is None or right is None:
        return _static_roi(det, h, w)
    real_left = not det.left_estimated and det.left_quality >= 0.15
    real_right = not det.right_estimated and det.right_quality >= 0.15
    if not (real_left or real_right):
        return _static_roi(det, h, w)
    # kalau salah satu sisi cuma hasil tebakan, koridor dibuat lebih longgar
    if not (real_left and real_right):
        side_margin = max(side_margin, 0.38)

    ys = np.linspace(int(h * top_u), h - 1, n)
    xl = np.array([det._x(left, float(y)) for y in ys])
    xr = np.array([det._x(right, float(y)) for y in ys])
    if not (np.all(np.isfinite(xl)) and np.all(np.isfinite(xr))):
        return _static_roi(det, h, w)

    width = np.maximum(xr - xl, 12.0)
    margin = np.clip(width * side_margin, min_margin_px, max_margin_px)
    left_x = np.clip(xl - margin, 0, w)
    right_x = np.clip(xr + margin, 0, w)
    if np.any(right_x - left_x < 8):
        return _static_roi(det, h, w)

    poly = np.array(
        [[x, y] for x, y in zip(left_x, ys)]
        + [[x, y] for x, y in zip(right_x[::-1], ys[::-1])],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # sedikit longgar supaya marka di tepi koridor tidak terpotong
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
    det._road_area_dynamic = True
    return mask


def is_dynamic(det: LaneDetector) -> bool:
    return bool(getattr(det, "_road_area_dynamic", False))
