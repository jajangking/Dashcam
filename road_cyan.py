"""Overlay CYAN untuk engine klasik (`lane_v2`).

Deteksinya tetap memakai `LaneDetector` klasik — tidak ada model UFLD yang
dipanggil. Yang diubah hanya gaya gambarnya supaya konsisten dengan tampilan
model: garis cyan + halo, dengan label `ROAD`. Sisi yang sedang diperkirakan
(profil lebar jalur klasik) digambar putus-putus dan diberi tanda `EST`. Saat
kedua sisi sementara hilang, kurva terakhir boleh ditahan (`HOLD`) dengan
kecepatan geser terakhir supaya tidak membeku.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from lane_v2 import LaneDetector

CYAN = (255, 255, 0)          # BGR
HALO = (150, 195, 195)
EST = (225, 235, 235)


def road_points(det: LaneDetector, coeff: Optional[np.ndarray], h: int, w: int, n: int = 40):
    if coeff is None:
        return None
    # urutan naik seperti kurva model: indeks 0 = dekat cakrawala, -1 = bawah
    ys = np.linspace(int(h * 0.60), h - 1, n)
    xs = np.array([det._x(coeff, float(y)) for y in ys])
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    keep = (pts[:, 0] >= -w) & (pts[:, 0] <= 2 * w)
    pts = pts[keep]
    return pts if len(pts) >= 2 else None


def draw_points(
    frame: np.ndarray,
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    estimated: Tuple[bool, bool] = (False, False),
    label: str = "ROAD",
) -> np.ndarray:
    """Gambar dua array titik (N,2) dengan gaya cyan."""
    out = frame.copy()
    h, w = out.shape[:2]
    for pts, est in ((left, estimated[0]), (right, estimated[1])):
        if pts is None or len(pts) < 2:
            continue
        p = np.asarray(pts).reshape(-1, 1, 2).astype(np.int32)
        if est:
            for i in range(0, len(p) - 3, 7):
                cv2.polylines(out, [p[i:i + 5]], False, HALO, 5, cv2.LINE_AA)
                cv2.polylines(out, [p[i:i + 5]], False, EST, 2, cv2.LINE_AA)
        else:
            cv2.polylines(out, [p], False, HALO, 7, cv2.LINE_AA)
            cv2.polylines(out, [p], False, CYAN, 3, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (w, 42), (16, 22, 30), -1)
    if left is None and right is None:
        cv2.putText(out, "TIDAK TERDETEKSI", (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, CYAN, 2, cv2.LINE_AA)
    return out


def draw_road(frame: np.ndarray, det: LaneDetector) -> np.ndarray:
    """Gambar kurva klasik dengan warna cyan tanpa menyentuh logika deteksi."""
    h, w = frame.shape[:2]
    left = road_points(det, det.left_draw, h, w)
    right = road_points(det, det.right_draw, h, w)
    any_est = bool(det.left_estimated or det.right_estimated)
    return draw_points(
        frame, left, right,
        estimated=(bool(det.left_estimated), bool(det.right_estimated)),
        label="ROAD + EST" if any_est else "ROAD",
    )


def bottom_pair(det: LaneDetector, h: int, w: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return (
        road_points(det, det.left_draw, h, w),
        road_points(det, det.right_draw, h, w),
    )
