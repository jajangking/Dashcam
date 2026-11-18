"""Detektor jalur klasik generasi berikutnya: BEV + RANSAC + Kalman.

Kenapa rewrite: detektor `lane_v2` mencari marka langsung di gambar perspective
memakai sliding-window warna. Di footage dashcam ini itu gampang terkunci pada
pagar/bodi mobil, lalu koefisiennya melintir. Strategi di sini:

1. **IPM** — image di-warp ke bird's-eye view (BEV). Marka yang sejajar di dunia
   nyata jadi hampir vertikal dan paralel di BEV.
2. **RANSAC** — fit kurva di BEV dengan voting inlier, bukan rata-rata semua
   piksel terang.
3. **Kalman/EMA** — koefisien dilacak lintas frame; update hanya kalau pengukuran
   masuk akal, kalau tidak pakai prediksi.
4. **Lebar BEV** — lebar jalur hampir konstan di BEV, jadi marka tertutup bisa
   disintesis tanpa menebak lebar perspektif.

Modul ini sengaja tidak menggambar overlay. Antarmukanya sengaja mirip
`LaneDetector`: `process(frame) -> (frame, info)` plus atribut
`left_draw/right_draw/left_coeff/right_coeff/_x/_slope`.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from numpy.exceptions import RankWarning
except ImportError:  # numpy lama
    RankWarning = np.RankWarning
warnings.filterwarnings("ignore", category=RankWarning)


class LaneDetectorBEV:
    def __init__(
        self,
        bev_w: int = 400,
        bev_h: int = 400,
        horizon_u: float = 0.40,
        max_hold: int = 20,
        side_margin: float = 0.35,
    ):
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.horizon_u = horizon_u
        self.max_hold = max_hold
        self.side_margin = side_margin

        self.left_coeff: Optional[np.ndarray] = None   # BEV x(y)
        self.right_coeff: Optional[np.ndarray] = None
        self.left_draw: Optional[np.ndarray] = None    # koefisien gambar asli
        self.right_draw: Optional[np.ndarray] = None
        self.left_estimated = False
        self.right_estimated = False
        self.left_quality = 0.0
        self.right_quality = 0.0
        self.left_missing = 0
        self.right_missing = 0
        self.w_bev: Optional[float] = None
        self.hold = {"left": None, "right": None, "vx": 0.0, "age": 0, "last_x": None}
        self.h, self.w = 1, 1
        self.sanity_note = ""
        self._road_area_dynamic = True
        self._rng = np.random.default_rng(0)

    # ---------- antarmuka kompatibel lane_v2 ----------
    def _x(self, coeff: Optional[np.ndarray], y: float) -> float:
        if coeff is None:
            return float("nan")
        return float(np.polyval(coeff, float(y) / float(self.h)))

    def _slope(self, coeff: Optional[np.ndarray], y: float) -> float:
        if coeff is None:
            return float("nan")
        c = np.asarray(coeff, dtype=np.float64)
        return float(np.polyval(np.polyder(c), float(y) / float(self.h)) / float(self.h))

    def roi_vertices(self, h: int, w: int) -> np.ndarray:
        return np.array([[
            (int(w * 0.02), h),
            (int(w * 0.30), int(h * 0.58)),
            (int(w * 0.70), int(h * 0.58)),
            (int(w * 0.98), h),
        ]], dtype=np.int32)

    # ---------- IPM ----------
    def _homography(self, h: int, w: int):
        cx = w / 2.0
        top_y = int(h * self.horizon_u) + 8
        half_top = max(20, int(w * 0.045))
        half_bottom = int(w * 0.46)
        src = np.float32([
            [cx - half_top, top_y],
            [cx + half_top, top_y],
            [cx + half_bottom, h],
            [cx - half_bottom, h],
        ])
        dst = np.float32([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h],
        ])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    # ---------- fit ----------
    def _ransac(self, xs: np.ndarray, ys: np.ndarray, deg: int = 1) -> Optional[np.ndarray]:
        if len(xs) < 8:
            return None
        best, best_in = None, 0
        for _ in range(45):
            idx = self._rng.choice(len(xs), size=deg + 1, replace=False)
            try:
                c = np.polyfit(ys[idx], xs[idx], deg)
            except Exception:
                continue
            res = np.abs(xs - np.polyval(c, ys))
            n = int((res < 7.0).sum())
            if n > best_in:
                best, best_in = c, n
        if best is None or best_in < 8:
            return None
        res = np.abs(xs - np.polyval(best, ys))
        keep = res < 9.0
        if keep.sum() < 6:
            return None
        return np.polyfit(ys[keep], xs[keep], deg)

    @staticmethod
    def _peaks(hist: np.ndarray, min_sep: int = 80, min_val: float = 12.0) -> List[int]:
        out: List[int] = []
        half = 9
        for i in range(half, len(hist) - half):
            if hist[i] < min_val or hist[i] != hist[i - half:i + half + 1].max():
                continue
            if not out or i - out[-1] > min_sep:
                out.append(i)
        return out

    def _vertical_segments(self, bev: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Segmen Hough yang hampir vertikal di BEV = kandidat marka."""
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=18,
            minLineLength=18, maxLineGap=30,
        )
        segs: List[Tuple[float, float, float, float, float]] = []
        if lines is None:
            return segs
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dy = float(y2 - y1)
            if abs(dy) < 10:
                continue
            m = float(x2 - x1) / dy
            if abs(m) > 0.5:
                continue
            ymax = max(y1, y2)
            if ymax < int(self.bev_h * 0.35):
                continue
            length = float(np.hypot(x2 - x1, dy))
            segs.append(((x1 + x2) / 2.0, float(x1), float(y1), float(x2), float(y2), length))
        return segs

    def _smooth(self, prev: Optional[np.ndarray], new: Optional[np.ndarray], alpha: float = 0.45):
        if new is None:
            return prev
        if prev is None or len(prev) != len(new):
            return new
        ys = np.array([0.15, 0.55, 1.0]) * self.bev_h
        v = np.polyval(prev, ys) + alpha * (np.polyval(new, ys) - np.polyval(prev, ys))
        return np.polyfit(ys, v, len(new) - 1)

    # ---------- proses ----------
    def process(self, frame: np.ndarray, debug_mask: bool = False) -> Tuple[np.ndarray, Dict]:
        h, w = frame.shape[:2]
        self.h, self.w = h, w
        M, Minv = self._homography(h, w)
        bev = cv2.warpPerspective(frame, M, (self.bev_w, self.bev_h))
        segs = self._vertical_segments(bev)
        raw: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        qual = {"left": 0.0, "right": 0.0}

        hist = np.zeros(self.bev_w, dtype=np.float32)
        for mid, x1, y1, x2, y2, length in segs:
            hist[int(np.clip(mid, 0, self.bev_w - 1))] += length
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 9).ravel()
        peaks = self._peaks(hist, min_sep=55, min_val=4)

        y0 = self.bev_h * 0.75
        pred = {}
        for side in ("left", "right"):
            c = self.__dict__[side + "_coeff"]
            pred[side] = float(np.polyval(c, y0)) if c is not None else (
                self.bev_w * (0.35 if side == "left" else 0.65)
            )
        real_peaks: Dict[str, float] = {}
        if self.left_coeff is not None or self.right_coeff is not None:
            for side in ("left", "right"):
                if peaks:
                    best = min(peaks, key=lambda p: abs(p - pred[side]))
                    if abs(best - pred[side]) < 65:
                        real_peaks[side] = best
        if not real_peaks and len(peaks) >= 2:
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    a, b = peaks[i], peaks[j]
                    width = b - a
                    center = (a + b) / 2.0
                    if 90 <= width <= 280 and abs(center - self.bev_w / 2) < 120:
                        real_peaks = {"left": a, "right": b}
                        break
                if real_peaks:
                    break
        if not real_peaks and peaks:
            p = peaks[0]
            real_peaks = {"left": p} if p < self.bev_w / 2 else {"right": p}

        for side, px in real_peaks.items():
            xs, ys = [], []
            for mid, x1, y1, x2, y2, length in segs:
                if abs(mid - px) > 32:
                    continue
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            if len(xs) < 12:
                continue
            c = self._ransac(np.array(xs), np.array(ys), deg=1)
            if c is None:
                continue
            if abs(float(np.polyval(c, y0)) - pred[side]) > 30:
                continue
            raw[side] = c
            span = (max(ys) - min(ys)) / self.bev_h
            qual[side] = min(1.0, 0.4 + 0.6 * span)

        # lebar jalur di BEV hampir konstan — dipelajari hanya dari dua sisi nyata
        if raw["left"] is not None and raw["right"] is not None:
            y0 = self.bev_h * 0.75
            wl = float(np.polyval(raw["left"], y0))
            wr = float(np.polyval(raw["right"], y0))
            width = wr - wl
            if 80 <= width <= 280:
                self.w_bev = width if self.w_bev is None else 0.8 * self.w_bev + 0.2 * width

        estimated = {"left": False, "right": False}
        for side, other in (("left", "right"), ("right", "left")):
            if raw[side] is None and raw[other] is not None and self.w_bev:
                c = raw[other].copy()
                c[-1] += -self.w_bev if other == "right" else self.w_bev
                raw[side] = c
                estimated[side] = True
                qual[side] = min(0.5, qual[other])

        held = False
        if raw["left"] is None and raw["right"] is None and self.hold["left"] is not None \
                and self.hold["age"] < self.max_hold:
            self.hold["age"] += 1
            shift = self.hold["vx"] * self.hold["age"]
            left = self.hold["left"].copy()
            right = self.hold["right"].copy()
            left[-1] += shift
            right[-1] += shift
            raw = {"left": left, "right": right}
            estimated = {"left": True, "right": True}
            held = True

        for side in ("left", "right"):
            if raw[side] is None:
                self.__dict__[side + "_missing"] += 1
                continue
            self.__dict__[side + "_missing"] = 0
            prev = self.__dict__[side + "_coeff"]
            new = self._smooth(prev, raw[side], 0.35 if estimated[side] else 0.5)
            self.__dict__[side + "_coeff"] = new
            self.__dict__[side + "_estimated"] = estimated[side]
            self.__dict__[side + "_quality"] = qual[side]

        # pelajari kecepatan geser dari dua sisi nyata
        if not estimated["left"] and not estimated["right"] and raw["left"] is not None and raw["right"] is not None:
            y0 = self.bev_h * 0.85
            xl = float(np.polyval(raw["left"], y0))
            xr = float(np.polyval(raw["right"], y0))
            if self.hold["last_x"] is not None:
                dx = ((xl - self.hold["last_x"][0]) + (xr - self.hold["last_x"][1])) / 2.0
                if abs(dx) < 0.12 * self.bev_w:
                    self.hold["vx"] = 0.7 * self.hold["vx"] + 0.3 * dx
            self.hold["last_x"] = (xl, xr)
            self.hold["left"] = self.left_coeff.copy() if self.left_coeff is not None else None
            self.hold["right"] = self.right_coeff.copy() if self.right_coeff is not None else None
            self.hold["age"] = 0
        if raw["left"] is None and raw["right"] is None:
            self.hold["left"] = self.hold["right"] = None
            self.hold["last_x"] = None
            self.hold["vx"] = 0.0
            self.hold["age"] = 0

        # ubah kurva BEV kembali ke gambar asli, lalu simpan koefisien kompatibel
        prev_draw = {"left": self.left_draw, "right": self.right_draw}
        ys_bev = np.linspace(self.bev_h * 0.38, self.bev_h - 1, 32)
        for side in ("left", "right"):
            coeff = self.__dict__[side + "_coeff"]
            if coeff is None:
                self.__dict__[side + "_draw"] = None
                continue
            xs = np.polyval(coeff, ys_bev)
            pts = np.stack([xs, ys_bev], axis=1).astype(np.float32).reshape(-1, 1, 2)
            orig = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
            keep = np.isfinite(orig).all(axis=1) & (orig[:, 1] > 0) & (orig[:, 1] < h)
            orig = orig[keep]
            if len(orig) < 4:
                self.__dict__[side + "_draw"] = None
                continue
            new_draw = np.polyfit(orig[:, 1] / h, orig[:, 0], 2)
            # gerbang akhir di ruang gambar: mobil tidak bisa bergeser >12 px/frame.
            # Kalau ada, itu masih loncatan deteksi — pakai kurva frame sebelumnya.
            if prev_draw[side] is not None:
                x_prev = self._x(prev_draw[side], h - 1)
                x_new = self._x(new_draw, h - 1)
                if np.isfinite(x_prev) and np.isfinite(x_new) and abs(x_new - x_prev) > 12:
                    new_draw = prev_draw[side]
            self.__dict__[side + "_draw"] = new_draw

        ok_l = self.left_draw is not None
        ok_r = self.right_draw is not None
        info = {
            "left": ok_l,
            "right": ok_r,
            "left_stale": False,
            "right_stale": False,
            "left_estimated": bool(self.left_estimated),
            "right_estimated": bool(self.right_estimated),
            "left_points": 0,
            "right_points": 0,
            "left_quality": round(float(self.left_quality), 2),
            "right_quality": round(float(self.right_quality), 2),
            "left_span_pct": 0.0,
            "right_span_pct": 0.0,
            "offset_pct": None,
            "turn": None,
            "road_area_dynamic": 0,
            "bev_held": int(held),
            "bev_w": round(float(self.w_bev), 1) if self.w_bev else None,
            "rejected": "",
        }
        return frame.copy(), info
