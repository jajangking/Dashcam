#!/usr/bin/env python3
"""
Deteksi jalur v2.1 — fokus pada KESTABILAN.

Perubahan penting dibanding v2.0:
- fitting dilakukan pada koordinat ternormalisasi (u = y/h) supaya tidak
  menghasilkan kurva "bergelombang" karena ketidakstabilan numerik;
- derajat maksimal 2 dan kurva wajib monoton (tidak boleh balik arah);
- regularisasi temporal: kalau marka yang terlihat hanya potongan kecil,
  bentuk kurva lebih percaya pada frame sebelumnya (tidak melompat-lompat);
- pencarian jendela dimulai dari posisi jalur frame sebelumnya (prediksi);
- fitting robust (IRLS) supaya objek putih lain tidak menarik garis;
- kualitas deteksi dilaporkan (kuat / sedang / lemah).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        n_windows: int = 9,
        margin: int = 100,
        minpix: int = 40,
        smooth_alpha: float = 0.45,
        max_missing: int = 8,
        estimate_missing: bool = True,
        max_jump_pct: float = 4.0,
        color_min_v: int = 145,
        color_max_s: int = 70,
        max_mask_density: float = 0.35,
        max_band_spread_px: float = 14.0,
        use_road_area: bool = False,
    ):
        self.n_windows = n_windows
        self.margin = margin
        self.minpix = minpix
        self.smooth_alpha = smooth_alpha
        self.max_missing = max_missing
        self.estimate_missing = estimate_missing
        self.max_jump_pct = max_jump_pct
        self.color_min_v = color_min_v
        self.color_max_s = color_max_s
        # penjaga "apakah ini gambar jalan?" (lihat _sanity)
        self.max_mask_density = max_mask_density
        self.max_band_spread_px = max_band_spread_px
        # opsi: batasi pencarian marka ke area jalan (koridor kurva terpercaya)
        self.use_road_area = use_road_area
        self.sanity_note = ""

        # koefisien selalu dalam koordinat ternormalisasi u = y / h
        # x(u) = c[0]*u^2 + c[1]*u + c[2]  (derajat 2) atau c[0]*u + c[1]
        self.left_coeff: Optional[np.ndarray] = None
        self.right_coeff: Optional[np.ndarray] = None
        self.left_missing = 0
        self.right_missing = 0
        self.left_quality = 0.0
        self.right_quality = 0.0
        self.left_span_pct = 0.0
        self.right_span_pct = 0.0
        # Lebar jalur dipelajari sebagai fungsi posisi vertikal (bukan satu angka!).
        # Karena perspektif, lebar jalur di layar menyempit ke arah horizon:
        #   W(u) = w_a * u + w_b      dengan u = y / h
        # Menyimpannya sebagai konstanta (perilaku lama) membuat sisi yang hilang
        # ditebak ratusan piksel meleset di tikungan.
        self.lane_w_a: Optional[float] = None
        self.lane_w_b: Optional[float] = None
        self.lane_w_samples = 0
        # SATU ukuran lebar jalur (piksel) untuk dipakai saat salah satu sisi
        # tidak terdeteksi. Hanya diperbarui dari frame yang KEDUA sisinya
        # terlihat (dan tidak sekedar hasil tebakan), lalu dihaluskan.
        self.lane_w_m = None
        self.lane_w_mirror = False
        self.left_estimated = False
        self.right_estimated = False
        self.left_stale = False
        self.right_stale = False
        # Rentang u (=y/h) yang BENAR-BENAR teramati per sisi. Ini kunci: kurva
        # derajat 2 yang dipakai jauh di luar rentang datanya akan melengkung ke
        # arah yang salah. Terukur di `case_belok_480`: galat 0.4 px di dalam
        # data, 189 px di luar data.
        self.left_obs_u: Optional[Tuple[float, float]] = None
        self._learned_this_frame = False
        self.right_obs_u: Optional[Tuple[float, float]] = None
        self.left_extended = False
        self.right_extended = False
        self.left_draw = None
        self.right_draw = None

        self.h = 1
        self.w = 1

    # ---------- ROI ----------
    @staticmethod
    def roi_vertices(h: int, w: int) -> np.ndarray:
        if h > w:  # portrait
            return np.array([[
                (int(w * 0.02), h),
                (int(w * 0.20), int(h * 0.52)),
                (int(w * 0.80), int(h * 0.52)),
                (int(w * 0.98), h),
            ]], dtype=np.int32)
        return np.array([[
            (int(w * 0.02), h),
            (int(w * 0.30), int(h * 0.58)),
            (int(w * 0.70), int(h * 0.58)),
            (int(w * 0.98), h),
        ]], dtype=np.int32)

    # ---------- bantu ----------
    def _x(self, coeff: Optional[np.ndarray], y: float) -> float:
        """Evaluasi garis pada posisi y (piksel) memakai koefisien ternormalisasi."""
        if coeff is None:
            return float("nan")
        return float(np.polyval(coeff, float(y) / float(self.h)))

    def _slope(self, coeff: Optional[np.ndarray], y: float) -> float:
        """dx/dy (piksel per piksel) di posisi y."""
        if coeff is None:
            return float("nan")
        c = np.asarray(coeff, dtype=np.float64)
        d = np.polyder(c)  # turunan terhadap u
        return float(np.polyval(d, float(y) / float(self.h)) / float(self.h))

    # ---------- masker warna ----------
    def color_mask(self, frame: np.ndarray, roi: np.ndarray, area_mask: Optional[np.ndarray] = None) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        white = ((s_ch < self.color_max_s) & (v_ch > self.color_min_v)).astype(np.uint8) * 255
        yellow = ((h_ch > 12) & (h_ch < 40) & (s_ch > 80) & (v_ch > 90)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(white, yellow)

        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, roi, 255)
        mask = cv2.bitwise_and(mask, roi_mask)
        if area_mask is not None:
            mask = cv2.bitwise_and(mask, area_mask)

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    # ---------- pencarian jendela ----------
    def _histogram_bases(self, mask: np.ndarray) -> Tuple[int, int]:
        h, w = mask.shape[:2]
        hist = np.sum(mask[int(h * 0.65):, :], axis=0)
        mid = w // 2
        left = int(np.argmax(hist[:mid])) if hist[:mid].any() else int(w * 0.25)
        right = mid + int(np.argmax(hist[mid:])) if hist[mid:].any() else int(w * 0.75)
        return left, right

    def _windows(self, mask: np.ndarray, lx: int, rx: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape[:2]
        win_h = max(1, int((h * 0.45) / self.n_windows))
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        nz_y, nz_x = np.nonzero(mask)
        for i in range(self.n_windows):
            y_high = h - i * win_h
            y_low = max(0, y_high - win_h)
            band = (nz_y >= y_low) & (nz_y < y_high)
            for side, cx, store in (("L", lx, left_pts), ("R", rx, right_pts)):
                x_lo = max(0, cx - self.margin)
                x_hi = min(w, cx + self.margin)
                sel = band & (nz_x >= x_lo) & (nz_x < x_hi)
                xs = nz_x[sel]
                ys = nz_y[sel]
                if len(xs) > self.minpix:
                    cx = int(np.mean(xs))
                    if side == "L":
                        lx = cx
                    else:
                        rx = cx
                    store.extend(zip(xs.tolist(), ys.tolist()))
        return (
            np.array(left_pts, dtype=np.float64).reshape(-1, 2),
            np.array(right_pts, dtype=np.float64).reshape(-1, 2),
        )

    def _window_search(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape[:2]
        lb, rb = self._histogram_bases(mask)
        lp, rp = self._windows(mask, lb, rb)

        # pencarian berbasis prediksi hanya dipakai kalau cara biasa gagal
        # (mencegah garis "ikut bergeser" mengikuti jendela sendiri)
        if len(lp) + len(rp) < 300 and (self.left_coeff is not None or self.right_coeff is not None):
            pl = lb
            pr = rb
            if self.left_coeff is not None and not self.left_estimated:
                pl = int(np.clip(self._x(self.left_coeff, h - 2), -w, 2 * w))
            if self.right_coeff is not None and not self.right_estimated:
                pr = int(np.clip(self._x(self.right_coeff, h - 2), -w, 2 * w))
            lp2, rp2 = self._windows(mask, pl, pr)
            if len(lp2) + len(rp2) > len(lp) + len(rp):
                lp, rp = lp2, rp2
        return lp, rp

    # ---------- fitting robust + regularisasi temporal ----------
    def _fit(
        self,
        points: np.ndarray,
        prior: Optional[np.ndarray],
        weights: Optional[np.ndarray] = None,
        min_span_pct: float = 5.0,
    ) -> Optional[np.ndarray]:
        if points is None or len(points) < 20:
            return None
        y = points[:, 1].astype(np.float64)
        x = points[:, 0].astype(np.float64)
        span = float(y.max() - y.min())
        if span < self.h * min_span_pct / 100.0:
            return None

        u = y / float(self.h)
        deg = 2 if (span >= self.h * 0.15 and len(points) >= 150) else 1

        A = np.stack([u ** k for k in range(deg, -1, -1)], axis=1)
        base_w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)

        # seberapa kuat bukti: rentang vertikal + banyak titik
        ev = min(1.0, span / (0.35 * self.h)) * min(1.0, len(x) / 1200.0)

        p = None
        if prior is not None:
            p = np.asarray(prior, dtype=np.float64)
            if len(p) > deg + 1:
                p = p[-(deg + 1):]
            elif len(p) < deg + 1:
                # naikkan derajat: kelengkungan awal 0
                p = np.concatenate([np.zeros(deg + 1 - len(p)), p])

        lam = np.zeros(deg + 1)
        if p is not None:
            n = float(len(x))
            if deg == 2:
                lam[0] = (1.0 - ev) * 6.0 * n        # kelengkungan: stabil
                lam[1] = (1.0 - ev) * 0.25 * n       # kemiringan: sedikit stabil
            else:
                lam[0] = (1.0 - ev) * 0.30 * n       # derajat 1: kemiringan
            # posisi (konstanta) selalu mengikuti data

        w = base_w.copy()
        c = None
        for _ in range(4):
            Aw = A * w[:, None]
            M = Aw.T @ A
            b = Aw.T @ x
            if p is not None:
                M = M + np.diag(lam)
                b = b + lam * p
            M = M + 1e-9 * np.eye(deg + 1)
            try:
                c = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                return None
            r = np.abs(A @ c - x)
            s = max(3.0, 0.012 * self.h)
            w = base_w / (1.0 + (r / s) ** 2)

        if c is None or not np.all(np.isfinite(c)):
            return None

        # batasi kelengkungan yang tidak masuk akal (deg2 tidak bisa membentuk S)
        if deg == 2:
            limit = 6.0 * self.w
            c[0] = float(np.clip(c[0], -limit, limit))
        return np.asarray(c, dtype=np.float64)

    def _plausible(self, coeff: np.ndarray, side: str,
                   max_slope: float = 3.0) -> bool:
        """Saringan kasar supaya garis tidak jatuh ke objek yang jelas bukan jalur.

        `max_slope` bisa dilonggarkan untuk kurva yang DIBANGUN (bukan dideteksi):
        pada tikungan tajam asli kemiringan marka memang bisa 3.2 px/px, terukur
        di `case_belok_480`.
        """
        x_top = float(np.polyval(coeff, 0.62))
        x_mid = float(np.polyval(coeff, 0.80))
        x_bot = float(np.polyval(coeff, 1.00))
        if not np.all(np.isfinite([x_top, x_mid, x_bot])):
            return False
        # jalur harus berada di dalam/sekitar gambar pada semua ketinggian
        if not (-0.25 * self.w <= x_mid <= 1.25 * self.w):
            return False
        if not (-0.30 * self.w <= x_bot <= 1.30 * self.w):
            return False
        if not (-0.10 * self.w <= x_top <= 1.10 * self.w):
            return False
        if side == "L" and x_bot > 0.90 * self.w:
            return False
        if side == "R" and x_bot < 0.10 * self.w:
            return False
        s2 = self._slope(coeff, self.h - 1)
        if not np.isfinite(s2) or abs(s2) > max_slope:
            return False
        return True

    # ---------- blob / garis putus-putus ----------
    def _blob_candidates(self, mask: np.ndarray, side: str) -> Tuple[List[dict], np.ndarray]:
        h, w = mask.shape[:2]
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < 25 or bh < 4:
                continue
            if bw > w * 0.6:
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            if side == "L" and cx >= w * 0.55:
                continue
            if side == "R" and cx <= w * 0.45:
                continue
            out.append({"cx": cx, "cy": cy, "area": area, "bh": bh, "bw": bw, "label": i})
        return out, labels

    def _fit_blobs(
        self,
        mask: np.ndarray,
        side: str,
        prior: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float]:
        blobs, labels = self._blob_candidates(mask, side)
        if not blobs:
            return None, 0.0

        if len(blobs) >= 2:
            pts = np.array([[b["cx"], b["cy"]] for b in blobs], dtype=np.float64)
            wts = np.sqrt(np.array([b["area"] for b in blobs], dtype=np.float64))
            c = self._fit(pts, prior, weights=wts, min_span_pct=4.0)
            if c is None:
                return None, 0.0
            span = pts[:, 1].max() - pts[:, 1].min()
            q = 0.55 * min(1.0, span / (0.30 * self.h)) + 0.45 * min(1.0, len(blobs) / 6.0)
            return c, float(min(0.95, q))

        # hanya satu potongan
        b = blobs[0]
        if b["cy"] < self.h * 0.70:
            return None, 0.0

        # potongan memanjang (arahnya bisa dipercaya) -> pakai arah potongan itu
        if b["bh"] >= 1.5 * b["bw"] and b["bh"] >= 0.02 * self.h:
            ys, xs = np.nonzero(labels == b["label"])
            if len(xs) < 15:
                return None, 0.0

            sample = np.stack([xs, ys], axis=1).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(sample, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            if abs(float(vy)) < 1e-3:
                return None, 0.0
            slope = float(vx) / float(vy)  # dx/dy
            if not (0.05 <= abs(slope) <= 6.0):
                return None, 0.0
            if side == "L" and slope > -0.05:
                return None, 0.0
            if side == "R" and slope < 0.05:
                return None, 0.0

            # konversi ke koefisien ternormalisasi (derajat 1)
            a1 = slope * self.h
            a0 = float(x0) - slope * float(y0)
            coeff = np.array([a1, a0], dtype=np.float64)
            # gabungkan dengan bentuk frame sebelumnya supaya tidak liar
            if prior is not None and len(prior) == 2:
                coeff = 0.5 * np.asarray(prior) + 0.5 * coeff
            return coeff, 0.22

        # potongan pipih/lebar (marka sangat dekat kamera): arahnya tidak bisa
        # dipakai karena garis jadi hampir datar lalu melejit saat diperpanjang.
        # Pakai bentuk frame sebelumnya, atau arahkan ke titik hilang.
        if prior is not None:
            coeff = np.array(prior, dtype=np.float64)
            shift = b["cx"] - self._x(coeff, b["cy"])
            coeff[-1] += shift
            return coeff, 0.25

        vp_y = 0.58 * self.h
        if b["cy"] - vp_y < 0.05 * self.h:
            return None, 0.0
        slope = (b["cx"] - 0.5 * self.w) / (b["cy"] - vp_y)
        if not (0.05 <= abs(slope) <= 6.0):
            return None, 0.0
        if side == "L" and slope > -0.05:
            return None, 0.0
        if side == "R" and slope < 0.05:
            return None, 0.0
        a1 = slope * self.h
        a0 = b["cx"] - slope * b["cy"]
        return np.array([a1, a0], dtype=np.float64), 0.22

    def _observed_range(self, mask: np.ndarray, coeff: np.ndarray,
                        tol_px: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """Rentang u yang benar-benar punya piksel marka di dekat kurva.

        Ini menandai batas "data" lawan "ekstrapolasi". Tanpa ini, sisi yang
        terhalang truk tetap digambar sampai bawah gambar seolah terdeteksi.
        """
        h, w = mask.shape[:2]
        tol = float(tol_px if tol_px is not None else max(12.0, 0.02 * w))
        rows = np.arange(int(0.55 * h), h, max(2, h // 120))
        # ambil hanya baris yang punya piksel mask di sekitar kurva
        nz_y, nz_x = np.nonzero(mask)
        if len(nz_y) == 0:
            return None
        keep = np.zeros(len(nz_y), dtype=bool)
        pred = np.polyval(coeff, nz_y / float(h))
        keep = np.abs(nz_x - pred) <= tol
        if keep.sum() < 40:
            return None
        us = nz_y[keep] / float(h)
        return float(us.min()), float(us.max())

    # ---------- dukungan piksel ----------
    def _support(self, dmap: np.ndarray, coeff: np.ndarray) -> float:
        """Fraksi titik kurva yang dekat dengan piksel marka (putih/kuning)."""
        h, w = dmap.shape[:2]
        tol = max(10.0, 0.015 * w)
        us = np.linspace(0.62, 1.0, 24)
        xs = np.polyval(coeff, us)
        ys = us * h
        ok = 0
        for x, y in zip(xs, ys):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h and dmap[yi, xi] <= tol:
                ok += 1
        return ok / float(len(us))

    # ---------- deteksi satu sisi ----------
    def _detect_side(
        self,
        mask: np.ndarray,
        side: str,
        pts: np.ndarray,
        prior: Optional[np.ndarray],
        dmap: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        h, w = mask.shape[:2]
        cand = None
        quality = 0.0
        span_pct = 0.0

        if pts is not None and len(pts) >= 150:
            span = float(pts[:, 1].max() - pts[:, 1].min())
            span_pct = span / h * 100.0
            c = self._fit(pts, prior)
            if c is not None and self._plausible(c, side):
                cand = c
                quality = 0.7 * min(1.0, span / (0.30 * h)) + 0.3 * min(1.0, len(pts) / 1200.0)
                # perbaikan: ambil ulang piksel di sekitar kurva (bantu tikungan)
                if span > 0.15 * h:
                    band = _band_points(mask, c, max(40, self.margin), h)
                    if len(band) >= max(150, len(pts) // 2):
                        c2 = self._fit(band, cand)
                        if c2 is not None and self._plausible(c2, side):
                            cand = c2

        cb, qb = self._fit_blobs(mask, side, prior if cand is None else cand)
        if cb is not None and (cand is None or qb > quality):
            if cand is None or self._plausible(cb, side):
                cand, quality = cb, qb

        if cand is None:
            return None, 0.0, span_pct

        # garis harus benar-benar berada di atas marka, bukan di area kosong
        if dmap is not None:
            sup = self._support(dmap, cand)
            if sup < 0.08:
                return None, 0.0, span_pct

        # gerbang: lompatan posisi mendadak hampir selalu salah deteksi
        if prior is not None and len(prior) > 0:
            pred = self._x(prior, h - 1)
            got = self._x(cand, h - 1)
            tol = (1.5 if quality >= 0.6 else 1.0) * self.max_jump_pct / 100.0 * w
            if abs(got - pred) > tol:
                return None, 0.0, span_pct
        return cand, float(min(1.0, quality)), span_pct

    # ---------- smoothing ----------
    def _smooth(
        self,
        prev: Optional[np.ndarray],
        new: Optional[np.ndarray],
        quality: float = 1.0,
    ) -> Optional[np.ndarray]:
        if new is None:
            return prev
        if prev is None or len(prev) != len(new):
            return np.asarray(new, dtype=np.float64)
        new = np.asarray(new, dtype=np.float64)
        prev = np.asarray(prev, dtype=np.float64)
        deg = len(new) - 1
        # haluskan NILAI kurva di beberapa titik (bukan koefisien satu-satu),
        # supaya bentuk kurva tidak berubah aneh saat koefisien besar.
        if deg == 2:
            us = np.array([0.62, 0.80, 1.0])
            a = np.array([0.16, 0.40, 0.52])
        else:
            us = np.array([0.80, 1.0])
            a = np.array([0.40, 0.52])
        a = a * float(np.clip(0.35 + 0.65 * quality, 0.35, 1.0))
        v_prev = np.polyval(prev, us)
        v_new = np.polyval(new, us)
        v = v_prev + a * (v_new - v_prev)
        try:
            return np.polyfit(us, v, deg)
        except Exception:
            return new

    # ---------- proses 1 frame ----------
    def _sanity(self, mask: np.ndarray, roi: np.ndarray) -> Tuple[bool, str]:
        """Periksa apakah masker masuk akal sebagai marka, sebelum fitting apa pun.

        Dua penyebab salah deteksi paling sering tidak ada hubungannya dengan
        fitting: gambar yang isinya bukan jalan sama sekali (noise, salju TV,
        lantai bertekstur, hujan ekstrem). Di kondisi itu hampir seluruh ROI
        berwarna "putih", dan metode apa pun akan menemukan "garis".
        Lebih baik menolak dengan jujur daripada menggambar jalur palsu.
        """
        h, w = mask.shape[:2]
        roi_area = float(cv2.contourArea(roi.astype(np.int32)))
        if roi_area <= 0:
            return True, ""
        density = float((mask > 0).sum()) / roi_area
        if density > self.max_mask_density:
            return False, f"terlalu banyak piksel terang di area jalan ({density:.0%}) — gambar bukan jalan?"

        # Pita di sekitar kurva harus RAMPING. Marka asli selebar belasan piksel;
        # noise/tekstur menyebar puluhan piksel karena jendela pencarian ikut
        # menggemuk. Ini penjaga kedua kalau kepadatan masih lolos.
        ys, xs = np.nonzero(mask)
        stds = []
        for coeff in (self.left_draw, self.right_draw):
            if coeff is None or len(ys) == 0:
                continue
            pred = np.polyval(coeff, ys / float(h))
            keep = np.abs(xs - pred) <= 40
            if keep.sum() < 80:
                continue
            yy, xx = ys[keep], xs[keep]
            per_row = [float(np.std(xx[yy == y])) for y in np.unique(yy) if (yy == y).sum() >= 3]
            if per_row:
                stds.append(float(np.median(per_row)))
        if stds and float(np.mean(stds)) > self.max_band_spread_px:
            return False, "piksel terang menyebar terlalu lebar di sekitar garis — bukan marka"
        return True, ""

    def process(self, frame: np.ndarray, debug_mask: bool = False) -> Tuple[np.ndarray, Dict]:
        h, w = frame.shape[:2]
        self.h, self.w = h, w
        roi = self.roi_vertices(h, w)
        area_mask = None
        if self.use_road_area:
            from road_area import is_dynamic, road_area_mask

            area_mask = road_area_mask(self, h, w)
            self._road_area_dynamic = is_dynamic(self)
        mask = self.color_mask(frame, roi, area_mask)

        # --- penjaga: apakah ini gambar jalan yang wajar? ---
        sane, why = self._sanity(mask, roi)
        if not sane:
            self.left_missing = self.max_missing + 1
            self.right_missing = self.max_missing + 1
            self.left_stale = self.left_coeff is not None
            self.right_stale = self.right_coeff is not None
            self.left_quality = 0.0
            self.right_quality = 0.0
            self.left_draw = None
            self.right_draw = None
            self.left_estimated = False
            self.right_estimated = False
            self.sanity_note = why
            out = self.draw(frame, roi, mask if debug_mask else None)
            return out, {
                "left": False, "right": False,
                "left_stale": self.left_stale, "right_stale": self.right_stale,
                "left_estimated": False, "right_estimated": False,
                "left_points": 0, "right_points": 0,
                "left_quality": 0.0, "right_quality": 0.0,
                "left_span_pct": 0.0, "right_span_pct": 0.0,
                "offset_pct": None, "turn": None,
                "road_area_dynamic": bool(getattr(self, "_road_area_dynamic", False)),
                "rejected": why,
            }
        self.sanity_note = ""

        left_pts, right_pts = self._window_search(mask)
        dmap = cv2.distanceTransform((255 - mask).astype(np.uint8), cv2.DIST_L2, 3)

        new_left, q_left, span_left = self._detect_side(
            mask, "L", left_pts, self.left_coeff, dmap
        )
        new_right, q_right, span_right = self._detect_side(
            mask, "R", right_pts, self.right_coeff, dmap
        )

        if new_left is not None:
            self.left_coeff = self._smooth(self.left_coeff, new_left, q_left)
            self.left_obs_u = self._observed_range(mask, self.left_coeff)
            self.left_missing = 0
            self.left_stale = False
            self.left_quality = 0.6 * self.left_quality + 0.4 * q_left if self.left_quality else q_left
            self.left_span_pct = span_left
        else:
            self.left_missing += 1
            self.left_quality *= 0.9
            if self.left_missing > self.max_missing:
                # bentuk terakhir tetap disimpan untuk prediksi, tapi tidak digambar
                self.left_stale = True
                self.left_quality = 0.0

        if new_right is not None:
            self.right_coeff = self._smooth(self.right_coeff, new_right, q_right)
            self.right_obs_u = self._observed_range(mask, self.right_coeff)
            self.right_missing = 0
            self.right_stale = False
            self.right_quality = 0.6 * self.right_quality + 0.4 * q_right if self.right_quality else q_right
            self.right_span_pct = span_right
        else:
            self.right_missing += 1
            self.right_quality *= 0.9
            if self.right_missing > self.max_missing:
                self.right_stale = True
                self.right_quality = 0.0

        self.left_extended = False
        self.right_extended = False
        # urutan penting: perluasan butuh model lebar, jadi pelajari dulu
        self._learned_this_frame = False
        if (self.left_coeff is not None and not self.left_stale
                and self.right_coeff is not None and not self.right_stale):
            self._learned_this_frame = self._learn_lane_width(w)
        self._extend_short_sides(h, w)
        self._reconstruct_missing_side(h, w)

        out = self.draw(frame, roi, mask if debug_mask else None)
        left_ok = self.left_draw is not None
        right_ok = self.right_draw is not None
        info = {
            "left": left_ok,
            "right": right_ok,
            "left_stale": self.left_stale,
            "right_stale": self.right_stale,
            "left_estimated": self.left_estimated,
            "right_estimated": self.right_estimated,
            "left_extended": self.left_extended,
            "right_extended": self.right_extended,
            "width_assumed": bool(self.lane_w_mirror),
            "left_obs_u": self.left_obs_u,
            "right_obs_u": self.right_obs_u,
            "left_points": len(left_pts),
            "right_points": len(right_pts),
            "left_quality": round(float(self.left_quality), 2),
            "right_quality": round(float(self.right_quality), 2),
            "left_span_pct": round(float(self.left_span_pct), 1),
            "right_span_pct": round(float(self.right_span_pct), 1),
            "offset_pct": self.center_offset(w, h),
            "turn": self.turn_label(w, h),
            "road_area_dynamic": bool(getattr(self, "_road_area_dynamic", False)),
            "rejected": "",
        }
        return out, info

    # ---------- rekonstruksi satu sisi ----------
    def _reconstruct_missing_side(self, h: int, w: int) -> None:
        """Tentukan kurva yang digambar; sisi yang hilang boleh diperkirakan.

        Koefisien hasil deteksi (left_coeff/right_coeff) tidak pernah ditimpa
        dengan perkiraan, supaya perkiraan tidak dianggap deteksi nyata.
        """
        ld_fresh = self.left_coeff is not None and not self.left_stale
        rd_fresh = self.right_coeff is not None and not self.right_stale
        self.left_estimated = False
        self.right_estimated = False
        self.left_draw = self.left_coeff if ld_fresh else None
        self.right_draw = self.right_coeff if rd_fresh else None
        if not self.estimate_missing:
            return

        if ld_fresh and rd_fresh:
            self._learn_single_width(w)
            # Lebar jalur dipelajari HANYA di rentang u tempat kedua marka
            # benar-benar teramati. Versi lama memakai dua titik (u=0.86 dan 1.0)
            # yang pada footage asli sering berupa ekstrapolasi -> model lebar
            # dengan titik nol di u=-0.58 (di luar gambar), lalu estimasi tikungan
            # meleset ratusan piksel.
            if not getattr(self, "_learned_this_frame", False):
                self._learned_this_frame = self._learn_lane_width(w)

        if self.lane_w_a is None:
            # Belum pernah lihat dua sisi        if self.lane_w_a is None:
            # Belum pernah lihat dua sisi: asumsikan jalur simetris terhadap
            # pusat gambar, dan lebarnya menyempit linier ke arah horizon.
            # Kalau asumsi ini meleset, sisi tebakan tetap ditandai "estimasi".
            if ld_fresh and self.left_quality >= 0.5:
                xl = self._x(self.left_coeff, h - 1)
                half_hi = w / 2.0 - xl
            elif rd_fresh and self.right_quality >= 0.5:
                xr = self._x(self.right_coeff, h - 1)
                half_hi = xr - w / 2.0
            else:
                return
            guess = 2.0 * half_hi
            if not (0.2 * w <= guess <= 0.9 * w):
                return
            # Model lebar untuk sisi yang tidak pernah terlihat.
            # CATATAN JUJUR: percobaan memakai cakrawala u=0.45 (a=1.82W, b=-0.82W)
            # membuat galat sisi kanan di footage asli MEMBURUK (median 89 -> 123 px,
            # diukur pada samples/mentah/mentah_1001183916.mp4), karena model yang
            # lebih sempit di atas menarik tebakan makin ke dalam. Jadi nilai lama
            # dipertahankan; lebar jalur di footage asli tetap TIDAK DIKETAHUI dan
            # ditandai `width_assumed`.
            self.lane_w_a, self.lane_w_b = guess * 0.75, guess * 0.25
            self.lane_w_mirror = True
        elif self.lane_w_samples < 5 and not self.lane_w_mirror:
            return

        est = self._estimate_other_side(h, ld_fresh)
        if est is None:
            return
        if ld_fresh and self.right_draw is None:
            self.right_draw = est
            self.right_estimated = True
        elif rd_fresh and self.left_draw is None:
            self.left_draw = est
            self.left_estimated = True

    def _learn_lane_width(self, w: int) -> bool:
        """Perbarui W(u) = a*u + b dari irisan rentang teramati kedua sisi."""
        if self.left_coeff is None or self.right_coeff is None:
            return False
        if self.left_obs_u is None or self.right_obs_u is None:
            return False
        lo = max(self.left_obs_u[0], self.right_obs_u[0])
        hi = min(self.left_obs_u[1], self.right_obs_u[1])
        if hi - lo < 0.10:                     # irisan terlalu pendek -> jangan belajar
            return False
        us = np.linspace(lo, hi, 32)
        widths = np.polyval(self.right_coeff, us) - np.polyval(self.left_coeff, us)
        if not np.all(np.isfinite(widths)) or np.min(widths) <= 0.05 * w:
            return False
        try:
            a_now, b_now = np.polyfit(us, widths, 1)
        except Exception:
            return False
        if not (np.isfinite(a_now) and np.isfinite(b_now)) or abs(a_now) < 1e-6:
            return False
        u0 = -b_now / a_now                     # tempat lebar jalur menjadi nol
        # cakrawala kamera dashcam wajar di 0.25-0.80 tinggi gambar. Nilai di luar
        # itu berarti model lebarnya bohong -> tolak daripada dipakai.
        if a_now <= 0 or not (0.25 <= u0 <= 0.80):
            return False
        if self.lane_w_a is None:
            self.lane_w_a, self.lane_w_b = float(a_now), float(b_now)
            self.lane_w_samples = 1
            self.lane_w_mirror = False
        else:
            alpha = 0.12                        # lebar jalur sangat halus
            self.lane_w_a = (1 - alpha) * self.lane_w_a + alpha * float(a_now)
            self.lane_w_b = (1 - alpha) * self.lane_w_b + alpha * float(b_now)
            self.lane_w_samples += 1
        return True

    def _extend_short_sides(self, h: int, w: int) -> None:
        """Lengkapi bagian bawah kurva yang tidak teramati (terhalang benda).

        Kalau marka kanan hanya terlihat sampai u=0.74 (terhalang truk), kurva
        derajat 2 yang dipakai sampai u=1.0 melengkung ke arah yang salah —
        terukur 189 px meleset. Dua batas jalur di jalan nyata hampir sejajar
        (kelengkungan sama, hanya bergeser selebar jalur), jadi bagian yang
        hilang lebih jujur dibangun dari sisi yang terlihat + W(u).
        """
        # Perluasan HANYA boleh memakai lebar jalur yang benar-benar terukur.
        # Di footage asli lebar jalur sering masih asumsi cermin; memakainya untuk
        # memperpanjang kurva ke bawah membuat galat sisi kanan MEMBURUK
        # (terukur median 89 -> 193 px, samples/mentah/mentah_1001183916.mp4).
        if (self.lane_w_a is None or self.lane_w_b is None
                or self.lane_w_mirror or self.lane_w_samples < 3):
            return
        for side in ("L", "R"):
            fresh = (self.left_coeff is not None and not self.left_stale) if side == "L" \
                else (self.right_coeff is not None and not self.right_stale)
            if not fresh:
                continue
            obs = self.left_obs_u if side == "L" else self.right_obs_u
            other_obs = self.right_obs_u if side == "L" else self.left_obs_u
            other = self.right_coeff if side == "L" else self.left_coeff
            if obs is None or obs[1] >= 0.96 or other is None or other_obs is None:
                continue
            if other_obs[1] < obs[1] + 0.05:     # sisi lain juga tidak lebih baik
                continue
            base = self.left_coeff if side == "L" else self.right_coeff
            us = np.linspace(min(obs[0], other_obs[0]), 1.0, 40)
            own = np.polyval(base, us)
            shifted = np.polyval(other, us) + (self.lane_w_a * us + self.lane_w_b) * \
                (1.0 if side == "R" else -1.0)
            xs = np.where(us <= obs[1], own, shifted)
            deg = len(base) - 1
            try:
                new = np.asarray(np.polyfit(us, xs, deg), dtype=np.float64)
            except Exception:
                continue
            if not np.all(np.isfinite(new)) or not self._plausible(new, side, max_slope=6.0):
                continue
            if side == "L":
                self.left_coeff = new
                self.left_extended = True
            else:
                self.right_coeff = new
                self.right_extended = True

    def _learn_single_width(self, w: int) -> bool:
        """Perbarui SATU ukuran lebar jalur dari frame yang kedua sisinya terlihat.

        Dasarnya: lebar fisik jalur tidak berubah, hanya tampilan di layarnya yang
        berubah karena perspektif. Satu angka di baris acuan (u_ref=0.95, dekat
        bawah gambar) cukup, dan jauh lebih tahan banting daripada model dua
        parameter yang harus diekstrapolasi ke bawah.

        Syarat kedua sisi terlihat sampai u=0.96 ternyata TIDAK PERNAH terpenuhi
        di footage asli (terukur 0 dari 444 frame) — di jalan nyata salah satu
        marka hampir selalu tertutup kendaraan. Jadi acuan diambil dari irisan
        rentang teramati lalu dievaluasi di u_ref; kalau irisannya tidak sampai
        dekat acuan, nilainya ditolak.
        """
        if self.left_coeff is None or self.right_coeff is None:
            return False
        if self.left_obs_u is None or self.right_obs_u is None:
            return False
        lo = max(self.left_obs_u[0], self.right_obs_u[0])
        hi = min(self.left_obs_u[1], self.right_obs_u[1])
        u_ref = 0.95
        # Acuan tidak harus di dalam irisan: yang dipelajari adalah KEMIRINGAN
        # lebar terhadap u, lalu dievaluasi di u_ref. Di footage asli irisan
        # kedua sisi hampir tidak pernah sampai bawah (terukur 442 dari 444 frame
        # berhenti di atas u=0.83), jadi menuntut acuan di dalam irisan berarti
        # lebar tidak pernah terukur sama sekali.
        if hi - lo < 0.08:
            return False
        us = np.linspace(lo, hi, 24)
        widths = np.polyval(self.right_coeff, us) - np.polyval(self.left_coeff, us)
        widths = widths[np.isfinite(widths)]
        if len(widths) < 6 or np.min(widths) <= 0.05 * w:
            return False
        try:
            a_now, b_now = np.polyfit(us, widths, 1)
        except Exception:
            return False
        if not (np.isfinite(a_now) and np.isfinite(b_now)) or a_now <= 0:
            return False
        at_ref = float(a_now * u_ref + b_now)
        if not (0.08 * w < at_ref < 1.10 * w):
            return False
        # Haluskan kuat. Pengukuran per frame bergoyang ~91 px (std) karena salah
        # satu sisi sering hanya terlihat sebagian; lebar fisik jalur sendiri
        # hampir tidak berubah dalam 15 detik rekaman, jadi dipercaya rata-rata.
        if self.lane_w_m is None:
            self.lane_w_m = at_ref
        elif abs(at_ref - self.lane_w_m) > 0.35 * self.lane_w_m:
            return False            # lompatan >35% hampir pasti salah ukur
        else:
            alpha = 0.06
            self.lane_w_m = (1 - alpha) * self.lane_w_m + alpha * at_ref
        return True

    def _width_at(self, y: float) -> Optional[float]:
        """Lebar jalur terukur (px) pada ketinggian y, dari kedua kurva yang terdeteksi."""
        if self.left_coeff is None or self.right_coeff is None:
            return None
        return self._x(self.right_coeff, y) - self._x(self.left_coeff, y)

    def _estimate_other_side(self, h: int, from_left: bool) -> Optional[np.ndarray]:
        """Bangun kurva sisi yang hilang dari sisi yang terlihat + lebar jalur W(u).

        Lebar jalur bergantung pada u, jadi tidak cukup menggeser koefisien
        konstanta: kita evaluasi posisi di beberapa ketinggian, tambah/kurangi
        W(u), lalu fitting ulang agar bentuknya tetap halus.
        """
        base = self.left_coeff if from_left else self.right_coeff
        if base is None or self.lane_w_a is None or self.lane_w_b is None:
            return None
        deg = len(base) - 1
        us = np.linspace(1.0, 0.58, 28)
        # SATU ukuran lebar, sesuai permintaan: kalau satu garis tidak terdeteksi,
        # pakai lebar yang sudah terukur sebelumnya.
        # Tapi jangan dipakai sebagai angka datar: di layar jalur menyempit ke
        # arah cakrawala (perspektif). Jadi satu angka itu diskalakan linier
        # terhadap u dengan titik nol di cakrawala u_h — tetap satu parameter.
        # SATU ukuran lebar, sesuai permintaan: kalau satu garis tidak terdeteksi,
        # pakai lebar yang sudah terukur sebelumnya.
        width = self.lane_w_a * us + self.lane_w_b
        if self.lane_w_m is not None:
            # Satu angka itu dijadikan acuan: bentuk perspektif tetap, tapi
            # skalanya dinormalkan supaya lebar di baris acuan sama dengan angka
            # terukur. (Memakai angka datar di semua baris justru lebih buruk:
            # galat sisi kanan 98.7 px vs 88.3 px, karena jalur di layar menyempit.)
            u_ref = 0.95
            w_ref = float(self.lane_w_a * u_ref + self.lane_w_b)
            if w_ref > 1e-6:
                width = width * (float(self.lane_w_m) / w_ref)
        if not np.all(np.isfinite(width)):
            return None
        xs = np.polyval(base, us) + (width if from_left else -width)
        try:
            est = np.polyfit(us, xs, deg)
        except Exception:
            return None
        if not np.all(np.isfinite(est)):
            return None
        return np.asarray(est, dtype=np.float64)

    def center_offset(self, w: int, h: Optional[int] = None) -> Optional[float]:
        if self.left_draw is None or self.right_draw is None:
            return None
        y = float(h - 1) if h else 1.0
        xl = self._x(self.left_draw, y)
        xr = self._x(self.right_draw, y)
        return float((((xl + xr) / 2.0) - w / 2.0) / w * 100.0)

    # ---------- overlay ----------
    def turn_label(self, w: int, h: int) -> Optional[str]:
        coeffs = [c for c in (self.left_draw, self.right_draw) if c is not None]
        if not coeffs:
            return None
        y1, y2, y3 = h * 0.70, h * 0.85, h - 1
        vals = []
        for c in coeffs:
            x1, x2, x3 = self._x(c, y1), self._x(c, y2), self._x(c, y3)
            pred = x1 + (x2 - x1) * (y3 - y1) / max(1e-6, (y2 - y1))
            vals.append(abs(x3 - pred) / w * 100.0)
        s = max(vals)
        if s < 0.5:
            return "lurus"
        if s < 2.0:
            return "landai"
        if s < 3.5:
            return "sedang"
        return "tajam"

    @staticmethod
    def _quality_word(q: float) -> str:
        if q >= 0.7:
            return "kuat"
        if q >= 0.45:
            return "sedang"
        return "lemah"

    def draw(self, frame: np.ndarray, roi: np.ndarray, dbg_mask: Optional[np.ndarray] = None) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        if dbg_mask is not None:
            tint = np.zeros_like(out)
            tint[dbg_mask > 0] = (60, 220, 255)
            out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)

        ys = np.linspace(h - 1, int(h * 0.60), 40)

        def curve(coeff):
            xs = np.array([self._x(coeff, y) for y in ys])
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            return pts[(pts[:, 0] >= -w) & (pts[:, 0] <= 2 * w)]

        lpts = curve(self.left_draw) if self.left_draw is not None else None
        rpts = curve(self.right_draw) if self.right_draw is not None else None

        if lpts is not None and rpts is not None and len(lpts) > 2 and len(rpts) > 2:
            poly = np.vstack([lpts, rpts[::-1]]).astype(np.int32)
            shade = out.copy()
            alpha = 0.10 if (self.left_estimated or self.right_estimated) else 0.20
            cv2.fillPoly(shade, [poly], (0, 190, 60))
            out = cv2.addWeighted(shade, alpha, out, 1 - alpha, 0)

        def draw_side(pts, estimated):
            if pts is None or len(pts) < 2:
                return
            if not estimated:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 8, cv2.LINE_AA)
                return
            color = (0, 150, 255)
            for i in range(0, len(pts) - 4, 8):
                seg = pts[i:i + 5]
                cv2.polylines(out, [seg.reshape(-1, 1, 2)], False, color, 5, cv2.LINE_AA)

        draw_side(lpts, self.left_estimated)
        draw_side(rpts, self.right_estimated)

        ld = self.left_draw is not None and not self.left_estimated
        rd = self.right_draw is not None and not self.right_estimated
        le = self.left_draw is not None and self.left_estimated
        re_ = self.right_draw is not None and self.right_estimated

        if ld and rd:
            text, color = "Jalur terdeteksi (kiri+kanan)", (0, 220, 0)
        elif (ld and re_) or (rd and le):
            side = "kanan" if re_ else "kiri"
            text, color = f"Jalur terdeteksi ({side} estimasi)", (0, 200, 255)
        elif ld or rd:
            side = "kiri" if ld else "kanan"
            text, color = f"Sebagian jalur (hanya {side})", (0, 170, 255)
        else:
            if self.left_stale or self.right_stale:
                text, color = "Marka tidak terlihat saat ini", (160, 160, 160)
            else:
                text, color = "Jalur tidak terdeteksi", (60, 60, 230)

        cv2.rectangle(out, (0, 0), (w, 52), (20, 24, 32), -1)
        cv2.putText(out, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

        off = self.center_offset(w, h)
        turn = self.turn_label(w, h)
        parts = []
        if off is not None:
            side = "kiri" if off < 0 else "kanan"
            suffix = " (est)" if (self.left_estimated or self.right_estimated) else ""
            parts.append(f"offset {abs(off):.1f}% ke {side}{suffix}")
        if turn:
            parts.append(f"tikungan: {turn}")
        q = max(self.left_quality if ld else 0.0, self.right_quality if rd else 0.0)
        if ld or rd:
            parts.append(f"kualitas: {self._quality_word(q)}")
        if parts:
            cv2.putText(
                out,
                "  ·  ".join(parts),
                (14, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
        return out


# ---------- util ----------
def _band_points(mask: np.ndarray, coeff: np.ndarray, band: int, h: int) -> np.ndarray:
    ys, xs = mask.nonzero()
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pred = np.polyval(coeff, ys / float(h))
    keep = np.abs(xs - pred) <= band
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)
