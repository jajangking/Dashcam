#!/usr/bin/env python3
"""
Geometri jalan: dari piksel ke meter.

Kenapa modul ini ada:
  Deteksi marka saja hanya memberi tahu "di mana garis di layar". Untuk menjawab
  "berapa jarak ke depan", "seberapa jauh saya dari tengah jalur", atau "saya baru
  pindah jalur", kita butuh skala metrik.

Kunci modelnya sederhana dan sudah terbukti di domain ini:
  Lebar jalur dalam piksel tumbuh LINIER terhadap baris gambar (kamera pinhole,
  jalan datar). Detektor sudah mempelajarinya sebagai  W(u) = a*u + b  dengan
  u = y/h. Dari situ cakrawala ada di  y_h = -b*h/a, dan

      W_px(y) = a*(y/h) + b

  Karena lebar jalur nyata diketahui (W_m, mis. 3.5 m), skala di baris y adalah
      s(y) = W_m / W_px(y)          [meter per piksel]

  dan jarak ke titik di baris y (kamera pinhole, tinggi H, fokus f_y):
      Z(y) = f_y * H / (y - y_h)

  f_y didapat dari sudut pandang vertikal kamera:  f_y = (h/2) / tan(vfov/2)

Batasan yang harus jujur disebut:
  - jalan dianggap DATAR (bukit/turunan membuat error besar),
  - kamera dianggap tidak miring (pitch=0) relatif sumbu kendaraan,
  - piksel blur di ketinggian >60% gambar (dekat cakrawala) tidak dihitung.

Semua angka di bawah adalah ESTIMASI, bukan pengukuran sensor. Tanpa kecepatan
dari GPS/OBD, jarak tempuh dihitung dari perubahan skala antar frame sehingga
error-nya menumpuk (lihat `Trajectory`).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CameraParams:
    """Parameter kamera. Nilai default = dashcam tipikal (HP di kaca depan)."""

    lane_width_m: float = 3.5        # lebar satu lajur (Indonesia: 3.0-3.75)
    height_m: float = 1.30           # tinggi kamera dari permukaan jalan
    vfov_deg: float = 55.0           # sudut pandang vertikal kamera
    # baris di atas fraksi ini diabaikan: terlalu dekat cakrawala, 1 piksel = puluhan meter
    max_row_frac: float = 0.95

    def focal_px(self, frame_h: int) -> float:
        return (frame_h / 2.0) / np.tan(np.deg2rad(self.vfov_deg) / 2.0)


class RoadModel:
    """Konversi piksel <-> meter untuk satu frame, memakai model lebar jalur detektor."""

    def __init__(self, width_a: float, width_b: float, frame_h: int, frame_w: int,
                 cam: Optional[CameraParams] = None):
        self.a = float(width_a)
        self.b = float(width_b)
        self.h = int(frame_h)
        self.w = int(frame_w)
        self.cam = cam or CameraParams()

    # ---------- validitas ----------
    def is_valid(self, min_row_frac: float = 0.60) -> bool:
        """Model lebar harus masuk akal: menyempit ke atas dan cakrawala di dalam gambar."""
        if not np.isfinite(self.a) or not np.isfinite(self.b):
            return False
        if self.a <= 0:                      # lebar harus bertambah ke bawah
            return False
        y_h = self.horizon_row()
        if not (0.0 < y_h < self.h * min_row_frac):
            return False
        if self.width_px(self.h - 1) <= 0:
            return False
        return True

    def horizon_row(self) -> float:
        """Baris cakrawala (tempat lebar jalur = 0)."""
        return -self.b * self.h / self.a

    def width_px(self, y: float) -> float:
        return self.a * (float(y) / self.h) + self.b

    # ---------- skala & jarak ----------
    def metres_per_px(self, y: float) -> float:
        wp = self.width_px(y)
        if wp <= 1e-6:
            return float("inf")
        return self.cam.lane_width_m / wp

    def distance_m(self, y: float) -> float:
        """Jarak (meter) ke titik jalan yang tergambar di baris y."""
        y_h = self.horizon_row()
        dy = float(y) - y_h
        if dy <= 1e-6:
            return float("inf")
        f_y = self.cam.focal_px(self.h)
        return f_y * self.cam.height_m / dy

    def row_at_distance(self, z_m: float) -> Optional[float]:
        """Baris gambar yang kira-kira berjarak z_m di depan (None kalau di luar gambar)."""
        if z_m <= 0:
            return None
        f_y = self.cam.focal_px(self.h)
        y = self.horizon_row() + f_y * self.cam.height_m / z_m
        if not (self.horizon_row() < y < self.h):
            return None
        return y

    # ---------- posisi lateral ----------
    def lateral_offset_m(self, left_x: float, right_x: float, y: float) -> float:
        """Posisi tengah kendaraan terhadap tengah jalur, dalam meter (positif = ke kanan)."""
        centre_lane = (left_x + right_x) / 2.0
        centre_img = self.w / 2.0
        return (centre_img - centre_lane) * self.metres_per_px(y)

    def lane_centre_px(self, left_x: float, right_x: float) -> float:
        return (left_x + right_x) / 2.0

    # ---------- jalur ke depan ----------
    def lane_path_m(self, left_at, right_at, distances=(4, 6, 9, 13, 18, 25)):
        """Titik tengah jalur ke depan dalam koordinat (maju_m, geser_m).

        left_at/right_at: fungsi y -> x (biasanya det._x(coeff, y)).
        """
        pts = []
        for z in distances:
            y = self.row_at_distance(z)
            if y is None:
                continue
            off = self.lateral_offset_m(left_at(y), right_at(y), y)
            pts.append((float(z), float(off)))
        return pts


def centre_path_px(model: RoadModel, left_at, right_at,
                   distances=(4, 6, 9, 13, 18, 25)):
    """Titik-titik tengah jalur ke depan, dalam koordinat layar (x, y).

    Ini yang digambar sebagai "path" di UI. Titik tengah jalur pada baris y
    adalah (x_kiri(y) + x_kanan(y)) / 2 — BUKAN sumbu tengah gambar ditambah
    offset. Offset terhadap sumbu kamera hanya berarti kalau kita ingin tahu
    posisi kendaraan, bukan posisi jalur.
    """
    pts = []
    for z in distances:
        y = model.row_at_distance(z)
        if y is None:
            continue
        xl, xr = left_at(y), right_at(y)
        if not (np.isfinite(xl) and np.isfinite(xr)):
            continue
        pts.append((float((xl + xr) / 2.0), float(y)))
    return pts


@dataclass
class LaneDrift:
    """Deteksi "drift bersama": kedua marka bergeser bersama padahal jalannya lurus.

    Terukur pada video tol contoh: korelasi pergeseran garis kiri vs kanan 0.994,
    keduanya bergeser ~59 px, sementara jarak antar keduanya (lebar jalur) hanya
    berubah 11 px (std 1.7 %). Itu tanda deteksi jendela ikut mengunci objek yang
    bergeser di layar (mis. bayangan/tepi jalan), bukan jalur yang benar-benar
    bergeser ke samping.

    Akibatnya kalau dipakai mentah: posisi lateral kendaraan dilaporkan
    mengembara sampai 2.3 m tanpa alasan.
    """

    window: int = 30
    trigger_px: float = 45.0        # geser bersama minimal (piksel, di baris bawah)
    max_width_change: float = 0.06  # lebar jalur boleh berubah 6% (relatif)

    _centres: List[float] = field(default_factory=list)
    _widths: List[float] = field(default_factory=list)

    def update(self, centre_x: float, width_px: float) -> bool:
        if np.isfinite(centre_x) and np.isfinite(width_px) and width_px > 1:
            self._centres.append(float(centre_x))
            self._widths.append(float(width_px))
            if len(self._centres) > self.window:
                self._centres.pop(0)
                self._widths.pop(0)
        if len(self._centres) < 10:
            return False
        shift = float(np.max(self._centres) - np.min(self._centres))
        width_change = float(np.max(self._widths) - np.min(self._widths)) / float(np.mean(self._widths))
        return shift >= self.trigger_px and width_change <= self.max_width_change

    @property
    def shift_px(self) -> float:
        if len(self._centres) < 2:
            return 0.0
        return float(self._centres[-1] - self._centres[0])

    def reset(self):
        self._centres.clear()
        self._widths.clear()


class Smoother:
    """Penstabil angka antar frame (EMA), supaya path & offset tidak melompat.

    Detektor sudah menghaluskan kurva, tapi besaran turunan (titik tengah jalur,
    jarak lateral) tetap bergetar karena satu-dua piksel galat di dekat cakrawala
    bisa berarti puluhan sentimeter. Terukur pada video contoh: simpangan baku
    titik path 4 m mencapai 40 px dan offset 0.72 m tanpa penstabil ini.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = float(alpha)
        self._path: List[List[float]] = []
        self._offset: Optional[float] = None
        self._width: Optional[float] = None

    def path(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not pts:
            return []
        if len(self._path) != len(pts):
            self._path = [list(p) for p in pts]
            return list(pts)
        out = []
        a = self.alpha
        for (x, y), prev in zip(pts, self._path):
            nx = prev[0] + a * (x - prev[0])
            ny = prev[1] + a * (y - prev[1])
            prev[0], prev[1] = nx, ny
            out.append((nx, ny))
        return out

    def offset(self, value: float) -> float:
        if self._offset is None or not np.isfinite(self._offset):
            self._offset = float(value)
        else:
            self._offset += self.alpha * (float(value) - self._offset)
        return self._offset

    def width(self, value: float) -> float:
        if self._width is None or not np.isfinite(self._width):
            self._width = float(value)
        else:
            self._width += self.alpha * (float(value) - self._width)
        return self._width

    def reset(self):
        self._path = []
        self._offset = None
        self._width = None


@dataclass
class Trajectory:
    """Jejak kendaraan yang diakumulasi dari geometri marka (tanpa GPS/OBD).

    Metodenya:
      - geser lateral  : selisih offset terhadap tengah jalur, pada jarak acuan tetap
      - maju           : dari perubahan skala (lebar jalur di baris acuan membesar
                         berarti kita maju). Ini setara monocular visual odometry
                         sederhana, jadi error menumpuk — ditandai di `drift_warning`.
    """

    reference_distance_m: float = 10.0
    smooth_alpha: float = 0.25
    points: List[Tuple[float, float]] = field(default_factory=list)  # (maju, geser)
    _prev_offset: Optional[float] = None
    _prev_width: Optional[float] = None
    _travelled: float = 0.0
    _smoother: Smoother = field(default_factory=Smoother)
    _shift: float = 0.0

    def reset(self):
        self.points.clear()
        self._prev_offset = None
        self._prev_width = None
        self._travelled = 0.0
        self._smoother = Smoother(self.smooth_alpha)
        self._shift = 0.0

    @property
    def travelled_m(self) -> float:
        return self._travelled

    @property
    def drift_warning(self) -> bool:
        # setelah ~50 m, error akumulatif mulai tidak bisa diabaikan
        return self._travelled > 50.0

    def update(self, model: RoadModel, left_at, right_at) -> Optional[Tuple[float, float]]:
        """Tambah satu sampel dari frame sekarang. Mengembalikan titik terbaru."""
        y = model.row_at_distance(self.reference_distance_m)
        if y is None:
            return None
        offset = self._smoother.offset(model.lateral_offset_m(left_at(y), right_at(y), y))
        width = self._smoother.width(model.width_px(model.h - 1))
        if not np.isfinite(offset) or not np.isfinite(width) or width <= 1e-6:
            return None

        maju = 0.0
        if self._prev_offset is None:
            geser = 0.0
        else:
            geser = offset - self._prev_offset
        if self._prev_width is not None and self._prev_width > 1e-6:
            # lebar membesar -> kita mendekat; jari-jari skala -> jarak tempuh
            z_ref = self.reference_distance_m
            ratio = self._prev_width / width
            maju = max(0.0, z_ref * (1.0 - ratio))
            # batasi lompatan yang tidak wajar (frame rusak / deteksi liar)
            maju = min(maju, 15.0)
            geser = float(np.clip(geser, -3.0, 3.0))

        self._prev_offset = offset
        self._prev_width = width
        self._travelled += maju

        self._shift += geser
        last = self.points[-1] if self.points else (0.0, 0.0)
        pt = (last[0] + maju, self._shift)
        self.points.append(pt)
        if len(self.points) > 600:
            self.points = self.points[-600:]
        return pt


def classify_lane_change(traj: Trajectory, window: int = 45,
                         min_shift_m: float = 1.5) -> Tuple[str, float]:
    """Tebak apakah sedang pindah jalur dari jejak terakhir.

    Konvensi tanda: `geser` positif = kendaraan bergeser ke KANAN relatif
    terhadap tengah jalur.

    Ambang 1.5 m dipakai karena lebar lajur ~3.5 m: perpindahan lajur penuh
    menggeser posisi sekitar 3.5 m, jadi 1.5 m sudah tidak mungkin goyangan
    kamera (terukur < 0.2 m pada uji sintetis).

    Sengaja TIDAK bergantung pada jarak tempuh: pindah lajur bisa terjadi saat
    kendaraan hampir berhenti, dan di banyak video uji kendaraan tidak maju.
    """
    pts = traj.points[-window:]
    if len(pts) < 8:
        return "tidak cukup data", 0.0

    ys = np.array([p[1] for p in pts], dtype=np.float64)
    # rata-rata 8 sampel terakhir di kedua ujung -> tahan terhadap satu frame liar
    k = min(8, len(ys) // 2)
    shift = float(np.mean(ys[-k:]) - np.mean(ys[:k]))
    # ayunan (bolak-balik) besar tapi nettonya kecil = goyangan, bukan pindah
    swing = float(np.max(ys) - np.min(ys))

    if abs(shift) >= min_shift_m:
        arah = "kanan" if shift > 0 else "kiri"
        return f"pindah jalur ke {arah}", shift
    if abs(shift) >= 0.5:
        arah = "kanan" if shift > 0 else "kiri"
        if swing > 2.0 * abs(shift):
            return "goyangan kamera", shift
        return f"bergeser ke {arah}", shift
    if swing > 0.6:
        return "goyangan kamera", shift
    return "di tengah jalur", shift
