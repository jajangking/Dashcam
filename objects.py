#!/usr/bin/env python3
"""
Objek di jalan (mobil, motor, orang) + estimasi jarak.

Modul ini SENGAJA tidak mengikat ke satu model AI. Bagian yang bisa diuji
dengan pasti adalah geometrinya: kalau sebuah objek menyentuh jalan di baris y,
jaraknya = distance_m(y) dari RoadModel (model pinhole, sudah divalidasi di
tests/test_road_geometry.py).

Model AI (YOLO, dsb.) tinggal mengisi daftar `DetectedObject`. Kontraknya kecil
supaya mudah diganti:

    class MyDetector:
        def detect(self, frame_bgr) -> list[DetectedObject]: ...

Kenapa tidak ada model AI di repo ini:
    Bobot YOLOv8 tidak bisa diunduh di lingkungan pengembangan kami
    (SSL certificate verify failed untuk host aset rilis). Menulis dekoder
    tanpa bisa mengujinya berisiko; jadi yang ada di sini hanya kontrak +
    geometri. Lihat AUDIT/diskusi untuk rencana integrasinya.

Batasan estimasi jarak dari satu kamera:
    - Akurat hanya kalau objek menyentuh jalan (kontak ban/kaki terlihat).
    - Objek yang terhalang, di tanjakan, atau bagian bawahnya terpotong gambar
      akan terbaca lebih jauh. Selalu tampilkan sebagai ESTIMASI.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from road_geometry import RoadModel

# Kelas yang relevan untuk dashcam (indeks mengikuti COCO untuk YOLO).
COCO_RELEVANT = {
    0: "orang",
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}


@dataclass
class DetectedObject:
    """Satu objek terdeteksi. Kotak dalam piksel: (x1, y1, x2, y2)."""

    label: str
    box: Tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def x1(self) -> float:
        return float(self.box[0])

    @property
    def y1(self) -> float:
        return float(self.box[1])

    @property
    def x2(self) -> float:
        return float(self.box[2])

    @property
    def y2(self) -> float:
        return float(self.box[3])

    @property
    def contact_point(self) -> Tuple[float, float]:
        """Titik kontak dengan jalan: tengah bawah kotak."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def width_px(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height_px(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class ObjectReading:
    """Objek + hasil pengukuran geometri."""

    obj: DetectedObject
    distance_m: Optional[float]
    lateral_m: Optional[float]
    trustworthy: bool


def measure(obj: DetectedObject, model: RoadModel,
            min_bottom_frac: float = 0.55) -> ObjectReading:
    """Ukur jarak & posisi lateral satu objek terhadap model jalan.

    `trustworthy` = False kalau titik kontak terlalu dekat cakrawala (piksel
    blur, 1 px bisa berarti puluhan meter) atau model jalan tidak valid.
    """
    _cx, cy = obj.contact_point
    if not model.is_valid():
        return ObjectReading(obj, None, None, False)
    if cy < model.h * min_bottom_frac:
        # kotak berakhir terlalu tinggi: kemungkinan objek jauh/menggantung,
        # jarak dari kontak ban tidak bisa dipercaya
        return ObjectReading(obj, None, None, False)

    jarak = model.distance_m(cy)
    if not np.isfinite(jarak):
        return ObjectReading(obj, None, None, False)
    lateral = (obj.contact_point[0] - model.w / 2.0) * model.metres_per_px(cy)
    return ObjectReading(obj, float(jarak), float(lateral), True)


def measure_all(objs: Sequence[DetectedObject], model: RoadModel) -> List[ObjectReading]:
    """Ukur semua objek, urut dari yang paling dekat."""
    out = [measure(o, model) for o in objs]
    out.sort(key=lambda r: (r.distance_m is None, r.distance_m or 0.0))
    return out


def nearest_in_path(readings: Sequence[ObjectReading], max_lateral_m: float = 1.6,
                    horizon_m: float = 80.0) -> Optional[ObjectReading]:
    """Objek terdekat yang berada di jalur kita (bukan di lajur sebelah).

    1.6 m ~ setengah lebar lajur; di luar itu kemungkinan kendaraan di lajur
    sebelah, jadi tidak perlu diwaspadai sebagai "di depan kita".
    """
    best = None
    for r in readings:
        if not r.trustworthy or r.distance_m is None or r.lateral_m is None:
            continue
        if r.distance_m > horizon_m or abs(r.lateral_m) > max_lateral_m:
            continue
        if best is None or r.distance_m < best.distance_m:
            best = r
    return best


def time_to_contact(distance_m: float, speed_mps: float) -> Optional[float]:
    """Perkiraan waktu sampai menyentuh objek (detik), kalau kecepatan diketahui.

    Kecepatan HARUS datang dari sumber nyata (GPS/OBD). Tanpa itu modul ini
    tidak menebak: menebak kecepatan lalu menampilkan "tabrak dalam 2 detik"
    jauh lebih berbahaya daripada tidak menampilkan apa pun.
    """
    if speed_mps <= 0.2 or distance_m <= 0:
        return None
    return float(distance_m / speed_mps)
