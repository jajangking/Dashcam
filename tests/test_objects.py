"""Test pengukuran objek.

Tidak ada model AI di sini (bobot tidak bisa diunduh di lingkungan dev), tapi
geometrinya bisa diuji dengan kotak buatan: objek yang menyentuh jalan di baris
yang lebih tinggi harus terbaca lebih jauh.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from lane_v2 import LaneDetector
from objects import (DetectedObject, measure, measure_all, nearest_in_path,
                     time_to_contact)
from road_geometry import RoadModel

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def model_and_size():
    img = cv2.imread(str(SAMPLES / "case_clear.jpg"))
    if img is None:
        pytest.skip("fixture tidak ada")
    h, w = img.shape[:2]
    det = LaneDetector()
    for _ in range(12):
        det.process(img.copy())
    if det.lane_w_a is None:
        pytest.skip("model lebar belum terbentuk")
    return RoadModel(det.lane_w_a, det.lane_w_b, h, w), h, w


def box_at_row(h, w, row, width=120, height=90):
    return (w / 2 - width / 2, row - height, w / 2 + width / 2, row)


def test_objek_lebih_atas_terbaca_lebih_jauh():
    model, h, w = model_and_size()
    dekat = DetectedObject("mobil", box_at_row(h, w, h - 5))
    jauh = DetectedObject("mobil", box_at_row(h, w, int(h * 0.70)))
    r_dekat = measure(dekat, model)
    r_jauh = measure(jauh, model)
    assert r_dekat.trustworthy and r_jauh.trustworthy
    assert r_dekat.distance_m < r_jauh.distance_m


def test_objek_di_tengah_jalur_lateralnya_kecil():
    model, h, w = model_and_size()
    r = measure(DetectedObject("mobil", box_at_row(h, w, h - 5)), model)
    assert abs(r.lateral_m) < 0.4


def test_objek_di_lajur_sebelah_lateralnya_besar():
    model, h, w = model_and_size()
    kotak = (w * 0.80, h - 100, w * 0.95, h - 5)
    r = measure(DetectedObject("motor", kotak), model)
    assert r.lateral_m is not None and r.lateral_m > 1.0


def test_objek_terlalu_dekat_cakrawala_tidak_dipercaya():
    model, h, w = model_and_size()
    r = measure(DetectedObject("mobil", box_at_row(h, w, int(h * 0.45))), model)
    assert not r.trustworthy
    assert r.distance_m is None


def test_urut_dari_yang_paling_dekat():
    model, h, w = model_and_size()
    objs = [
        DetectedObject("mobil", box_at_row(h, w, int(h * 0.72))),
        DetectedObject("truk", box_at_row(h, w, h - 4)),
        DetectedObject("orang", box_at_row(h, w, int(h * 0.85))),
    ]
    jarak = [r.distance_m for r in measure_all(objs, model) if r.distance_m is not None]
    assert jarak == sorted(jarak)


def test_nearest_in_path_mengabaikan_lajur_sebelah():
    model, h, w = model_and_size()
    kanan_jauh = DetectedObject("mobil", (w * 0.80, h - 120, w * 0.95, h - 60))
    tengah_dekat = DetectedObject("mobil", box_at_row(h, w, h - 5))
    readings = measure_all([kanan_jauh, tengah_dekat], model)
    best = nearest_in_path(readings)
    assert best is not None
    assert abs(best.lateral_m) < 1.6


def test_time_to_contact_tidak_menebak_kecepatan():
    assert time_to_contact(20.0, 0.0) is None      # kecepatan tidak diketahui
    assert time_to_contact(20.0, 10.0) == pytest.approx(2.0)
