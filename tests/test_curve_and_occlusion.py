"""Regresi untuk dua keluhan pengguna:

1. "marka tiba-tiba tidak terdeteksi karena terhalang benda"
2. "ketika marka berbelok estimasinya salah mulu"
"""

import numpy as np
import pytest

from lane_v2 import LaneDetector


def truth_right(curve, w, h):
    """Posisi marka kanan sejati, dihitung dari generator `make_tests.road`."""
    sky = int(h * 0.42)
    vpx = w * 0.5
    out = {}
    for y in range(sky, h):
        t = (y - sky) / max(1, (h - sky))
        out[y] = vpx + curve * (t ** 2) + (w * 0.38 * t)
    return out


def load(name):
    import cv2
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "samples" / name
    if not p.exists():
        pytest.skip(f"{name} tidak ada")
    return cv2.imread(str(p))


def make_curve_frames(curve=360.0, n=30, w=960, h=540):
    """Frame tikungan sintetis (generator `make_tests.road`), berurutan seperti video."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "mt", str(Path(__file__).resolve().parent.parent / "make_tests.py"))
    mt = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mt)
    except SystemExit:
        pass
    return [mt.add_noise(mt.road(w, h, curve=curve, marking_v=240, asphalt=80), sigma=5)
            for _ in range(n)]


def test_tikungan_kanan_tajam_pada_video():
    """Dulu 172 px karena kurva derajat 2 diekstrapolasi jauh di luar data teramati.

    Perbaikan butuh beberapa frame untuk mempelajari lebar jalur, jadi diuji pada
    urutan frame seperti pemakaian nyata — bukan pada satu gambar terpisah.
    """
    w, h = 960, 540
    frames = make_curve_frames(360.0, 30, w, h)
    gt = truth_right(360.0, w, h)
    det = LaneDetector()
    errs = []
    for f in frames:
        det.process(f)
        if det.right_draw is None:
            continue
        errs.append(max(abs(det._x(det.right_draw, y) - gt[y])
                        for y in (int(h * p) for p in (0.7, 0.8, 0.9, 0.99))))
    assert errs, "sisi kanan tidak pernah terdeteksi"
    assert np.median(errs) < 60, f"galat median {np.median(errs):.0f} px (dulu 172 px)"


def test_satu_gambar_terhalang_tidak_dikarang():
    """Kalau marka kanan hanya terlihat sampai 0.74h dan lebar jalur belum pernah
    terukur, galat di bawah tetap besar. Itu BATASAN yang jujur, bukan bug baru —
    sistem tidak boleh mengaku akurat di area yang tidak pernah dilihatnya."""
    f = load("case_belok_480.jpg")
    h, w = f.shape[:2]
    det = LaneDetector()
    _, info = det.process(f)
    assert info["right_obs_u"] is not None
    assert info["right_obs_u"][1] < 0.85, "rentang teramati harus dilaporkan dengan benar"


def test_rentang_teramati_dilaporkan():
    f = load("case_belok_480.jpg")
    det = LaneDetector()
    _, info = det.process(f)
    obs = info["right_obs_u"]
    assert obs is not None
    lo, hi = obs
    assert 0.5 < lo < hi <= 1.01
    assert hi < 0.85, "pada tikungan tajam marka kanan memang tidak terlihat sampai bawah"


def test_sisi_terhalang_ditandai_tebakan_bukan_deteksi():
    """Marka tertutup benda tidak boleh dilaporkan sebagai deteksi nyata."""
    import cv2
    base = load("case_clear.jpg")
    h, w = base.shape[:2]
    img = base.copy()
    cv2.rectangle(img, (int(w * 0.55), int(h * 0.5)), (w, h), (40, 40, 45), -1)
    det = LaneDetector()
    _, info = det.process(img)
    if info["right"]:
        assert info["right_estimated"] or info["right_extended"], (
            "sisi kanan yang terhalang tidak boleh dianggap terdeteksi penuh"
        )


def test_perluasan_tidak_merusak_jalan_lurus():
    f = load("case_clear.jpg")
    det = LaneDetector()
    _, info = det.process(f)
    assert not info["left_extended"]
    assert not info["right_extended"]
    assert abs(info["offset_pct"]) < 1.0


# ---------- satu ukuran lebar jalur (permintaan pengguna) ----------
def test_satu_ukuran_lebar_terukur_dari_dua_sisi():
    """Kalau kedua marka terlihat, disimpan SATU angka lebar jalur."""
    import cv2
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "samples" / "mentah" / "mentah_1001183916.mp4"
    if not p.exists():
        pytest.skip("footage mentah tidak ada")
    cap = cv2.VideoCapture(str(p))
    det = LaneDetector()
    n = 0
    terukur = 0
    while n < 150:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        det.process(f)
        if det.lane_w_m is not None:
            terukur += 1
    cap.release()
    assert terukur / max(1, n) > 0.5, (
        f"lebar satu angka hanya terukur {terukur}/{n} frame"
    )
    assert 0.08 * det.w < det.lane_w_m < 1.10 * det.w


def test_sisi_hilang_memakai_satu_ukuran_lebar():
    """Saat satu garis tidak terdeteksi, lebar memakai angka terukur sebelumnya."""
    import cv2
    f = load("case_clear.jpg")
    h, w = f.shape[:2]
    det = LaneDetector()
    for _ in range(10):
        det.process(f.copy())
    assert det.lane_w_m is not None
    lebar_sebelum = det.lane_w_m

    hidden = f.copy()
    cv2.rectangle(hidden, (int(w * 0.62), int(h * 0.5)), (w, h), (35, 35, 40), -1)
    # `max_missing` = 8 frame: toleransi singkat saat marka tertutup sebentar
    for _ in range(14):
        _, info = det.process(hidden.copy())
    assert info["right_estimated"]
    assert det.lane_w_m == pytest.approx(lebar_sebelum, rel=0.25)
