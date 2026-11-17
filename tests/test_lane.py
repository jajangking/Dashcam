"""Test untuk lane_v2.

Fokus pada dua bug yang paling sering muncul di pemakaian nyata:
  1. estimasi sisi yang hilang harus mengikuti perspektif (lebar jalur menyempit),
  2. gambar yang bukan jalan tidak boleh dilaporkan "jalur terdeteksi".
Ditambah regresi akurasi supaya perbaikan berikutnya tidak merusak yang sudah jalan.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_v2 import LaneDetector

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ---------- helper ----------
def load(name: str) -> np.ndarray:
    img = cv2.imread(str(SAMPLES / name))
    if img is None:
        pytest.skip(f"fixture {name} tidak ada")
    return img


def truth_edge(h: int, w: int, curve: float, lane_w: float = 0.38):
    """Geometri marka asli, rumusnya sama dengan make_tests.road()."""
    vp_x, vp_y = w * 0.5, int(h * 0.42)

    def edge(y: float, side: int) -> float:
        t = (y - vp_y) / max(1, (h - vp_y))
        return vp_x + curve * (t ** 2) + side * w * lane_w * t

    return edge


def hide_side(img: np.ndarray, keep: str) -> np.ndarray:
    """Hapus marka di satu sisi (dicat warna aspal) supaya sisi itu harus ditebak."""
    h, w = img.shape[:2]
    det = LaneDetector()
    mask = det.color_mask(img, det.roi_vertices(h, w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = img.copy()
    for i in range(1, n):
        x, _y, bw = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH]
        centre = x + bw / 2.0
        if (keep == "left" and centre > w * 0.5) or (keep == "right" and centre <= w * 0.5):
            out[labels == i] = (85, 85, 85)
    return out


# ---------- 1. estimasi mengikuti perspektif ----------
@pytest.mark.parametrize("name", ["case_curve_left", "case_curve_right", "case_clear", "case_night"])
def test_estimasi_sisi_hilang_mengikuti_perspektif(name):
    """Dulu galatnya sampai 754 px karena lebar jalur dianggap konstan."""
    img = load(name + ".jpg")
    h, _w = img.shape[:2]

    det = LaneDetector()
    for _ in range(12):                       # pelajari lebar jalur dari dua sisi
        det.process(img.copy())
    true_right = det.right_coeff.copy()

    hidden = hide_side(img, keep="left")
    info = None
    for _ in range(15):                       # marka kanan benar-benar hilang
        _, info = det.process(hidden.copy())

    assert info["right_estimated"], "sisi kanan seharusnya ditebak"
    galat = [
        abs(det._x(det.right_draw, y) - det._x(true_right, y))
        for y in np.linspace(h - 1, h * 0.62, 40)
    ]
    assert max(galat) < 4.0, f"estimasi meleset {max(galat):.1f} px (target < 4 px)"


def test_lebar_jalur_bukan_konstanta():
    """Model lebar harus menyempit ke arah horizon, bukan satu angka datar."""
    img = load("case_clear.jpg")
    h, _w = img.shape[:2]
    det = LaneDetector()
    for _ in range(12):
        det.process(img.copy())

    assert det.lane_w_a is not None and det.lane_w_b is not None
    lebar_bawah = det._width_at(h - 1)
    lebar_atas = det._width_at(h * 0.62)
    assert lebar_bawah > lebar_atas * 1.8, "lebar jalur harus menyempit jelas ke arah horizon"


# ---------- 2. penolakan gambar yang bukan jalan ----------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_noise_tidak_dilaporkan_sebagai_jalur(seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    _, info = LaneDetector().process(noise)
    assert not info["left"] and not info["right"], "noise dilaporkan punya jalur"
    assert info.get("rejected"), "alasan penolakan harus diisi supaya bisa ditampilkan"


def test_dinding_terang_tidak_dilaporkan_sebagai_jalur():
    _, info = LaneDetector().process(np.full((480, 640, 3), 235, np.uint8))
    assert not info["left"] and not info["right"]


# ---------- 3. regresi: fixture asli tidak boleh ikut tertolak ----------
@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in SAMPLES.glob("case_*.jpg") if p.stem != "case_nomark"),
)
def test_fixture_asli_tetap_terdeteksi(name):
    _, info = LaneDetector().process(load(name).copy())
    assert info["left"] or info["right"], f"{name} jadi tidak terdeteksi (regresi penjaga)"
    assert not info.get("rejected", ""), f"{name} salah ditolak: {info.get('rejected')}"


def test_tanpa_marka_tidak_terdeteksi():
    _, info = LaneDetector().process(load("case_nomark.jpg").copy())
    assert not info["left"] and not info["right"]


# ---------- 4. regresi akurasi ----------
@pytest.mark.parametrize(
    "name,curve",
    [
        ("case_clear", 0.0),
        ("case_dashed", 0.0),
        ("case_night", 0.0),
        ("case_faded", 0.0),
        ("case_curve_left", -60.0),
        ("case_curve_right", 60.0),
        ("case_belok_120", 120.0),
    ],
)
def test_akurasi_terhadap_geometri_asli(name, curve):
    img = load(name + ".jpg")
    h, w = img.shape[:2]
    det = LaneDetector()
    _, info = det.process(img.copy())
    assert info["left"] and info["right"]

    edge = truth_edge(h, w, curve)
    for side, coeff, sign in (("kiri", det.left_draw, -1), ("kanan", det.right_draw, +1)):
        galat = []
        for y in np.linspace(h * 0.72, h - 1, 60):
            x_asli = edge(y, sign)
            if 0 <= x_asli <= w:              # hanya baris yang markanya terlihat
                galat.append(det._x(coeff, y) - x_asli)
        rms = float(np.sqrt(np.mean(np.square(galat))))
        assert rms < 2.0, f"{name} sisi {side}: RMS {rms:.2f} px (target < 2 px)"
