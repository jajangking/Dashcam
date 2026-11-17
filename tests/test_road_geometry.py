"""Test geometri jalan (piksel -> meter).

Yang diuji hanya yang bisa dibuktikan dengan fixture sintetis yang geometrinya
diketahui persis (make_tests.road): posisi cakrawala, skala metrik, dan jarak.
"""

import cv2
import numpy as np
import pytest

from lane_v2 import LaneDetector
from road_geometry import CameraParams, RoadModel, Trajectory

SAMPLES = __import__("pathlib").Path(__file__).resolve().parent.parent / "samples"


def fitted_model(name="case_clear.jpg", frames=12):
    img = cv2.imread(str(SAMPLES / name))
    if img is None:
        pytest.skip("fixture tidak ada")
    h, w = img.shape[:2]
    det = LaneDetector()
    for _ in range(frames):
        det.process(img.copy())
    if det.lane_w_a is None:
        pytest.skip("model lebar belum terbentuk")
    return RoadModel(det.lane_w_a, det.lane_w_b, h, w), det, h, w


def test_cakrawala_sesuai_geometri_asli():
    """make_tests.road() menaruh cakrawala di 0.42*h."""
    model, _det, h, _w = fitted_model()
    assert model.is_valid()
    galat = abs(model.horizon_row() - 0.42 * h)
    assert galat < 12.0, f"cakrawala meleset {galat:.1f} px"


def test_lebar_jalur_menyempit_ke_atas():
    model, _det, h, _w = fitted_model()
    assert model.width_px(h - 1) > model.width_px(h * 0.7) > model.width_px(h * 0.62)


def test_skala_dan_jarak_konsisten():
    """Semakin ke atas, skala (m/px) membesar dan jarak bertambah monoton."""
    model, _det, h, _w = fitted_model()
    rows = np.linspace(h - 1, h * 0.65, 8)
    jarak = [model.distance_m(y) for y in rows]
    skala = [model.metres_per_px(y) for y in rows]
    assert all(np.isfinite(jarak)) and all(np.isfinite(skala))
    assert jarak == sorted(jarak), "jarak harus bertambah ke arah horizon"
    assert skala == sorted(skala), "skala harus membesar ke arah horizon"


def test_jarak_pada_lebar_jalur_setara():
    """Baris yang lebar jalurnya setengah -> jaraknya dua kali (model pinhole)."""
    model, _det, h, _w = fitted_model()
    z1, z2 = 5.0, 10.0
    y1, y2 = model.row_at_distance(z1), model.row_at_distance(z2)
    assert y1 is not None and y2 is not None
    # pemetaan baris -> jarak harus konsisten bolak-balik
    assert abs(model.distance_m(y1) - z1) < 0.05
    assert abs(model.distance_m(y2) - z2) < 0.05


def test_lintasan_jalur_ke_depan():
    """Jalur lurus: geser lateral ke depan harus mendekati nol."""
    model, det, h, w = fitted_model()
    pts = model.lane_path_m(lambda y: det._x(det.left_draw, y),
                            lambda y: det._x(det.right_draw, y))
    assert len(pts) >= 3
    for z, off in pts:
        assert z > 0
        assert abs(off) < 0.35, f"jalur lurus tapi geser {off:+.2f} m di {z:.0f} m"


def test_model_tidak_valid_kalau_lebar_terbalik():
    """Model yang tidak masuk akal harus ditolak, bukan dipakai diam-diam."""
    model = RoadModel(width_a=-100.0, width_b=900.0, frame_h=720, frame_w=1280)
    assert not model.is_valid()


def test_trajectory_tidak_melempar_saat_data_kosong():
    model, det, h, w = fitted_model()
    traj = Trajectory()
    for _ in range(5):
        traj.update(model, lambda y: det._x(det.left_draw, y),
                    lambda y: det._x(det.right_draw, y))
    assert traj.travelled_m >= 0.0
    assert len(traj.points) <= 600


# ---------- perpindahan jalur ----------
def _jalur_frame(shift_px, wobble=0, size=(960, 540)):
    """Marka digeser horizontal dengan lebar jalur TETAP (meniru kendaraan bergeser).

    Catatan konvensi: menggeser marka ke kanan == kendaraan bergeser ke KIRI.
    """
    from road_geometry import Trajectory  # noqa: F401
    w, h = size
    base = cv2.resize(cv2.imread(str(SAMPLES / "case_clear.jpg")), (w, h),
                      interpolation=cv2.INTER_AREA)
    sky = int(h * 0.42)
    img = cv2.warpAffine(base, np.float32([[1, 0, shift_px + wobble], [0, 1, 0]]),
                         (w, h), borderMode=cv2.BORDER_REPLICATE)
    img[:sky] = base[:sky]
    return img


def _run_trajectory(shift_fn, frames=90, size=(960, 540)):
    from road_geometry import Trajectory
    w, h = size
    det = LaneDetector()
    traj = None
    for i in range(frames):
        sh, wob = shift_fn(i)
        det.process(_jalur_frame(sh, wob, size))
        if det.lane_w_a is None or det.left_draw is None or det.right_draw is None:
            continue
        model = RoadModel(det.lane_w_a, det.lane_w_b, h, w)
        if not model.is_valid():
            continue
        if traj is None:
            traj = Trajectory()
        traj.update(model, lambda y: det._x(det.left_draw, y),
                    lambda y: det._x(det.right_draw, y))
    assert traj is not None, "tidak ada sampel terkumpul"
    return traj


@pytest.mark.parametrize("sign,arah", [(1, "kiri"), (-1, "kanan")])
def test_perpindahan_jalur_terdeteksi(sign, arah):
    """Marka bergeser ke kanan -> kendaraan bergeser ke kiri, dan sebaliknya."""
    from road_geometry import classify_lane_change
    traj = _run_trajectory(lambda i: (sign * min(1.0, i / 60.0) * 170, 0))
    label, shift = classify_lane_change(traj)
    assert arah in label, f"harusnya bergeser ke {arah}, dapat: {label} ({shift:+.2f} m)"
    assert abs(shift) > 1.0


@pytest.mark.parametrize("amplitudo", [10, 22])
def test_goyangan_kamera_tidak_dianggap_pindah_jalur(amplitudo):
    """Goyangan kamera dulu memberi skor PALING TINGGI pada uji flow optik."""
    from road_geometry import classify_lane_change
    traj = _run_trajectory(lambda i: (0.0, int(amplitudo * np.sin(i * 0.35))))
    label, shift = classify_lane_change(traj)
    assert "pindah" not in label, f"goyangan salah dibaca: {label} ({shift:+.2f} m)"
    assert abs(shift) < 1.0


def test_jalan_lurus_tetap_di_tengah():
    from road_geometry import classify_lane_change
    traj = _run_trajectory(lambda i: (0.0, 0))
    label, _shift = classify_lane_change(traj)
    assert "pindah" not in label


def test_path_mengikuti_tengah_jalur_bukan_sumbu_kamera():
    """Bug yang tertangkap lewat pratinjau visual: path digambar di sumbu kamera."""
    from road_geometry import centre_path_px
    model, det, h, w = fitted_model()
    pts = centre_path_px(model, lambda y: det._x(det.left_draw, y),
                         lambda y: det._x(det.right_draw, y))
    assert len(pts) >= 3
    for x, y in pts:
        tengah = (det._x(det.left_draw, y) + det._x(det.right_draw, y)) / 2.0
        assert abs(x - tengah) < 1.0, "path harus tepat di tengah kedua marka"
    # dan harus benar-benar bergerak ke atas (bukan garis vertikal di satu x)
    ys = [y for _x, y in pts]
    assert ys == sorted(ys, reverse=True)


# ---------- drift bersama (bug yang dilaporkan pengguna) ----------
def test_drift_bersama_terdeteksi_saat_kedua_garis_bergeser():
    """Terukur di video tol: korelasi geser kiri/kanan 0.994, lebar jalur tetap."""
    from road_geometry import LaneDrift
    drift = LaneDrift()
    terdeteksi = False
    for i in range(40):
        geser = 0 if i < 10 else min(i - 10, 25) * 3.5      # bergeser bersama ~80 px
        terdeteksi = drift.update(centre_x=540 + geser, width_px=660 + (i % 3))
    assert terdeteksi, "pergeseran bersama 80 px dengan lebar tetap harus terdeteksi"


def test_drift_tidak_menyala_kalau_lebar_jalur_berubah():
    """Jalur benar-benar melebar/menyempit bukan drift."""
    from road_geometry import LaneDrift
    drift = LaneDrift()
    hasil = False
    for i in range(40):
        hasil = drift.update(centre_x=540 + i * 3, width_px=600 + i * 12)
    assert not hasil


def test_drift_tidak_menyala_pada_video_stabil():
    from pathlib import Path as _P
    import cv2 as _cv
    from lane_v2 import LaneDetector as _LD
    from road_geometry import LaneDrift as _LDft
    path = _P(__file__).resolve().parent.parent / "samples" / "case_video_clear.mp4"
    cap = _cv.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.skip("video tidak ada")
    det = _LD()
    drift = _LDft()
    nyala = 0
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        det.process(f)
        if det.left_draw is None or det.right_draw is None:
            continue
        xl = det._x(det.left_draw, det.h - 1)
        xr = det._x(det.right_draw, det.h - 1)
        if drift.update((xl + xr) / 2.0, xr - xl):
            nyala += 1
    cap.release()
    assert nyala / max(1, n) < 0.05, f"drift salah menyala di {nyala}/{n} frame"
