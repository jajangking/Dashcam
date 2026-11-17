import numpy as np

from lane_v2 import LaneDetector
from road_area import is_dynamic, road_area_mask


def test_tanpa_kurva_pakai_roi_statis():
    det = LaneDetector()
    det.h, det.w = 400, 800
    mask = road_area_mask(det, det.h, det.w)

    assert not is_dynamic(det)
    assert mask.shape == (400, 800)
    assert mask[int(det.h * 0.9), det.w // 2] > 0


def test_kurva_terpercaya_membentuk_koridor():
    det = LaneDetector()
    det.h, det.w = 400, 800
    det.left_coeff = np.array([200.0, 0.0])
    det.right_coeff = np.array([600.0, 0.0])
    det.left_quality = det.right_quality = 0.8

    mask = road_area_mask(det, det.h, det.w)

    assert is_dynamic(det)
    assert mask[390, 400] > 0        # tengah jalan, dekat bawah
    assert mask[390, 20] == 0        # jauh di luar koridor kiri
    assert mask[390, 780] == 0       # jauh di luar koridor kanan
