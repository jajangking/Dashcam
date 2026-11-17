import numpy as np

from lane_v2 import LaneDetector
from road_cyan import CYAN, draw_points, road_points


def test_road_points_urutan_bawah_ke_atas():
    det = LaneDetector()
    det.h, det.w = 400, 800
    coeff = np.array([0.0, 120.0])       # x konstan 120 px
    pts = road_points(det, coeff, det.h, det.w)

    assert pts is not None
    assert len(pts) >= 2
    assert pts[0, 1] < pts[-1, 1]        # y naik: indeks -1 = bawah gambar
    assert pts[-1, 0] == 120
    assert abs(float(pts[-1, 1]) - (det.h - 1)) <= 1


def test_draw_points_memakai_cyan_bukan_kuning():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    line = np.stack([np.arange(20, 180), np.full(160, 100)], axis=1)

    out = draw_points(frame, line, None, label="ROAD")

    # core garis digambar paling akhir, jadi warnanya persis CYAN (BGR).
    assert tuple(int(v) for v in out[100, 90]) == CYAN
