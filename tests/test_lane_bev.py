from pathlib import Path

import cv2
import pytest

from lane_bev import LaneDetectorBEV


def test_api_compatibel_dengan_lane_v2():
    det = LaneDetectorBEV()
    det.h, det.w = 544, 864
    coeff = [100.0, 20.0]
    assert abs(det._x(coeff, 544) - 120.0) < 1e-6
    assert det._slope(coeff, 544) > 0


@pytest.mark.filterwarnings("ignore:Polyfit may be poorly conditioned")
def test_tidak_melintir_pada_30_frame_pertama():
    video = Path(__file__).resolve().parents[1] / "samples" / "mentah" / "mentah_1001183916.mp4"
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    det = LaneDetectorBEV()
    prev = None
    detected = 0
    for _ in range(30):
        ok, frame = cap.read()
        if not ok:
            break
        det.process(frame)
        h = frame.shape[0]
        cur = det._x(det.right_draw, h - 1) if det.right_draw is not None else None
        if cur is not None:
            detected += 1
        if prev is not None and cur is not None:
            assert abs(cur - prev) <= 15, f"loncatan {abs(cur - prev):.1f} px/frame"
        prev = cur
    cap.release()
    assert detected >= 10
