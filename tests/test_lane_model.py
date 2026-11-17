"""Uji jalur model ONNX (UFLD/UFLDv2) — tanpa bobot asli.

Bobot asli tidak disertakan karena sandbox proyek ini tidak punya akses ke host
penyimpan bobot. Yang diuji: model uji berbentuk IDENTIK UFLD dipakai untuk
memastikan praproses, pembacaan bentuk, dan pasca-prosesnya benar. Tanpa uji ini,
modul seperti ini biasanya baru ketahuan salah saat pengguna memasang bobot asli.
"""

import os
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from lane_model import OnnxLaneDetector, find_model  # noqa: E402

G, L, R = 200, 4, 18


@pytest.fixture(scope="module")
def dummy_model(tmp_path_factory):
    from onnx import helper, TensorProto, numpy_helper
    d = tmp_path_factory.mktemp("model")
    pw = numpy_helper.from_array(
        np.random.RandomState(0).randn(800, G * L * R).astype(np.float32) * 0.01, "pw")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 288, 800])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, G * L, R])
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["m"], axes=[1, 2], keepdims=0),
        helper.make_node("Flatten", ["m"], ["mf"], axis=0),
        helper.make_node("MatMul", ["mf", "pw"], ["proj"]),
        helper.make_node("Reshape", ["proj", "shape"], ["output"]),
    ]
    shape = numpy_helper.from_array(np.array([1, G * L, R], dtype=np.int64), "shape")
    graph = helper.make_graph(nodes, "uji", [inp], [out], [pw, shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = os.path.join(str(d), "uji.onnx")
    onnx.save(model, path)
    return path


def test_model_hilang_dilaporkan_jujur():
    det = OnnxLaneDetector("models/tidak_ada.onnx")
    assert not det.ready
    assert "tidak ditemukan" in det.info
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    assert det.detect(frame) is None          # tidak mengarang angka


def test_bentuk_keluaran_ufld_dibaca_benar(dummy_model):
    import cv2
    det = OnnxLaneDetector(dummy_model)
    assert det.ready
    frame = cv2.imread("samples/case_clear.jpg")
    if frame is None:
        pytest.skip("sampel tidak ada")
    lanes = det.detect(frame)
    assert lanes is not None and len(lanes) == 4
    # (jumlah baris anchor, 2) — bukan (800, 2) seperti bug pertama modul ini
    assert lanes[0].shape == (R, 2), f"bentuk salah: {lanes[0].shape}"
    h, w = frame.shape[:2]
    for lane in lanes:
        assert lane[:, 1].min() > 0.3 * h
        assert lane[:, 1].max() <= h
        assert lane[:, 0].min() >= 0 and lane[:, 0].max() <= w


def test_praproses_normalisasi(dummy_model):
    det = OnnxLaneDetector(dummy_model)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    x = det.preprocess(frame)
    assert x.shape == (1, 3, 288, 800)
    # piksel hitam setelah normalisasi ImageNet: (0 - 0.485)/0.229
    assert x[0, 0].mean() == pytest.approx(-0.485 / 0.229, abs=1e-3)


def test_find_model_kosong():
    assert find_model("folder_tidak_ada") is None
