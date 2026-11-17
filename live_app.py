#!/usr/bin/env python3
"""
Mode LIVE: deteksi jalur real-time dari kamera (webcam HP / USB / dashcam).

Cara kerja:
  Browser mengambil frame dari kamera (getUserMedia), mengirim JPEG kecil ke
  /live/frame, server mengembalikan koordinat kurva + angka TANPA mengirim balik
  gambar (jauh lebih hemat kuota daripada streaming video).

Catatan penting:
  - getUserMedia HANYA jalan di secure context (https:// atau http://localhost).
    Membuka http://<ip-hp>:5001 dari HP lain TIDAK akan memunculkan izin kamera.
  - Kalau kamera gagal, halaman tetap berguna: ada "Mode file" untuk menguji
    frame gambar/video lewat pipeline yang sama.

Pakai:
  PORT=5001 python -u live_app.py
"""

import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from lane_detect import process_video
from lane_v2 import LaneDetector

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FRAME_BYTES = 4 * 1024 * 1024      # frame kamera: kecil saja, tolak yang aneh
TARGET_WIDTH = 640                     # sisi kamera di-resize ke lebar ini

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

# ---------- sesi deteksi ----------
# Satu LaneDetector per klien: detektor menyimpan state antar-frame
# (koefisien sebelumnya, kualitas, lebar jalur). Kalau dipakai bersama,
# dua orang yang membuka halaman akan saling merusak smooth-nya.
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
SESSION_TTL = 15 * 60          # detik tanpa frame -> sesi dibuang
MAX_SESSIONS = 32              # cegah kebocoran memori


def _prune_sessions():
    now = time.time()
    with SESSIONS_LOCK:
        for sid, sess in list(SESSIONS.items()):
            if now - sess["last_seen"] > SESSION_TTL:
                SESSIONS.pop(sid, None)
        if len(SESSIONS) > MAX_SESSIONS:
            oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1]["last_seen"])
            for sid, _ in oldest[: len(SESSIONS) - MAX_SESSIONS]:
                SESSIONS.pop(sid, None)


def _get_session(sid):
    with SESSIONS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            # Kalau penuh, usir sesi yang paling lama tidak dipakai — bukan
            # menolak pengguna baru. Sesi lama biasanya sudah mati (tab ditutup)
            # tapi belum lewat TTL.
            if len(SESSIONS) >= MAX_SESSIONS:
                oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1]["last_seen"])
                for old_sid, _ in oldest[: len(SESSIONS) - MAX_SESSIONS + 1]:
                    SESSIONS.pop(old_sid, None)
            sess = {"det": LaneDetector(), "last_seen": time.time(), "frames": 0}
            SESSIONS[sid] = sess
        sess["last_seen"] = time.time()
        sess["frames"] += 1
        return sess


def _decode_frame(raw):
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > TARGET_WIDTH * 1.5:            # klien nakal / resolusi besar
        scale = TARGET_WIDTH / float(w)
        frame = cv2.resize(frame, (TARGET_WIDTH, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def _curve_points(coeff, h, w, steps=32):
    if coeff is None:
        return None
    ys = np.linspace(h - 1, h * 0.60, steps)
    xs = [float(np.polyval(coeff, y / float(h))) for y in ys]
    pts = [[round(x, 1), round(float(y), 1)] for x, y in zip(xs, ys) if -0.2 * w <= x <= 1.2 * w]
    return pts or None


@app.route("/")
def index():
    return render_template(
        "live.html",
        target_width=TARGET_WIDTH,
        has_ffmpeg=bool(__import__("shutil").which("ffmpeg")),
    )


# ---------- endpoint kamera ----------
@app.route("/live/frame", methods=["POST"])
def live_frame():
    _prune_sessions()
    sid = (request.form.get("sid") or request.headers.get("X-Session-Id") or "").strip()
    if not sid or len(sid) > 64:
        sid = uuid.uuid4().hex
    if "frame" not in request.files:
        return jsonify({"error": "field 'frame' tidak ada"}), 400
    raw = request.files["frame"].read(MAX_FRAME_BYTES + 1)
    if not raw:
        return jsonify({"error": "frame kosong"}), 400
    if len(raw) > MAX_FRAME_BYTES:
        return jsonify({"error": "frame terlalu besar"}), 413

    t0 = time.perf_counter()
    frame = _decode_frame(raw)
    if frame is None:
        return jsonify({"error": "frame tidak bisa dibaca sebagai gambar"}), 400

    sess = _get_session(sid)

    det = sess["det"]
    try:
        _, info = det.process(frame)
    except Exception as e:                      # jangan biarkan 1 frame jelek mematikan live
        return jsonify({"error": f"gagal memproses frame: {e}"}), 500

    h, w = frame.shape[:2]
    payload = {
        "sid": sid,
        "w": w,
        "h": h,
        "left": _curve_points(det.left_draw, h, w),
        "right": _curve_points(det.right_draw, h, w),
        "left_estimated": bool(info["left_estimated"]),
        "right_estimated": bool(info["right_estimated"]),
        "status": _status_text(info),
        "title": _title_text(info),
        "offset_pct": info["offset_pct"],
        "turn": info["turn"],
        "left_quality": info["left_quality"],
        "right_quality": info["right_quality"],
        "left_span_pct": info["left_span_pct"],
        "right_span_pct": info["right_span_pct"],
        "server_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }
    resp = jsonify(payload)
    if not (request.form.get("sid") or request.headers.get("X-Session-Id")):
        resp.headers["X-Session-Id"] = sid
    return resp


def _title_text(info):
    ld, rd = info["left"], info["right"]
    le, re_ = info["left_estimated"], info["right_estimated"]
    if ld and rd and not le and not re_:
        return "Jalur terdeteksi (kiri+kanan)"
    if (ld and re_) or (rd and le):
        return "Jalur terdeteksi (satu sisi estimasi)"
    if ld or rd:
        return "Sebagian jalur"
    if info["left_stale"] or info["right_stale"]:
        return "Marka tidak terlihat"
    return "Jalur tidak terdeteksi"


def _status_text(info):
    parts = []
    if info["offset_pct"] is not None:
        side = "kiri" if info["offset_pct"] < 0 else "kanan"
        parts.append(f"offset {abs(info['offset_pct']):.1f}% ke {side}")
    if info["turn"]:
        parts.append(f"tikungan {info['turn']}")
    return " · ".join(parts)


# ---------- mode file (cadangan kalau kamera diblokir) ----------
JOBS = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id, **kw):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(kw)


def _run_job(job_id, in_path, out_path):
    try:
        _set_job(job_id, status="running", started_at=time.time())
        stats = {}
        process_video(
            str(in_path), str(out_path),
            progress_cb=lambda f, t: _set_job(
                job_id, frames=f, total=t,
                percent=(100.0 * f / t) if t else 0.0),
            status_cb=lambda m: _set_job(job_id, stage=m),
            stats_out=stats,
        )
        _set_job(job_id, status="done", percent=100.0, stats=stats,
                 result=Path(out_path).name, finished_at=time.time())
    except Exception as e:
        _set_job(job_id, status="error", error=str(e), finished_at=time.time())


@app.route("/live/file", methods=["POST"])
def live_file():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file tidak dipilih"}), 400
    ext = Path(f.filename).suffix.lower()          # ambil dari nama ASLI (nama non-Latin tetap jalan)
    if ext not in ALLOWED_IMAGE | ALLOWED_VIDEO:
        return jsonify({"error": f"format tidak didukung: {ext or '(tanpa ekstensi)'}"}), 400
    stem = secure_filename(Path(f.filename).stem)[:40] or "berkas"
    job_id = uuid.uuid4().hex[:12]
    in_path = UPLOAD_DIR / f"{job_id}_{stem}{ext}"

    if ext in ALLOWED_IMAGE:
        f.save(in_path)
        frame = cv2.imread(str(in_path))
        if frame is None:
            in_path.unlink(missing_ok=True)
            return jsonify({"error": "gambar tidak bisa dibaca"}), 400
        det = LaneDetector()
        _, info = det.process(frame)
        h, w = frame.shape[:2]
        in_path.unlink(missing_ok=True)
        return jsonify({
            "kind": "image", "sid": job_id, "w": w, "h": h,
            "left": _curve_points(det.left_draw, h, w),
            "right": _curve_points(det.right_draw, h, w),
            "left_estimated": bool(info["left_estimated"]),
            "right_estimated": bool(info["right_estimated"]),
            "title": _title_text(info), "status": _status_text(info),
        })

    f.save(in_path)
    out_path = RESULT_DIR / f"{job_id}_result.mp4"
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "percent": 0.0, "frames": 0, "total": 0,
                        "stage": "menunggu", "error": None, "result": None,
                        "in_path": str(in_path), "out_path": str(out_path)}
    threading.Thread(target=_run_job, args=(job_id, in_path, out_path), daemon=True).start()
    return jsonify({"kind": "video", "job": job_id})


@app.route("/live/job/<job_id>")
def live_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        data = dict(job) if job else None
    if data is None:
        return jsonify({"error": "job tidak ditemukan"}), 404
    data.pop("in_path", None)
    data.pop("out_path", None)
    return jsonify(data)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


@app.errorhandler(404)
def _404(_):
    return jsonify({"error": "tidak ditemukan"}), 404


@app.errorhandler(413)
def _413(_):
    return jsonify({"error": "file terlalu besar (maks 300 MB)"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
