#!/usr/bin/env python3
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from lane_detect import detect_lanes, draw_overlay, process_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB

JOBS = {}
JOBS_LOCK = threading.Lock()

INDEX_HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prototipe Deteksi Jalan</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .card { max-width: 900px; margin: 0 auto; background: #111827; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,.3); }
    h1 { margin-top: 0; font-size: 1.5rem; }
    .row { margin: 12px 0; }
    input[type=file] { width: 100%; padding: 10px; background: #0b1220; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; }
    button { background: #2563eb; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 1rem; }
    button:hover { background: #1d4ed8; }
    .muted { color: #94a3b8; font-size: .95rem; }
    .error { color: #fecaca; background: #7f1d1d; padding: 10px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Prototipe Deteksi Jalan / Marka Jalur</h1>
    <p class="muted">Unggah gambar atau video dashcam. Setelah dikirim, halaman progres akan menampilkan status pemrosesan secara real-time.</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="post" action="/process" enctype="multipart/form-data">
      <div class="row"><input type="file" name="file" accept="image/*,video/*" required></div>
      <div class="row"><label><input type="checkbox" name="fallback_model" value="1"> Pakai model UFLD saja — garis cyan, tanpa garis kuning klasik (halus, tapi lebih lambat)</label></div>
      <div class="row"><button type="submit">Proses</button></div>
    </form>
    <p class="muted">Mode model butuh <code>models/ufld.tflite</code> (178 MB, tidak ikut di-repo). Kalau tidak ada, proses otomatis memakai engine klasik.</p>
    <p class="muted"><a href="/tests">Lihat contoh uji (input vs hasil) →</a></p>
  </div>
</body>
</html>
"""

JOB_HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {% if auto_refresh %}<meta http-equiv="refresh" content="3">{% endif %}
  <title>Status Proses - Deteksi Jalan</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .card { max-width: 900px; margin: 0 auto; background: #111827; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,.3); }
    h1 { margin-top: 0; font-size: 1.4rem; }
    .muted { color: #94a3b8; font-size: .95rem; }
    .bar-wrap { width: 100%; height: 18px; background: #1f2937; border-radius: 999px; overflow: hidden; margin: 10px 0; border: 1px solid #334155; }
    .bar { height: 100%; width: 0%; background: linear-gradient(90deg, #2563eb, #22c55e); transition: width .3s ease; }
    .stat { display: grid; grid-template-columns: 160px 1fr; gap: 6px 12px; margin: 12px 0; }
    .k { color: #94a3b8; }
    .log { background: #0b1220; border: 1px solid #334155; border-radius: 8px; padding: 10px; max-height: 260px; overflow: auto; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; white-space: pre-wrap; }
    .error { color: #fecaca; background: #7f1d1d; padding: 10px; border-radius: 8px; margin: 10px 0; }
    .ok { color: #bbf7d0; }
    img, video { max-width: 100%; border-radius: 8px; margin-top: 12px; background: #000; }
    a { color: #93c5fd; }
    button { background: #334155; color: white; border: none; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Status Pemrosesan</h1>
    <p class="muted">Job ID: <code>{{ job_id }}</code> · <a href="/">Unggah file lain</a></p>

    <div class="bar-wrap"><div class="bar" style="width: {{ bar_pct }}%"></div></div>
    <div class="stat">
      <div class="k">Status</div><div><b>{{ status }}</b></div>
      <div class="k">Tahap</div><div>{{ stage }}</div>
      <div class="k">Progres</div><div>{{ progress_text }}</div>
      <div class="k">Kecepatan</div><div>{{ speed_text }}</div>
      <div class="k">Waktu berjalan</div><div>{{ elapsed_text }}</div>
      <div class="k">File</div><div>{{ filename }}</div>
      <div class="k">Info halaman</div><div class="muted">{{ ui_info }}</div>
    </div>

    {% if stats_rows %}
    <h3>Stabilitas &amp; cakupan garis</h3>
    <div class="stat">
      {% for k, v in stats_rows %}<div class="k">{{ k }}</div><div>{{ v }}</div>{% endfor %}
    </div>
    <p class="muted">Goyangan = median pergeseran posisi garis bawah antar frame. Makin kecil makin stabil (di bawah ~2 px/frame sudah halus).</p>
    {% endif %}

    {% if error %}<div class="error">{{ error }}</div>{% endif %}

    <h3>Log proses</h3>
    <div class="log">{{ log_text }}</div>

    {% if result_url %}
      <h3 class="ok">Hasil selesai</h3>
      {% if kind == 'image' %}
        <img src="{{ result_url }}" alt="Hasil deteksi">
        {% if result_mask_url %}
          <h3>Masker deteksi</h3>
          <p class="muted">Area yang dianggap marka (putih/kuning terang di dalam ROI). Kalau garis terlihat aneh, biasanya karena masker ini ikut menangkap objek lain (dinding, mobil, bayangan terang).</p>
          <img src="{{ result_mask_url }}" alt="Masker deteksi">
        {% endif %}
      {% else %}
        <video controls preload="metadata" src="{{ result_url }}"></video>
      {% endif %}
      <p><a href="{{ result_url }}" download>Unduh hasil</a> · <a href="{{ result_url }}" target="_blank" rel="noopener">Buka file</a></p>
    {% endif %}

    <p class="muted" style="margin-top:16px;">
      <a href="{{ url_for('job_page', job_id=job_id) }}">Refresh sekarang</a>
      {% if auto_refresh %}· halaman ini auto-refresh tiap 3 detik{% endif %}
    </p>
  </div>
</body>
</html>
"""


def _log(job_id, msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(line)
        if len(job["logs"]) > 400:
            job["logs"] = job["logs"][-400:]
        job["stage"] = msg


def _set(job_id, **kw):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(kw)


def process_job(job_id, in_path, out_path, kind, orig_name, use_model=False):
    started = time.time()
    _set(job_id, status="running", started_at=started)
    _log(job_id, f"Mulai memproses: {orig_name} ({kind})")
    if use_model:
        _log(job_id, "Diminta: model UFLD untuk semua frame (cyan tanpa klasik)")
    try:
        if kind == "image":
            _log(job_id, "Membaca dan mendeteksi jalur pada gambar")
            from lane_detect import process_image

            mask_out = RESULT_DIR / f"{job_id}_mask.jpg"
            process_image(str(in_path), str(out_path), mask_path=str(mask_out))
            _log(job_id, "Gambar hasil disimpan")
            if mask_out.exists():
                _set(job_id, result_mask=mask_out.name)
                _log(job_id, "Gambar debug masker disimpan (lihat di bawah)")
            _set(job_id, processed_frames=1, total_frames=1, percent=100.0)
            _log(job_id, "Selesai")
        else:
            cap = cv2.VideoCapture(str(in_path))
            if not cap.isOpened():
                raise RuntimeError("Gagal membuka video.")
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            _log(job_id, f"Info video: {w}x{h}, fps={fps_in:.2f}, total_frame={total or 'tidak diketahui'}")
            _set(job_id, total_frames=total)

            last_t = {"t": time.time(), "f": 0}

            def cb(frames, total_frames):
                now = time.time()
                dt = now - last_t["t"]
                df = frames - last_t["f"]
                inst_fps = (df / dt) if dt > 0 else 0
                last_t["t"] = now
                last_t["f"] = frames
                pct = (frames / total_frames * 100.0) if total_frames else 0.0
                elapsed = now - started
                avg_fps = (frames / elapsed) if elapsed > 0 else 0
                _set(job_id, processed_frames=frames, total_frames=total_frames, percent=pct, fps=avg_fps)
                if frames == 1 or frames % 30 == 0 or (total_frames and frames >= total_frames):
                    if total_frames:
                        _log(job_id, f"Memproses frame {frames}/{total_frames} ({pct:.1f}%) · {avg_fps:.2f} fps")
                    else:
                        _log(job_id, f"Memproses frame {frames} · {avg_fps:.2f} fps")

            _log(job_id, "Memulai deteksi jalur per-frame")
            stats = {}
            if use_model:
                try:
                    from hybrid_video import process_video_model

                    process_video_model(
                        str(in_path),
                        str(out_path),
                        progress_cb=cb,
                        status_cb=lambda m: _log(job_id, m),
                        stats_out=stats,
                    )
                except Exception as exc:
                    _log(job_id, f"Mode model gagal ({exc}); pakai engine klasik")
                    stats = {}
                    process_video(
                        str(in_path),
                        str(out_path),
                        show=False,
                        progress_cb=cb,
                        status_cb=lambda m: _log(job_id, m),
                        stats_out=stats,
                    )
            else:
                process_video(
                    str(in_path),
                    str(out_path),
                    show=False,
                    progress_cb=cb,
                    status_cb=lambda m: _log(job_id, m),
                    stats_out=stats,
                )
            _log(job_id, "Output video selesai")
            if stats.get("model_frames") is not None:
                _log(
                    job_id,
                    f"Model terdeteksi {stats['model_frames']}/{stats['frames']} frame "
                    f"· {stats.get('model_failed', 0)} kosong · "
                    f"goyakan lebar {stats.get('model_width_jitter_px') or 0:.2f} px/frame",
                )
            if stats:
                _set(job_id, stats=stats)
                jl = stats.get("left_jitter_px")
                jr = stats.get("right_jitter_px")
                _log(
                    job_id,
                    "Stabilitas garis (median geser per frame): "
                    f"kiri {('%.2f px' % jl) if jl is not None else '-'} · "
                    f"kanan {('%.2f px' % jr) if jr is not None else '-'}",
                )
                _log(
                    job_id,
                    "Terdeteksi: kiri %d%% (%d%% estimasi) · kanan %d%% (%d%% estimasi)"
                    % (
                        round(100 * stats.get("left_frames", 0) / max(1, stats.get("frames", 1))),
                        round(100 * stats.get("left_est", 0) / max(1, stats.get("frames", 1))),
                        round(100 * stats.get("right_frames", 0) / max(1, stats.get("frames", 1))),
                        round(100 * stats.get("right_est", 0) / max(1, stats.get("frames", 1))),
                    ),
                )

        elapsed = time.time() - started
        _set(job_id, status="done", percent=100.0, elapsed_sec=elapsed, finished_at=time.time())
        _log(job_id, f"Job selesai dalam {elapsed:.1f} detik")
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        elapsed = time.time() - started
        _set(job_id, status="error", error=str(e), elapsed_sec=elapsed)
        _log(job_id, f"ERROR: {e}")
        _log(job_id, tb.strip())
    finally:
        try:
            _set(job_id, input_path=str(in_path))
            _log(job_id, f"File asli disimpan untuk pengujian ulang: {in_path.name}")
            files = sorted(UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[20:]:
                old.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/", methods=["GET"])
def index():
    error = request.args.get("error")
    return render_template_string(INDEX_HTML, error=error)


@app.route("/process", methods=["POST"])
def process():
    f = request.files.get("file")
    if not f or not f.filename:
        return redirect(url_for("index", error="File tidak dipilih."))
    name = secure_filename(f.filename)
    ext = Path(name).suffix.lower()
    job_id = uuid.uuid4().hex[:12]
    in_path = UPLOAD_DIR / f"{job_id}_{name}"

    if ext in ALLOWED_IMAGE:
        kind = "image"
        out_name = f"{job_id}_result.jpg"
    elif ext in ALLOWED_VIDEO:
        kind = "video"
        out_name = f"{job_id}_result.mp4"
    else:
        return redirect(url_for("index", error=f"Format tidak didukung: {ext}"))

    out_path = RESULT_DIR / out_name
    f.save(in_path)
    use_model = bool(request.form.get("fallback_model")) and kind == "video"

    job = {
        "id": job_id,
        "status": "queued",
        "stage": "Menunggu antrean",
        "filename": name,
        "kind": kind,
        "use_model": use_model,
        "result": out_name,
        "percent": 0.0,
        "processed_frames": 0,
        "total_frames": 0,
        "fps": 0.0,
        "logs": [],
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "elapsed_sec": 0.0,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    _log(job_id, "File diterima, job dibuat")

    t = threading.Thread(
        target=process_job,
        args=(job_id, in_path, out_path, kind, name, use_model),
        daemon=True,
    )
    t.start()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        data = dict(job) if job else None
    if not data:
        return redirect(url_for("index", error="Job tidak ditemukan (status bisa hilang setelah server restart; file hasil lama tetap ada di folder results/)."))

    status = data.get("status") or "-"
    pct = float(data.get("percent") or 0.0)
    pct = max(0.0, min(100.0, pct))

    elapsed = float(data.get("elapsed_sec") or 0.0)
    if status == "running" and data.get("started_at"):
        elapsed = time.time() - float(data["started_at"])

    pf = int(data.get("processed_frames") or 0)
    tf = int(data.get("total_frames") or 0)

    progress_text = f"{pct:.1f}%"
    if tf:
        progress_text += f" ({pf}/{tf} frame)"
    elif pf:
        progress_text += f" ({pf} frame)"

    logs = data.get("logs") or []
    log_text = "\n".join(logs) if logs else "Belum ada log."

    try:
        speed = float(data.get("fps") or 0.0)
    except Exception:
        speed = 0.0

    result_url = None
    if status == "done" and data.get("result"):
        result_url = url_for("serve_result", filename=data["result"])
    result_mask_url = None
    if status == "done" and data.get("result_mask"):
        result_mask_url = url_for("serve_result", filename=data["result_mask"])

    stats_rows = []
    st = data.get("stats") or {}
    if st:
        def _px(v):
            return f"{v:.2f} px/frame" if isinstance(v, (int, float)) else "-"
        frames = max(1, int(st.get("frames") or 1))
        for key, label in (("left", "kiri"), ("right", "kanan")):
            j = st.get(key + "_jitter_px")
            if j is not None:
                stats_rows.append((f"Goyangan garis {label}", _px(j)))
            fr = int(st.get(key + "_frames") or 0)
            es = int(st.get(key + "_est") or 0)
            if fr:
                stats_rows.append(
                    (
                        f"Cakupan {label}",
                        f"{100 * fr / frames:.0f}% frame" + (f" (dari itu {100 * es / frames:.0f}% estimasi)" if es else ""),
                    )
                )
            q = st.get(key + "_quality")
            if q is not None:
                stats_rows.append((f"Kualitas {label}", f"{float(q):.2f}"))
        if "model_frames" in st:
            stats_rows.append(
                (
                    "Model (cyan)",
                    f"{int(st['model_frames'])}/{int(st.get('frames') or 0)} frame · "
                    f"{int(st.get('model_failed') or 0)} kosong",
                )
            )
            wj = st.get("model_width_jitter_px")
            if wj is not None:
                stats_rows.append(("Goyakan lebar jalur", _px(wj)))

    return render_template_string(
        JOB_HTML,
        job_id=job_id,
        status=status,
        stage=data.get("stage") or "-",
        bar_pct=f"{pct:.1f}",
        progress_text=progress_text,
        speed_text=(f"{speed:.2f} frame/detik" if speed > 0 else "-"),
        elapsed_text=(f"{int(elapsed // 60)}:{int(elapsed) % 60:02d}" if elapsed > 0 else "-"),
        filename=data.get("filename") or "-",
        ui_info=f"dirender server pada {time.strftime('%H:%M:%S')}",
        error=data.get("error"),
        log_text=log_text,
        result_url=result_url,
        result_mask_url=result_mask_url,
        kind=data.get("kind"),
        auto_refresh=(status in ("queued", "running")),
        stats_rows=stats_rows,
    )


@app.route("/status/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        data = dict(job)
    if data.get("status") == "running" and data.get("started_at"):
        data["elapsed_sec"] = time.time() - data["started_at"]
    return jsonify(data)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

TESTS_HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Contoh Uji Deteksi Jalur</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    .card { background: #111827; padding: 16px; border-radius: 12px; margin-bottom: 18px; }
    h1 { font-size: 1.4rem; }
    h2 { font-size: 1.05rem; margin: 0 0 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
    img, video { width: 100%; border-radius: 8px; background: #000; }
    .cap { font-size: .82rem; color: #94a3b8; margin-bottom: 4px; }
    a { color: #93c5fd; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Contoh Uji Deteksi Jalur</h1>
  <p class="muted">Kiri: input sintetis. Kanan: hasil deteksi. <a href="/">← kembali</a></p>

  {% for it in items %}
  <div class="card">
    <h2>{{ it.name }}</h2>
    <div class="grid">
      <div>
        <div class="cap">INPUT</div>
        <img src="{{ it.src }}" alt="input">
      </div>
      <div>
        <div class="cap">HASIL DETEKSI</div>
        {% if it.out %}<img src="{{ it.out }}" alt="hasil">{% else %}<div class="cap">belum diproses</div>{% endif %}
      </div>
    </div>
  </div>
  {% endfor %}

  {% for v in videos %}
  <div class="card">
    <h2>{{ v.name }}</h2>
    <div class="cap">HASIL VIDEO</div>
    {% if v.out %}<video controls preload="metadata" src="{{ v.out }}"></video>{% else %}<div class="cap">belum diproses</div>{% endif %}
  </div>
  {% endfor %}
</div>
</body>
</html>
"""


@app.route("/tests")
def tests_page():
    items = []
    for s in sorted(SAMPLES_DIR.glob("*.jpg")):
        r = RESULT_DIR / f"test_{s.stem}.jpg"
        items.append({
            "name": s.stem.replace("case_", "").replace("_", " "),
            "src": url_for("serve_sample", filename=s.name),
            "out": (url_for("serve_result", filename=r.name) if r.exists() else None),
        })

    videos = []
    for v in sorted(SAMPLES_DIR.glob("*.mp4")):
        r = RESULT_DIR / f"test_{v.stem}.mp4"
        videos.append({
            "name": v.stem.replace("case_", "").replace("_", " "),
            "out": (url_for("serve_result", filename=r.name) if r.exists() else None),
        })

    return render_template_string(TESTS_HTML, items=items, videos=videos)


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
