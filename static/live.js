/* Mode LIVE: ambil frame dari kamera -> POST ke /live/frame -> gambar overlay lokal. */
(() => {
  const canvas   = document.getElementById('view');
  const ctx      = canvas.getContext('2d');
  const video    = document.getElementById('cam');
  const placeholder = document.getElementById('placeholder');

  const btnCam   = document.getElementById('btn-cam');
  const fileInput= document.getElementById('file-input');
  const alertBox = document.getElementById('alert');

  const hudTitle = document.getElementById('hud-title');
  const hudStatus= document.getElementById('hud-status');
  const sCap     = document.getElementById('s-cap');
  const sProc    = document.getElementById('s-proc');
  const sLat     = document.getElementById('s-lat');
  const sSpan    = document.getElementById('s-span');

  const optOverlay = document.getElementById('opt-overlay');
  const optHz    = document.getElementById('opt-hz');
  const optQ     = document.getElementById('opt-q');
  const lblHz    = document.getElementById('lbl-hz');
  const lblQ     = document.getElementById('lbl-q');
  const optDevice= document.getElementById('opt-device');

  const videoResult = document.getElementById('video-result');
  const videoProg   = document.getElementById('video-progress');
  const videoBar    = document.getElementById('video-bar');
  const videoOut    = document.getElementById('video-out');

  const TARGET_WIDTH = Number(document.querySelector('script[data-target-width]')?.dataset.targetWidth || 640);

  let stream = null, running = false;
  let sid = null;
  let lastResult = null;
  let sentFrames = 0, procFrames = 0, capFrames = 0;
  let lastCapT = performance.now(), lastProcT = performance.now();
  let errorStreak = 0;

  const showAlert = (msg, kind) => {
    alertBox.textContent = msg;
    alertBox.className = 'alert' + (kind === 'info' ? ' info' : '');
  };
  const clearAlert = () => alertBox.className = 'alert hidden';

  // ---------- kamera ----------
  async function listDevices() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter(d => d.kind === 'videoinput');
      optDevice.innerHTML = '<option value="">default</option>';
      cams.forEach((d, i) => {
        const o = document.createElement('option');
        o.value = d.deviceId;
        o.textContent = d.label || `kamera ${i + 1}`;
        optDevice.appendChild(o);
      });
    } catch (_) { /* label kosong sebelum izin diberikan: tidak masalah */ }
  }

  async function startCamera(deviceId) {
    if (!window.isSecureContext) {
      showAlert('Kamera diblokir: halaman ini bukan HTTPS. Buat browser mengizinkan kamera ' +
                'hanya lewat https:// atau http://localhost. Pakai Mode file sebagai gantinya.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      showAlert('Browser ini tidak mendukung akses kamera (getUserMedia). Pakai Mode file.');
      return;
    }
    stopCamera();
    clearAlert();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play().catch(() => {});
      placeholder.classList.add('hidden');
      btnCam.textContent = 'Matikan kamera';
      btnCam.classList.add('danger');
      btnCam.classList.remove('primary');
      running = true;
      errorStreak = 0;
      await listDevices();
      loop();
    } catch (err) {
      const name = err?.name || '';
      const inFrame = window.self !== window.top;
      let msg = `Gagal mengakses kamera (${name || err}).`;
      if (name === 'NotAllowedError') {
        msg = 'Izin kamera ditolak. Cek ikon kamera di address bar, lalu muat ulang.';
        // Di preview yang di-embed sebagai iframe, kebijakan izin induk sering
        // memblokir kamera walau originnya sudah HTTPS.
        if (inFrame) {
          msg = 'Izin kamera ditolak. Halaman ini berjalan di dalam iframe, dan banyak induk ' +
                'memblokir kamera untuknya — coba buka preview di tab baru, lalu tekan ' +
                '“Nyalakan kamera” lagi.';
        }
      } else if (name === 'NotFoundError') msg = 'Tidak ada kamera terdeteksi di perangkat ini.';
      else if (name === 'NotReadableError') msg = 'Kamera sedang dipakai aplikasi lain.';
      else if (name === 'SecurityError' && inFrame) {
        msg = 'Iframe ini tidak diizinkan memakai kamera. Buka preview di tab baru.';
      }
      showAlert(msg + ' Anda masih bisa memakai Mode file.');
    }
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    btnCam.textContent = 'Nyalakan kamera';
    btnCam.classList.remove('danger');
    btnCam.classList.add('primary');
    placeholder.classList.remove('hidden');
  }

  // ---------- loop kirim frame ----------
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function loop() {
    const off = document.createElement('canvas');
    const offCtx = off.getContext('2d');

    while (running) {
      const t0 = performance.now();
      if (video.readyState >= 2 && video.videoWidth) {
        const scale = TARGET_WIDTH / video.videoWidth;
        const w = Math.round(video.videoWidth * scale);
        const h = Math.round(video.videoHeight * scale);
        off.width = w; off.height = h;
        offCtx.drawImage(video, 0, 0, w, h);
        capFrames++;

        const blob = await new Promise(res => off.toBlob(res, 'image/jpeg', Number(optQ.value) / 100));
        if (blob) {
          const fd = new FormData();
          fd.append('frame', blob, 'f.jpg');
          if (sid) fd.append('sid', sid);
          try {
            const r = await fetch('/live/frame', { method: 'POST', body: fd });
            if (r.ok) {
              lastResult = await r.json();
              if (!sid) sid = lastResult.sid;
              sentFrames++; procFrames++;
              errorStreak = 0;
              clearAlert();
              draw(lastResult);
              updateHud(lastResult);
            } else {
              errorStreak++;
              const body = await r.json().catch(() => ({}));
              if (errorStreak >= 3) showAlert('Server menolak frame: ' + (body.error || r.status));
            }
          } catch (e) {
            errorStreak++;
            if (errorStreak >= 3) showAlert('Tidak bisa menghubungi server: ' + e.message);
          }
        }
      }
      // jaga laju
      const target = 1000 / Number(optHz.value);
      const spent = performance.now() - t0;
      await sleep(Math.max(0, target - spent));
      updateRates();
    }
  }

  function updateRates() {
    const now = performance.now();
    if (now - lastCapT > 1000) {
      sCap.textContent = capFrames.toFixed(0) + ' fps';
      capFrames = 0; lastCapT = now;
    }
    if (now - lastProcT > 1000) {
      sProc.textContent = procFrames.toFixed(0) + ' fps';
      procFrames = 0; lastProcT = now;
    }
  }

  // ---------- gambar overlay ----------
  function draw(res) {
    const w = canvas.width, h = canvas.height;
    // gambar frame kamera terakhir ke canvas (ukuran internal 16:9 tetap)
    if (video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    }
    if (!optOverlay.checked) return;
    if (!res || !res.left && !res.right) return;

    const sx = w / (res.w || w), sy = h / (res.h || h);
    const toXY = p => [p[0] * sx, p[1] * sy];

    // area jalur
    if (res.left && res.right) {
      ctx.beginPath();
      res.left.forEach((p, i) => { const [x, y] = toXY(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      [...res.right].reverse().forEach(p => { const [x, y] = toXY(p); ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.fillStyle = (res.left_estimated || res.right_estimated) ? 'rgba(0,190,60,.10)' : 'rgba(0,190,60,.20)';
      ctx.fill();
    }

    const line = (pts, estimated, color) => {
      if (!pts || pts.length < 2) return;
      ctx.beginPath();
      pts.forEach((p, i) => { const [x, y] = toXY(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.setLineDash(estimated ? [10, 7] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(res.left, res.left_estimated, '#facc15');
    line(res.right, res.right_estimated, '#facc15');

    // lintasan tengah jalur ke depan (oranye putus-putus)
    const g = res.geometry;
    if (g && g.path && g.path.length > 1 && !g.drifting) {
      ctx.beginPath();
      g.path.forEach((p, i) => { const [x, y] = toXY(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#fb923c';
      ctx.setLineDash([4, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function updateHud(res) {
    hudTitle.textContent = res.title || '—';
    const okSides = (res.left ? 1 : 0) + (res.right ? 1 : 0);
    const weak = !res.left && !res.right;
    hudTitle.className = 'pill ' + (res.rejected ? 'bad' : (weak ? 'bad' : (okSides === 2 && !res.left_estimated && !res.right_estimated ? 'ok' : 'warn')));
    const g = res.geometry;
    let extra = '';
    if (g && g.valid) {
      if (g.drifting) {
        extra = ' · estimasi lateral ditahan (garis bergeser bersama, bukan pindah lajur)';
      } else if (g.lane_change && g.lane_change !== 'di tengah jalur') {
        extra = ' · ' + g.lane_change;
        if (g.moving) extra += ` (${g.shift_m >= 0 ? '+' : ''}${g.shift_m} m)`;
      }
      if (g.drift_warning && !g.drifting) extra += ' · perkiraan jarak tempuh mulai melenceng';
      if (g.width_assumed) extra += ' · lebar jalur diasumsikan, jarak bisa meleset ±20%';
    }
    hudStatus.textContent = (res.status || '') + extra;
    sLat.textContent = res.server_ms != null ? res.server_ms.toFixed(0) + ' ms' : '–';
    sSpan.textContent = `${res.left_span_pct ?? '–'}% / ${res.right_span_pct ?? '–'}%`;
  }

  // ---------- mode file ----------
  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    stopCamera();
    clearAlert();
    videoResult.classList.remove('hidden');
    videoOut.classList.add('hidden');
    videoProg.textContent = 'Mengirim berkas…';
    videoBar.style.width = '2%';

    const fd = new FormData();
    fd.append('file', f);
    let res;
    try {
      const r = await fetch('/live/file', { method: 'POST', body: fd });
      res = await r.json();
      if (!r.ok) throw new Error(res.error || r.status);
    } catch (e) {
      videoProg.textContent = '';
      showAlert('Gagal memproses berkas: ' + e.message);
      return;
    }
    errorStreak = 0;

    if (res.kind === 'image') {
      videoProg.textContent = `Gambar ${res.w}×${res.h} — ${res.title || ''}`;
      videoBar.style.width = '100%';
      const img = new Image();
      // gambar input tidak disimpan server; tampilkan hasil overlay dari koordinat
      canvas.width = res.w; canvas.height = res.h;
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 0, res.w, res.h);
      placeholder.classList.add('hidden');
      draw({ ...res, w: res.w, h: res.h });
      videoResult.classList.add('hidden');
      hudTitle.textContent = res.title || '—';
      const g = res.geometry;
    let extra = '';
    if (g && g.valid) {
      if (g.drifting) {
        extra = ' · estimasi lateral ditahan (garis bergeser bersama, bukan pindah lajur)';
      } else if (g.lane_change && g.lane_change !== 'di tengah jalur') {
        extra = ' · ' + g.lane_change;
        if (g.moving) extra += ` (${g.shift_m >= 0 ? '+' : ''}${g.shift_m} m)`;
      }
      if (g.drift_warning && !g.drifting) extra += ' · perkiraan jarak tempuh mulai melenceng';
      if (g.width_assumed) extra += ' · lebar jalur diasumsikan, jarak bisa meleset ±20%';
    }
    hudStatus.textContent = (res.status || '') + extra;
    } else {
      const job = res.job;
      pollJob(job);
    }
  });

  async function pollJob(job) {
    videoProg.textContent = 'Memproses video…';
    for (;;) {
      await sleep(700);
      let d;
      try {
        const r = await fetch('/live/job/' + job);
        d = await r.json();
        if (!r.ok) throw new Error(d.error || r.status);
      } catch (e) {
        videoProg.textContent = '';
        showAlert('Job hilang: ' + e.message);
        return;
      }
      const pct = Math.max(0, Math.min(100, d.percent || 0));
      videoBar.style.width = pct + '%';
      videoProg.textContent = `${d.status} · ${pct.toFixed(1)}%` +
        (d.total ? ` (${d.frames}/${d.total} frame)` : '') + (d.stage ? ` · ${d.stage}` : '');
      if (d.status === 'done') {
        videoOut.src = '/results/' + d.result;
        videoOut.classList.remove('hidden');
        videoProg.textContent = 'Selesai.';
        return;
      }
      if (d.status === 'error') {
        showAlert('Gagal: ' + d.error);
        return;
      }
    }
  }

  // ---------- kontrol ----------
  btnCam.addEventListener('click', () => {
    if (running) { stopCamera(); clearAlert(); return; }
    startCamera(optDevice.value || undefined);
  });

  optHz.addEventListener('input', () => { lblHz.textContent = optHz.value; });
  optQ.addEventListener('input', () => { lblQ.textContent = optQ.value; });
  optDevice.addEventListener('change', () => { if (running) startCamera(optDevice.value || undefined); });
  optOverlay.addEventListener('change', () => {
    if (lastResult) draw(lastResult); else if (running) draw({ w: canvas.width, h: canvas.height });
  });

  window.addEventListener('beforeunload', stopCamera);
})();
