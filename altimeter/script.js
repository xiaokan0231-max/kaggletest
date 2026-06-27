/* ============================================================
   Altimeter PWA — script.js
   Vanilla JS. No frameworks, no dependencies.
   Handles geolocation, compass, chart, history, export, PWA.
   ============================================================ */
'use strict';

(function () {
  const APP_VERSION = 'v1.0.0';
  const HISTORY_LIMIT = 50;
  const STORAGE_KEY = 'altimeter.history.v1';
  const THEME_KEY = 'altimeter.theme';

  /* ---------- DOM helpers ---------- */
  const $ = (id) => document.getElementById(id);

  const el = {
    statusPill: $('status-pill'),
    statusText: $('status-text'),
    live: $('live-indicator'),
    altValue: $('altitude-value'),
    altFeet: $('altitude-feet'),
    altMessage: $('altitude-message'),
    accDot: $('accuracy-dot'),
    accLabel: $('accuracy-label'),
    chart: $('altitude-chart'),
    trendBadge: $('trend-badge'),
    statHigh: $('stat-high'),
    statLow: $('stat-low'),
    statAvg: $('stat-avg'),
    statCount: $('stat-count'),
    lat: $('info-lat'),
    lng: $('info-lng'),
    hacc: $('info-hacc'),
    vacc: $('info-vacc'),
    speed: $('info-speed'),
    heading: $('info-heading'),
    time: $('info-time'),
    appleMaps: $('apple-maps'),
    googleMaps: $('google-maps'),
    compassRose: $('compass-rose'),
    compassDeg: $('compass-deg'),
    compassCard: $('compass-card'),
    compassEnable: $('compass-enable'),
    copyCoords: $('copy-coords'),
    shareLoc: $('share-loc'),
    exportJson: $('export-json'),
    exportCsv: $('export-csv'),
    clearHistory: $('clear-history'),
    themeToggle: $('theme-toggle'),
    toast: $('toast'),
    version: $('app-version'),
    moreToggle: $('more-toggle'),
    deviceList: $('device-list'),
    devNetwork: $('dev-network'),
    devBattery: $('dev-battery'),
    devWakelock: $('dev-wakelock'),
    devFps: $('dev-fps'),
    devPlatform: $('dev-platform'),
    devScreen: $('dev-screen'),
    toggleWakelock: $('toggle-wakelock'),
    toggleFullscreen: $('toggle-fullscreen'),
    installBanner: $('install-banner'),
    installBtn: $('install-btn'),
  };

  /* ---------- App state ---------- */
  const state = {
    history: [],          // { t, lat, lng, alt, acc, vacc, speed, heading }
    lastPosition: null,
    watchId: null,
    wakeLock: null,
    displayedAlt: null,    // for smooth animation
  };

  /* ============================================================
     STATUS
     ============================================================ */
  function setStatus(text, kind) {
    el.statusText.textContent = text;
    el.statusPill.className = 'status-pill status-' + (kind || 'waiting');
  }
  function setLive(kind) {
    // kind: 'on' | 'error' | 'idle'
    el.live.className = 'live-indicator' + (kind === 'on' ? ' live-on' : kind === 'error' ? ' live-error' : '');
    el.live.querySelector('.live-text').textContent =
      kind === 'on' ? 'Live' : kind === 'error' ? 'Error' : 'Idle';
  }

  /* ============================================================
     TOAST
     ============================================================ */
  let toastTimer = null;
  function toast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.remove('show'), 2200);
  }

  /* ============================================================
     HAPTICS
     ============================================================ */
  function vibrate(pattern) {
    if (navigator.vibrate) {
      try { navigator.vibrate(pattern); } catch (e) { /* ignore */ }
    }
  }

  /* ============================================================
     FORMATTING
     ============================================================ */
  function fmt(n, digits) {
    return (n === null || n === undefined || isNaN(n)) ? '—' : Number(n).toFixed(digits === undefined ? 1 : digits);
  }
  function metersToFeet(m) { return m * 3.28084; }

  function cardinal(deg) {
    const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    return dirs[Math.round(((deg % 360) / 22.5)) % 16];
  }

  /* ============================================================
     HISTORY (localStorage)
     ============================================================ */
  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) state.history = JSON.parse(raw) || [];
    } catch (e) { state.history = []; }
  }
  function saveHistory() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.history)); } catch (e) { /* quota */ }
  }
  function pushHistory(sample) {
    state.history.push(sample);
    if (state.history.length > HISTORY_LIMIT) {
      state.history = state.history.slice(-HISTORY_LIMIT);
    }
    saveHistory();
  }

  /* ============================================================
     STATS + TREND
     ============================================================ */
  function updateStats() {
    const alts = state.history.map((h) => h.alt).filter((a) => a !== null && a !== undefined && !isNaN(a));
    el.statCount.textContent = String(state.history.length);
    if (!alts.length) {
      el.statHigh.textContent = el.statLow.textContent = el.statAvg.textContent = '--';
      setTrend(0);
      return;
    }
    const high = Math.max(...alts);
    const low = Math.min(...alts);
    const avg = alts.reduce((a, b) => a + b, 0) / alts.length;
    el.statHigh.textContent = fmt(high) + ' m';
    el.statLow.textContent = fmt(low) + ' m';
    el.statAvg.textContent = fmt(avg) + ' m';

    // Trend: compare mean of last 3 vs previous window
    if (alts.length >= 4) {
      const recent = alts.slice(-3).reduce((a, b) => a + b, 0) / 3;
      const prev = alts.slice(-6, -3);
      const prevAvg = prev.length ? prev.reduce((a, b) => a + b, 0) / prev.length : recent;
      setTrend(recent - prevAvg);
    } else {
      setTrend(0);
    }
  }
  function setTrend(delta) {
    if (delta > 0.8) {
      el.trendBadge.className = 'trend-badge trend-up';
      el.trendBadge.textContent = '▲ Ascending';
    } else if (delta < -0.8) {
      el.trendBadge.className = 'trend-badge trend-down';
      el.trendBadge.textContent = '▼ Descending';
    } else {
      el.trendBadge.className = 'trend-badge trend-flat';
      el.trendBadge.textContent = '— Steady';
    }
  }

  /* ============================================================
     CANVAS CHART (custom, no Chart.js)
     ============================================================ */
  const ctx = el.chart.getContext('2d');
  function resizeChart() {
    const dpr = window.devicePixelRatio || 1;
    const rect = el.chart.getBoundingClientRect();
    el.chart.width = Math.max(1, rect.width * dpr);
    el.chart.height = Math.max(1, rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawChart();
  }
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function drawChart() {
    const rect = el.chart.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const data = state.history
      .map((h) => h.alt)
      .filter((a) => a !== null && a !== undefined && !isNaN(a));

    const pad = { l: 8, r: 8, t: 14, b: 18 };
    const gridColor = cssVar('--stroke') || 'rgba(255,255,255,0.08)';
    const accent = cssVar('--accent') || '#0a84ff';
    const textColor = cssVar('--text-tertiary') || 'rgba(235,235,245,0.3)';

    // Empty-state message
    if (data.length < 2) {
      ctx.fillStyle = textColor;
      ctx.font = '13px ' + getFontStack();
      ctx.textAlign = 'center';
      ctx.fillText('Collecting altitude samples…', W / 2, H / 2);
      return;
    }

    let min = Math.min(...data), max = Math.max(...data);
    if (min === max) { min -= 5; max += 5; }
    const range = max - min || 1;
    // Add 8% headroom
    min -= range * 0.08; max += range * 0.08;

    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const xOf = (i) => pad.l + (data.length === 1 ? plotW / 2 : (i / (data.length - 1)) * plotW);
    const yOf = (v) => pad.t + plotH - ((v - min) / (max - min)) * plotH;

    // Horizontal grid lines + labels
    ctx.strokeStyle = gridColor;
    ctx.fillStyle = textColor;
    ctx.lineWidth = 1;
    ctx.font = '10px ' + getFontStack();
    ctx.textAlign = 'left';
    const rows = 4;
    for (let i = 0; i <= rows; i++) {
      const y = pad.t + (plotH / rows) * i;
      const val = max - ((max - min) / rows) * i;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(0) + 'm', pad.l + 2, y - 3);
    }

    // Area gradient under line
    const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
    grad.addColorStop(0, hexToRgba(accent, 0.32));
    grad.addColorStop(1, hexToRgba(accent, 0));
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(xOf(i), yOf(data[i]));
    ctx.lineTo(xOf(data.length - 1), H - pad.b);
    ctx.lineTo(xOf(0), H - pad.b);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line (smooth)
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(data[0]));
    for (let i = 1; i < data.length; i++) {
      const xc = (xOf(i - 1) + xOf(i)) / 2;
      const yc = (yOf(data[i - 1]) + yOf(data[i])) / 2;
      ctx.quadraticCurveTo(xOf(i - 1), yOf(data[i - 1]), xc, yc);
    }
    ctx.lineTo(xOf(data.length - 1), yOf(data[data.length - 1]));
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Last point marker
    const lx = xOf(data.length - 1), ly = yOf(data[data.length - 1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lx, ly, 7, 0, Math.PI * 2);
    ctx.strokeStyle = hexToRgba(accent, 0.4);
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  function getFontStack() {
    return '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  }
  function hexToRgba(hex, a) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('');
    const r = parseInt(hex.substr(0, 2), 16) || 10;
    const g = parseInt(hex.substr(2, 2), 16) || 132;
    const b = parseInt(hex.substr(4, 2), 16) || 255;
    return `rgba(${r},${g},${b},${a})`;
  }

  /* ============================================================
     ALTITUDE ANIMATION (smooth count-up)
     ============================================================ */
  let animRaf = null;
  function animateAltitude(target) {
    if (target === null || target === undefined || isNaN(target)) {
      el.altValue.textContent = '--';
      el.altFeet.textContent = '-- ft';
      return;
    }
    const start = state.displayedAlt === null ? target : state.displayedAlt;
    const startTime = performance.now();
    const dur = 600;
    cancelAnimationFrame(animRaf);
    function step(now) {
      const p = Math.min(1, (now - startTime) / dur);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      const val = start + (target - start) * eased;
      el.altValue.textContent = val.toFixed(1);
      el.altFeet.textContent = metersToFeet(val).toFixed(0) + ' ft';
      if (p < 1) animRaf = requestAnimationFrame(step);
      else state.displayedAlt = target;
    }
    animRaf = requestAnimationFrame(step);
  }

  /* ============================================================
     ACCURACY INDICATOR
     ============================================================ */
  function updateAccuracy(acc) {
    if (acc === null || acc === undefined || isNaN(acc)) {
      el.accDot.className = 'accuracy-dot';
      el.accLabel.textContent = 'Accuracy: —';
      return;
    }
    let cls, label;
    if (acc <= 10) { cls = 'acc-green'; label = 'Excellent'; }
    else if (acc <= 30) { cls = 'acc-yellow'; label = 'Moderate'; }
    else { cls = 'acc-red'; label = 'Low'; }
    el.accDot.className = 'accuracy-dot ' + cls;
    el.accLabel.textContent = `Accuracy: ${label} (±${acc.toFixed(0)} m)`;
  }

  /* ============================================================
     GEOLOCATION
     ============================================================ */
  function startGeolocation() {
    if (!('geolocation' in navigator)) {
      setStatus('Browser unsupported', 'error');
      setLive('error');
      el.altMessage.textContent = 'Geolocation is not supported by this browser.';
      return;
    }
    // HTTPS requirement (except localhost)
    if (location.protocol !== 'https:' &&
        location.hostname !== 'localhost' &&
        location.hostname !== '127.0.0.1') {
      setStatus('HTTPS required', 'error');
      setLive('error');
      el.altMessage.textContent = 'Location access requires a secure (HTTPS) connection.';
      return;
    }

    setStatus('Locating…', 'waiting');
    const opts = { enableHighAccuracy: true, maximumAge: 1000, timeout: 27000 };
    state.watchId = navigator.geolocation.watchPosition(onPosition, onError, opts);
  }

  function onPosition(pos) {
    const c = pos.coords;
    state.lastPosition = pos;
    setStatus('GPS Ready', 'ready');
    setLive('on');

    // Altitude
    if (c.altitude !== null && c.altitude !== undefined && !isNaN(c.altitude)) {
      animateAltitude(c.altitude);
      el.altMessage.textContent = '';
    } else {
      el.altValue.textContent = '--';
      el.altFeet.textContent = '-- ft';
      el.altMessage.textContent = 'Altitude unavailable on this device or region.';
      setStatus('Altitude Not Supported', 'warn');
    }

    // GPS info
    el.lat.textContent = fmt(c.latitude, 6) + '°';
    el.lng.textContent = fmt(c.longitude, 6) + '°';
    el.hacc.textContent = c.accuracy != null ? '±' + fmt(c.accuracy, 0) + ' m' : '—';
    el.vacc.textContent = c.altitudeAccuracy != null ? '±' + fmt(c.altitudeAccuracy, 0) + ' m' : '—';
    el.speed.textContent = c.speed != null && !isNaN(c.speed) ? fmt(c.speed * 3.6, 1) + ' km/h' : '—';
    el.heading.textContent = c.heading != null && !isNaN(c.heading) ? fmt(c.heading, 0) + '°' : '—';
    el.time.textContent = new Date(pos.timestamp).toLocaleTimeString();

    updateAccuracy(c.accuracy);

    // GPS-based compass fallback (when device orientation unavailable & moving)
    if (!compassActive && c.heading != null && !isNaN(c.heading)) {
      setCompass(c.heading);
    }

    // Map links
    updateMapLinks(c.latitude, c.longitude);

    // History — store one sample per position update
    pushHistory({
      t: pos.timestamp,
      lat: c.latitude,
      lng: c.longitude,
      alt: (c.altitude != null && !isNaN(c.altitude)) ? c.altitude : null,
      acc: c.accuracy != null ? c.accuracy : null,
      vacc: c.altitudeAccuracy != null ? c.altitudeAccuracy : null,
      speed: c.speed != null ? c.speed : null,
      heading: c.heading != null ? c.heading : null,
    });
    updateStats();
    drawChart();
  }

  function onError(err) {
    setLive('error');
    switch (err.code) {
      case err.PERMISSION_DENIED:
        setStatus('Permission Denied', 'denied');
        el.altMessage.textContent = 'Location permission was denied. Enable it in your browser settings.';
        break;
      case err.POSITION_UNAVAILABLE:
        setStatus('Location Error', 'error');
        el.altMessage.textContent = 'GPS position is unavailable. Move to an open area and try again.';
        break;
      case err.TIMEOUT:
        setStatus('Location Error', 'warn');
        el.altMessage.textContent = 'Location request timed out. Retrying…';
        break;
      default:
        setStatus('Location Error', 'error');
        el.altMessage.textContent = 'An unknown location error occurred.';
    }
  }

  /* ============================================================
     MAP LINKS
     ============================================================ */
  function updateMapLinks(lat, lng) {
    if (lat == null || lng == null) return;
    el.appleMaps.href = `https://maps.apple.com/?ll=${lat},${lng}&q=Current+Location`;
    el.googleMaps.href = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    el.appleMaps.removeAttribute('aria-disabled');
    el.googleMaps.removeAttribute('aria-disabled');
  }

  /* ============================================================
     COMPASS (DeviceOrientation)
     ============================================================ */
  let compassActive = false;
  function setCompass(heading) {
    // Rotate rose so that the heading points up (rotate opposite direction)
    el.compassRose.style.transform = `rotate(${-heading}deg)`;
    el.compassDeg.textContent = Math.round(heading) + '°';
    el.compassCard.textContent = cardinal(heading);
  }
  function onOrientation(e) {
    let heading = null;
    if (typeof e.webkitCompassHeading === 'number') {
      heading = e.webkitCompassHeading; // iOS — already true heading
    } else if (e.absolute && typeof e.alpha === 'number') {
      heading = 360 - e.alpha; // alpha is counter-clockwise from north
    }
    if (heading !== null && !isNaN(heading)) {
      compassActive = true;
      setCompass(heading);
    }
  }
  function initCompass() {
    if (!('DeviceOrientationEvent' in window)) {
      el.compassDeg.textContent = 'N/A';
      el.compassCard.textContent = 'Unsupported';
      return;
    }
    // iOS 13+ requires explicit permission via a user gesture
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      el.compassEnable.hidden = false;
      el.compassEnable.addEventListener('click', async () => {
        try {
          const res = await DeviceOrientationEvent.requestPermission();
          if (res === 'granted') {
            window.addEventListener('deviceorientation', onOrientation, true);
            el.compassEnable.hidden = true;
            vibrate(10);
          } else {
            toast('Compass permission denied');
          }
        } catch (err) {
          toast('Compass unavailable');
        }
      });
    } else {
      window.addEventListener('deviceorientationabsolute', onOrientation, true);
      window.addEventListener('deviceorientation', onOrientation, true);
    }
  }

  /* ============================================================
     EXPORT (JSON / CSV)
     ============================================================ */
  function download(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function exportJSON() {
    if (!state.history.length) return toast('No data to export');
    const rows = state.history.map((h) => ({
      time: new Date(h.t).toISOString(),
      latitude: h.lat,
      longitude: h.lng,
      altitude: h.alt,
      accuracy: h.acc,
    }));
    download('altimeter-history.json', JSON.stringify(rows, null, 2), 'application/json');
    toast('Exported JSON');
  }
  function exportCSV() {
    if (!state.history.length) return toast('No data to export');
    const header = 'time,latitude,longitude,altitude,accuracy';
    const lines = state.history.map((h) =>
      [new Date(h.t).toISOString(), h.lat, h.lng, h.alt == null ? '' : h.alt, h.acc == null ? '' : h.acc].join(',')
    );
    download('altimeter-history.csv', [header, ...lines].join('\n'), 'text/csv');
    toast('Exported CSV');
  }

  /* ============================================================
     COPY / SHARE
     ============================================================ */
  async function copyCoords() {
    if (!state.lastPosition) return toast('No location yet');
    const c = state.lastPosition.coords;
    const text = `${c.latitude.toFixed(6)}, ${c.longitude.toFixed(6)}`;
    try {
      await navigator.clipboard.writeText(text);
      toast('Coordinates copied');
      vibrate(10);
    } catch (e) {
      toast(text);
    }
  }
  async function shareLocation() {
    if (!state.lastPosition) return toast('No location yet');
    const c = state.lastPosition.coords;
    const url = `https://maps.apple.com/?ll=${c.latitude},${c.longitude}`;
    const altText = (c.altitude != null && !isNaN(c.altitude)) ? ` at ${c.altitude.toFixed(1)} m altitude` : '';
    const shareData = {
      title: 'My Location',
      text: `I'm at ${c.latitude.toFixed(5)}, ${c.longitude.toFixed(5)}${altText}.`,
      url,
    };
    if (navigator.share) {
      try { await navigator.share(shareData); vibrate(10); }
      catch (e) { /* user cancelled */ }
    } else {
      try { await navigator.clipboard.writeText(`${shareData.text} ${url}`); toast('Location link copied'); }
      catch (e) { toast('Share not supported'); }
    }
  }

  /* ============================================================
     THEME
     ============================================================ */
  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    el.themeToggle.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme');
      const next = cur === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem(THEME_KEY, next);
      vibrate(8);
      drawChart(); // re-render with new colors
    });
  }

  /* ============================================================
     WAKE LOCK
     ============================================================ */
  async function toggleWakeLock() {
    if (!('wakeLock' in navigator)) return toast('Wake Lock not supported');
    try {
      if (state.wakeLock) {
        await state.wakeLock.release();
        state.wakeLock = null;
        el.devWakelock.textContent = 'Off';
        el.toggleWakelock.textContent = 'Keep Screen On';
        toast('Screen may sleep');
      } else {
        state.wakeLock = await navigator.wakeLock.request('screen');
        el.devWakelock.textContent = 'On';
        el.toggleWakelock.textContent = 'Allow Sleep';
        toast('Screen will stay on');
        state.wakeLock.addEventListener('release', () => {
          el.devWakelock.textContent = 'Off';
          el.toggleWakelock.textContent = 'Keep Screen On';
          state.wakeLock = null;
        });
      }
    } catch (e) {
      toast('Wake Lock failed');
    }
  }
  // Re-acquire wake lock when returning to the tab
  document.addEventListener('visibilitychange', async () => {
    if (state.wakeLock !== null && document.visibilityState === 'visible') {
      try { state.wakeLock = await navigator.wakeLock.request('screen'); } catch (e) { /* ignore */ }
    }
  });

  /* ============================================================
     FULLSCREEN
     ============================================================ */
  function toggleFullscreen() {
    const d = document;
    if (!d.fullscreenElement && !d.webkitFullscreenElement) {
      const root = d.documentElement;
      const req = root.requestFullscreen || root.webkitRequestFullscreen;
      if (req) req.call(root); else toast('Fullscreen not supported');
    } else {
      const exit = d.exitFullscreen || d.webkitExitFullscreen;
      if (exit) exit.call(d);
    }
  }

  /* ============================================================
     DEVICE / SYSTEM INFO (bonus)
     ============================================================ */
  function initDeviceInfo() {
    // Platform & screen
    el.devPlatform.textContent = navigator.platform || navigator.userAgent.slice(0, 24) || '—';
    el.devScreen.textContent = `${window.screen.width}×${window.screen.height} @${window.devicePixelRatio || 1}x`;

    // Network status
    function updateNetwork() {
      const online = navigator.onLine;
      let label = online ? 'Online' : 'Offline';
      const conn = navigator.connection || navigator.webkitConnection;
      if (online && conn && conn.effectiveType) label += ` · ${conn.effectiveType}`;
      el.devNetwork.textContent = label;
    }
    updateNetwork();
    window.addEventListener('online', () => { updateNetwork(); toast('Back online'); });
    window.addEventListener('offline', () => { updateNetwork(); toast('You are offline'); });
    const conn = navigator.connection || navigator.webkitConnection;
    if (conn && conn.addEventListener) conn.addEventListener('change', updateNetwork);

    // Battery
    if (navigator.getBattery) {
      navigator.getBattery().then((b) => {
        function upd() {
          el.devBattery.textContent = `${Math.round(b.level * 100)}%${b.charging ? ' ⚡' : ''}`;
        }
        upd();
        b.addEventListener('levelchange', upd);
        b.addEventListener('chargingchange', upd);
      }).catch(() => { el.devBattery.textContent = 'N/A'; });
    } else {
      el.devBattery.textContent = 'N/A';
    }

    // Wake lock support label
    el.devWakelock.textContent = ('wakeLock' in navigator) ? 'Off' : 'N/A';

    // FPS monitor
    let frames = 0, lastFps = performance.now();
    function fpsLoop(now) {
      frames++;
      if (now - lastFps >= 1000) {
        el.devFps.textContent = frames + ' fps';
        frames = 0;
        lastFps = now;
      }
      requestAnimationFrame(fpsLoop);
    }
    requestAnimationFrame(fpsLoop);

    // Expand/collapse toggle
    el.moreToggle.addEventListener('click', () => {
      const hidden = el.deviceList.hidden;
      el.deviceList.hidden = !hidden;
      el.moreToggle.textContent = hidden ? 'Hide' : 'Show';
      el.moreToggle.setAttribute('aria-expanded', String(hidden));
    });
  }

  /* ============================================================
     PWA — service worker + install
     ============================================================ */
  let deferredPrompt = null;
  function initPWA() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('service-worker.js').catch(() => { /* ignore */ });
      });
    }
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      el.installBanner.hidden = false;
    });
    el.installBtn.addEventListener('click', async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      if (outcome === 'accepted') toast('Installing…');
      deferredPrompt = null;
      el.installBanner.hidden = true;
    });
    window.addEventListener('appinstalled', () => {
      el.installBanner.hidden = true;
      toast('Altimeter installed');
    });
  }

  /* ============================================================
     INIT
     ============================================================ */
  function init() {
    el.version.textContent = APP_VERSION;
    setStatus('Waiting for GPS…', 'waiting');
    setLive('idle');

    loadHistory();
    updateStats();
    resizeChart();

    initTheme();
    initCompass();
    initDeviceInfo();
    initPWA();

    // Wire up buttons
    el.exportJson.addEventListener('click', exportJSON);
    el.exportCsv.addEventListener('click', exportCSV);
    el.clearHistory.addEventListener('click', () => {
      state.history = [];
      saveHistory();
      updateStats();
      drawChart();
      toast('History cleared');
      vibrate(10);
    });
    el.copyCoords.addEventListener('click', copyCoords);
    el.shareLoc.addEventListener('click', shareLocation);
    el.toggleWakelock.addEventListener('click', toggleWakeLock);
    el.toggleFullscreen.addEventListener('click', toggleFullscreen);

    window.addEventListener('resize', resizeChart);

    // Start GPS
    startGeolocation();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
