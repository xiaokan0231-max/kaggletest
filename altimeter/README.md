# 🏔️ Altimeter — Professional Mobile Altitude PWA

A beautiful, **native-feeling altimeter web app** that measures your current
altitude on mobile devices using the browser **Geolocation API**. Designed in
the **Apple Human Interface** style — glassmorphism, dark mode, rounded cards,
smooth animations — and built as a **fully static Progressive Web App** with
**zero dependencies, zero frameworks, and no backend**.

> Pure **HTML5 + CSS3 + Vanilla JavaScript**. Deployable directly to GitHub
> Pages, Cloudflare Pages, or Netlify.

---

## ✨ Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Current Altitude** | Large animated readout in meters (and feet), with graceful "unavailable" messaging |
| 2 | **GPS Information** | Latitude, longitude, horizontal accuracy, altitude accuracy, speed, heading, timestamp |
| 3 | **Compass** | Rotating compass via `DeviceOrientation` API (with iOS permission flow) + GPS heading fallback |
| 4 | **Real-time Updates** | Continuous `navigator.geolocation.watchPosition()` with smooth UI animation |
| 5 | **Apple Maps** | One-tap "Open Current Location" via `maps.apple.com` |
| 6 | **Google Maps** | One-tap "Open in Google Maps" |
| 7 | **Status** | Waiting / Locating / GPS Ready / Permission Denied / Altitude Not Supported / Location Error |
| 8 | **Accuracy Indicator** | Color-coded Green / Yellow / Red dot based on GPS accuracy |
| 9 | **History** | Last 50 samples — highest, lowest, average altitude + current trend |
| 10 | **Altitude Chart** | Real-time line chart drawn on **HTML Canvas** (no Chart.js) |
| 11 | **Export** | Download history as **JSON** or **CSV** (time, latitude, longitude, altitude, accuracy) |
| 12 | **PWA** | `manifest.json`, `service-worker.js`, offline support, Add-to-Home-Screen, Apple icons |
| 13 | **Icons** | Hand-built **SVG** + generated PNG app icons, no external icon libraries |
| 14 | **Error Handling** | Permission denied, GPS unavailable, HTTPS required, altitude unavailable, unsupported browser, timeout, region restriction |
| 15 | **Compatibility** | Safari, Chrome, Edge, Firefox, Android, iOS — with graceful degradation |

### 🎁 Bonus features

Dark / Light mode toggle · Live GPS indicator · Copy Coordinates · Share
Location · Fullscreen mode · Vibration (haptic) feedback · Battery status ·
Network status · Wake Lock (keep screen on) · Install button · Device
information · Version number · FPS monitor.

---

## 📱 Browser Compatibility

| Browser / OS | Altitude | Compass | Install (PWA) | Notes |
|--------------|----------|---------|---------------|-------|
| iOS Safari | ⚠️ Often null* | ✅ (tap *Enable*) | ✅ | Best experience; altitude may be unavailable (see limitations) |
| Android Chrome | ✅ | ✅ | ✅ | Full feature support |
| Desktop Chrome / Edge | ⚠️ Usually null | ❌ (no sensors) | ✅ | Graceful degradation — coordinates still work |
| Firefox | ⚠️ Varies | ⚠️ Varies | ➖ | Geolocation works; install limited |
| iPad | ⚠️ Varies | ✅ | ✅ | Same as iOS Safari |

\* On many iPhones and in some regions the OS/browser does **not** expose
altitude even when location is available — this is a platform limitation, not a
bug in the app. The app detects this and shows
*"Altitude unavailable on this device or region."*

---

## 🚀 Deployment

This app is **100% static** — just serve the files over **HTTPS** (required by
the Geolocation API). It lives in the [`altimeter/`](.) directory of the
repository.

### Local preview

```bash
cd altimeter
python3 -m http.server 8080
# open http://localhost:8080
```

> Geolocation works on `localhost` without HTTPS; everywhere else needs HTTPS.

### GitHub Pages

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** = *Deploy from a branch*,
   **Branch** = `main`, folder = `/ (root)`.
4. Save. After a minute your site is live at:
   `https://<username>.github.io/<repo>/altimeter/`

### Cloudflare Pages

1. **Create a project → Connect to Git**, select this repository.
2. Build command: *(none)*. Build output directory: `altimeter` (or root).
3. Deploy. Cloudflare serves over HTTPS automatically.

### Netlify

1. **Add new site → Import an existing project**, pick this repo.
2. Build command: *(none)*. Publish directory: `altimeter`.
3. Deploy. HTTPS is provided automatically.

---

## ⚠️ Known Limitations

- **Altitude availability:** On some iPhones and in some regions, the browser
  or operating system may **not provide altitude information even when location
  is available**. The Geolocation API simply returns `null` for
  `coords.altitude` in those cases. This app degrades gracefully and tells the
  user when altitude cannot be read.
- **GPS altitude accuracy:** Even when present, GPS-derived altitude is far less
  accurate than horizontal position (errors of tens of meters are common). It is
  **not** barometric — phones with a barometer don't expose it to the browser.
- **Compass on iOS:** Requires an explicit permission tap (Apple restriction).
- **Desktop devices** typically have no GPS or magnetometer — location is
  IP/Wi-Fi based and altitude/compass are usually unavailable.
- **HTTPS required:** Geolocation only works over secure origins (`https://` or
  `localhost`).

---

## 🧱 Project Structure

```
altimeter/
├── index.html          # App shell / markup (no inline JS or CSS)
├── style.css           # All styling — Apple HIG, glassmorphism, dark/light
├── script.js           # All logic — geolocation, compass, chart, export, PWA
├── manifest.json       # PWA manifest
├── service-worker.js   # Offline caching
├── assets/
│   ├── icon.svg            # Scalable app icon
│   ├── apple-touch-icon.svg/.png
│   ├── icon-192.png
│   └── icon-512.png
└── README.md
```

---

## 🔮 Suggested Future Improvements

- Optional **barometric calibration** input (set known altitude to offset GPS drift).
- **Unit toggle** (meters / feet) in the UI.
- **Map preview tile** embedded in-app (privacy-respecting, e.g. static tiles).
- **Session recording** with GPX export for hikes.
- **Background tracking** with periodic notifications (where supported).
- **Internationalisation** (multi-language UI).

---

## 🔒 Privacy

All processing happens **on your device**. Location and history never leave the
browser — history is stored only in `localStorage`, and there is **no analytics,
no tracking, and no backend**.

---

*Built with HTML5, CSS3, and Vanilla JavaScript. No frameworks. No npm. No external dependencies.*
