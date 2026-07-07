/* ============================================================
   Altimeter PWA — service-worker.js
   Cache-first offline support for the static app shell.
   ============================================================ */
'use strict';

const CACHE_NAME = 'altimeter-v1.0.3';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './script.js',
  './manifest.json',
  './assets/icon.svg',
  './assets/apple-touch-icon.svg',
  './assets/icon-192.png',
  './assets/icon-512.png',
];

/* Install — pre-cache the app shell */
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS).catch(() => { /* tolerate missing optional assets */ }))
      .then(() => self.skipWaiting())
  );
});

/* Activate — clean old caches */
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/* Fetch — cache-first for same-origin GET, network passthrough otherwise */
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // don't cache cross-origin (maps, etc.)

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req)
        .then((res) => {
          // Cache a copy of successful responses for next time
          if (res && res.status === 200 && res.type === 'basic') {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
          }
          return res;
        })
        .catch(() => {
          // Offline fallback to the cached shell for navigations
          if (req.mode === 'navigate') return caches.match('./index.html');
        });
    })
  );
});
