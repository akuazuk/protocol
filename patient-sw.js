/* PWA shell для patient.html. HTML/JS/CSS — только network-first (без устаревшего кэша). */
const CACHE = "protocol-patient-v10";
const OFFLINE_ASSETS = ["/patient-manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(OFFLINE_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k.startsWith("protocol-patient-") && k !== CACHE).map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

function networkFirst(request) {
  return fetch(request)
    .then((res) => {
      if (res && res.ok) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(request, copy));
      }
      return res;
    })
    .catch(() => caches.match(request));
}

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  const path = url.pathname;

  if (
    path === "/patient.html" ||
    path === "/patient-ui.js" ||
    path === "/patient-tokens.css" ||
    path.startsWith("/patient-ui.js") ||
    path.startsWith("/patient-tokens.css")
  ) {
    e.respondWith(fetch(e.request));
    return;
  }

  if (path.endsWith("patient-manifest.webmanifest")) {
    e.respondWith(networkFirst(e.request));
  }
});
