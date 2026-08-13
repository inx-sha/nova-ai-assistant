const CACHE_NAME = "nova-cache-v4";
const CACHE_FILES = [
  "/",
  "/static/manifest.json",
  "/static/icon-192.png",
  "/static/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CACHE_FILES))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  // Only intercept simple page-load requests (GET) for the app shell --
  // never intercept API calls (/chat, /sessions, etc). Those must always
  // go live to the server; NOVA's actual functionality needs Ollama +
  // the backend running, this cache only helps the page shell load.
  if (event.request.method !== "GET") return;

  const url = new URL(event.request.url);
  if (!CACHE_FILES.includes(url.pathname)) return;

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});