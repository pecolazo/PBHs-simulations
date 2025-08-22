// sw.js
const CACHE_NAME = "pbh-cache-v1";

// Agregá acá los archivos que querés disponibles offline
const ASSETS = [
  "/PBHs-simulations/",
  "/PBHs-simulations/index.html",
  "/PBHs-simulations/README.md",      // si querés
  "/PBHs-simulations/CDM.gif",        // ideal convertir a .webm (ver optimización)
  "/PBHs-simulations/FCT.gif",
  "/PBHs-simulations/NB.gif",
  "/PBHs-simulations/comparacion_3modelos.gif",
  "/PBHs-simulations/comparacion_triple.html",
  "/PBHs-simulations/comparacion_triple_sync.html",
  "/PBHs-simulations/icon-192.png",
  "/PBHs-simulations/icon-512.png"
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))))
    )
  );
});

self.addEventListener("fetch", (e) => {
  // Estrategia: cache-first (ideal para sitio estático)
  e.respondWith(
    caches.match(e.request).then((cached) => {
      return (
        cached ||
        fetch(e.request).then((resp) => {
          // Cachea en segundo plano (opcional)
          const copy = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(e.request, copy));
          return resp;
        }).catch(() => cached) // si no hay red, devolvé lo del cache
      );
    })
  );
});
