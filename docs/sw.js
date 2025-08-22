// sw.js — scope relativo a /docs (GitHub Pages: Folder /docs)
const CACHE_NAME = "pbh-cache-v1";

// Si más tarde agregás recursos extra (CSS/otras páginas), sumalos acá.
// IMPORTANTE: solo listar archivos que EXISTEN en docs/ para evitar fallos de instalación.
const ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png"
];

// Instalación: cachear assets base
self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

// Activación: limpiar cachés viejas si cambiás el nombre
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))))
    )
  );
});

// Estrategia cache-first para sitio estático
self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((resp) => {
        // Cachear en segundo plano lo que se vaya pidiendo
        const copy = resp.clone();
        caches.open(CACHE_NAME).then((c) => c.put(e.request, copy));
        return resp;
      }).catch(() => cached);
    })
  );
});
