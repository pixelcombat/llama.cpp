/* BDH mobile: replace Workbox SW — unregister + clear caches (dist + server must match). */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil((async () => {
    await self.registration.unregister();
    for (const key of await caches.keys()) await caches.delete(key);
})()));