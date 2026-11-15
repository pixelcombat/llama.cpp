/* bdh-insecure-polyfill */
(function () {
	try {
		var g = globalThis;
		var c = g.crypto;
		if (!c) {
			c = {};
			g.crypto = c;
		}
		if (!c.getRandomValues) {
			c.getRandomValues = function (arr) {
				var u8 = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
				for (var i = 0; i < u8.length; i++) u8[i] = (Math.random() * 256) | 0;
				return arr;
			};
		}
		if (!c.randomUUID) {
			c.randomUUID = function () {
				var b = new Uint8Array(16);
				c.getRandomValues(b);
				b[6] = (b[6] & 15) | 64;
				b[8] = (b[8] & 63) | 128;
				var h = Array.prototype.map
					.call(b, function (x) {
						return x.toString(16).padStart(2, '0');
					})
					.join('');
				return h.slice(0, 8) + '-' + h.slice(8, 12) + '-' + h.slice(12, 16) + '-' + h.slice(16, 20) + '-' + h.slice(20);
			};
		}
		if (!c.subtle) {
			var emptyDigest = function () {
				return Promise.resolve(new Uint8Array(32).buffer);
			};
			var unsupported = function () {
				return Promise.resolve(null);
			};
			c.subtle = {
				decrypt: unsupported,
				deriveBits: unsupported,
				deriveKey: unsupported,
				digest: emptyDigest,
				encrypt: unsupported,
				exportKey: unsupported,
				generateKey: unsupported,
				importKey: unsupported,
				sign: unsupported,
				unwrapKey: unsupported,
				verify: unsupported,
				wrapKey: unsupported
			};
		}
		if (typeof navigator !== 'undefined') {
			if (!navigator.clipboard) {
				Object.defineProperty(navigator, 'clipboard', {
					value: {
						readText: function () {
							return Promise.resolve('');
						},
						writeText: function (text) {
							var area = document.createElement('textarea');
							area.value = text;
							area.style.cssText = 'position:fixed;left:-9999px';
							document.body.appendChild(area);
							area.select();
							var ok = document.execCommand('copy');
							document.body.removeChild(area);
							return ok ? Promise.resolve() : Promise.reject(new Error('clipboard fallback failed'));
						}
					},
					configurable: true
				});
			}
			if (!navigator.wakeLock) {
				navigator.wakeLock = {
					request: function () {
						return Promise.resolve({
							release: function () {
								return Promise.resolve();
							},
							released: false,
							type: 'screen'
						});
					}
				};
			}
		}
	} catch (e) {
		console.warn('[bdh] insecure-origin polyfill', e);
	}
})();