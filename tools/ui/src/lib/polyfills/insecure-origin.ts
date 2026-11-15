/**
 * Polyfills for APIs that are missing or restricted on plain HTTP (non-localhost).
 * Runs before the rest of the client bundle via hooks.client.ts and app.html.
 */
export function installInsecureOriginPolyfills(): void {
	if (typeof globalThis === 'undefined') return;

	const g = globalThis as typeof globalThis & { crypto?: Crypto };

	let cryptoObj = g.crypto;
	if (!cryptoObj) {
		cryptoObj = {} as Crypto;
		g.crypto = cryptoObj;
	}

	if (typeof cryptoObj.getRandomValues !== 'function') {
		cryptoObj.getRandomValues = <T extends ArrayBufferView>(array: T): T => {
			const view = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
			for (let i = 0; i < view.length; i++) {
				view[i] = (Math.random() * 256) | 0;
			}
			return array;
		};
	}

	if (typeof cryptoObj.randomUUID !== 'function') {
		cryptoObj.randomUUID = (): `${string}-${string}-${string}-${string}-${string}` => {
			const bytes = new Uint8Array(16);
			cryptoObj.getRandomValues!(bytes);
			bytes[6] = (bytes[6] & 0x0f) | 0x40;
			bytes[8] = (bytes[8] & 0x3f) | 0x80;
			const hex = [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
			return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
		};
	}

	if (!cryptoObj.subtle) {
		const noop = async (): Promise<null> => null;
		const digest = async (): Promise<ArrayBuffer> => new Uint8Array(32).buffer;
		const subtle = {
			decrypt: noop,
			deriveBits: noop,
			deriveKey: noop,
			digest,
			encrypt: noop,
			exportKey: noop,
			generateKey: noop,
			importKey: noop,
			sign: noop,
			unwrapKey: noop,
			verify: noop,
			wrapKey: noop
		};
		Object.defineProperty(cryptoObj, 'subtle', { value: subtle, configurable: true });
	}

	if (typeof navigator !== 'undefined') {
		const nav = navigator as Navigator & { wakeLock?: { request: (type: string) => Promise<unknown> } };
		if (!nav.clipboard) {
			Object.defineProperty(nav, 'clipboard', {
				value: {
					readText: async () => '',
					writeText: async (text: string) => {
						const area = document.createElement('textarea');
						area.value = text;
						area.style.cssText = 'position:fixed;left:-9999px';
						document.body.appendChild(area);
						area.select();
						const ok = document.execCommand('copy');
						document.body.removeChild(area);
						if (!ok) throw new Error('clipboard.writeText fallback failed');
					}
				},
				configurable: true
			});
		}

		if (!nav.wakeLock) {
			nav.wakeLock = {
				request: async () => ({
					release: async () => {},
					addEventListener: () => {},
					dispatchEvent: () => false,
					removeEventListener: () => {},
					released: false,
					type: 'screen'
				})
			};
		}
	}
}