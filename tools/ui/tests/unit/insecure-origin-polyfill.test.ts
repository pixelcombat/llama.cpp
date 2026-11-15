import { describe, expect, it } from 'vitest';
import { installInsecureOriginPolyfills } from '$lib/polyfills/insecure-origin';

describe('insecure-origin polyfills', () => {
	it('provides randomUUID when missing', () => {
		const saved = globalThis.crypto;
		// @ts-expect-error test stub
		globalThis.crypto = { getRandomValues: undefined, randomUUID: undefined, subtle: undefined };
		installInsecureOriginPolyfills();
		expect(typeof globalThis.crypto.randomUUID).toBe('function');
		const id = globalThis.crypto.randomUUID();
		expect(id).toMatch(
			/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
		);
		globalThis.crypto = saved;
	});
});