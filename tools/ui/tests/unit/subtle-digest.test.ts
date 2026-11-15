import { describe, expect, it } from 'vitest';
import {
	SUBTLE_DIGEST_SHA256_EMPTY_HEX,
	subtleDigest,
	subtleDigestSha256Hex
} from '$lib/polyfills/subtle-digest';

describe('subtle-digest polyfill', () => {
	it('SHA-256("") matches empty-string vector', async () => {
		const buf = await subtleDigest('SHA-256', new Uint8Array());
		const hex = [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
		expect(hex).toBe(SUBTLE_DIGEST_SHA256_EMPTY_HEX);
	});

	it('SHA-256 helper matches hashlib reference string', async () => {
		expect(await subtleDigestSha256Hex('bdh-insecure-test')).toBe(
			'08bbb53a47915f62ecb67885770b56b916e0952f2c7c303fd2f4887d7ea1eb94'
		);
	});

	it('supports SHA-1 and SHA-512', async () => {
		const data = new TextEncoder().encode('bdh');
		const s1 = await subtleDigest('SHA-1', data);
		const s512 = await subtleDigest('SHA-512', data);
		expect(s1.byteLength).toBe(20);
		expect(s512.byteLength).toBe(64);
	});
});