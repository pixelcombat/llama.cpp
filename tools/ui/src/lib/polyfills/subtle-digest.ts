import { sha1 } from '@noble/hashes/sha1';
import { sha256 } from '@noble/hashes/sha256';
import { sha384, sha512 } from '@noble/hashes/sha2';

const DIGEST_FNS: Record<string, (msg: Uint8Array) => Uint8Array> = {
	'SHA-1': sha1,
	'SHA-256': sha256,
	'SHA-384': sha384,
	'SHA-512': sha512
};

function toBytes(data: BufferSource): Uint8Array {
	if (data instanceof ArrayBuffer) {
		return new Uint8Array(data);
	}
	return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function normalizeDigestAlgorithm(algorithm: AlgorithmIdentifier | string): string {
	return typeof algorithm === 'string' ? algorithm : algorithm.name;
}

/** Web Crypto–compatible digest for environments without `crypto.subtle` (plain HTTP). */
export async function subtleDigest(
	algorithm: AlgorithmIdentifier | string,
	data: BufferSource
): Promise<ArrayBuffer> {
	const name = normalizeDigestAlgorithm(algorithm);
	const hash = DIGEST_FNS[name];
	if (!hash) {
		throw new DOMException(`Unsupported digest algorithm: ${name}`, 'NotSupportedError');
	}
	const out = hash(toBytes(data));
	return new Uint8Array(out).buffer;
}

/** SHA-256('') — RFC 6234 / FIPS 180-4 empty-string vector. */
export const SUBTLE_DIGEST_SHA256_EMPTY_HEX =
	'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';

export async function subtleDigestSha256Hex(message: string): Promise<string> {
	const buf = await subtleDigest('SHA-256', new TextEncoder().encode(message));
	return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}