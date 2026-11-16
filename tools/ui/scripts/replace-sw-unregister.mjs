#!/usr/bin/env node
/**
 * Post-build: overwrite Workbox sw.js with BDH unregister SW; remove workbox-*.js chunks.
 * Bundle insecure-origin polyfill (esbuild) into index.html + client bundle prepend.
 */
import { readFileSync, writeFileSync, readdirSync, unlinkSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.env.LLAMA_UI_OUT_DIR || resolve(__dirname, '..', 'dist'));
const src = resolve(__dirname, '..', 'static', 'bdh-sw-unregister.js');
const polyfillEntry = resolve(__dirname, '..', 'src/lib/polyfills/insecure-polyfill-prepend.ts');

if (!existsSync(outDir)) {
	console.error(`replace-sw-unregister: missing out dir ${outDir}`);
	process.exit(1);
}

function buildInsecurePolyfillPrepend() {
	const result = esbuild.buildSync({
		banner: { js: '/* bdh-insecure-polyfill */' },
		bundle: true,
		entryPoints: [polyfillEntry],
		format: 'iife',
		minify: true,
		platform: 'browser',
		write: false
	});
	return result.outputFiles[0].text;
}

const body = readFileSync(src, 'utf8');
writeFileSync(resolve(outDir, 'sw.js'), body, 'utf8');

let removed = 0;
for (const name of readdirSync(outDir)) {
	if (/^workbox-.*\.js$/i.test(name)) {
		unlinkSync(resolve(outDir, name));
		removed++;
	}
}

const poly = buildInsecurePolyfillPrepend();
const indexPath = resolve(outDir, 'index.html');
if (existsSync(indexPath)) {
	let html = readFileSync(indexPath, 'utf8');
	html = html.replace(/<script>\s*\/\* bdh-insecure-polyfill[\s\S]*?<\/script>\s*/g, '');
	const snippet = `<script>\n${poly}\n\t\t</script>`;
	if (!html.includes('bdh-insecure-polyfill')) {
		html = html.replace('<meta charset="utf-8" />', `<meta charset="utf-8" />\n\t\t${snippet}`);
	}
	const hook = process.env.LLAMA_UI_INDEX_HOOK;
	if (hook) {
		const { spawnSync } = await import('node:child_process');
		const py = spawnSync('python3', ['-c', hook], { input: html, encoding: 'utf8' });
		if (py.status === 0 && py.stdout) {
			html = py.stdout;
		}
	}
	writeFileSync(indexPath, html, 'utf8');
}
const imm = resolve(outDir, '_app', 'immutable');
if (existsSync(imm)) {
	for (const name of readdirSync(imm)) {
		if (!/^bundle\..*\.js$/i.test(name)) continue;
		const bp = resolve(imm, name);
		let js = readFileSync(bp, 'utf8');
		const marker = '/* bdh-insecure-polyfill */';
		const idx = js.indexOf(marker);
		if (idx !== -1) {
			const after = js.indexOf('})();', idx);
			js = after !== -1 ? js.slice(after + 5).replace(/^\s*/, '') : js.slice(idx);
		}
		if (!js.includes(marker)) {
			js = `${poly}\n${js}`;
		}
		writeFileSync(bp, js, 'utf8');
	}
}

if (body.includes('registration.unregister')) {
	console.log(`replace-sw-unregister: ok out=${outDir} removed_workbox=${removed}`);
} else {
	console.error('replace-sw-unregister: source missing unregister');
	process.exit(1);
}