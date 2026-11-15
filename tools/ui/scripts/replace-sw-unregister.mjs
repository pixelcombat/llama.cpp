#!/usr/bin/env node
/**
 * Post-build: overwrite Workbox sw.js with BDH unregister SW; remove workbox-*.js chunks.
 * CMake sets LLAMA_UI_OUT_DIR; vite build uses the same via svelte.config.js.
 */
import { readFileSync, writeFileSync, readdirSync, unlinkSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.env.LLAMA_UI_OUT_DIR || resolve(__dirname, '..', 'dist'));
const src = resolve(__dirname, '..', 'static', 'bdh-sw-unregister.js');
const polyfillPrepend = resolve(__dirname, '..', 'static', 'bdh-insecure-polyfill.prepend.js');

if (!existsSync(outDir)) {
  console.error(`replace-sw-unregister: missing out dir ${outDir}`);
  process.exit(1);
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

const indexPath = resolve(outDir, 'index.html');
if (existsSync(indexPath) && existsSync(polyfillPrepend)) {
  const poly = readFileSync(polyfillPrepend, 'utf8');
  let html = readFileSync(indexPath, 'utf8');
  if (!html.includes('bdh-insecure-polyfill')) {
    const snippet = `<script>\n${poly}\n\t\t</script>`;
    html = html.replace('<meta charset="utf-8" />', `<meta charset="utf-8" />\n\t\t${snippet}`);
    writeFileSync(indexPath, html, 'utf8');
  }
  const imm = resolve(outDir, '_app', 'immutable');
  if (existsSync(imm)) {
    const polyMin = poly.replace(/\n/g, '');
    for (const name of readdirSync(imm)) {
      if (!/^bundle\..*\.js$/i.test(name)) continue;
      const bp = resolve(imm, name);
      let js = readFileSync(bp, 'utf8');
      if (!js.includes('bdh-insecure-polyfill')) {
        writeFileSync(bp, polyMin + js, 'utf8');
      }
    }
  }
}

if (body.includes('registration.unregister')) {
  console.log(`replace-sw-unregister: ok out=${outDir} removed_workbox=${removed}`);
} else {
  console.error('replace-sw-unregister: source missing unregister');
  process.exit(1);
}