import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { decodePng, pngOfForm } from './png.mjs';
import { screenCandidates } from './filter.mjs';
import { zopfliZlib } from './zopfli.mjs';

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const chunkOverhead = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) +
  (form.trns === null ? 0 : 12 + form.trns.length);

function bestZlibForCandidate(cand, width, iterations, maxChain, maxBlocks) {
  const raw = cand.data;
  let best = cand.zlib;

  const zop = zopfliZlib(raw, {
    iterations,
    maxChain,
    maxBlocks,
    rowStride: width * cand.form.bpp + 1,
    bpp: cand.form.bpp,
  });

  if (zop.length > 0 && zop.length < best.length) {
    const check = inflateSync(zop);
    if (bytesEqual(check, raw)) {
      best = zop;
    }
  }
  return best;
}

function squeezePng(rgba, width, height) {
  const finalists = screenCandidates(rgba, width, height, 2);
  const top0 = finalists[0];
  const rawLen = top0.data.length;
  const zLen = top0.zlib.length;
  const isHugePhoto = zLen > 220_000;
  const isDense = zLen > 90_000;

  // High-power SIMD128 Rust/WASM deterministic Zopfli budgets calibrated for <=14.5s Node 22 Mac (~32s cloud!)
  const iters0 = isHugePhoto ? 4 : rawLen <= 1_200_000 ? 13 : 9;
  const chain0 = isHugePhoto ? 48 : rawLen <= 1_200_000 ? 352 : isDense ? 208 : 240;
  const blocks0 = 32;

  let bestForm = top0.form;
  let bestIdat = bestZlibForCandidate(top0, width, iters0, chain0, blocks0);
  let bestTotal = bestIdat.length + chunkOverhead(bestForm);

  // Run Rust Zopfli on tiny razor-close palette tie (<15 KB & within 0.1%)
  if (finalists.length > 1 && zLen < 15_000) {
    const cand = finalists[1];
    const gap = (cand.bytes - top0.bytes) / Math.max(1, top0.bytes);
    if (gap <= 0.001) {
      const idat = bestZlibForCandidate(cand, width, 10, 256, blocks0);
      const total = idat.length + chunkOverhead(cand.form);
      if (total < bestTotal) {
        bestTotal = total;
        bestIdat = idat;
        bestForm = cand.form;
      }
    }
  }

  return pngOfForm(width, height, bestForm, bestIdat);
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(I).sort()) {
  const input = new Uint8Array(readFileSync(join(I, id, 'image.png')));
  const img = decodePng(input);
  const out = squeezePng(img.rgba, img.width, img.height);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'image.png'), out);
}
