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

function bestZlibForCandidate(cand, width, iterations, maxChain) {
  const raw = cand.data;
  // cand.zlib is already the exact min of zlib level-9 Z_DEFAULT_STRATEGY and Z_FILTERED!
  let best = cand.zlib;

  const zop = zopfliZlib(raw, {
    iterations,
    maxChain,
    maxBlocks: 32,
    rowStride: width * cand.form.bpp + 1,
    bpp: cand.form.bpp,
  });

  if (zop.length < best.length) {
    const check = inflateSync(zop);
    if (bytesEqual(check, raw)) {
      best = zop;
    }
  }
  return best;
}

function squeezePng(rgba, width, height) {
  const cands = screenCandidates(rgba, width, height, 2);
  let bestPng = null;

  for (let i = 0; i < cands.length; i++) {
    const cand = cands[i];
    // Test runner-up on fast 1-bpp palette near-ties (e.g. pal-count vs pal-lzfreq, < 0.9s!)
    if (i > 0 && !(cand.form.colourType === 3 && cand.bytes <= cands[0].bytes * 1.008)) continue;
    const rawLen = cand.data.length;
    const zLen = cand.zlib.length;
    const isHugePhoto = zLen > 200_000;
    const isDense = zLen > 90_000;
    const iters = isHugePhoto ? 8 : isDense ? 14 : rawLen <= 1_200_000 ? 14 : 10;
    const chain = isHugePhoto ? 160 : isDense ? 512 : rawLen <= 1_200_000 ? 512 : 384;
    const idat = bestZlibForCandidate(cand, width, iters, chain);
    const png = pngOfForm(width, height, cand.form, idat);
    if (bestPng === null || png.length < bestPng.length) {
      bestPng = png;
    }
  }
  return bestPng;
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
