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
  const cand = screenCandidates(rgba, width, height, 1)[0];
  const rawLen = cand.data.length;
  const zLen = cand.zlib.length;
  // Deterministic work budget calibrated for <=19.5s Mac / <=44s cloud 1-vCPU (rock-solid margin under 60s!)
  const isHugePhoto = zLen > 190_000;
  const isDense = zLen > 90_000;
  const iters = isHugePhoto ? 6 : isDense ? 10 : rawLen <= 1_200_000 ? 11 : 9;
  const chain = isHugePhoto ? 80 : isDense ? 256 : rawLen <= 1_200_000 ? 384 : 256;
  const idat = bestZlibForCandidate(cand, width, iters, chain);
  return pngOfForm(width, height, cand.form, idat);
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
