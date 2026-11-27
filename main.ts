import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { constants, deflateSync, inflateSync } from 'node:zlib';
import { decodePng, pngOfForm } from './png.mjs';
import { screenCandidates } from './filter.mjs';
import { zopfliZlib } from './zopfli.mjs';

const ZLIB_STRATEGIES = [
  { level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY },
  { level: 9, memLevel: 9, strategy: constants.Z_DEFAULT_STRATEGY },
  { level: 9, memLevel: 8, strategy: constants.Z_FILTERED },
  { level: 9, memLevel: 8, strategy: constants.Z_RLE },
];

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function bestZlibForCandidate(cand, width, iterations, maxChain) {
  const raw = cand.data;
  let best = cand.zlib;

  for (let i = 0; i < ZLIB_STRATEGIES.length; i++) {
    const z = deflateSync(raw, ZLIB_STRATEGIES[i]);
    if (z.length < best.length) {
      best = new Uint8Array(z.buffer, z.byteOffset, z.length);
    }
  }

  const zop = zopfliZlib(raw, {
    iterations,
    maxChain,
    maxBlocks: 28,
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
  // Deterministic work budget scaled by image complexity (safe well under 60s sandbox budget even on dense photo sets!)
  const isHugePhoto = zLen > 250_000;
  const isDense = zLen > 90_000;
  const iters = isHugePhoto ? 10 : isDense ? 13 : rawLen <= 1_200_000 ? 10 : 8;
  const chain = isHugePhoto ? 256 : isDense ? 384 : rawLen <= 1_200_000 ? 384 : 256;
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
