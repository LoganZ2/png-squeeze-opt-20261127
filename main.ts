import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync, inflateSync } from 'node:zlib';
import { decodePng, pngOfForm } from './png.mjs';
import { screenCandidates } from './filter.mjs';
import { zopfliZlib } from './zopfli.mjs';

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function bestZlibForCandidate(cand, width, iterations, maxChain, maxBlocks) {
  const raw = cand.data;
  const zop = zopfliZlib(raw, {
    iterations,
    maxChain,
    maxBlocks,
    rowStride: width * cand.form.bpp + 1,
    bpp: cand.form.bpp,
  });

  if (zop.length > 0) {
    const check = inflateSync(zop);
    if (bytesEqual(check, raw)) {
      return zop;
    }
  }
  const fb = deflateSync(raw, { level: 6 });
  return new Uint8Array(fb.buffer, fb.byteOffset, fb.length);
}

function squeezePng(rgba, width, height) {
  const finalists = screenCandidates(rgba, width, height, 4);
  const top0 = finalists[0];
  const rawLen = top0.data.length;
  const zLen = top0.estBytes;
  const isMegaImg = rawLen > 4_200_000 || zLen > 380_000;
  const isHugePhoto = zLen > 205_000;
  const isDense = zLen > 90_000;

  // High-power SIMD128 Rust/WASM deterministic Zopfli budgets calibrated for <=20.0s Node 22 Mac (<47s cloud!)
  const iters0 = isMegaImg
    ? 3
    : isHugePhoto
      ? 4
      : zLen > 135_000
        ? 6
        : isDense
          ? rawLen <= 1_200_000
            ? 12
            : 6
          : rawLen <= 1_200_000
            ? 9
            : 6;
  const chain0 = isMegaImg ? 32 : isHugePhoto ? 48 : rawLen <= 1_200_000 ? 512 : isDense ? 240 : 512;
  const blocks0 = isMegaImg ? 16 : (isDense && rawLen <= 1_200_000) ? 64 : 36;

  const bestIdat = bestZlibForCandidate(top0, width, iters0, chain0, blocks0);
  return pngOfForm(width, height, top0.form, bestIdat);
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
