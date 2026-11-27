import { readFileSync } from 'node:fs';

const wasmBytes = readFileSync(new URL('./squeeze.wasm', import.meta.url));
const wasmModule = new WebAssembly.Module(wasmBytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, {});
const wasm = wasmInstance.exports;

function copyToWasm(bytes) {
  const ptr = wasm.alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

export function zopfliZlib(data, { iterations = 20, maxChain = 384, maxBlocks = 32, rowStride = 0, bpp = 1 } = {}) {
  const inLen = data.length;
  const outCap = inLen + Math.max(4096, (inLen >>> 3) + 1024);
  const inPtr = copyToWasm(data);
  const outPtr = wasm.alloc(outCap);
  try {
    const written = wasm.deflate_zopfli(
      inPtr,
      inLen,
      outPtr,
      outCap,
      iterations,
      maxChain,
      maxBlocks,
      rowStride,
      bpp
    );
    if (written === 0) return new Uint8Array(0);
    return new Uint8Array(wasm.memory.buffer, outPtr, written).slice();
  } finally {
    wasm.dealloc(outPtr, outCap);
    wasm.dealloc(inPtr, inLen);
  }
}

export function fastLzLiteralCounts(data, rowStride = 0) {
  const inLen = data.length;
  const inPtr = copyToWasm(data);
  const outPtr = wasm.alloc(256 * 4);
  try {
    wasm.fast_lz_lit_counts(inPtr, inLen, rowStride, outPtr);
    return new Uint32Array(wasm.memory.buffer, outPtr, 256).slice();
  } finally {
    wasm.dealloc(outPtr, 256 * 4);
    wasm.dealloc(inPtr, inLen);
  }
}

export function generateFilterCandidatesWasm(samples, width, height, bpp, includeBrute = false, topK = 3) {
  const rowLen = width * bpp + 1;
  const imgBytes = height * rowLen;
  const maxCands = topK;
  const inPtr = copyToWasm(samples);
  const outBytes = imgBytes * maxCands;
  const outPtr = wasm.alloc(outBytes);
  try {
    const count = wasm.generate_filter_candidates(
      inPtr,
      width,
      height,
      bpp,
      includeBrute ? 1 : 0,
      outPtr,
      maxCands
    );
    const mem = new Uint8Array(wasm.memory.buffer, outPtr, count * imgBytes);
    const res = new Array(count);
    for (let i = 0; i < count; i++) {
      res[i] = mem.subarray(i * imgBytes, (i + 1) * imgBytes).slice();
    }
    return res;
  } finally {
    wasm.dealloc(outPtr, outBytes);
    wasm.dealloc(inPtr, samples.length);
  }
}

export function optimizePaletteTspWasm(idx, width, height, k, numTrans) {
  const inPtr = copyToWasm(idx);
  const outPtr = wasm.alloc(k);
  try {
    wasm.optimize_palette_tsp(inPtr, width, height, k, numTrans, outPtr);
    return new Uint8Array(wasm.memory.buffer, outPtr, k).slice();
  } finally {
    wasm.dealloc(outPtr, k);
    wasm.dealloc(inPtr, idx.length);
  }
}
