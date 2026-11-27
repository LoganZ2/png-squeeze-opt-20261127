import { constants, deflateSync } from 'node:zlib';
import { filterRow } from './png.mjs';
import { fastLzLiteralCounts } from './zopfli.mjs';

function wordsOf(rgba) {
  const at = rgba.byteOffset % 4 === 0 ? rgba : rgba.slice();
  return new Uint32Array(at.buffer, at.byteOffset, at.length >>> 2);
}

export function analyse(rgba) {
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  let opaque = true;
  let grey = true;
  const counts = new Map();
  let many = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!many) {
      const c = u32[i];
      const k = counts.get(c);
      if (k === undefined) {
        if (counts.size === 256) many = true;
        else counts.set(c, 1);
      } else counts.set(c, k + 1);
    }
  }
  const colours = many ? null : [...counts.keys()].map((c) => {
    const b = new Uint8Array(Uint32Array.of(c).buffer);
    return { rgba: [b[0], b[1], b[2], b[3]], word: c, count: counts.get(c) };
  });
  return { opaque, grey, colours };
}

export function transparentKey(rgba, grey) {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}

export function samplesOf(rgba, ct) {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) { out[s] = rgba[d]; out[s + 1] = rgba[d + 3]; }
    else { out[s] = rgba[d]; out[s + 1] = rgba[d + 1]; out[s + 2] = rgba[d + 2]; }
  }
  return out;
}

function cooccurrenceOrder(rgba, width, height, colours) {
  const k = colours.length;
  const sorted = [...colours].sort((x, y) => {
    const ta = x.rgba[3] === 255 ? 1 : 0;
    const tb = y.rgba[3] === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
    return y.count - x.count || x.word - y.word;
  });
  let numTrans = 0;
  while (numTrans < k && sorted[numTrans].rgba[3] !== 255) numTrans++;
  const numOpaque = k - numTrans;
  if (numOpaque <= 2) return sorted;

  const idMap = new Map();
  for (let i = 0; i < k; i++) idMap.set(sorted[i].word, i);
  const u32 = wordsOf(rgba);
  const idx = new Uint8Array(width * height);
  for (let i = 0; i < idx.length; i++) idx[i] = idMap.get(u32[i]);

  const W = new Float64Array(k * k);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 1; x < width; x++) {
      const a = idx[row + x - 1], b = idx[row + x];
      if (a !== b) { W[a * k + b] += 1; W[b * k + a] += 1; }
    }
    if (y > 0) {
      const prevRow = row - width;
      for (let x = 0; x < width; x++) {
        const a = idx[prevRow + x], b = idx[row + x];
        if (a !== b) { W[a * k + b] += 1; W[b * k + a] += 1; }
      }
    }
  }

  const used = new Uint8Array(k);
  const seq = new Int32Array(numOpaque);
  let first = numTrans, bestDeg = -1;
  for (let u = numTrans; u < k; u++) {
    let deg = 0;
    for (let v = numTrans; v < k; v++) deg += W[u * k + v];
    if (deg > bestDeg) { bestDeg = deg; first = u; }
  }
  seq[0] = first; used[first] = 1;
  for (let s = 1; s < numOpaque; s++) {
    const last = seq[s - 1];
    let bestU = -1, bestScore = -1;
    for (let u = numTrans; u < k; u++) {
      if (used[u]) continue;
      let sc = W[last * k + u] * 4;
      if (s >= 2) sc += W[seq[s - 2] * k + u] * 2;
      if (s >= 3) sc += W[seq[s - 3] * k + u];
      if (sc > bestScore) { bestScore = sc; bestU = u; }
    }
    seq[s] = bestU; used[bestU] = 1;
  }

  const adj = Array.from({ length: k }, () => []);
  for (let u = numTrans; u < k; u++) {
    for (let v = numTrans; v < k; v++) if (W[u * k + v] > 0) adj[u].push([v, W[u * k + v]]);
  }
  const pos = new Int32Array(k);
  for (let s = 0; s < numOpaque; s++) pos[seq[s]] = s;
  const logTbl = new Float64Array(numOpaque + 1);
  for (let d = 0; d <= numOpaque; d++) logTbl[d] = Math.log2(1 + d);

  const nodeCost = (u, p) => {
    let c = 0;
    const list = adj[u];
    for (let i = 0; i < list.length; i++) {
      const d = p - pos[list[i][0]];
      c += list[i][1] * logTbl[d < 0 ? -d : d];
    }
    return c;
  };

  for (let pass = 0; pass < 3; pass++) {
    let improved = false;
    for (let i = 0; i < numOpaque; i++) {
      for (let j = i + 1; j < numOpaque; j++) {
        const u = seq[i], v = seq[j];
        const oldC = nodeCost(u, i) + nodeCost(v, j);
        pos[u] = j; pos[v] = i;
        const newC = nodeCost(u, j) + nodeCost(v, i);
        if (newC + 1e-6 < oldC) { seq[i] = v; seq[j] = u; improved = true; }
        else { pos[u] = i; pos[v] = j; }
      }
    }
    if (!improved) break;
  }

  const out = new Array(k);
  for (let i = 0; i < numTrans; i++) out[i] = sorted[i];
  for (let s = 0; s < numOpaque; s++) out[numTrans + s] = sorted[seq[s]];
  return out;
}

function buildPaletteFromEntries(rgba, entries) {
  const index = new Map(entries.map((c, i) => [c.word, i]));
  const plte = new Uint8Array(entries.length * 3);
  let transparent = 0;
  for (let i = 0; i < entries.length; i++) {
    const c = entries[i];
    plte[i * 3] = c.rgba[0]; plte[i * 3 + 1] = c.rgba[1]; plte[i * 3 + 2] = c.rgba[2];
    if (c.rgba[3] !== 255) transparent = i + 1;
  }
  const trns = transparent === 0 ? null : Uint8Array.from(entries.slice(0, transparent), (c) => c.rgba[3]);
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  const indices = new Uint8Array(n);
  for (let i = 0; i < n; i++) indices[i] = index.get(u32[i]);
  return { plte, trns, indices, entries };
}

function paletteForm(rgba, width, height, colours, order) {
  let entries;
  if (order === 'cooccur') entries = cooccurrenceOrder(rgba, width, height, colours);
  else {
    const lum = (c) => 299 * c.rgba[0] + 587 * c.rgba[1] + 114 * c.rgba[2];
    entries = [...colours].sort((x, y) => {
      const ta = x.rgba[3] === 255 ? 1 : 0;
      const tb = y.rgba[3] === 255 ? 1 : 0;
      if (ta !== tb) return ta - tb;
      if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
      if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
      return y.count - x.count || x.word - y.word;
    });
  }
  return buildPaletteFromEntries(rgba, entries);
}

/**
 * Re-order opaque palette entries by actual LZ77 literal frequency descending!
 * Aligns monotonic Huffman code lengths across 0..K-1, shrinking Dynamic Huffman headers on every block.
 */
function paletteLzFreqForm(rgba, width, height, baseForm) {
  const rowLen = width + 1;
  const raw0 = new Uint8Array(height * rowLen);
  const edgeCnt = new Int32Array(256);
  for (let y = 0; y < height; y++) {
    const row = baseForm.samples.subarray(y * width, (y + 1) * width);
    raw0.set(row, y * rowLen + 1);
    edgeCnt[row[0]]++;
    edgeCnt[row[width - 1]]++;
  }
  const litFreq = fastLzLiteralCounts(raw0, rowLen);
  const entries = baseForm.entries;
  const k = entries.length;
  let numTrans = 0;
  while (numTrans < k && entries[numTrans].rgba[3] !== 255) numTrans++;
  const opaqueIdx = [];
  for (let i = numTrans; i < k; i++) opaqueIdx.push(i);

  // If no transparent entries, lock index 0 to the dominant border color so filter-byte 0 merges with scanline edge runs!
  let borderIdx = -1;
  if (numTrans === 0 && opaqueIdx.length > 1) {
    let bestE = -1;
    for (const i of opaqueIdx) {
      if (edgeCnt[i] > bestE) { bestE = edgeCnt[i]; borderIdx = i; }
    }
  }
  opaqueIdx.sort((a, b) => {
    if (a === borderIdx) return -1;
    if (b === borderIdx) return 1;
    return litFreq[b] - litFreq[a] || entries[b].count - entries[a].count || entries[a].word - entries[b].word;
  });

  const reordered = new Array(k);
  for (let i = 0; i < numTrans; i++) reordered[i] = entries[i];
  for (let i = 0; i < opaqueIdx.length; i++) reordered[numTrans + i] = entries[opaqueIdx[i]];
  const p = buildPaletteFromEntries(rgba, reordered);
  return { name: 'pal-lzfreq', colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, entries: p.entries };
}

export function formsOf(rgba, width, height) {
  const { opaque, grey, colours } = analyse(rgba);
  const forms = [];
  const key = opaque ? null : transparentKey(rgba, grey);
  // On lossless Type-0 greyscale images, Type 0 (1 bpp, zero PLTE overhead, natural luma order) always dominates Type 3!
  if (colours !== null && !(grey && (opaque || key !== null))) {
    const pCount = paletteForm(rgba, width, height, colours, 'count');
    const fCount = { name: 'pal-count', colourType: 3, bpp: 1, samples: pCount.indices, plte: pCount.plte, trns: pCount.trns, entries: pCount.entries };
    forms.push(fCount);
    forms.push(paletteLzFreqForm(rgba, width, height, fCount));
    for (const order of ['luminance', 'cooccur']) {
      const p = paletteForm(rgba, width, height, colours, order);
      forms.push({ name: `pal-${order}`, colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, entries: p.entries });
    }
    if (!grey) return forms;
  }
  let ct;
  let trns = null;
  if (grey) {
    ct = opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    ct = opaque || key !== null ? 2 : 6;
    if (key !== null) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[ct];
  forms.push({ name: `ct${ct}`, colourType: ct, bpp, samples: samplesOf(rgba, ct), plte: null, trns });
  return forms;
}

function signedSum(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) { const v = row[i]; s += v < 128 ? v : 256 - v; }
  return s;
}

function pairsOf(row, seen, stamp) {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const p = (row[i - 1] << 8) | row[i];
    if (seen[p] !== stamp) { seen[p] = stamp; n++; }
  }
  return n;
}

function rowEntropy(row, counts) {
  counts.fill(0);
  const n = row.length;
  for (let i = 0; i < n; i++) counts[row[i]]++;
  let e = 0;
  for (let i = 0; i < 256; i++) { const c = counts[i]; if (c > 0) e -= c * Math.log2(c / n); }
  return e;
}

function buildRowBank(samples, w, h, bpp, includeEntropy = true) {
  const stride = w * bpp;
  const rowLen = stride + 1;
  const numModes = includeEntropy ? 8 : 7;
  const modes = Array.from({ length: numModes }, () => new Uint8Array(h * rowLen));
  const seen = new Uint32Array(65536);
  const counts = includeEntropy ? new Int32Array(256) : null;
  let stamp = 0;
  for (let y = 0; y < h; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * rowLen;
    let bestP = 0, bestPC = Infinity;
    let bestM = 0, bestMC = Infinity;
    let bestE = 0, bestEC = Infinity;
    for (let t = 0; t < 5; t++) {
      modes[t][at] = t;
      const dst = modes[t].subarray(at + 1, at + rowLen);
      filterRow(t, cur, prev, bpp, dst);
      const pc = pairsOf(dst, seen, ++stamp);
      if (pc < bestPC) { bestPC = pc; bestP = t; }
      const mc = signedSum(dst);
      if (mc < bestMC) { bestMC = mc; bestM = t; }
      if (includeEntropy) {
        const ec = rowEntropy(dst, counts);
        if (ec < bestEC) { bestEC = ec; bestE = t; }
      }
    }
    modes[5].set(modes[bestP].subarray(at, at + rowLen), at);
    modes[6].set(modes[bestM].subarray(at, at + rowLen), at);
    if (includeEntropy) modes[7].set(modes[bestE].subarray(at, at + rowLen), at);
  }
  return modes;
}

// Fast lookup tables for pure-JS DEFLATE bit-cost estimation (zero zlib syscalls!)
const EST_LEN_SYM = new Uint16Array(259);
const EST_LEN_EX = new Uint8Array(259);
{
  const LB = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
  const LE = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
  for (let s = 0; s < 29; s++) {
    const nxt = s === 28 ? 259 : LB[s + 1];
    for (let l = LB[s]; l < nxt; l++) { EST_LEN_SYM[l] = 257 + s; EST_LEN_EX[l] = LE[s]; }
  }
}
const EST_DIST_SYM = new Uint8Array(32769);
const EST_DIST_EX = new Uint8Array(30);
{
  const DB = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
  const DE = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
  for (let s = 0; s < 30; s++) {
    EST_DIST_EX[s] = DE[s];
    const nxt = s === 29 ? 32769 : DB[s + 1];
    for (let d = DB[s]; d < nxt; d++) EST_DIST_SYM[d] = s;
  }
}
const EST_LOG2 = new Float64Array(65537);
for (let i = 1; i <= 65536; i++) EST_LOG2[i] = Math.log2(i);
const fastLog2 = (v) => (v <= 65536 ? EST_LOG2[v] : Math.log2(v));

const H_BITS = 15;
const H_SIZE = 1 << H_BITS;
const H_MASK = H_SIZE - 1;
const W_SIZE = 32768;
const W_MASK = W_SIZE - 1;

function hash3Buf(buf, p) {
  return (((buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16)) * 0x1e35a7bd) >>> (32 - H_BITS)) & H_MASK;
}

/**
 * Pure-JS zero-syscall incremental per-row LZ77 + adaptive literal-entropy filter selector!
 * Directly models 32 KiB sliding-window LZ77 match savings + global literal Huffman entropy
 * without making ANY native zlib.deflateSync calls (zero gVisor syscall overhead).
 */
function filterLzRow(rowBank, w, h, bpp, out) {
  const rowLen = w * bpp + 1;
  const head = new Int32Array(H_SIZE).fill(-1);
  const prev = new Int32Array(W_SIZE).fill(-1);
  const hist = new Int32Array(256);
  let histTot = 0;

  const rowHead = new Int32Array(H_SIZE);
  const rowStamp = new Uint32Array(H_SIZE);
  let stamp = 0;

  // Approximate distance cost (symbol + extra bits)
  const distCost = new Float64Array(30);
  for (let s = 0; s < 30; s++) distCost[s] = 4.2 + EST_DIST_EX[s];
  const litCost = new Float64Array(256);

  for (let y = 0; y < h; y++) {
    const at = y * rowLen;
    const limit = at > W_SIZE ? at - W_SIZE : 0;
    const logDen = fastLog2(histTot + 256);
    for (let i = 0; i < 256; i++) litCost[i] = logDen - fastLog2(hist[i] + 1);

    let bestT = 0;
    let bestBits = Infinity;

    for (let t = 0; t < 5; t++) {
      const row = rowBank[t].subarray(at, at + rowLen);
      stamp++;
      let bits = litCost[t]; // filter byte
      let x = 1;
      while (x < rowLen) {
        const b0 = row[x];
        const rem = rowLen - x;
        if (rem >= 3) {
          let bestL = 2;
          let bestD = 0;
          // RLE dist=1
          if (x > 1 && b0 === row[x - 1]) {
            let l = 1;
            const lim = rem < 258 ? rem : 258;
            while (l < lim && row[x + l] === b0) l++;
            if (l > bestL) { bestL = l; bestD = 1; }
          }
          // RLE dist=bpp
          if (bpp > 1 && x > bpp && b0 === row[x - bpp] && bestL < 258) {
            let l = 1;
            const lim = rem < 258 ? rem : 258;
            while (l < lim && row[x + l] === row[x - bpp + l]) l++;
            if (l > bestL) { bestL = l; bestD = bpp; }
          }
          // Same-row & previous-rows hash match (skip when RLE already found >=16 bytes)
          if (bestL < 16) {
            const h3 = hash3Buf(row, x);
            if (rowStamp[h3] === stamp) {
              const rx = rowHead[h3];
              const lim = rem < 258 ? rem : 258;
              if (row[rx] === b0 && row[rx + 1] === row[x + 1] && row[rx + 2] === row[x + 2]) {
                let l = 3;
                while (l < lim && row[rx + l] === row[x + l]) l++;
                if (l > bestL) { bestL = l; bestD = x - rx; }
              }
            }
            if (at > 0 && bestL < 16) {
              const b1 = row[x + 1], b2 = row[x + 2];
              const lim = rem < 258 ? rem : 258;
              let cand = head[h3];
              let probes = 4;
              while (cand >= limit && probes-- > 0) {
                const cEnd = at - cand;
                const maxL = cEnd < lim ? cEnd : lim;
                if (maxL > bestL && out[cand + bestL] === row[x + bestL] && out[cand] === b0 && out[cand + 1] === b1 && out[cand + 2] === b2) {
                  let l = 3;
                  while (l < maxL && out[cand + l] === row[x + l]) l++;
                  if (l > bestL) {
                    bestL = l;
                    bestD = (at + x) - cand;
                    if (bestL >= 16) break;
                  }
                }
                const nxt = prev[cand & W_MASK];
                if (nxt >= cand) break;
                cand = nxt;
              }
            }
            rowStamp[h3] = stamp;
            rowHead[h3] = x;
          }

          if (bestL >= 3) {
            const mBits = 6.2 + EST_LEN_EX[bestL] + distCost[EST_DIST_SYM[bestD]];
            // Compare against emitting literals if short match
            let litSum = 0;
            if (bestL <= 5) {
              for (let k = 0; k < bestL; k++) litSum += litCost[row[x + k]];
            } else {
              litSum = 1e9;
            }
            if (mBits < litSum) {
              bits += mBits;
              x += bestL;
              continue;
            }
          }
        }
        bits += litCost[b0];
        x++;
      }
      if (bits < bestBits) { bestBits = bits; bestT = t; }
    }

    const winRow = rowBank[bestT].subarray(at, at + rowLen);
    out.set(winRow, at);
    // Commit row into 32 KiB sliding hash & adaptive histogram
    for (let x = 0; x < rowLen; x++) {
      const p = at + x;
      hist[out[p]]++;
      if (x + 2 < rowLen) {
        const h3 = hash3Buf(out, p);
        prev[p & W_MASK] = head[h3];
        head[h3] = p;
      }
    }
    histTot += rowLen;
    if (histTot > 65536) {
      let s = 0;
      for (let i = 0; i < 256; i++) { hist[i] >>>= 1; s += hist[i]; }
      histTot = s;
    }
  }
  return out;
}

/**
 * Pure-JS zero-syscall banded filter selector!
 * Evaluates all candidate filter modes per band using fast greedy LZ77 + joint Shannon Huffman
 * entropy over (preceding 16 KiB context + band), followed by pure-JS row-level boundary
 * refinement between adjacent segments. Zero zlib.deflateSync calls!
 */
function filterBandPureJs(rowBank, w, h, bpp, out, bandH, hysteresis = 16) {
  const rowLen = w * bpp + 1;
  const numModes = rowBank.length;
  const maxWin = Math.min(16384, Math.floor(16384 / rowLen) * rowLen);
  const numBands = Math.ceil(h / bandH);
  const bandModes = new Uint8Array(numBands);

  const ctxHead = new Int32Array(H_SIZE);
  const ctxStamp = new Uint32Array(H_SIZE);
  const ctxPrev = new Int32Array(W_SIZE);
  const bandHead = new Int32Array(H_SIZE);
  const bandStamp = new Uint32Array(H_SIZE);
  const bandPrev = new Int32Array(W_SIZE);
  let cStamp = 0;
  let bStamp = 0;

  const baseLL = new Int32Array(286);
  const baseD = new Int32Array(30);
  const llFreq = new Int32Array(286);
  const dFreq = new Int32Array(30);

  // Fast greedy LZ77 accumulating token stats on `src[0..len]` with optional read-only context `ctx[0..ctxLen]`
  const parseSlice = (ctx, ctxLen, src, srcLen, llOut, dOut, hasCtx) => {
    bStamp++;
    const bs = bStamp;
    const cs = cStamp;
    let extra = 0;
    let llTot = 0;
    let dTot = 0;

    for (let p = 0; p < srcLen; ) {
      const rem = srcLen - p;
      const b0 = src[p];
      if (rem >= 3) {
        let bestL = 2;
        let bestD = 0;
        const maxL = rem < 258 ? rem : 258;
        // Period-1 RLE
        if ((p > 0 ? src[p - 1] : (ctxLen > 0 ? ctx[ctxLen - 1] : -1)) === b0) {
          let l = 1;
          while (l < maxL && src[p + l] === b0) l++;
          if (l > bestL) { bestL = l; bestD = 1; }
        }
        // Period-bpp RLE
        if (bpp > 1 && p >= bpp && src[p - bpp] === b0 && bestL < maxL) {
          let l = 1;
          while (l < maxL && src[p + l] === src[p - bpp + l]) l++;
          if (l > bestL) { bestL = l; bestD = bpp; }
        }
        // Period-rowLen (above row in same slice)
        if (p >= rowLen && src[p - rowLen] === b0 && bestL < maxL) {
          let l = 1;
          while (l < maxL && src[p + l] === src[p - rowLen + l]) l++;
          if (l > bestL) { bestL = l; bestD = rowLen; }
        }

        const h3 = hash3Buf(src, p);
        if (bestL < 64) {
          const b1 = src[p + 1], b2 = src[p + 2];
          // Check in-band hash chain (up to 8 probes)
          if (bandStamp[h3] === bs) {
            let cand = bandHead[h3];
            const bLim = p > W_SIZE ? p - W_SIZE : 0;
            let probes = 8;
            while (cand >= bLim && probes-- > 0) {
              if (src[cand + bestL] === src[p + bestL] && src[cand] === b0 && src[cand + 1] === b1 && src[cand + 2] === b2) {
                let l = 3;
                while (l < maxL && src[cand + l] === src[p + l]) l++;
                if (l > bestL) { bestL = l; bestD = p - cand; if (bestL >= 64) break; }
              }
              const nxt = bandPrev[cand & W_MASK];
              if (nxt >= cand) break;
              cand = nxt;
            }
          }
          // Check preceding context hash chain (up to 6 probes)
          if (hasCtx && bestL < 64 && ctxStamp[h3] === cs) {
            let cand = ctxHead[h3];
            const minCtx = p + ctxLen > W_SIZE ? p + ctxLen - W_SIZE : 0;
            let probes = 6;
            while (cand >= minCtx && probes-- > 0) {
              const avail = ctxLen - cand;
              const lim = avail < maxL ? avail : maxL;
              if (lim > bestL && ctx[cand + bestL] === src[p + bestL] && ctx[cand] === b0 && ctx[cand + 1] === b1 && ctx[cand + 2] === b2) {
                let l = 3;
                while (l < lim && ctx[cand + l] === src[p + l]) l++;
                if (l > bestL) { bestL = l; bestD = (ctxLen + p) - cand; if (bestL >= 64) break; }
              }
              const nxt = ctxPrev[cand & W_MASK];
              if (nxt >= cand) break;
              cand = nxt;
            }
          }
        }

        // Insert current position into band hash
        bandPrev[p & W_MASK] = bandStamp[h3] === bs ? bandHead[h3] : -1;
        bandStamp[h3] = bs;
        bandHead[h3] = p;

        if (bestL >= 3) {
          llOut[EST_LEN_SYM[bestL]]++; llTot++;
          const ds = EST_DIST_SYM[bestD];
          dOut[ds]++; dTot++;
          extra += EST_LEN_EX[bestL] + EST_DIST_EX[ds];
          const end = p + bestL;
          if (bestL <= 24) {
            for (let q = p + 1; q < end && q + 2 < srcLen; q++) {
              const hq = hash3Buf(src, q);
              bandPrev[q & W_MASK] = bandStamp[hq] === bs ? bandHead[hq] : -1;
              bandStamp[hq] = bs;
              bandHead[hq] = q;
            }
          }
          p = end;
          continue;
        }
      }
      llOut[b0]++; llTot++;
      p++;
    }
    return { extra, llTot, dTot };
  };

  const entropyBits = (ll, d, llTot, dTot, extra) => {
    let bits = extra + 64;
    const logLL = fastLog2(llTot + 1);
    let actLL = 0;
    for (let i = 0; i < 286; i++) {
      const c = ll[i];
      if (c > 0) { actLL++; const l = logLL - fastLog2(c); bits += c * (l > 1 ? l : 1); }
    }
    let actD = 0;
    if (dTot > 0) {
      const logD = fastLog2(dTot);
      for (let i = 0; i < 30; i++) {
        const c = d[i];
        if (c > 0) { actD++; const l = logD - fastLog2(c); bits += c * (l > 1 ? l : 1); }
      }
    }
    return bits + actLL * 2.2 + actD * 2.8;
  };

  for (let bi = 0; bi < numBands; bi++) {
    const y0 = bi * bandH;
    const bh = Math.min(bandH, h - y0);
    const at = y0 * rowLen;
    const bBytes = bh * rowLen;
    const wLen = Math.min(at, maxWin);
    const ctx = wLen > 0 ? out.subarray(at - wLen, at) : null;

    // Index preceding context ONCE per band + get its base token stats
    baseLL.fill(0);
    baseD.fill(0);
    baseLL[256] = 1;
    let baseExtra = 0, baseLLTot = 1, baseDTot = 0;
    if (wLen > 0) {
      cStamp++;
      const cs = cStamp;
      for (let p = 0; p + 2 < wLen; p++) {
        const h3 = hash3Buf(ctx, p);
        ctxPrev[p & W_MASK] = ctxStamp[h3] === cs ? ctxHead[h3] : -1;
        ctxStamp[h3] = cs;
        ctxHead[h3] = p;
      }
      const st = parseSlice(null, 0, ctx, wLen, baseLL, baseD, false);
      baseExtra = st.extra;
      baseLLTot += st.llTot;
      baseDTot += st.dTot;
    }

    const prevMode = bi > 0 ? bandModes[bi - 1] : -1;
    let bestScore = Infinity;
    let bestMode = 0;
    for (let mode = 0; mode < numModes; mode++) {
      if (mode === 3) continue;
      llFreq.set(baseLL);
      dFreq.set(baseD);
      const st = parseSlice(ctx, wLen, rowBank[mode].subarray(at, at + bBytes), bBytes, llFreq, dFreq, wLen > 0);
      const bits = entropyBits(llFreq, dFreq, baseLLTot + st.llTot, baseDTot + st.dTot, baseExtra + st.extra);
      const score = mode === prevMode ? bits - hysteresis : bits;
      if (score < bestScore) { bestScore = score; bestMode = mode; }
    }
    bandModes[bi] = bestMode;
    out.set(rowBank[bestMode].subarray(at, at + bBytes), at);
  }

  // Merge adjacent bands with identical modes into segments [y0, y1, mode]
  const segs = [];
  for (let bi = 0; bi < numBands; bi++) {
    const y0 = bi * bandH;
    const y1 = Math.min(h, (bi + 1) * bandH);
    const m = bandModes[bi];
    if (segs.length > 0 && segs[segs.length - 1].mode === m) segs[segs.length - 1].y1 = y1;
    else segs.push({ y0, y1, mode: m });
  }

  // Pure-JS coarse-to-fine row-level boundary refinement between adjacent segments (zero syscalls!)
  const halfSpan = Math.min(bandH, 20);
  const spanWork = new Uint8Array((halfSpan * 2 + 2) * rowLen);
  for (let si = 0; si < segs.length - 1; si++) {
    const sA = segs[si], sB = segs[si + 1];
    const yMid = sA.y1;
    const yStart = Math.max(sA.y0, yMid - halfSpan);
    const yEnd = Math.min(sB.y1, yMid + halfSpan);
    const spanH = yEnd - yStart;
    if (spanH < 6) continue;
    const atStart = yStart * rowLen;
    const spanBytes = spanH * rowLen;
    const bankA = rowBank[sA.mode];
    const bankB = rowBank[sB.mode];

    const evalSplit = (splitY) => {
      const k = splitY - yStart;
      if (k > 0) spanWork.set(bankA.subarray(atStart, atStart + k * rowLen), 0);
      if (k < spanH) spanWork.set(bankB.subarray(atStart + k * rowLen, yEnd * rowLen), k * rowLen);
      llFreq.fill(0); dFreq.fill(0); llFreq[256] = 1;
      const st = parseSlice(null, 0, spanWork, spanBytes, llFreq, dFreq, false);
      return entropyBits(llFreq, dFreq, 1 + st.llTot, st.dTot, st.extra);
    };

    let bestY = yMid;
    let bestC = evalSplit(yMid);
    for (let y = yStart + 4; y <= yEnd - 4; y += 5) {
      if (y === yMid) continue;
      const c = evalSplit(y);
      if (c < bestC) { bestC = c; bestY = y; }
    }
    const center = bestY;
    for (const dy of [-2, -1, 1, 2]) {
      const y = center + dy;
      if (y > yStart && y < yEnd) {
        const c = evalSplit(y);
        if (c < bestC) { bestC = c; bestY = y; }
      }
    }
    if (bestY !== yMid) {
      const k = bestY - yStart;
      if (k > 0) out.set(bankA.subarray(atStart, atStart + k * rowLen), atStart);
      if (k < spanH) out.set(bankB.subarray(atStart + k * rowLen, yEnd * rowLen), atStart + k * rowLen);
      sA.y1 = bestY;
      sB.y0 = bestY;
    }
  }
  return out;
}

const chunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) +
  (form.trns === null ? 0 : 12 + form.trns.length);

const SCREEN_DEF = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const SCREEN_FAST = Object.freeze({ level: 4, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const RANK_DEF = Object.freeze({ level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const RANK_FILT = Object.freeze({ level: 9, memLevel: 8, strategy: constants.Z_FILTERED });

export function screenCandidates(rgba, width, height, topK = 2) {
  const forms = formsOf(rgba, width, height);
  const pool = [];
  let maxPool = Math.max(2, topK + 1);

  const consider = (form, choice, data, zlibBuf, bytes, clone = true) => {
    for (let i = 0; i < pool.length; i++) {
      if (pool[i].bytes === bytes && pool[i].form === form) {
        const a = pool[i].data;
        if (a.length === data.length) {
          let same = true;
          for (let j = 0; j < a.length; j += 127) {
            if (a[j] !== data[j]) { same = false; break; }
          }
          if (same) return;
        }
      }
    }
    if (pool.length < maxPool || bytes < pool[pool.length - 1].bytes) {
      pool.push({
        form,
        choice,
        data: clone ? data.slice() : data,
        zlib: new Uint8Array(zlibBuf.buffer, zlibBuf.byteOffset, zlibBuf.length),
        bytes,
      });
      pool.sort((a, b) => a.bytes - b.bytes);
      if (pool.length > maxPool) pool.length = maxPool;
    }
  };

  for (const form of forms) {
    const size = height * (width * form.bpp + 1);
    const chunks = chunkBytes(form);

    if (form.colourType === 3) {
      const bank = buildRowBank(form.samples, width, height, 1, false);
      const modes = form.name === 'pal-count' || form.name === 'pal-lzfreq' ? [0, 1, 2, 5] : [0, 1, 5, 6];
      const names = ['0', '1', '2', '3', '4', 'pairs', 'minsum'];
      for (const m of modes) {
        const z = deflateSync(bank[m], SCREEN_DEF);
        consider(form, names[m], bank[m], z, z.length + chunks, false);
      }
      continue;
    }

    const bank = buildRowBank(form.samples, width, height, form.bpp, false);
    // Pure-JS zero-syscall per-row LZ+entropy filter (replaces slow 3,600-syscall filterZlibBrute!)
    const lzRowBuf = new Uint8Array(size);
    filterLzRow(bank, width, height, form.bpp, lzRowBuf);
    bank.push(lzRowBuf);

    // Fast deterministic density probe on Paeth to avoid slow level-6 screening on huge noisy photos (>220 KB)
    let isHuge = false;
    if (form.bpp >= 3) {
      const zp = deflateSync(bank[4], SCREEN_FAST);
      if (zp.length > 220_000) {
        isHuge = true;
        maxPool = Math.max(2, topK);
        consider(form, '4', bank[4], zp, zp.length + chunks, false);
      }
    }

    const screenOpt = isHuge ? SCREEN_FAST : SCREEN_DEF;
    const names = ['0', '1', '2', '3', '4', 'pairs', 'minsum', 'lzrow'];
    for (let m = 0; m < bank.length; m++) {
      if (isHuge && (m === 4 || m === 3)) continue;
      const z = deflateSync(bank[m], screenOpt);
      consider(form, names[m], bank[m], z, z.length + chunks, false);
    }

    // Single fast pure-JS zero-syscall hybrid banded filter pass (skip on huge noisy photos where lzrow wins!)
    if (!isHuge) {
      const bh = form.bpp === 1 ? 24 : 40;
      const buf = new Uint8Array(size);
      filterBandPureJs(bank, width, height, form.bpp, buf, bh, 16);
      const zb = deflateSync(buf, screenOpt);
      consider(form, `band${bh}`, buf, zb, zb.length + chunks, true);
    }
  }

  // Re-rank top 3 pool entries by min(z9_default, z9_filtered when dense)
  for (let i = 0; i < pool.length; i++) {
    const c = pool[i];
    const ch = chunkBytes(c.form);
    const zd = deflateSync(c.data, RANK_DEF);
    let bestZ = zd;
    if (zd.length > 75_000) {
      const zf = deflateSync(c.data, RANK_FILT);
      if (zf.length < bestZ.length) bestZ = zf;
    }
    const bestB = bestZ.length + ch;
    if (bestB < c.bytes) {
      c.zlib = new Uint8Array(bestZ.buffer, bestZ.byteOffset, bestZ.length);
      c.bytes = bestB;
    }
  }
  pool.sort((a, b) => a.bytes - b.bytes);
  return pool.slice(0, topK);
}
