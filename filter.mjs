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

function filterZlibBrute(rowBank, w, h, bpp, out, level = 3, context = 32768) {
  const rowLen = w * bpp + 1;
  for (let y = 0; y < h; y++) {
    const at = y * rowLen;
    const before = out.subarray(Math.max(0, at - context), at);
    let bestT = 0, bestLen = Infinity;
    for (let t = 0; t < 5; t++) {
      const row = rowBank[t].subarray(at, at + rowLen);
      const len = deflateSync(
        row,
        before.length === 0 ? { level, memLevel: 8 } : { level, memLevel: 8, dictionary: before }
      ).length;
      if (len < bestLen) { bestLen = len; bestT = t; }
    }
    out.set(rowBank[bestT].subarray(at, at + rowLen), at);
  }
  return out;
}

/**
 * Hybrid banded filter selector with:
 * - Modes: fixed 0..4, per-row pairs, minsum, entropy, AND per-row brute!
 * - Joint-window zlib selection (preserving shared Huffman trees & LZ77 context)
 * - Exact coarse-to-fine row-level boundary refinement with preceding 8 KiB context!
 * - Final bidirectional segment-mode verification pass (seeing BOTH prev & next segments)!
 */
function filterBandHybrid(rowBank, w, h, bpp, out, bandH, winBytes = 32768, level = 4, strategy = constants.Z_DEFAULT_STRATEGY, hysteresis = 10) {
  const rowLen = w * bpp + 1;
  const numModes = rowBank.length;
  const maxWin = Math.floor(winBytes / rowLen) * rowLen;
  const work = new Uint8Array(maxWin + bandH * rowLen);
  const numBands = Math.ceil(h / bandH);
  const bandModes = new Uint8Array(numBands);

  for (let bi = 0; bi < numBands; bi++) {
    const y0 = bi * bandH;
    const bh = Math.min(bandH, h - y0);
    const at = y0 * rowLen;
    const bBytes = bh * rowLen;
    const wLen = Math.min(at, maxWin);
    if (wLen > 0) work.set(out.subarray(at - wLen, at), 0);
    const slice = work.subarray(0, wLen + bBytes);
    const prevMode = bi > 0 ? bandModes[bi - 1] : -1;

    let bestScore = Infinity;
    let bestMode = 0;
    for (let mode = 0; mode < numModes; mode++) {
      if (mode === 3) continue; // Avg is dominated by Paeth/pairs/minsum inside banded search
      work.set(rowBank[mode].subarray(at, at + bBytes), wLen);
      const len = deflateSync(slice, { level, memLevel: 8, strategy }).length;
      // Small hysteresis bonus to keep previous mode unless switching saves meaningfully
      const score = mode === prevMode ? len - hysteresis : len;
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

  // Exact local boundary refinement between every pair of adjacent segments!
  const halfSpan = Math.min(bandH, 40);
  const ctxMax = Math.floor(8192 / rowLen) * rowLen;
  const spanWork = new Uint8Array(ctxMax + (halfSpan * 2 + 2) * rowLen);

  for (let si = 0; si < segs.length - 1; si++) {
    const sA = segs[si], sB = segs[si + 1];
    const yMid = sA.y1;
    const yStart = Math.max(sA.y0, yMid - halfSpan);
    const yEnd = Math.min(sB.y1, yMid + halfSpan);
    const spanH = yEnd - yStart;
    if (spanH < 6) continue;
    const atStart = yStart * rowLen;
    const ctxLen = Math.min(atStart, ctxMax);
    if (ctxLen > 0) spanWork.set(out.subarray(atStart - ctxLen, atStart), 0);
    const spanSlice = spanWork.subarray(0, ctxLen + spanH * rowLen);
    const bankA = rowBank[sA.mode];
    const bankB = rowBank[sB.mode];

    const evalSplit = (splitY, lv = 2) => {
      const k = splitY - yStart;
      if (k > 0) spanWork.set(bankA.subarray(atStart, atStart + k * rowLen), ctxLen);
      if (k < spanH) spanWork.set(bankB.subarray(atStart + k * rowLen, yEnd * rowLen), ctxLen + k * rowLen);
      return deflateSync(spanSlice, { level: lv, memLevel: 8, strategy }).length;
    };

    let bestY = yMid;
    let bestC = evalSplit(yMid, 2);
    for (let y = yStart + 4; y <= yEnd - 4; y += 6) {
      if (y === yMid) continue;
      const c = evalSplit(y, 2);
      if (c < bestC) { bestC = c; bestY = y; }
    }
    let center = bestY;
    bestC = evalSplit(bestY, 3);
    for (let y = Math.max(yStart + 2, center - 4); y <= Math.min(yEnd - 2, center + 4); y += 2) {
      if (y === bestY) continue;
      const c = evalSplit(y, 3);
      if (c < bestC) { bestC = c; bestY = y; }
    }
    center = bestY;
    for (const dy of [-1, 1]) {
      const y = center + dy;
      if (y > yStart && y < yEnd) {
        const c = evalSplit(y, 3);
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

  // Bidirectional segment-mode re-check at level 3 (capped to <=220 KiB for fast screening!)
  if (segs.length >= 2) {
    const padRows = Math.min(12, Math.max(4, Math.floor(12288 / rowLen)));
    for (let si = 0; si < segs.length; si++) {
      const seg = segs[si];
      if (seg.y1 <= seg.y0) continue;
      const wStart = Math.max(0, seg.y0 - padRows);
      const wEnd = Math.min(h, seg.y1 + padRows);
      const wBytes = (wEnd - wStart) * rowLen;
      if (wBytes > 420 * 1024) continue;
      const buf = out.subarray(wStart * rowLen, wEnd * rowLen).slice();
      const segOff = (seg.y0 - wStart) * rowLen;
      let bestM = seg.mode;
      let bestC = deflateSync(buf, { level: 3, memLevel: 8, strategy }).length;
      for (let m = 0; m < numModes; m++) {
        if (m === seg.mode || m === 3) continue;
        buf.set(rowBank[m].subarray(seg.y0 * rowLen, seg.y1 * rowLen), segOff);
        const c = deflateSync(buf, { level: 3, memLevel: 8, strategy }).length;
        if (c + 4 < bestC) { bestC = c; bestM = m; }
      }
      if (bestM !== seg.mode) {
        seg.mode = bestM;
        out.set(rowBank[bestM].subarray(seg.y0 * rowLen, seg.y1 * rowLen), seg.y0 * rowLen);
      }
    }
  }
  return out;
}

const chunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) +
  (form.trns === null ? 0 : 12 + form.trns.length);

const SCREEN_DEF = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const RANK_DEF = Object.freeze({ level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const RANK_FILT = Object.freeze({ level: 9, memLevel: 8, strategy: constants.Z_FILTERED });

export function screenCandidates(rgba, width, height, topK = 2) {
  const forms = formsOf(rgba, width, height);
  const pool = [];

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
    const maxPool = Math.max(3, topK + 2);
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
    if (form.bpp <= 2) {
      const bruteBuf = new Uint8Array(size);
      filterZlibBrute(bank, width, height, form.bpp, bruteBuf, 3, 32768);
      bank.push(bruteBuf);
    }

    const names = ['0', '1', '2', '3', '4', 'pairs', 'minsum', 'brute'];
    for (let m = 0; m < bank.length; m++) {
      const z = deflateSync(bank[m], SCREEN_DEF);
      consider(form, names[m], bank[m], z, z.length + chunks, false);
    }

    const buf = new Uint8Array(size);
    for (const [bh, hyst] of [[24, 0], [40, 8]]) {
      filterBandHybrid(bank, width, height, form.bpp, buf, bh, 32768, 4, constants.Z_DEFAULT_STRATEGY, hyst);
      const z = deflateSync(buf, SCREEN_DEF);
      consider(form, `band${bh}`, buf, z, z.length + chunks, true);
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
