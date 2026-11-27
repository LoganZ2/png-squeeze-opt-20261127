import { constants, deflateSync } from 'node:zlib';
import { filterRow } from './png.mjs';

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

function paletteOrder(rgba, width, height, colours, order) {
  if (order === 'cooccur') return cooccurrenceOrder(rgba, width, height, colours);
  const lum = (c) => 299 * c.rgba[0] + 587 * c.rgba[1] + 114 * c.rgba[2];
  return [...colours].sort((x, y) => {
    const ta = x.rgba[3] === 255 ? 1 : 0;
    const tb = y.rgba[3] === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
}

function paletteForm(rgba, width, height, colours, order) {
  const entries = paletteOrder(rgba, width, height, colours, order);
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
  return { plte, trns, indices };
}

export function formsOf(rgba, width, height) {
  const { opaque, grey, colours } = analyse(rgba);
  const forms = [];
  if (colours !== null) {
    for (const order of ['count', 'luminance', 'cooccur']) {
      const p = paletteForm(rgba, width, height, colours, order);
      forms.push({ name: `pal-${order}`, colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns });
    }
    if (!grey) return forms;
  }
  const key = opaque ? null : transparentKey(rgba, grey);
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

function buildRowBank(samples, w, h, bpp) {
  const stride = w * bpp;
  const rowLen = stride + 1;
  const modes = Array.from({ length: 7 }, () => new Uint8Array(h * rowLen));
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < h; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * rowLen;
    let bestP = 0, bestPC = Infinity;
    let bestM = 0, bestMC = Infinity;
    for (let t = 0; t < 5; t++) {
      modes[t][at] = t;
      const dst = modes[t].subarray(at + 1, at + rowLen);
      filterRow(t, cur, prev, bpp, dst);
      const pc = pairsOf(dst, seen, ++stamp);
      if (pc < bestPC) { bestPC = pc; bestP = t; }
      const mc = signedSum(dst);
      if (mc < bestMC) { bestMC = mc; bestM = t; }
    }
    modes[5].set(modes[bestP].subarray(at, at + rowLen), at);
    modes[6].set(modes[bestM].subarray(at, at + rowLen), at);
  }
  return modes;
}

/**
 * Fast hybrid banded filter selector with bounded local-window boundary refinement!
 * Uses level 4 for band choices (~120 ms) and a compact local window (+/- 24 rows, coarse-to-fine ~25 ms)
 * for exact row-level transition alignment!
 */
function filterBandHybrid(rowBank, w, h, bpp, out, bandH, winBytes = 32768, level = 4) {
  const rowLen = w * bpp + 1;
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

    let bestLen = Infinity;
    let bestMode = 0;
    for (let mode = 0; mode < 7; mode++) {
      work.set(rowBank[mode].subarray(at, at + bBytes), wLen);
      const len = deflateSync(slice, { level, memLevel: 8 }).length;
      if (len < bestLen) { bestLen = len; bestMode = mode; }
    }
    bandModes[bi] = bestMode;
    out.set(rowBank[bestMode].subarray(at, at + bBytes), at);
  }

  // Fast local boundary refinement with preceding 8 KiB context window
  const halfSpan = Math.min(bandH, 40);
  const ctxMax = Math.floor(8192 / rowLen) * rowLen;
  const spanWork = new Uint8Array(ctxMax + (halfSpan * 2 + 2) * rowLen);
  for (let bi = 0; bi < numBands - 1; bi++) {
    const mA = bandModes[bi];
    const mB = bandModes[bi + 1];
    if (mA === mB) continue;

    const yMid = (bi + 1) * bandH;
    const yStart = Math.max(bi * bandH, yMid - halfSpan);
    const yEnd = Math.min(h, Math.min((bi + 2) * bandH, yMid + halfSpan));
    const spanH = yEnd - yStart;
    if (spanH < 6) continue;
    const atStart = yStart * rowLen;
    const ctxLen = Math.min(atStart, ctxMax);
    if (ctxLen > 0) spanWork.set(out.subarray(atStart - ctxLen, atStart), 0);
    const spanSlice = spanWork.subarray(0, ctxLen + spanH * rowLen);
    const bankA = rowBank[mA];
    const bankB = rowBank[mB];

    const evalSplit = (splitY, lv = 2) => {
      const k = splitY - yStart;
      if (k > 0) spanWork.set(bankA.subarray(atStart, atStart + k * rowLen), ctxLen);
      if (k < spanH) spanWork.set(bankB.subarray(atStart + k * rowLen, yEnd * rowLen), ctxLen + k * rowLen);
      return deflateSync(spanSlice, { level: lv, memLevel: 8 }).length;
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
    }
  }
  return out;
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

const chunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) +
  (form.trns === null ? 0 : 12 + form.trns.length);

const SCREEN_ZLIB = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const RANK_ZLIB = Object.freeze({ level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });

export function screenCandidates(rgba, width, height, topK = 1) {
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
    const maxPool = Math.max(4, topK + 2);
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
      const bank = buildRowBank(form.samples, width, height, 1);
      const modes = form.name === 'pal-count' ? [0, 1, 2, 5, 6] : [0, 1, 5, 6];
      const names = ['0', '1', '2', '3', '4', 'pairs', 'minsum'];
      for (const m of modes) {
        const z = deflateSync(bank[m], SCREEN_ZLIB);
        consider(form, names[m], bank[m], z, z.length + chunks, false);
      }
      continue;
    }

    const bank = buildRowBank(form.samples, width, height, form.bpp);
    const names = ['0', '1', '2', '3', '4', 'pairs', 'minsum'];
    for (let m = 0; m < 7; m++) {
      const z = deflateSync(bank[m], SCREEN_ZLIB);
      consider(form, names[m], bank[m], z, z.length + chunks, false);
    }
    const buf = new Uint8Array(size);
    for (const bh of [40, 72]) {
      filterBandHybrid(bank, width, height, form.bpp, buf, bh, 32768, 4);
      const z = deflateSync(buf, SCREEN_ZLIB);
      consider(form, `band${bh}`, buf, z, z.length + chunks, true);
    }
    // Per-row brute shines on 1-bpp/2-bpp images where 32 KiB spans 12-25 rows!
    if (form.bpp <= 2) {
      filterZlibBrute(bank, width, height, form.bpp, buf, 3, 32768);
      const zb = deflateSync(buf, SCREEN_ZLIB);
      consider(form, 'brute', buf, zb, zb.length + chunks, true);
    }
  }

  for (let i = 0; i < pool.length; i++) {
    const c = pool[i];
    const z9 = deflateSync(c.data, RANK_ZLIB);
    const z9Bytes = z9.length + chunkBytes(c.form);
    if (z9Bytes < c.bytes) {
      c.zlib = new Uint8Array(z9.buffer, z9.byteOffset, z9.length);
      c.bytes = z9Bytes;
    }
  }
  pool.sort((a, b) => a.bytes - b.bytes);
  return pool.slice(0, topK);
}
