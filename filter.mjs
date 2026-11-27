import { constants, deflateSync } from 'node:zlib';
import { filterRow } from './png.mjs';
import {
  fastLzLiteralCounts,
  fastZopfliScreenWasm,
  generateFilterCandidatesWasm,
  optimizePaletteTspWasm,
  scoutZopfliSizeWasm,
} from './zopfli.mjs';

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

function cooccurrenceOrder(rgba, width, height, colours, lockBorderZero = false) {
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
  if (k - numTrans <= 2) return sorted;

  const idMap = new Map();
  for (let i = 0; i < k; i++) idMap.set(sorted[i].word, i);
  const u32 = wordsOf(rgba);
  const idx = new Uint8Array(width * height);
  for (let i = 0; i < idx.length; i++) idx[i] = idMap.get(u32[i]);

  let lockedPrefix = numTrans;
  if (lockBorderZero && numTrans === 0 && k > 2) {
    const edgeCnt = new Int32Array(k);
    for (let y = 0; y < height; y++) {
      edgeCnt[idx[y * width]]++;
      edgeCnt[idx[(y + 1) * width - 1]]++;
    }
    let bIdx = 0, bestE = -1;
    for (let i = 0; i < k; i++) {
      if (edgeCnt[i] > bestE) { bestE = edgeCnt[i]; bIdx = i; }
    }
    if (bIdx !== 0) {
      const tmp = sorted[0]; sorted[0] = sorted[bIdx]; sorted[bIdx] = tmp;
      for (let i = 0; i < idx.length; i++) {
        if (idx[i] === 0) idx[i] = bIdx;
        else if (idx[i] === bIdx) idx[i] = 0;
      }
    }
    lockedPrefix = 1;
  }

  const order = optimizePaletteTspWasm(idx, width, height, k, lockedPrefix);
  const out = new Array(k);
  for (let i = 0; i < k; i++) out[i] = sorted[order[i]];
  return out;
}

function paletteForm(rgba, width, height, colours, order) {
  let entries;
  if (order === 'cooccur') entries = cooccurrenceOrder(rgba, width, height, colours, false);
  else if (order === 'bcooccur') entries = cooccurrenceOrder(rgba, width, height, colours, true);
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

function paletteLzFreqForms(rgba, width, height, baseForm) {
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

  let borderIdx = -1;
  if (numTrans === 0 && opaqueIdx.length > 1) {
    let bestE = -1;
    for (const i of opaqueIdx) {
      if (edgeCnt[i] > bestE) { bestE = edgeCnt[i]; borderIdx = i; }
    }
  }

  const out = [];
  for (const [name, lockBorder] of [['pal-lzfreq', true], ['pal-lzpure', false]]) {
    const ord = [...opaqueIdx].sort((a, b) => {
      if (lockBorder && a === borderIdx) return -1;
      if (lockBorder && b === borderIdx) return 1;
      return litFreq[b] - litFreq[a] || entries[b].count - entries[a].count || entries[a].word - entries[b].word;
    });
    const reordered = new Array(k);
    for (let i = 0; i < numTrans; i++) reordered[i] = entries[i];
    for (let i = 0; i < ord.length; i++) reordered[numTrans + i] = entries[ord[i]];
    const p = buildPaletteFromEntries(rgba, reordered);
    out.push({ name, colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, entries: p.entries });
  }
  return out;
}

export function formsOf(rgba, width, height) {
  const { opaque, grey, colours } = analyse(rgba);
  const forms = [];
  const key = opaque ? null : transparentKey(rgba, grey);
  if (colours !== null && !(grey && (opaque || key !== null) && colours.length > 64)) {
    const pCount = paletteForm(rgba, width, height, colours, 'count');
    const fCount = { name: 'pal-count', colourType: 3, bpp: 1, samples: pCount.indices, plte: pCount.plte, trns: pCount.trns, entries: pCount.entries };
    forms.push(fCount);
    for (const f of paletteLzFreqForms(rgba, width, height, fCount)) forms.push(f);
    for (const order of ['bcooccur', 'cooccur', 'luminance']) {
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

function buildFixedRowMode(samples, w, h, mode) {
  const rowLen = w + 1;
  const out = new Uint8Array(h * rowLen);
  for (let y = 0; y < h; y++) {
    const at = y * rowLen;
    out[at] = mode;
    const cur = samples.subarray(y * w, (y + 1) * w);
    const prev = y === 0 ? null : samples.subarray((y - 1) * w, y * w);
    filterRow(mode, cur, prev, 1, out.subarray(at + 1, at + rowLen));
  }
  return out;
}

const chunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) +
  (form.trns === null ? 0 : 12 + form.trns.length);

const SCREEN_PAL = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const SCREEN_HI = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const SCREEN_LO = Object.freeze({ level: 4, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });

export function screenCandidates(rgba, width, height, topK = 3) {
  const forms = formsOf(rgba, width, height);
  const pool = [];
  const maxPool = Math.max(7, topK + 3);

  const consider = (form, choice, data, estZlibBytes, bytes) => {
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
        data,
        estBytes: estZlibBytes,
        bytes,
      });
      pool.sort((a, b) => a.bytes - b.bytes);
      if (pool.length > maxPool) pool.length = maxPool;
    }
  };

  for (const form of forms) {
    const chunks = chunkBytes(form);
    if (form.colourType === 3) {
      if (form.name === 'pal-count') {
        const cands = generateFilterCandidatesWasm(form.samples, width, height, 1, 2, 4);
        const names = ['eu0', 'band24', '0', '1'];
        for (let m = 0; m < cands.length; m++) {
          const z = deflateSync(cands[m], SCREEN_PAL).length;
          consider(form, names[m], cands[m], z, z + chunks);
        }
      } else if (form.name === 'pal-lzfreq' || form.name === 'pal-lzpure' || form.name === 'pal-bcooccur') {
        const cands = generateFilterCandidatesWasm(form.samples, width, height, 1, 3, 2);
        const names = ['eu0', '0'];
        for (let m = 0; m < cands.length; m++) {
          const z = deflateSync(cands[m], SCREEN_PAL).length;
          consider(form, names[m], cands[m], z, z + chunks);
        }
      } else {
        const modes = form.name === 'pal-cooccur' ? [0, 1] : [0];
        for (const m of modes) {
          const buf = buildFixedRowMode(form.samples, width, height, m);
          const z = deflateSync(buf, SCREEN_PAL).length;
          consider(form, String(m), buf, z, z + chunks);
        }
      }
      continue;
    }

    // Native WASM returns elite finalists including split fine/macro Viterbi DP & adaptive LZ-Row!
    const mode = form.bpp <= 2 ? 1 : 0;
    const cands = generateFilterCandidatesWasm(form.samples, width, height, form.bpp, mode, 6);
    const names = mode === 1
      ? ['band16', 'viterbi', 'lzrow', '0', '4', 'vitM']
      : ['band40', 'viterbi', 'lzrow', '0', '4', 'vitM'];
    const sOpt = form.bpp <= 2 ? SCREEN_HI : SCREEN_LO;
    for (let m = 0; m < cands.length; m++) {
      const z = deflateSync(cands[m], sOpt).length;
      consider(form, names[m] ?? `w${m}`, cands[m], z, z + chunks);
    }
  }

  // Hybrid Zopfli-aligned WASM tie-breaker:
  // - On standard images (estBytes <= 135 KB): ultra-fast lazy+DP-split WASM screen on close non-fixed finalists!
  // - On dense multi-regime 3/4-bpp images (135 KB < estBytes <= 240 KB): lazily generate clean separate-block
  //   band20c and scout #1 vs band20c via native WASM hash3+DP scout so true Zopfli winners always win!
  if (
    pool.length > 1 &&
    pool[0].form.colourType !== 3 &&
    pool[1].bytes <= pool[0].bytes * 1.035
  ) {
    const topScreenBytes = pool[0].bytes;
    if (pool[0].estBytes <= 135_000) {
      const maxScore = pool[0].form.bpp === 1 ? 3 : 4;
      const tol = pool[0].form.bpp === 1 ? 1.02 : 1.035;
      let scoredCount = 0;
      for (let i = 0; i < pool.length && scoredCount < maxScore; i++) {
        const c = pool[i];
        if (i > 0 && c.bytes > topScreenBytes * tol) break;
        if (i > 0 && c.form.bpp > 1 && (c.choice === '0' || c.choice === '4')) continue;
        const z = fastZopfliScreenWasm(c.data, width * c.form.bpp + 1, c.form.bpp);
        c.zopScreen = z + chunkBytes(c.form);
        scoredCount++;
      }
    } else if (pool[0].estBytes <= 240_000 && pool[0].form.bpp >= 3) {
      const form0 = pool[0].form;
      const chunks0 = chunkBytes(form0);
      const extra = generateFilterCandidatesWasm(form0.samples, width, height, form0.bpp, 4, 1);
      if (extra.length > 0) {
        const b20Data = extra[0];
        const b20Z = deflateSync(b20Data, SCREEN_LO).length;
        const b20Bytes = b20Z + chunks0;
        if (b20Bytes <= topScreenBytes * 1.02) {
          const b20Cand = { form: form0, choice: 'band20c', data: b20Data, estBytes: b20Z, bytes: b20Bytes };
          pool.push(b20Cand);
          for (const c of [pool[0], b20Cand]) {
            const z = scoutZopfliSizeWasm(c.data, width * c.form.bpp + 1, c.form.bpp);
            c.zopScreen = z + chunkBytes(c.form);
          }
        }
      }
    }
    const scored = [];
    const unscored = [];
    for (let i = 0; i < pool.length; i++) {
      if (pool[i].zopScreen !== undefined) scored.push(pool[i]);
      else unscored.push(pool[i]);
    }
    if (scored.length > 1) {
      scored.sort((a, b) => a.zopScreen - b.zopScreen);
      pool.length = 0;
      pool.push(...scored, ...unscored);
    }
  }

  return pool.slice(0, topK);
}
