import { adler32 } from './png.mjs';

const LEN_BASE = Uint16Array.of(
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
  35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
);
const LEN_EXTRA = Uint8Array.of(
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
  3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
);
const DIST_BASE = Uint16Array.of(
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
  257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
);
const DIST_EXTRA = Uint8Array.of(
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
  7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
);
const CL_ORDER = Uint8Array.of(16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15);
const LONG_LEN_STOPS = Uint16Array.of(14, 18, 26, 34, 50, 66, 98, 130, 162, 194, 226);
const FINAL_LEN_STOPS = Uint16Array.of(
  40, 42, 44, 48, 50, 56, 58, 64, 66, 72, 80, 82, 96, 98, 112, 114, 128, 130, 144, 160, 162, 176, 192, 194, 208, 224, 226, 240
);

const LEN_SYM = new Uint16Array(259);
const LEN_EX = new Uint8Array(259);
for (let s = 0; s < 29; s++) {
  const base = LEN_BASE[s];
  const next = s === 28 ? 259 : LEN_BASE[s + 1];
  for (let l = base; l < next; l++) {
    LEN_SYM[l] = 257 + s;
    LEN_EX[l] = LEN_EXTRA[s];
  }
}
LEN_SYM[258] = 285;
LEN_EX[258] = 0;

const DIST_SYM = new Uint8Array(32769);
for (let s = 0; s < 30; s++) {
  const base = DIST_BASE[s];
  const next = s === 29 ? 32769 : DIST_BASE[s + 1];
  for (let d = base; d < next; d++) DIST_SYM[d] = s;
}

const FIXED_LL_LENS = new Uint8Array(288);
for (let i = 0; i <= 143; i++) FIXED_LL_LENS[i] = 8;
for (let i = 144; i <= 255; i++) FIXED_LL_LENS[i] = 9;
for (let i = 256; i <= 279; i++) FIXED_LL_LENS[i] = 7;
for (let i = 280; i <= 287; i++) FIXED_LL_LENS[i] = 8;
const FIXED_D_LENS = new Uint8Array(30).fill(5);

function buildCanonicalCodes(lens) {
  const n = lens.length;
  const blCount = new Int32Array(16);
  for (let i = 0; i < n; i++) blCount[lens[i]]++;
  blCount[0] = 0;
  const nextCode = new Int32Array(16);
  let code = 0;
  for (let bits = 1; bits <= 15; bits++) {
    code = (code + blCount[bits - 1]) << 1;
    nextCode[bits] = code;
  }
  const codes = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const l = lens[i];
    if (l !== 0) {
      let c = nextCode[l]++;
      let r = 0;
      for (let b = 0; b < l; b++) {
        r = (r << 1) | (c & 1);
        c >>>= 1;
      }
      codes[i] = r;
    }
  }
  return codes;
}

const FIXED_LL_CODES = buildCanonicalCodes(FIXED_LL_LENS);
const FIXED_D_CODES = buildCanonicalCodes(FIXED_D_LENS);

export function packageMerge(freqs, maxBits, outLens = new Uint8Array(freqs.length)) {
  outLens.fill(0);
  const syms = [];
  const wts = [];
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] > 0) { syms.push(i); wts.push(freqs[i]); }
  }
  const k = syms.length;
  if (k === 0) return outLens;
  if (k === 1) { outLens[syms[0]] = 1; return outLens; }
  if (k === 2) { outLens[syms[0]] = 1; outLens[syms[1]] = 1; return outLens; }

  const order = new Int32Array(k);
  for (let i = 0; i < k; i++) order[i] = i;
  order.sort((a, b) => wts[a] - wts[b] || syms[a] - syms[b]);
  const sSym = new Int32Array(k);
  const sWt = new Float64Array(k);
  for (let i = 0; i < k; i++) { sSym[i] = syms[order[i]]; sWt[i] = wts[order[i]]; }

  const levelType = new Array(maxBits);
  const levelLeft = new Array(maxBits);
  const levelRight = new Array(maxBits);
  const levelLen = new Int32Array(maxBits);

  let prevWt = sWt;
  let prevCount = k;
  levelType[0] = new Int16Array(k);
  for (let i = 0; i < k; i++) levelType[0][i] = i;
  levelLen[0] = k;

  for (let lv = 1; lv < maxBits; lv++) {
    const pkgCount = prevCount >>> 1;
    const total = k + pkgCount;
    const curWt = new Float64Array(total);
    const curType = new Int16Array(total);
    const curL = new Int16Array(total);
    const curR = new Int16Array(total);

    let li = 0, pi = 0, oi = 0;
    while (li < k && pi < pkgCount) {
      const pw = prevWt[pi * 2] + prevWt[pi * 2 + 1];
      if (sWt[li] <= pw) { curWt[oi] = sWt[li]; curType[oi] = li; li++; }
      else { curWt[oi] = pw; curType[oi] = -1; curL[oi] = pi * 2; curR[oi] = pi * 2 + 1; pi++; }
      oi++;
    }
    while (li < k) { curWt[oi] = sWt[li]; curType[oi] = li; li++; oi++; }
    while (pi < pkgCount) {
      curWt[oi] = prevWt[pi * 2] + prevWt[pi * 2 + 1];
      curType[oi] = -1; curL[oi] = pi * 2; curR[oi] = pi * 2 + 1; pi++; oi++;
    }
    levelType[lv] = curType; levelLeft[lv] = curL; levelRight[lv] = curR; levelLen[lv] = total;
    prevWt = curWt; prevCount = total;
  }

  const leafLens = new Uint8Array(k);
  let curUse = new Int32Array(levelLen[maxBits - 1]);
  const take = 2 * k - 2;
  for (let i = 0; i < take; i++) curUse[i] = 1;

  for (let lv = maxBits - 1; lv >= 1; lv--) {
    const types = levelType[lv];
    const lefts = levelLeft[lv];
    const rights = levelRight[lv];
    const nextUse = new Int32Array(levelLen[lv - 1]);
    const n = levelLen[lv];
    for (let i = 0; i < n; i++) {
      const u = curUse[i];
      if (u === 0) continue;
      const t = types[i];
      if (t >= 0) leafLens[t] += u;
      else { nextUse[lefts[i]] += u; nextUse[rights[i]] += u; }
    }
    curUse = nextUse;
  }
  for (let i = 0; i < k; i++) leafLens[i] += curUse[i];
  for (let i = 0; i < k; i++) outLens[sSym[i]] = leafLens[i];
  return outLens;
}

function optimizeCountsForRle(counts, length) {
  const out = new Int32Array(length);
  for (let i = 0; i < length; i++) out[i] = counts[i];
  let n = length;
  while (n > 0 && out[n - 1] === 0) n--;
  if (n === 0) return out;

  const goodForRle = new Uint8Array(n);
  let symbol = out[0];
  let stride = 0;
  for (let i = 0; i <= n; i++) {
    if (i === n || out[i] !== symbol) {
      if ((symbol === 0 && stride >= 5) || (symbol !== 0 && stride >= 7)) {
        for (let k = 0; k < stride; k++) goodForRle[i - k - 1] = 1;
      }
      stride = 1;
      if (i < n) symbol = out[i];
    } else stride++;
  }

  stride = 0;
  let limit = out[0];
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    if (i === n || goodForRle[i] || (i > 0 && goodForRle[i - 1]) || Math.abs(out[i] - limit) >= 4) {
      if (stride >= 4 || (stride >= 3 && sum === 0)) {
        let count = Math.floor((sum + stride / 2) / stride);
        if (count < 1) count = 1;
        if (sum === 0) count = 0;
        for (let k = 0; k < stride; k++) out[i - k - 1] = count;
      }
      stride = 0; sum = 0;
      if (i < n - 3) limit = Math.floor((out[i] + out[i + 1] + out[i + 2] + out[i + 3] + 2) / 4);
      else if (i < n) limit = out[i];
      else limit = 0;
    }
    stride++;
    if (i < n) sum += out[i];
  }
  return out;
}

function rleEncodeLengths(combined, n, use16, use17) {
  const syms = [];
  const ex = [];
  const clFreq = new Int32Array(19);
  for (let i = 0; i < n; ) {
    const v = combined[i];
    let run = 1;
    while (i + run < n && combined[i + run] === v) run++;
    if (v === 0 && use17 && run >= 3) {
      let rem = run;
      while (rem >= 11) {
        const take = rem > 138 ? 138 : rem;
        const adj = rem - take > 0 && rem - take < 3 && take > 13 ? rem - 3 : take;
        syms.push(18); ex.push(adj - 11); clFreq[18]++; rem -= adj;
      }
      while (rem >= 3) {
        const take = rem > 10 ? 10 : rem;
        syms.push(17); ex.push(take - 3); clFreq[17]++; rem -= take;
      }
      while (rem > 0) { syms.push(0); ex.push(0); clFreq[0]++; rem--; }
    } else if (v !== 0 && use16 && run >= 4) {
      syms.push(v); ex.push(0); clFreq[v]++;
      let rem = run - 1;
      while (rem >= 3) {
        const take = rem > 6 ? 6 : rem;
        const adj = rem - take > 0 && rem - take < 3 && take > 4 ? rem - 3 : take;
        syms.push(16); ex.push(adj - 3); clFreq[16]++; rem -= adj;
      }
      while (rem > 0) { syms.push(v); ex.push(0); clFreq[v]++; rem--; }
    } else {
      for (let k = 0; k < run; k++) { syms.push(v); ex.push(0); clFreq[v]++; }
    }
    i += run;
  }
  return { syms, ex, clFreq };
}

function rleEncodeDP(combined, n, clLens) {
  const dp = new Float64Array(n + 1).fill(Infinity);
  const prevPos = new Int32Array(n + 1);
  const prevSym = new Uint8Array(n + 1);
  const prevEx = new Uint8Array(n + 1);
  dp[0] = 0;

  for (let i = 0; i < n; i++) {
    const base = dp[i];
    if (!Number.isFinite(base)) continue;
    const v = combined[i];
    if (clLens[v] > 0) {
      const c = base + clLens[v];
      if (c < dp[i + 1]) { dp[i + 1] = c; prevPos[i + 1] = i; prevSym[i + 1] = v; prevEx[i + 1] = 0; }
    }
    if (i > 0 && combined[i - 1] === v && clLens[16] > 0) {
      const cost16 = base + clLens[16] + 2;
      let maxR = 0;
      while (maxR < 6 && i + maxR < n && combined[i + maxR] === v) maxR++;
      for (let r = 3; r <= maxR; r++) {
        if (cost16 < dp[i + r]) { dp[i + r] = cost16; prevPos[i + r] = i; prevSym[i + r] = 16; prevEx[i + r] = r - 3; }
      }
    }
    if (v === 0) {
      let maxZ = 0;
      while (maxZ < 138 && i + maxZ < n && combined[i + maxZ] === 0) maxZ++;
      if (maxZ >= 3 && clLens[17] > 0) {
        const cost17 = base + clLens[17] + 3;
        const lim = maxZ < 10 ? maxZ : 10;
        for (let r = 3; r <= lim; r++) {
          if (cost17 < dp[i + r]) { dp[i + r] = cost17; prevPos[i + r] = i; prevSym[i + r] = 17; prevEx[i + r] = r - 3; }
        }
      }
      if (maxZ >= 11 && clLens[18] > 0) {
        const cost18 = base + clLens[18] + 7;
        for (let r = 11; r <= maxZ; r++) {
          if (cost18 < dp[i + r]) { dp[i + r] = cost18; prevPos[i + r] = i; prevSym[i + r] = 18; prevEx[i + r] = r - 11; }
        }
      }
    }
  }
  if (!Number.isFinite(dp[n])) return null;
  const syms = [];
  const ex = [];
  const clFreq = new Int32Array(19);
  for (let p = n; p > 0; p = prevPos[p]) {
    const s = prevSym[p];
    syms.push(s); ex.push(prevEx[p]); clFreq[s]++;
  }
  syms.reverse();
  ex.reverse();
  return { syms, ex, clFreq };
}

function buildDynamicHeader(llLens, dLens, fast = false) {
  let hlit = 286;
  while (hlit > 257 && llLens[hlit - 1] === 0) hlit--;
  let hdist = 30;
  while (hdist > 1 && dLens[hdist - 1] === 0) hdist--;

  const totalLen = hlit + hdist;
  const combined = new Uint8Array(totalLen);
  combined.set(llLens.subarray(0, hlit), 0);
  combined.set(dLens.subarray(0, hdist), hlit);

  let best = null;
  const evalRle = (rle) => {
    const clLens = packageMerge(rle.clFreq, 7);
    let hclen = 19;
    while (hclen > 4 && clLens[CL_ORDER[hclen - 1]] === 0) hclen--;
    let bits = 14 + 3 * hclen;
    for (let s = 0; s < 19; s++) {
      const f = rle.clFreq[s];
      if (f > 0) {
        const eb = s === 16 ? 2 : s === 17 ? 3 : s === 18 ? 7 : 0;
        bits += f * (clLens[s] + eb);
      }
    }
    if (best === null || bits < best.headerBits) {
      best = { hlit, hdist, hclen, clLens, syms: rle.syms, ex: rle.ex, headerBits: bits };
    }
    return clLens;
  };

  const c1 = evalRle(rleEncodeLengths(combined, totalLen, true, true));
  if (fast) return best;
  evalRle(rleEncodeLengths(combined, totalLen, true, false));
  evalRle(rleEncodeLengths(combined, totalLen, false, true));
  evalRle(rleEncodeLengths(combined, totalLen, false, false));
  const dp1 = rleEncodeDP(combined, totalLen, best.clLens);
  if (dp1 !== null) {
    const cl2 = evalRle(dp1);
    const dp2 = rleEncodeDP(combined, totalLen, cl2);
    if (dp2 !== null) evalRle(dp2);
  }
  if (c1 !== best.clLens) {
    const dp3 = rleEncodeDP(combined, totalLen, c1);
    if (dp3 !== null) evalRle(dp3);
  }
  return best;
}

function buildBestDynamicTree(llFreq, dFreq, extraBitsTotal, fast = false) {
  const evalCounts = (lf, df) => {
    const llLens = packageMerge(lf, 15);
    const dLens = packageMerge(df, 15);
    let anyD = false;
    for (let i = 0; i < 30; i++) if (dLens[i] > 0) { anyD = true; break; }
    if (!anyD) dLens[0] = 1;

    let dataBits = extraBitsTotal;
    for (let i = 0; i < 286; i++) if (llFreq[i] > 0) dataBits += llFreq[i] * llLens[i];
    for (let i = 0; i < 30; i++) if (dFreq[i] > 0) dataBits += dFreq[i] * dLens[i];
    const hdr = buildDynamicHeader(llLens, dLens, fast);
    return { llLens, dLens, hdr, totalBits: 3 + hdr.headerBits + dataBits };
  };

  let best = evalCounts(llFreq, dFreq);
  if (fast) return best;

  const optLL = optimizeCountsForRle(llFreq, 286);
  const optD = optimizeCountsForRle(dFreq, 30);
  for (let i = 0; i < 286; i++) if (llFreq[i] > 0 && optLL[i] === 0) optLL[i] = 1;
  for (let i = 0; i < 30; i++) if (dFreq[i] > 0 && optD[i] === 0) optD[i] = 1;

  // Test both (optLL, rawD) and (optLL, optD) — smoothing dFreq alone often hurts data bits!
  const c1 = evalCounts(optLL, dFreq);
  if (c1.totalBits < best.totalBits) best = c1;
  const c2 = evalCounts(optLL, optD);
  if (c2.totalBits < best.totalBits) best = c2;
  return best;
}

function fixedBlockBits(llFreq, dFreq, extraBitsTotal) {
  let bits = 3 + extraBitsTotal;
  for (let i = 0; i < 286; i++) if (llFreq[i] > 0) bits += llFreq[i] * FIXED_LL_LENS[i];
  for (let i = 0; i < 30; i++) if (dFreq[i] > 0) bits += dFreq[i] * 5;
  return bits;
}

class BitWriter {
  constructor(initialCap = 16384) {
    this.buf = new Uint8Array(initialCap);
    this.pos = 0;
    this.bitBuf = 0;
    this.bitCnt = 0;
  }
  ensure(extra) {
    if (this.pos + extra <= this.buf.length) return;
    let next = this.buf.length * 2;
    while (next < this.pos + extra) next *= 2;
    const nb = new Uint8Array(next);
    nb.set(this.buf.subarray(0, this.pos));
    this.buf = nb;
  }
  writeBits(val, n) {
    this.bitBuf |= val << this.bitCnt;
    this.bitCnt += n;
    if (this.bitCnt >= 16) {
      this.ensure(4);
      while (this.bitCnt >= 8) {
        this.buf[this.pos++] = this.bitBuf & 0xff;
        this.bitBuf >>>= 8;
        this.bitCnt -= 8;
      }
    }
  }
  flushBytes() {
    this.ensure(4);
    while (this.bitCnt >= 8) {
      this.buf[this.pos++] = this.bitBuf & 0xff;
      this.bitBuf >>>= 8;
      this.bitCnt -= 8;
    }
  }
  alignByte() {
    this.flushBytes();
    if (this.bitCnt > 0) {
      this.ensure(1);
      this.buf[this.pos++] = this.bitBuf & 0xff;
      this.bitBuf = 0;
      this.bitCnt = 0;
    }
  }
  writeByte(b) { this.ensure(1); this.buf[this.pos++] = b; }
  writeBytes(arr) { this.ensure(arr.length); this.buf.set(arr, this.pos); this.pos += arr.length; }
  finish() { this.alignByte(); return this.buf.subarray(0, this.pos); }
}

function statsFromRange(litLens, dists, start, end) {
  const llFreq = new Int32Array(286);
  const dFreq = new Int32Array(30);
  let extraTotal = 0;
  for (let i = start; i < end; i++) {
    const d = dists[i];
    if (d === 0) llFreq[litLens[i]]++;
    else {
      const l = litLens[i];
      llFreq[LEN_SYM[l]]++;
      const ds = DIST_SYM[d];
      dFreq[ds]++;
      extraTotal += LEN_EX[l] + DIST_EXTRA[ds];
    }
  }
  llFreq[256] = 1;
  return { llFreq, dFreq, extraTotal };
}

function writeBlockRange(bw, isFinal, rawSlice, litLens, dists, start, end, dynTree = null) {
  const rawLen = rawSlice.length;
  const { llFreq, dFreq, extraTotal } = statsFromRange(litLens, dists, start, end);
  const dyn = dynTree ?? buildBestDynamicTree(llFreq, dFreq, extraTotal, false);
  const fixBits = fixedBlockBits(llFreq, dFreq, extraTotal);
  const storedBits = 50 + rawLen * 8;

  if (storedBits < dyn.totalBits && storedBits < fixBits && rawLen <= 65535) {
    bw.writeBits(isFinal ? 1 : 0, 1);
    bw.writeBits(0, 2);
    bw.alignByte();
    bw.writeByte(rawLen & 0xff);
    bw.writeByte(rawLen >>> 8);
    bw.writeByte(~rawLen & 0xff);
    bw.writeByte((~rawLen >>> 8) & 0xff);
    bw.writeBytes(rawSlice);
    return;
  }

  let llCodes, llLens, dCodes, dLens;
  if (fixBits < dyn.totalBits) {
    bw.writeBits(isFinal ? 1 : 0, 1);
    bw.writeBits(1, 2);
    llCodes = FIXED_LL_CODES; llLens = FIXED_LL_LENS;
    dCodes = FIXED_D_CODES; dLens = FIXED_D_LENS;
  } else {
    bw.writeBits(isFinal ? 1 : 0, 1);
    bw.writeBits(2, 2);
    const hdr = dyn.hdr;
    bw.writeBits(hdr.hlit - 257, 5);
    bw.writeBits(hdr.hdist - 1, 5);
    bw.writeBits(hdr.hclen - 4, 4);
    for (let i = 0; i < hdr.hclen; i++) bw.writeBits(hdr.clLens[CL_ORDER[i]], 3);
    const clCodes = buildCanonicalCodes(hdr.clLens);
    for (let i = 0; i < hdr.syms.length; i++) {
      const s = hdr.syms[i];
      bw.writeBits(clCodes[s], hdr.clLens[s]);
      if (s === 16) bw.writeBits(hdr.ex[i], 2);
      else if (s === 17) bw.writeBits(hdr.ex[i], 3);
      else if (s === 18) bw.writeBits(hdr.ex[i], 7);
    }
    llLens = dyn.llLens; dLens = dyn.dLens;
    llCodes = buildCanonicalCodes(llLens);
    dCodes = buildCanonicalCodes(dLens);
  }

  for (let i = start; i < end; i++) {
    const d = dists[i];
    if (d === 0) {
      const sym = litLens[i];
      bw.writeBits(llCodes[sym], llLens[sym]);
    } else {
      const l = litLens[i];
      const lsym = LEN_SYM[l];
      bw.writeBits(llCodes[lsym], llLens[lsym]);
      const lex = LEN_EX[l];
      if (lex > 0) bw.writeBits(l - LEN_BASE[lsym - 257], lex);
      const dsym = DIST_SYM[d];
      bw.writeBits(dCodes[dsym], dLens[dsym]);
      const dex = DIST_EXTRA[dsym];
      if (dex > 0) bw.writeBits(d - DIST_BASE[dsym], dex);
    }
  }
  bw.writeBits(llCodes[256], llLens[256]);
}

const HASH_BITS = 16;
const HASH_SIZE = 1 << HASH_BITS;
const HASH_MASK = HASH_SIZE - 1;
const WIN_SIZE = 32768;
const WIN_MASK = WIN_SIZE - 1;

function hash3(data, pos) {
  return (((data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)) * 0x1e35a7bd) >>> (32 - HASH_BITS)) & HASH_MASK;
}

function hashRle(runLen, byteVal, nextByte) {
  const r = runLen < 258 ? runLen : 258;
  return (((r | (byteVal << 9) | (nextByte << 17)) * 0x9e3779b1) >>> (32 - HASH_BITS)) & HASH_MASK;
}

function computeRuns(data, bpp, rowStride) {
  const n = data.length;
  const runLeft = new Uint16Array(n);
  if (n > 0) {
    runLeft[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
      if (data[i] === data[i + 1]) {
        const v = runLeft[i + 1] + 1;
        runLeft[i] = v < 65535 ? v : 65535;
      } else runLeft[i] = 1;
    }
  }
  let runBpp = null;
  if (bpp > 1) {
    runBpp = new Uint16Array(n);
    for (let i = n - 1; i >= bpp; i--) {
      if (data[i] === data[i - bpp]) {
        const v = i + 1 < n ? runBpp[i + 1] + 1 : 1;
        runBpp[i] = v < 65535 ? v : 65535;
      }
    }
  }
  let runRow = null;
  if (rowStride > 1 && rowStride <= WIN_SIZE) {
    runRow = new Uint16Array(n);
    for (let i = n - 1; i >= rowStride; i--) {
      if (data[i] === data[i - rowStride]) {
        const v = i + 1 < n ? runRow[i + 1] + 1 : 1;
        runRow[i] = v < 65535 ? v : 65535;
      }
    }
  }
  return { runLeft, runBpp, runRow };
}

export function fastLzLiteralCounts(data, rowStride = 0) {
  const { runLeft, runRow } = computeRuns(data, 1, rowStride);
  const lz = fastLazyLZ77(data, runLeft, null, runRow, rowStride, 1, 24);
  const litFreq = new Int32Array(256);
  for (let i = 0; i < lz.count; i++) {
    if (lz.dists[i] === 0) litFreq[lz.litLens[i]]++;
  }
  return litFreq;
}

function fastLazyLZ77(data, runLeft, runBpp, runRow, rowStride, bpp, maxChain = 48) {
  const n = data.length;
  const head = new Int32Array(HASH_SIZE).fill(-1);
  const prev = new Int32Array(WIN_SIZE).fill(-1);

  let cap = Math.max(4096, n >>> 2);
  let litLens = new Uint16Array(cap);
  let dists = new Uint16Array(cap);
  let posArr = new Int32Array(cap);
  let count = 0;

  const push = (ll, d, p) => {
    if (count === cap) {
      cap *= 2;
      const nll = new Uint16Array(cap); nll.set(litLens); litLens = nll;
      const nd = new Uint16Array(cap); nd.set(dists); dists = nd;
      const np = new Int32Array(cap); np.set(posArr); posArr = np;
    }
    litLens[count] = ll; dists[count] = d; posArr[count] = p; count++;
  };

  const insert = (p) => {
    if (p + 2 >= n) return;
    const h = hash3(data, p);
    prev[p & WIN_MASK] = head[h];
    head[h] = p;
  };

  const findBest = (p) => {
    const maxLen = n - p < 258 ? n - p : 258;
    if (maxLen < 3) return [0, 0];
    if (p > 0 && data[p] === data[p - 1] && runLeft[p] >= maxLen) return [maxLen, 1];
    if (runBpp !== null && runBpp[p] >= maxLen) return [maxLen, bpp];

    let bestL = 2, bestD = 0;
    const limit = p > WIN_SIZE ? p - WIN_SIZE : 0;
    const b0 = data[p], b1 = data[p + 1], b2 = data[p + 2];

    if (p > 0 && data[p] === data[p - 1]) {
      const rl = runLeft[p] < maxLen ? runLeft[p] : maxLen;
      if (rl > bestL) { bestL = rl; bestD = 1; }
    }
    if (runBpp !== null && runBpp[p] > bestL) {
      const rb = runBpp[p] < maxLen ? runBpp[p] : maxLen;
      if (rb > bestL) { bestL = rb; bestD = bpp; }
    }
    if (runRow !== null && runRow[p] > bestL) {
      const rr = runRow[p] < maxLen ? runRow[p] : maxLen;
      if (rr > bestL) { bestL = rr; bestD = rowStride; }
    }
    if (bestL === maxLen) return [bestL, bestD];

    const h = hash3(data, p);
    let cand = head[h];
    let chain = maxChain;
    while (cand >= limit && chain-- > 0 && bestL < maxLen) {
      if (data[cand + bestL] === data[p + bestL] && data[cand] === b0 && data[cand + 1] === b1 && data[cand + 2] === b2) {
        let l = 3;
        while (l < maxLen && data[cand + l] === data[p + l]) l++;
        if (l > bestL) { bestL = l; bestD = p - cand; if (bestL === maxLen) break; }
      }
      const next = prev[cand & WIN_MASK];
      if (next >= cand) break;
      cand = next;
    }
    return bestL >= 3 ? [bestL, bestD] : [0, 0];
  };

  for (let p = 0; p < n; ) {
    if (p > 0 && data[p] === data[p - 1] && runLeft[p] >= 258) {
      push(258, 1, p);
      if (runLeft[p] < 516) { for (let k = 0; k < 258; k++) insert(p + k); }
      else insert(p);
      p += 258; continue;
    }
    if (runBpp !== null && runBpp[p] >= 258 && runLeft[p] < 258) {
      push(258, bpp, p);
      if (runBpp[p] < 516) { for (let k = 0; k < 258; k++) insert(p + k); }
      else { for (let k = 0; k < bpp; k++) insert(p + k); }
      p += 258; continue;
    }
    const [l1, d1] = findBest(p);
    insert(p);
    if (l1 >= 3 && l1 < 128 && p + 1 < n) {
      const [l2] = findBest(p + 1);
      if (l2 > l1 + 1) { push(data[p], 0, p); p++; continue; }
    }
    if (l1 >= 3) {
      push(l1, d1, p);
      const end = p + l1;
      for (let q = p + 1; q < end; q++) insert(q);
      p = end;
    } else {
      push(data[p], 0, p);
      p++;
    }
  }
  return { litLens, dists, posArr, count };
}

function fastEntropyBlockBits(litLens, dists, l, r) {
  const llFreq = new Int32Array(286);
  const dFreq = new Int32Array(30);
  let extra = 0, llTot = 1, dTot = 0;
  llFreq[256] = 1;
  for (let i = l; i < r; i++) {
    const d = dists[i];
    if (d === 0) { llFreq[litLens[i]]++; llTot++; }
    else {
      const len = litLens[i];
      llFreq[LEN_SYM[len]]++; llTot++;
      const ds = DIST_SYM[d];
      dFreq[ds]++; dTot++;
      extra += LEN_EX[len] + DIST_EXTRA[ds];
    }
  }
  let bits = extra + 64;
  const logLL = Math.log2(llTot);
  let activeLL = 0;
  for (let i = 0; i < 286; i++) {
    const c = llFreq[i];
    if (c > 0) { activeLL++; bits += c * Math.max(1.0, logLL - Math.log2(c)); }
  }
  let activeD = 0;
  if (dTot > 0) {
    const logD = Math.log2(dTot);
    for (let i = 0; i < 30; i++) {
      const c = dFreq[i];
      if (c > 0) { activeD++; bits += c * Math.max(1.0, logD - Math.log2(c)); }
    }
  }
  bits += activeLL * 2.2 + activeD * 2.8;
  return bits;
}

function exactRangeBits(litLens, dists, l, r, fast = true) {
  const { llFreq, dFreq, extraTotal } = statsFromRange(litLens, dists, l, r);
  const dyn = buildBestDynamicTree(llFreq, dFreq, extraTotal, fast);
  const fix = fixedBlockBits(llFreq, dFreq, extraTotal);
  return dyn.totalBits < fix ? dyn.totalBits : fix;
}

function findBestSplitFast(litLens, dists, l, r) {
  const numPts = 11;
  let lo = l, hi = r, bestPos = -1, bestCost = Infinity;
  let lastStep = 64;
  let rounds = 0;
  while (hi - lo > 64 && rounds++ < 16) {
    const step = (hi - lo) / (numPts + 1);
    lastStep = Math.max(24, Math.floor(step));
    let localBest = -1, localCost = Infinity;
    for (let k = 1; k <= numPts; k++) {
      const mid = Math.floor(lo + k * step);
      if (mid <= l + 64 || mid >= r - 64) continue;
      const c = fastEntropyBlockBits(litLens, dists, l, mid) + fastEntropyBlockBits(litLens, dists, mid, r);
      if (c < localCost) { localCost = c; localBest = mid; }
    }
    if (localBest < 0) break;
    if (localCost < bestCost) { bestCost = localCost; bestPos = localBest; }
    lo = Math.max(l, Math.floor(localBest - step));
    hi = Math.min(r, Math.ceil(localBest + step));
    if (step < 24) break;
  }
  if (bestPos < 0) return { pos: -1, cost: Infinity };
  // Exact dynamic Huffman verification + local 3-point polish around bestPos
  let exactPos = bestPos;
  let exactCost = exactRangeBits(litLens, dists, l, bestPos, true) + exactRangeBits(litLens, dists, bestPos, r, true);
  const delta = Math.min(lastStep, 96);
  for (const off of [-delta, delta]) {
    const p = bestPos + off;
    if (p > l + 64 && p < r - 64) {
      const c = exactRangeBits(litLens, dists, l, p, true) + exactRangeBits(litLens, dists, p, r, true);
      if (c < exactCost) { exactCost = c; exactPos = p; }
    }
  }
  return { pos: exactPos, cost: exactCost };
}

function splitTokenStream(litLens, dists, count, maxBlocks = 32, minTokens = 256) {
  const splits = [0, count];
  const done = new Set();
  for (let iter = 1; iter < maxBlocks; iter++) {
    let bestSeg = -1, bestLen = 0;
    for (let i = 0; i < splits.length - 1; i++) {
      const a = splits[i], b = splits[i + 1];
      if (done.has(a)) continue;
      if (b - a > bestLen) { bestLen = b - a; bestSeg = i; }
    }
    if (bestSeg < 0 || bestLen < minTokens) break;
    const a = splits[bestSeg], b = splits[bestSeg + 1];
    const origCost = exactRangeBits(litLens, dists, a, b, true);
    const { pos, cost } = findBestSplitFast(litLens, dists, a, b);
    if (pos > a && pos < b && cost + 16 < origCost) splits.splice(bestSeg + 1, 0, pos);
    else done.add(a);
  }
  return splits;
}

function splitInputBlocks(data, runLeft, runBpp, runRow, rowStride, bpp, maxBlocks = 32, maxBlockBytes = 524288) {
  const n = data.length;
  if (n <= 4096) return [0, n];

  const lz = fastLazyLZ77(data, runLeft, runBpp, runRow, rowStride, bpp, 36);
  const splits = splitTokenStream(lz.litLens, lz.dists, lz.count, maxBlocks, 256);

  const byteBounds = [0];
  for (let i = 1; i < splits.length - 1; i++) {
    const bp = lz.posArr[splits[i]];
    if (bp > byteBounds[byteBounds.length - 1]) byteBounds.push(bp);
  }
  byteBounds.push(n);

  const finalBounds = [0];
  for (let i = 0; i < byteBounds.length - 1; i++) {
    let cur = byteBounds[i];
    const end = byteBounds[i + 1];
    while (end - cur > maxBlockBytes) { cur += maxBlockBytes; finalBounds.push(cur); }
    finalBounds.push(end);
  }
  return finalBounds;
}

/**
 * Exact Pareto-optimal (len, dist) match step precomputation using minDist[3..258]!
 * Combines dist=1, dist=bpp, primary 3-byte hash (nearest-to-farthest), RLE-run hash (head2),
 * and row-above probes (rowStride, 2*rowStride) while ALWAYS keeping the exact minimum distance
 * for every match length 3..maxFound!
 */
function precomputeBlockMatches(
  data, runLeft, runBpp, runRow, bStart, bEnd,
  head, prev, head2, prev2,
  rowStride, bpp, maxChain, pool
) {
  const B = bEnd - bStart;
  const off = pool.off;
  let mLen = pool.mLen;
  let mDist = pool.mDist;
  let cap = mLen.length;
  let mCnt = 0;

  const ensureMatch = (need) => {
    if (mCnt + need <= cap) return;
    while (cap < mCnt + need) cap *= 2;
    const nl = new Uint16Array(cap); nl.set(mLen.subarray(0, mCnt)); mLen = nl;
    const nd = new Uint16Array(cap); nd.set(mDist.subarray(0, mCnt)); mDist = nd;
    pool.mLen = mLen;
    pool.mDist = mDist;
  };

  const n = data.length;
  const minDist = new Uint16Array(259);
  const tmpLen = new Uint16Array(260);
  const tmpDist = new Uint16Array(260);

  const insertHashes = (p) => {
    if (p + 2 < n) {
      const h = hash3(data, p);
      prev[p & WIN_MASK] = head[h];
      head[h] = p;
    }
    const r = runLeft[p];
    if (r >= 3 && r < 258 && p + r < n) {
      const h2 = hashRle(r, data[p], data[p + r]);
      prev2[p & WIN_MASK] = head2[h2];
      head2[h2] = p;
    }
  };

  for (let i = 0; i < B; i++) {
    off[i] = mCnt;
    const p = bStart + i;
    const rem = bEnd - p;
    const maxLen = rem < 258 ? rem : 258;

    // Fast path 1: Period-1 RLE full match (maxLen, dist=1)
    if (p > 0 && data[p] === data[p - 1] && runLeft[p] >= maxLen && maxLen >= 3) {
      ensureMatch(1);
      mLen[mCnt] = maxLen; mDist[mCnt] = 1; mCnt++;
      if (runLeft[p] < 516 || i === 0 || i === B - 1) insertHashes(p);
      continue;
    }

    // Fast path 2: Period-bpp full match (flat RGB/RGBA in Filter 0)
    if (runBpp !== null && runBpp[p] >= maxLen && maxLen >= 3) {
      let steps = 0;
      if (p > 0 && data[p] === data[p - 1] && runLeft[p] >= 3) {
        tmpLen[steps] = runLeft[p] < maxLen ? runLeft[p] : maxLen;
        tmpDist[steps] = 1;
        steps++;
      }
      tmpLen[steps] = maxLen; tmpDist[steps] = bpp; steps++;
      ensureMatch(steps);
      for (let s = 0; s < steps; s++) { mLen[mCnt] = tmpLen[s]; mDist[mCnt] = tmpDist[s]; mCnt++; }
      if (runBpp[p] < 516 || i < bpp || i >= B - bpp) insertHashes(p);
      continue;
    }

    if (maxLen < 3) { insertHashes(p); continue; }

    const limit = p > WIN_SIZE ? p - WIN_SIZE : 0;
    const b0 = data[p], b1 = data[p + 1], b2 = data[p + 2];
    const myRun = runLeft[p];
    const rowLenAbove = runRow !== null ? (runRow[p] < maxLen ? runRow[p] : maxLen) : 0;
    const useRleHash = myRun >= 3 && myRun < maxLen && p + myRun < n;

    // ZERO-ALLOCATION O(1) FAST PATH when no out-of-order sources (rowAbove / RLE hash) exist!
    // Primary 3-byte hash walks strictly nearest-to-farthest, so every l > bestL is already Pareto-optimal!
    if (!useRleHash && rowLenAbove < 3) {
      let bestL = 2;
      let steps = 0;
      if (p > 0 && data[p - 1] === b0 && myRun >= 3) {
        bestL = myRun < maxLen ? myRun : maxLen;
        tmpLen[steps] = bestL; tmpDist[steps] = 1; steps++;
      }
      if (runBpp !== null && runBpp[p] > bestL) {
        const rb = runBpp[p] < maxLen ? runBpp[p] : maxLen;
        if (rb > bestL) { bestL = rb; tmpLen[steps] = rb; tmpDist[steps] = bpp; steps++; }
      }
      const h = hash3(data, p);
      let cand = head[h];
      let chain = maxChain;
      while (cand >= limit && chain-- > 0 && bestL < maxLen) {
        if (data[cand + bestL] === data[p + bestL] && data[cand] === b0 && data[cand + 1] === b1 && data[cand + 2] === b2) {
          let l = 3;
          while (l < maxLen && data[cand + l] === data[p + l]) l++;
          if (l > bestL) {
            bestL = l; tmpLen[steps] = l; tmpDist[steps] = p - cand; steps++;
            if (bestL === maxLen) break;
            if (bestL >= 48 && chain > 12) chain = 12;
            else if (bestL >= 24 && chain > 32) chain = 32;
          }
        }
        const next = prev[cand & WIN_MASK];
        if (next >= cand) break;
        cand = next;
      }
      const r2 = rowStride * 2;
      if (bestL < maxLen && rowStride > 1 && r2 <= WIN_SIZE && p >= r2) {
        const rc = p - r2;
        if (data[rc + bestL] === data[p + bestL] && data[rc] === b0 && data[rc + 1] === b1 && data[rc + 2] === b2) {
          let l = 3;
          while (l < maxLen && data[rc + l] === data[p + l]) l++;
          if (l > bestL) { bestL = l; tmpLen[steps] = l; tmpDist[steps] = r2; steps++; }
        }
      }
      insertHashes(p);
      if (steps > 0) {
        ensureMatch(steps);
        for (let s = 0; s < steps; s++) { mLen[mCnt] = tmpLen[s]; mDist[mCnt] = tmpDist[s]; mCnt++; }
      }
      continue;
    }

    // Exact Pareto minDist[3..258] path (ZERO closure allocations!) when rowAbove or RLE hash is active
    let maxFound = 2;
    if (p > 0 && data[p - 1] === b0 && myRun >= 3) {
      const rl = myRun < maxLen ? myRun : maxLen;
      for (let k = 3; k <= rl; k++) minDist[k] = 1;
      maxFound = rl;
    }
    if (runBpp !== null && runBpp[p] >= 3) {
      const rb = runBpp[p] < maxLen ? runBpp[p] : maxLen;
      if (rb > maxFound) { for (let k = maxFound + 1; k <= rb; k++) minDist[k] = bpp; maxFound = rb; }
      for (let k = rb; k >= 3 && bpp < minDist[k]; k--) minDist[k] = bpp;
    }
    if (rowLenAbove >= 3) {
      if (rowLenAbove > maxFound) { for (let k = maxFound + 1; k <= rowLenAbove; k++) minDist[k] = rowStride; maxFound = rowLenAbove; }
      for (let k = rowLenAbove; k >= 3 && rowStride < minDist[k]; k--) minDist[k] = rowStride;
    }

    // Fast path 3: Identical row-above full match -> only walk closer candidates (< rowStride) to improve dist!
    if (rowLenAbove === maxLen) {
      const closeLimit = p - rowStride + 1;
      const h = hash3(data, p);
      let cand = head[h];
      let chain = 20;
      let closeBest = minDist[3] < rowStride ? 3 : 2;
      while (closeBest < maxLen && minDist[closeBest + 1] < rowStride) closeBest++;
      while (cand >= closeLimit && chain-- > 0 && closeBest < maxLen) {
        if (data[cand + closeBest] === data[p + closeBest] && data[cand] === b0 && data[cand + 1] === b1 && data[cand + 2] === b2) {
          let l = 3;
          while (l < maxLen && data[cand + l] === data[p + l]) l++;
          if (l > closeBest) {
            closeBest = l;
            const d = p - cand;
            for (let k = l; k >= 3 && d < minDist[k]; k--) minDist[k] = d;
          }
        }
        const next = prev[cand & WIN_MASK];
        if (next >= cand) break;
        cand = next;
      }
      insertHashes(p);
      let steps = 0;
      for (let k = 3; k <= maxFound; k++) {
        if (k === maxFound || minDist[k] !== minDist[k + 1]) {
          tmpLen[steps] = k; tmpDist[steps] = minDist[k]; steps++;
        }
      }
      ensureMatch(steps);
      for (let s = 0; s < steps; s++) { mLen[mCnt] = tmpLen[s]; mDist[mCnt] = tmpDist[s]; mCnt++; }
      continue;
    }

    // Primary 3-byte hash walk FIRST (strictly nearest-to-farthest -> preserves all close matches!)
    const h = hash3(data, p);
    {
      let cand = head[h];
      let chain = useRleHash ? Math.min(64, maxChain >>> 1) : maxChain;
      let closeBest = 2;
      if (p > 0 && data[p - 1] === b0 && myRun >= 3) closeBest = myRun < maxLen ? myRun : maxLen;
      if (runBpp !== null && runBpp[p] > closeBest) closeBest = runBpp[p] < maxLen ? runBpp[p] : maxLen;
      while (cand >= limit && chain-- > 0 && closeBest < maxLen) {
        const dist = p - cand;
        if (dist >= rowStride && rowLenAbove > closeBest) closeBest = rowLenAbove;
        if (data[cand + closeBest] === data[p + closeBest] && data[cand] === b0 && data[cand + 1] === b1 && data[cand + 2] === b2) {
          let l = 3;
          if (myRun > 3 && runLeft[cand] >= myRun) l = myRun < maxLen ? myRun : maxLen;
          while (l < maxLen && data[cand + l] === data[p + l]) l++;
          if (l > closeBest) {
            closeBest = l;
            if (l > maxFound) { for (let k = maxFound + 1; k <= l; k++) minDist[k] = dist; maxFound = l; }
            for (let k = l; k >= 3 && dist < minDist[k]; k--) minDist[k] = dist;
            if (closeBest === maxLen) break;
            if (closeBest >= 48 && chain > 12) chain = 12;
            else if (closeBest >= 24 && chain > 32) chain = 32;
          }
        }
        const next = prev[cand & WIN_MASK];
        if (next >= cand) break;
        cand = next;
      }
    }

    // Secondary RLE-run hash walk (jumps across long zero runs to find deep 32 KiB matches without erasing closer matches!)
    if (useRleHash && maxFound < maxLen) {
      const h2 = hashRle(myRun, b0, data[p + myRun]);
      let cand = head2[h2];
      let chain2 = maxChain;
      let rleBest = myRun;
      while (cand >= limit && chain2-- > 0 && rleBest < maxLen) {
        const dist = p - cand;
        if (
          runLeft[cand] === myRun &&
          data[cand] === b0 &&
          data[cand + myRun] === data[p + myRun] &&
          (rleBest < maxFound ? dist < minDist[rleBest + 1] || data[cand + maxFound] === data[p + maxFound] : data[cand + rleBest] === data[p + rleBest])
        ) {
          let l = myRun + 1;
          while (l < maxLen && data[cand + l] === data[p + l]) l++;
          if (l > rleBest) {
            rleBest = l;
            if (l > maxFound) { for (let k = maxFound + 1; k <= l; k++) minDist[k] = dist; maxFound = l; }
            for (let k = l; k >= 3 && dist < minDist[k]; k--) minDist[k] = dist;
            if (rleBest === maxLen) break;
          }
        }
        const next = prev2[cand & WIN_MASK];
        if (next >= cand) break;
        cand = next;
      }
    }

    // O(1) 2*rowStride probe
    const r2 = rowStride * 2;
    if (rowStride > 1 && r2 <= WIN_SIZE && p >= r2 && (maxFound < maxLen || r2 < minDist[maxFound])) {
      const rc = p - r2;
      if (data[rc] === b0 && data[rc + 1] === b1 && data[rc + 2] === b2) {
        let l = 3;
        while (l < maxLen && data[rc + l] === data[p + l]) l++;
        if (l > maxFound) { for (let k = maxFound + 1; k <= l; k++) minDist[k] = r2; maxFound = l; }
        for (let k = l; k >= 3 && r2 < minDist[k]; k--) minDist[k] = r2;
      }
    }

    insertHashes(p);

    if (maxFound >= 3) {
      let steps = 0;
      for (let k = 3; k <= maxFound; k++) {
        if (k === maxFound || minDist[k] !== minDist[k + 1]) {
          tmpLen[steps] = k; tmpDist[steps] = minDist[k]; steps++;
        }
      }
      ensureMatch(steps);
      for (let s = 0; s < steps; s++) { mLen[mCnt] = tmpLen[s]; mDist[mCnt] = tmpDist[s]; mCnt++; }
    }
  }
  off[B] = mCnt;
  return { off, mLen, mDist };
}

function optimalParseBlock(
  data, runLeft, runBpp, bpp, bStart, bEnd,
  matches, llCost, dCost, dpCost, dpLen, dpDist,
  subStart = 0, subEnd = bEnd - bStart, exactAllLengths = true
) {
  const lenFullCost = new Float64Array(259);
  for (let l = 3; l <= 258; l++) lenFullCost[l] = llCost[LEN_SYM[l]] + LEN_EX[l];
  const distFullCost = new Float64Array(30);
  for (let s = 0; s < 30; s++) distFullCost[s] = dCost[s] + DIST_EXTRA[s];

  const { off, mLen, mDist } = matches;
  const bppDsym = DIST_SYM[bpp];
  const cost258_1 = lenFullCost[258] + distFullCost[0];
  const cost258_bpp = lenFullCost[258] + distFullCost[bppDsym];

  const subLen = subEnd - subStart;
  dpCost.fill(1e30, 0, subLen + 1);
  dpCost[0] = 0;

  for (let rel = 0; rel < subLen; rel++) {
    const c = dpCost[rel];
    if (c >= 1e29) continue;

    const i = subStart + rel;
    const p = bStart + i;
    const rem = subLen - rel;

    if (p > 0 && data[p] === data[p - 1] && runLeft[p] >= 516 && rem >= 516) {
      const mc = c + cost258_1;
      if (mc < dpCost[rel + 258]) { dpCost[rel + 258] = mc; dpLen[rel + 258] = 258; dpDist[rel + 258] = 1; }
      continue;
    }

    if (runBpp !== null && runBpp[p] >= 516 && runLeft[p] < 3 && rem >= 516) {
      const mc = c + cost258_bpp;
      if (mc < dpCost[rel + 258]) { dpCost[rel + 258] = mc; dpLen[rel + 258] = 258; dpDist[rel + 258] = bpp; }
      continue;
    }

    const lc = c + llCost[data[p]];
    if (lc < dpCost[rel + 1]) { dpCost[rel + 1] = lc; dpLen[rel + 1] = 1; dpDist[rel + 1] = 0; }

    const s0 = off[i];
    const s1 = off[i + 1];
    let prevL = 2;
    for (let s = s0; s < s1; s++) {
      let endL = mLen[s];
      if (endL > rem) endL = rem;
      if (endL < 3) continue;
      const dist = mDist[s];
      const base = c + distFullCost[DIST_SYM[dist]];
      const startL = endL > prevL ? prevL + 1 : 3;
      if ((exactAllLengths === 2) || endL <= (exactAllLengths === 1 ? 32 : 12)) {
        for (let l = startL; l <= endL; l++) {
          const mc = base + lenFullCost[l];
          const ni = rel + l;
          if (mc < dpCost[ni]) { dpCost[ni] = mc; dpLen[ni] = l; dpDist[ni] = dist; }
        }
      } else {
        const denseLim = exactAllLengths === 1 ? 32 : 12;
        const stops = exactAllLengths === 1 ? FINAL_LEN_STOPS : LONG_LEN_STOPS;
        const tailPad = exactAllLengths === 1 ? 3 : 1;
        const denseEnd = denseLim < endL ? denseLim : endL;
        for (let l = startL; l <= denseEnd; l++) {
          const mc = base + lenFullCost[l];
          const ni = rel + l;
          if (mc < dpCost[ni]) { dpCost[ni] = mc; dpLen[ni] = l; dpDist[ni] = dist; }
        }
        const loStop = startL > denseLim + 1 ? startL : denseLim + 1;
        const hiStop = endL - tailPad - 1;
        for (let k = 0; k < stops.length; k++) {
          const l = stops[k];
          if (l < loStop) continue;
          if (l > hiStop) break;
          const mc = base + lenFullCost[l];
          const ni = rel + l;
          if (mc < dpCost[ni]) { dpCost[ni] = mc; dpLen[ni] = l; dpDist[ni] = dist; }
        }
        const tailStart = endL - tailPad > startL ? endL - tailPad : startL;
        for (let l = tailStart; l <= endL; l++) {
          const mc = base + lenFullCost[l];
          const ni = rel + l;
          if (mc < dpCost[ni]) { dpCost[ni] = mc; dpLen[ni] = l; dpDist[ni] = dist; }
        }
      }
      if (endL > prevL) prevL = endL;
    }
  }

  let tokCnt = 0;
  for (let rel = subLen; rel > 0; rel -= dpLen[rel]) tokCnt++;
  const litLens = new Uint16Array(tokCnt);
  const dists = new Uint16Array(tokCnt);
  const posArr = new Int32Array(tokCnt);
  let idx = tokCnt - 1;
  for (let rel = subLen; rel > 0; ) {
    const l = dpLen[rel];
    const d = dpDist[rel];
    posArr[idx] = bStart + subStart + rel - l;
    if (d === 0) { litLens[idx] = data[bStart + subStart + rel - 1]; dists[idx] = 0; }
    else { litLens[idx] = l; dists[idx] = d; }
    idx--;
    rel -= l;
  }
  return { litLens, dists, posArr, tokCnt };
}

function greedyStatsFromMatches(data, bStart, matches, subStart, subEnd) {
  const { off, mLen, mDist } = matches;
  const llFreq = new Int32Array(286);
  const dFreq = new Int32Array(30);
  llFreq[256] = 1;
  for (let i = subStart; i < subEnd; ) {
    const s0 = off[i], s1 = off[i + 1];
    if (s1 > s0) {
      let l = mLen[s1 - 1];
      if (i + l > subEnd) l = subEnd - i;
      if (l >= 3) {
        llFreq[LEN_SYM[l]]++;
        dFreq[DIST_SYM[mDist[s1 - 1]]]++;
        i += l;
        continue;
      }
    }
    llFreq[data[bStart + i]]++;
    i++;
  }
  return { llFreq, dFreq };
}

function entropyCostsFromStats(llStats, dStats, llCost, dCost) {
  let llSum = 0;
  for (let i = 0; i < 286; i++) llSum += llStats[i];
  const logLL = llSum > 0 ? Math.log2(llSum) : 1;
  for (let i = 0; i < 286; i++) {
    const v = llStats[i];
    llCost[i] = v > 0 ? Math.max(1.0, logLL - Math.log2(v)) : logLL + 1.0;
  }
  let dSum = 0;
  for (let i = 0; i < 30; i++) dSum += dStats[i];
  const logD = dSum > 0 ? Math.log2(dSum) : 1;
  for (let i = 0; i < 30; i++) {
    const v = dStats[i];
    dCost[i] = v > 0 ? Math.max(1.0, logD - Math.log2(v)) : logD + 1.0;
  }
}

function zopfliOptimizeRange(
  data, runLeft, runBpp, bpp, bStart, bEnd,
  matches, iterations, dpCost, dpLen, dpDist,
  subStart = 0, subEnd = bEnd - bStart, initTokens = null
) {
  const llCost = new Float64Array(286);
  const dCost = new Float64Array(30);
  const llSmooth = new Float64Array(286);
  const dSmooth = new Float64Array(30);

  let bestBits = Infinity;
  let bestLitLens = null;
  let bestDists = null;
  let bestPosArr = null;
  let bestTokCnt = 0;
  let bestDyn = null;

  if (initTokens !== null) {
    const { llFreq, dFreq, extraTotal } = statsFromRange(initTokens.litLens, initTokens.dists, 0, initTokens.litLens.length);
    const dyn = buildBestDynamicTree(llFreq, dFreq, extraTotal, false);
    const fix = fixedBlockBits(llFreq, dFreq, extraTotal);
    bestBits = dyn.totalBits < fix ? dyn.totalBits : fix;
    bestLitLens = initTokens.litLens;
    bestDists = initTokens.dists;
    bestTokCnt = initTokens.litLens.length;
    bestDyn = dyn;
    for (let i = 0; i < 286; i++) llSmooth[i] = llFreq[i];
    for (let i = 0; i < 30; i++) dSmooth[i] = dFreq[i];
    entropyCostsFromStats(llSmooth, dSmooth, llCost, dCost);
  } else {
    // Initialize from ACTUAL raw byte literal entropy of the block + realistic match cost!
    // Immediately prunes unprofitable short matches on filtered photo/shadow blocks (10x faster convergence!).
    const rawFreq = new Int32Array(256);
    const subLen = subEnd - subStart;
    for (let i = subStart; i < subEnd; i++) rawFreq[data[bStart + i]]++;
    const logR = Math.log2(Math.max(2, subLen));
    for (let i = 0; i < 256; i++) {
      llCost[i] = rawFreq[i] > 0 ? Math.max(1.0, logR - Math.log2(rawFreq[i])) : logR + 1.0;
    }
    llCost[256] = logR;
    const initMatchCost = bpp === 1 ? 8.0 : 9.5;
    for (let i = 257; i < 286; i++) llCost[i] = initMatchCost;
    for (let i = 0; i < 30; i++) dCost[i] = 4.5;
  }

  let lastBits = bestBits;
  let rng = 0x9e3779b9 ^ (bStart + subStart + 1) ^ ((subEnd - subStart) * 2654435761);
  const nextRand = () => { rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5; return rng >>> 0; };

  let lastImproved = false;
  for (let iter = 0; iter < iterations; iter++) {
    // 0 = fast warm-up (<=20 + stops), 1 = rich final (<=36 + dense stops + tail-4), 2 = full 3..258 (1-bpp only)
    const exactMode = iter === iterations - 1 ? (data.length <= 1_200_000 ? 2 : 1) : 0;
    const isExactHeader = iter >= iterations - 2;
    const { litLens, dists, posArr, tokCnt } = optimalParseBlock(
      data, runLeft, runBpp, bpp, bStart, bEnd,
      matches, llCost, dCost, dpCost, dpLen, dpDist, subStart, subEnd, exactMode
    );
    const { llFreq, dFreq, extraTotal } = statsFromRange(litLens, dists, 0, tokCnt);
    const dyn = buildBestDynamicTree(llFreq, dFreq, extraTotal, !isExactHeader);
    const fix = fixedBlockBits(llFreq, dFreq, extraTotal);
    const bits = dyn.totalBits < fix ? dyn.totalBits : fix;

    lastImproved = false;
    if (bits < bestBits) {
      bestBits = bits;
      bestLitLens = litLens;
      bestDists = dists;
      bestPosArr = posArr;
      bestTokCnt = tokCnt;
      bestDyn = dyn;
      lastImproved = true;
    }

    if (iter === iterations - 1) break;

    if (iter < 4 && initTokens === null) {
      for (let i = 0; i < 286; i++) llSmooth[i] = llFreq[i];
      for (let i = 0; i < 30; i++) dSmooth[i] = dFreq[i];
    } else if (bits >= lastBits && iter < iterations - 2) {
      for (let i = 0; i < 286; i++) {
        let v = llSmooth[i] * 0.4 + llFreq[i] * 0.6;
        if ((nextRand() & 15) === 0) v = llSmooth[nextRand() % 286];
        llSmooth[i] = v;
      }
      for (let i = 0; i < 30; i++) {
        let v = dSmooth[i] * 0.4 + dFreq[i] * 0.6;
        if ((nextRand() & 15) === 0) v = dSmooth[nextRand() % 30];
        dSmooth[i] = v;
      }
    } else {
      for (let i = 0; i < 286; i++) llSmooth[i] = llSmooth[i] * 0.35 + llFreq[i] * 0.65;
      for (let i = 0; i < 30; i++) dSmooth[i] = dSmooth[i] * 0.35 + dFreq[i] * 0.65;
    }
    lastBits = bits;

    if (iter === iterations - 2 && bestDyn !== null) {
      for (let i = 0; i < 286; i++) llCost[i] = bestDyn.llLens[i] > 0 ? bestDyn.llLens[i] : 16;
      for (let i = 0; i < 30; i++) dCost[i] = bestDyn.dLens[i] > 0 ? bestDyn.dLens[i] : 16;
    } else {
      entropyCostsFromStats(llSmooth, dSmooth, llCost, dCost);
    }
  }

  // Bonus exact-Huffman continuation pass only when high iteration budget is requested!
  if (lastImproved && bestDyn !== null && iterations >= 8 && data.length <= 1_200_000) {
    for (let i = 0; i < 286; i++) llCost[i] = bestDyn.llLens[i] > 0 ? bestDyn.llLens[i] : 16;
    for (let i = 0; i < 30; i++) dCost[i] = bestDyn.dLens[i] > 0 ? bestDyn.dLens[i] : 16;
    const { litLens, dists, posArr, tokCnt } = optimalParseBlock(
      data, runLeft, runBpp, bpp, bStart, bEnd,
      matches, llCost, dCost, dpCost, dpLen, dpDist, subStart, subEnd, 2
    );
    const { llFreq, dFreq, extraTotal } = statsFromRange(litLens, dists, 0, tokCnt);
    const dyn = buildBestDynamicTree(llFreq, dFreq, extraTotal, false);
    const fix = fixedBlockBits(llFreq, dFreq, extraTotal);
    const bits = dyn.totalBits < fix ? dyn.totalBits : fix;
    if (bits < bestBits) {
      bestBits = bits;
      bestLitLens = litLens;
      bestDists = dists;
      bestPosArr = posArr;
      bestTokCnt = tokCnt;
    }
  }

  return { litLens: bestLitLens, dists: bestDists, posArr: bestPosArr, tokCnt: bestTokCnt };
}

export function zopfliZlib(data, { iterations = 13, maxChain = 448, maxBlocks = 32, rowStride = 0, bpp = 1 } = {}) {
  const n = data.length;
  const { runLeft, runBpp, runRow } = computeRuns(data, bpp, rowStride);
  const bounds = splitInputBlocks(data, runLeft, runBpp, runRow, rowStride, bpp, maxBlocks, 524288);
  const numBlocks = bounds.length - 1;

  let maxB = 0;
  for (let b = 0; b < numBlocks; b++) {
    const len = bounds[b + 1] - bounds[b];
    if (len > maxB) maxB = len;
  }
  const dpCost = new Float64Array(maxB + 1);
  const dpLen = new Uint16Array(maxB + 1);
  const dpDist = new Uint16Array(maxB + 1);
  const matchPool = {
    off: new Int32Array(maxB + 1),
    mLen: new Uint16Array(Math.max(4096, maxB * 2)),
    mDist: new Uint16Array(Math.max(4096, maxB * 2)),
  };

  const head = new Int32Array(HASH_SIZE).fill(-1);
  const prev = new Int32Array(WIN_SIZE).fill(-1);
  const head2 = new Int32Array(HASH_SIZE).fill(-1);
  const prev2 = new Int32Array(WIN_SIZE).fill(-1);

  const finalBlocks = [];
  for (let b = 0; b < numBlocks; b++) {
    const bStart = bounds[b];
    const bEnd = bounds[b + 1];
    const matches = precomputeBlockMatches(
      data, runLeft, runBpp, runRow, bStart, bEnd, head, prev, head2, prev2, rowStride, bpp, maxChain, matchPool
    );
    const res = zopfliOptimizeRange(
      data, runLeft, runBpp, bpp, bStart, bEnd, matches, iterations, dpCost, dpLen, dpDist
    );

    if (res.tokCnt >= 768) {
      const subSplits = splitTokenStream(res.litLens, res.dists, res.tokCnt, 4, 256);
      if (subSplits.length > 2) {
        for (let s = 0; s < subSplits.length - 1; s++) {
          const ts = subSplits[s];
          const te = subSplits[s + 1];
          const rawStart = res.posArr[ts];
          const rawEnd = te < res.tokCnt ? res.posArr[te] : bEnd;
          finalBlocks.push({
            litLens: res.litLens.subarray(ts, te),
            dists: res.dists.subarray(ts, te),
            rawStart,
            rawEnd,
          });
        }
      } else {
        finalBlocks.push({ litLens: res.litLens, dists: res.dists, rawStart: bStart, rawEnd: bEnd });
      }
    } else {
      finalBlocks.push({ litLens: res.litLens, dists: res.dists, rawStart: bStart, rawEnd: bEnd });
    }
  }

  // Stage 3: Adjacent block merge check
  let merged = finalBlocks;
  let didMerge = true;
  let mergeGuard = 0;
  while (didMerge && merged.length > 1 && mergeGuard++ < 32) {
    didMerge = false;
    const next = [];
    for (let i = 0; i < merged.length; ) {
      if (i + 1 < merged.length) {
        const a = merged[i], b = merged[i + 1];
        const sa = statsFromRange(a.litLens, a.dists, 0, a.litLens.length);
        const sb = statsFromRange(b.litLens, b.dists, 0, b.litLens.length);
        const ca = buildBestDynamicTree(sa.llFreq, sa.dFreq, sa.extraTotal, false).totalBits;
        const cb = buildBestDynamicTree(sb.llFreq, sb.dFreq, sb.extraTotal, false).totalBits;
        const jLL = new Int32Array(286);
        const jD = new Int32Array(30);
        for (let k = 0; k < 286; k++) jLL[k] = sa.llFreq[k] + sb.llFreq[k];
        jLL[256] = 1;
        for (let k = 0; k < 30; k++) jD[k] = sa.dFreq[k] + sb.dFreq[k];
        const cj = buildBestDynamicTree(jLL, jD, sa.extraTotal + sb.extraTotal, false).totalBits;
        if (cj <= ca + cb) {
          const nll = new Uint16Array(a.litLens.length + b.litLens.length);
          nll.set(a.litLens, 0); nll.set(b.litLens, a.litLens.length);
          const nd = new Uint16Array(a.dists.length + b.dists.length);
          nd.set(a.dists, 0); nd.set(b.dists, a.dists.length);
          next.push({ litLens: nll, dists: nd, rawStart: a.rawStart, rawEnd: b.rawEnd });
          i += 2; didMerge = true; continue;
        }
      }
      next.push(merged[i]); i++;
    }
    merged = next;
  }

  const bw = new BitWriter(Math.max(8192, n >>> 3));
  bw.writeByte(0x78);
  bw.writeByte(0xda);
  for (let b = 0; b < merged.length; b++) {
    const blk = merged[b];
    writeBlockRange(bw, b === merged.length - 1, data.subarray(blk.rawStart, blk.rawEnd), blk.litLens, blk.dists, 0, blk.litLens.length);
  }
  bw.alignByte();
  const a32 = adler32(data);
  bw.writeByte((a32 >>> 24) & 0xff);
  bw.writeByte((a32 >>> 16) & 0xff);
  bw.writeByte((a32 >>> 8) & 0xff);
  bw.writeByte(a32 & 0xff);
  return bw.finish();
}
