#![allow(clippy::needless_range_loop)]

use std::alloc::{alloc as std_alloc, dealloc as std_dealloc, Layout};
use std::slice;
use std::sync::OnceLock;

// ============================================================================
// WASM Memory FFI
// ============================================================================

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let size = len.max(1);
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { std_alloc(layout) }
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let size = len.max(1);
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { std_dealloc(ptr, layout) }
}

// ============================================================================
// DEFLATE Tables & Exact Canonical Huffman + Package-Merge
// ============================================================================

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
const WARMUP_STOPS: [usize; 11] = [12, 16, 20, 26, 34, 50, 66, 98, 130, 162, 226];
const FINAL_STOPS: [usize; 28] = [
    40, 42, 44, 48, 50, 56, 58, 64, 66, 72, 80, 82, 96, 98, 112, 114, 128, 130, 144, 160, 162, 176,
    192, 194, 208, 224, 226, 240,
];

static TABLES: OnceLock<Tables> = OnceLock::new();
#[inline(always)]
fn get_tables() -> &'static Tables {
    TABLES.get_or_init(Tables::new)
}

const HASH_BITS: usize = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const WIN_SIZE: usize = 32768;
const WIN_MASK: usize = WIN_SIZE - 1;
const BT_SIZE: usize = 65536;
const BT_MASK: usize = BT_SIZE - 1;

struct Tables {
    len_sym: [u16; 259],
    len_ex: [u8; 259],
    dist_sym: Vec<u8>,
    fixed_ll_lens: [u8; 288],
    fixed_ll_codes: [u16; 288],
    fixed_d_lens: [u8; 30],
    fixed_d_codes: [u16; 30],
    log2_tbl: Vec<f64>,
}

impl Tables {
    fn new() -> Self {
        let mut len_sym = [0u16; 259];
        let mut len_ex = [0u8; 259];
        for s in 0..29 {
            let base = LEN_BASE[s] as usize;
            let next = if s == 28 { 259 } else { LEN_BASE[s + 1] as usize };
            for l in base..next {
                len_sym[l] = (257 + s) as u16;
                len_ex[l] = LEN_EXTRA[s];
            }
        }
        len_sym[258] = 285;
        len_ex[258] = 0;

        let mut dist_sym = vec![0u8; 32769];
        for s in 0..30 {
            let base = DIST_BASE[s] as usize;
            let next = if s == 29 { 32769 } else { DIST_BASE[s + 1] as usize };
            for d in base..next {
                dist_sym[d] = s as u8;
            }
        }

        let mut fixed_ll_lens = [0u8; 288];
        for i in 0..=143 {
            fixed_ll_lens[i] = 8;
        }
        for i in 144..=255 {
            fixed_ll_lens[i] = 9;
        }
        for i in 256..=279 {
            fixed_ll_lens[i] = 7;
        }
        for i in 280..=287 {
            fixed_ll_lens[i] = 8;
        }
        let fixed_d_lens = [5u8; 30];
        let mut fixed_ll_codes = [0u16; 288];
        let mut fixed_d_codes = [0u16; 30];
        build_canonical_codes(&fixed_ll_lens, &mut fixed_ll_codes);
        build_canonical_codes(&fixed_d_lens, &mut fixed_d_codes);

        let mut log2_tbl = vec![0.0f64; 65537];
        for i in 1..=65536 {
            log2_tbl[i] = (i as f64).log2();
        }

        Self {
            len_sym,
            len_ex,
            dist_sym,
            fixed_ll_lens,
            fixed_ll_codes,
            fixed_d_lens,
            fixed_d_codes,
            log2_tbl,
        }
    }

    #[inline(always)]
    fn fast_log2(&self, v: u32) -> f64 {
        let u = v as usize;
        if u <= 65536 {
            self.log2_tbl[u]
        } else {
            (v as f64).log2()
        }
    }
}

#[inline(always)]
fn hash3(data: &[u8], pos: usize) -> usize {
    let v = (data[pos] as u32) | ((data[pos + 1] as u32) << 8) | ((data[pos + 2] as u32) << 16);
    ((v.wrapping_mul(0x1e35a7bd)) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

#[inline(always)]
fn hash_rle(run_len: usize, byte_val: u8, next_byte: u8) -> usize {
    let r = if run_len < 258 { run_len as u32 } else { 258 };
    let v = r | ((byte_val as u32) << 9) | ((next_byte as u32) << 17);
    ((v.wrapping_mul(0x9e3779b1)) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

#[inline(always)]
fn hash_bpp_run(run_len: usize, data: &[u8], pos: usize, bpp: usize, next_byte: u8) -> usize {
    let r = if run_len < 258 { run_len as u32 } else { 258 };
    let mut pix = (data[pos] as u32) | ((data[pos + 1] as u32) << 8);
    if bpp >= 3 {
        pix ^= (data[pos + 2] as u32) << 16;
    }
    if bpp >= 4 {
        pix ^= (data[pos + 3] as u32) << 24;
    }
    let v = pix
        .wrapping_mul(0x85ebca6b)
        ^ (r.wrapping_mul(0xc2b2ae35))
        ^ ((next_byte as u32).wrapping_mul(0x27d4eb2f));
    ((v.wrapping_mul(0x9e3779b1)) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    let mut i = 0;
    let n = data.len();
    while i < n {
        let end = (i + 5552).min(n);
        while i < end {
            a += data[i] as u32;
            b += a;
            i += 1;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

fn build_canonical_codes(lens: &[u8], codes: &mut [u16]) {
    let mut bl_count = [0u32; 16];
    for &l in lens {
        bl_count[l as usize] += 1;
    }
    bl_count[0] = 0;
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (i, &l) in lens.iter().enumerate() {
        if l != 0 {
            let mut c = next_code[l as usize];
            next_code[l as usize] += 1;
            let mut r = 0u32;
            for _ in 0..l {
                r = (r << 1) | (c & 1);
                c >>= 1;
            }
            codes[i] = r as u16;
        } else {
            codes[i] = 0;
        }
    }
}

fn package_merge(freqs: &[u32], max_bits: usize, out_lens: &mut [u8]) {
    for x in out_lens.iter_mut() {
        *x = 0;
    }
    let mut pairs: [(u32, u16); 288] = [(0, 0); 288];
    let mut k = 0usize;
    for (i, &f) in freqs.iter().enumerate() {
        if f > 0 {
            pairs[k] = (f, i as u16);
            k += 1;
        }
    }
    if k == 0 {
        return;
    }
    if k == 1 {
        out_lens[pairs[0].1 as usize] = 1;
        return;
    }
    if k == 2 {
        out_lens[pairs[0].1 as usize] = 1;
        out_lens[pairs[1].1 as usize] = 1;
        return;
    }
    pairs[..k].sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut s_sym = [0u16; 288];
    let mut s_wt = [0u64; 288];
    for i in 0..k {
        s_wt[i] = pairs[i].0 as u64;
        s_sym[i] = pairs[i].1;
    }

    // Zero-heap-allocation fixed stack buffers (k <= 286 -> stride <= 572, max_bits <= 15)
    let stride = 2 * k;
    let mut level_type = [0i16; 15 * 576];
    let mut level_left = [0u16; 15 * 576];
    let mut level_right = [0u16; 15 * 576];
    let mut level_len = [0usize; 16];

    let mut prev_wt = [0u64; 576];
    let mut cur_wt = [0u64; 576];
    for i in 0..k {
        prev_wt[i] = s_wt[i];
        level_type[i] = i as i16;
    }
    let mut prev_count = k;
    level_len[0] = k;

    for lv in 1..max_bits {
        let pkg_count = prev_count >> 1;
        let base = lv * stride;
        let mut li = 0usize;
        let mut pi = 0usize;
        let mut oi = 0usize;
        while li < k && pi < pkg_count {
            let pw = prev_wt[pi * 2] + prev_wt[pi * 2 + 1];
            if s_wt[li] <= pw {
                cur_wt[oi] = s_wt[li];
                level_type[base + oi] = li as i16;
                li += 1;
            } else {
                cur_wt[oi] = pw;
                level_type[base + oi] = -1;
                level_left[base + oi] = (pi * 2) as u16;
                level_right[base + oi] = (pi * 2 + 1) as u16;
                pi += 1;
            }
            oi += 1;
        }
        while li < k {
            cur_wt[oi] = s_wt[li];
            level_type[base + oi] = li as i16;
            li += 1;
            oi += 1;
        }
        while pi < pkg_count {
            cur_wt[oi] = prev_wt[pi * 2] + prev_wt[pi * 2 + 1];
            level_type[base + oi] = -1;
            level_left[base + oi] = (pi * 2) as u16;
            level_right[base + oi] = (pi * 2 + 1) as u16;
            pi += 1;
            oi += 1;
        }
        level_len[lv] = oi;
        prev_count = oi;
        prev_wt[..oi].copy_from_slice(&cur_wt[..oi]);
    }

    let mut leaf_lens = [0u8; 288];
    let mut cur_use = [0u32; 576];
    let mut next_use = [0u32; 576];
    let take = 2 * k - 2;
    for i in 0..take {
        cur_use[i] = 1;
    }
    for lv in (1..max_bits).rev() {
        let n = level_len[lv];
        let prev_n = level_len[lv - 1];
        for x in &mut next_use[..prev_n] {
            *x = 0;
        }
        let base = lv * stride;
        for i in 0..n {
            let u = cur_use[i];
            if u == 0 {
                continue;
            }
            let t = level_type[base + i];
            if t >= 0 {
                leaf_lens[t as usize] += u as u8;
            } else {
                next_use[level_left[base + i] as usize] += u;
                next_use[level_right[base + i] as usize] += u;
            }
        }
        cur_use[..prev_n].copy_from_slice(&next_use[..prev_n]);
    }
    for i in 0..k {
        leaf_lens[i] += cur_use[i] as u8;
        out_lens[s_sym[i] as usize] = leaf_lens[i];
    }
}

fn optimize_counts_for_rle(counts: &[u32], out: &mut [u32]) {
    let mut n = counts.len();
    for i in 0..n {
        out[i] = counts[i];
    }
    while n > 0 && out[n - 1] == 0 {
        n -= 1;
    }
    if n == 0 {
        return;
    }
    let mut good = [0u8; 288];
    let mut symbol = out[0];
    let mut stride = 0usize;
    for i in 0..=n {
        if i == n || out[i] != symbol {
            if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
                for k in 0..stride {
                    good[i - k - 1] = 1;
                }
            }
            stride = 1;
            if i < n {
                symbol = out[i];
            }
        } else {
            stride += 1;
        }
    }

    stride = 0;
    let mut limit = out[0] as i64;
    let mut sum = 0u64;
    for i in 0..=n {
        if i == n
            || good[i] != 0
            || (i > 0 && good[i - 1] != 0)
            || (out[i] as i64 - limit).abs() >= 4
        {
            if stride >= 4 || (stride >= 3 && sum == 0) {
                let mut cnt = ((sum + (stride as u64) / 2) / (stride as u64)) as u32;
                if cnt < 1 {
                    cnt = 1;
                }
                if sum == 0 {
                    cnt = 0;
                }
                for k in 0..stride {
                    out[i - k - 1] = cnt;
                }
            }
            stride = 0;
            sum = 0;
            if i + 3 < n {
                limit = ((out[i] + out[i + 1] + out[i + 2] + out[i + 3] + 2) / 4) as i64;
            } else if i < n {
                limit = out[i] as i64;
            } else {
                limit = 0;
            }
        }
        stride += 1;
        if i < n {
            sum += out[i] as u64;
        }
    }
    for i in 0..counts.len() {
        if counts[i] > 0 && out[i] == 0 {
            out[i] = 1;
        }
    }
}

struct RleResult {
    syms: Vec<u8>,
    ex: Vec<u8>,
    cl_freq: [u32; 19],
}

fn rle_encode_lengths(combined: &[u8], use16: bool, use17: bool) -> RleResult {
    let n = combined.len();
    let mut syms = Vec::with_capacity(n);
    let mut ex = Vec::with_capacity(n);
    let mut cl_freq = [0u32; 19];
    let mut i = 0;
    while i < n {
        let v = combined[i];
        let mut run = 1;
        while i + run < n && combined[i + run] == v {
            run += 1;
        }
        if v == 0 && use17 && run >= 3 {
            let mut rem = run;
            while rem >= 11 {
                let take = rem.min(138);
                let adj = if rem - take > 0 && rem - take < 3 && take > 13 {
                    rem - 3
                } else {
                    take
                };
                syms.push(18);
                ex.push((adj - 11) as u8);
                cl_freq[18] += 1;
                rem -= adj;
            }
            while rem >= 3 {
                let take = rem.min(10);
                syms.push(17);
                ex.push((take - 3) as u8);
                cl_freq[17] += 1;
                rem -= take;
            }
            while rem > 0 {
                syms.push(0);
                ex.push(0);
                cl_freq[0] += 1;
                rem -= 1;
            }
        } else if v != 0 && use16 && run >= 4 {
            syms.push(v);
            ex.push(0);
            cl_freq[v as usize] += 1;
            let mut rem = run - 1;
            while rem >= 3 {
                let take = rem.min(6);
                let adj = if rem - take > 0 && rem - take < 3 && take > 4 {
                    rem - 3
                } else {
                    take
                };
                syms.push(16);
                ex.push((adj - 3) as u8);
                cl_freq[16] += 1;
                rem -= adj;
            }
            while rem > 0 {
                syms.push(v);
                ex.push(0);
                cl_freq[v as usize] += 1;
                rem -= 1;
            }
        } else {
            for _ in 0..run {
                syms.push(v);
                ex.push(0);
                cl_freq[v as usize] += 1;
            }
        }
        i += run;
    }
    RleResult { syms, ex, cl_freq }
}

fn rle_encode_dp(combined: &[u8], cl_lens: &[u8; 19]) -> Option<RleResult> {
    let n = combined.len();
    let mut dp = [u32::MAX / 2; 320];
    let mut prev_pos = [0u16; 320];
    let mut prev_sym = [0u8; 320];
    let mut prev_ex = [0u8; 320];
    dp[0] = 0;

    for i in 0..n {
        let base = dp[i];
        if base >= u32::MAX / 4 {
            continue;
        }
        let v = combined[i];
        if cl_lens[v as usize] > 0 {
            let c = base + cl_lens[v as usize] as u32;
            if c < dp[i + 1] {
                dp[i + 1] = c;
                prev_pos[i + 1] = i as u16;
                prev_sym[i + 1] = v;
                prev_ex[i + 1] = 0;
            }
        }
        if i > 0 && combined[i - 1] == v && cl_lens[16] > 0 {
            let cost16 = base + cl_lens[16] as u32 + 2;
            let mut max_r = 0;
            while max_r < 6 && i + max_r < n && combined[i + max_r] == v {
                max_r += 1;
            }
            for r in 3..=max_r {
                if cost16 < dp[i + r] {
                    dp[i + r] = cost16;
                    prev_pos[i + r] = i as u16;
                    prev_sym[i + r] = 16;
                    prev_ex[i + r] = (r - 3) as u8;
                }
            }
        }
        if v == 0 {
            let mut max_z = 0;
            while max_z < 138 && i + max_z < n && combined[i + max_z] == 0 {
                max_z += 1;
            }
            if max_z >= 3 && cl_lens[17] > 0 {
                let cost17 = base + cl_lens[17] as u32 + 3;
                let lim = max_z.min(10);
                for r in 3..=lim {
                    if cost17 < dp[i + r] {
                        dp[i + r] = cost17;
                        prev_pos[i + r] = i as u16;
                        prev_sym[i + r] = 17;
                        prev_ex[i + r] = (r - 3) as u8;
                    }
                }
            }
            if max_z >= 11 && cl_lens[18] > 0 {
                let cost18 = base + cl_lens[18] as u32 + 7;
                for r in 11..=max_z {
                    if cost18 < dp[i + r] {
                        dp[i + r] = cost18;
                        prev_pos[i + r] = i as u16;
                        prev_sym[i + r] = 18;
                        prev_ex[i + r] = (r - 11) as u8;
                    }
                }
            }
        }
    }
    if dp[n] >= u32::MAX / 4 {
        return None;
    }
    let mut syms = Vec::with_capacity(n);
    let mut ex = Vec::with_capacity(n);
    let mut cl_freq = [0u32; 19];
    let mut p = n;
    while p > 0 {
        let s = prev_sym[p];
        syms.push(s);
        ex.push(prev_ex[p]);
        cl_freq[s as usize] += 1;
        p = prev_pos[p] as usize;
    }
    syms.reverse();
    ex.reverse();
    Some(RleResult { syms, ex, cl_freq })
}

#[derive(Clone)]
struct DynHeader {
    hlit: usize,
    hdist: usize,
    hclen: usize,
    cl_lens: [u8; 19],
    syms: Vec<u8>,
    ex: Vec<u8>,
    header_bits: u64,
}

#[derive(Clone)]
struct DynTree {
    ll_lens: [u8; 286],
    d_lens: [u8; 30],
    hdr: DynHeader,
    total_bits: u64,
}

fn build_dynamic_header(ll_lens: &[u8; 286], d_lens: &[u8; 30], fast: bool) -> DynHeader {
    let mut hlit = 286;
    while hlit > 257 && ll_lens[hlit - 1] == 0 {
        hlit -= 1;
    }
    let mut hdist = 30;
    while hdist > 1 && d_lens[hdist - 1] == 0 {
        hdist -= 1;
    }
    let total_len = hlit + hdist;
    let mut combined = [0u8; 316];
    combined[..hlit].copy_from_slice(&ll_lens[..hlit]);
    combined[hlit..total_len].copy_from_slice(&d_lens[..hdist]);
    let comb = &combined[..total_len];

    let mut best: Option<DynHeader> = None;
    let eval_rle = |rle: RleResult, best: &mut Option<DynHeader>| -> [u8; 19] {
        let mut cl_lens = [0u8; 19];
        package_merge(&rle.cl_freq, 7, &mut cl_lens);
        let mut hclen = 19;
        while hclen > 4 && cl_lens[CL_ORDER[hclen - 1]] == 0 {
            hclen -= 1;
        }
        let mut bits = (14 + 3 * hclen) as u64;
        for s in 0..19 {
            let f = rle.cl_freq[s] as u64;
            if f > 0 {
                let eb = match s {
                    16 => 2,
                    17 => 3,
                    18 => 7,
                    _ => 0,
                };
                bits += f * (cl_lens[s] as u64 + eb);
            }
        }
        if best.as_ref().map_or(true, |b| bits < b.header_bits) {
            *best = Some(DynHeader {
                hlit,
                hdist,
                hclen,
                cl_lens,
                syms: rle.syms,
                ex: rle.ex,
                header_bits: bits,
            });
        }
        cl_lens
    };

    let c1 = eval_rle(rle_encode_lengths(comb, true, true), &mut best);
    if fast {
        return best.unwrap();
    }
    eval_rle(rle_encode_lengths(comb, true, false), &mut best);
    eval_rle(rle_encode_lengths(comb, false, true), &mut best);
    eval_rle(rle_encode_lengths(comb, false, false), &mut best);

    let best_cl = best.as_ref().unwrap().cl_lens;
    if let Some(dp1) = rle_encode_dp(comb, &best_cl) {
        let cl2 = eval_rle(dp1, &mut best);
        if let Some(dp2) = rle_encode_dp(comb, &cl2) {
            eval_rle(dp2, &mut best);
        }
    }
    if c1 != best_cl {
        if let Some(dp3) = rle_encode_dp(comb, &c1) {
            eval_rle(dp3, &mut best);
        }
    }
    best.unwrap()
}

fn build_best_dynamic_tree(
    ll_freq: &[u32; 286],
    d_freq: &[u32; 30],
    extra_bits_total: u64,
    fast: bool,
) -> DynTree {
    let eval_counts = |lf: &[u32; 286], df: &[u32; 30]| -> DynTree {
        let mut ll_lens = [0u8; 286];
        let mut d_lens = [0u8; 30];
        package_merge(lf, 15, &mut ll_lens);
        package_merge(df, 15, &mut d_lens);
        if !d_lens.iter().any(|&v| v > 0) {
            d_lens[0] = 1;
        }
        let mut data_bits = extra_bits_total;
        for i in 0..286 {
            if ll_freq[i] > 0 {
                data_bits += ll_freq[i] as u64 * ll_lens[i] as u64;
            }
        }
        for i in 0..30 {
            if d_freq[i] > 0 {
                data_bits += d_freq[i] as u64 * d_lens[i] as u64;
            }
        }
        let hdr = build_dynamic_header(&ll_lens, &d_lens, fast);
        let total_bits = 3 + hdr.header_bits + data_bits;
        DynTree {
            ll_lens,
            d_lens,
            hdr,
            total_bits,
        }
    };

    let mut best = eval_counts(ll_freq, d_freq);
    if fast {
        return best;
    }

    let mut opt_ll = [0u32; 286];
    let mut opt_d = [0u32; 30];
    optimize_counts_for_rle(ll_freq, &mut opt_ll);
    optimize_counts_for_rle(d_freq, &mut opt_d);

    let c1 = eval_counts(&opt_ll, d_freq);
    if c1.total_bits < best.total_bits {
        best = c1;
    }
    let c2 = eval_counts(&opt_ll, &opt_d);
    if c2.total_bits < best.total_bits {
        best = c2;
    }
    let c3 = eval_counts(ll_freq, &opt_d);
    if c3.total_bits < best.total_bits {
        best = c3;
    }

    // Exact Kraft-preserving (sum == 32768) code-length RLE repair & 0-hole bridging!
    // Guaranteed valid complete canonical Huffman tree:
    //   Move A (Swap): lens[i] (L -> L+1) & lens[j] (L+1 -> L) => net Kraft delta = 0
    //   Move B (0-hole): lens[hole] (0 -> L+1) & lens[j] (L -> L+1) => net Kraft delta = 0
    let mut ll = best.ll_lens;
    let dl = best.d_lens;
    let mut best_bits = best.total_bits;
    let mut improved_rle = false;

    let calc_data_bits = |ll_a: &[u8; 286], d_a: &[u8; 30]| -> u64 {
        let mut db = extra_bits_total;
        for i in 0..286 {
            if ll_freq[i] > 0 {
                db += ll_freq[i] as u64 * ll_a[i] as u64;
            }
        }
        for i in 0..30 {
            if d_freq[i] > 0 {
                db += d_freq[i] as u64 * d_a[i] as u64;
            }
        }
        db
    };

    let active_ll = ll.iter().filter(|&&v| v > 0).count();
    if active_ll >= 3 {
        let mut hlit = 286usize;
        while hlit > 257 && ll[hlit - 1] == 0 {
            hlit -= 1;
        }
        let mut cur_data = calc_data_bits(&ll, &dl);
        for _pass in 0..2 {
            let mut pass_imp = false;
            // Find cheapest-to-lengthen symbol at each length L in 1..15 (min ll_freq[j])
            let mut min_freq_sym = [usize::MAX; 16];
            for idx in 0..hlit {
                let l = ll[idx] as usize;
                if l >= 1 && l < 15 {
                    let prev = min_freq_sym[l];
                    if prev == usize::MAX || ll_freq[idx] < ll_freq[prev] {
                        min_freq_sym[l] = idx;
                    }
                }
            }
            // Move B: Bridge single 0-holes `v, 0, v` (or `v, 0, v, v`) where v >= 2
            for i in 1..hlit.saturating_sub(1) {
                if ll[i] == 0 && ll[i - 1] >= 2 && ll[i - 1] == ll[i + 1] {
                    let v = ll[i - 1] as usize;
                    let donor = min_freq_sym[v - 1];
                    if donor != usize::MAX && ll_freq[donor] <= 20 && donor != i - 1 && donor != i + 1 {
                        let new_data = cur_data + ll_freq[donor] as u64;
                        if new_data < best_bits {
                            ll[i] = v as u8;
                            ll[donor] = v as u8;
                            let hdr = build_dynamic_header(&ll, &dl, false);
                            let tot = 3 + hdr.header_bits + new_data;
                            if tot < best_bits {
                                best_bits = tot;
                                cur_data = new_data;
                                best = DynTree { ll_lens: ll, d_lens: dl, hdr, total_bits: tot };
                                improved_rle = true;
                                pass_imp = true;
                                // Refresh donor for v-1
                                min_freq_sym[v - 1] = usize::MAX;
                                for idx in 0..hlit {
                                    if ll[idx] as usize == v - 1 {
                                        let prev = min_freq_sym[v - 1];
                                        if prev == usize::MAX || ll_freq[idx] < ll_freq[prev] {
                                            min_freq_sym[v - 1] = idx;
                                        }
                                    }
                                }
                            } else {
                                ll[i] = 0;
                                ll[donor] = (v - 1) as u8;
                            }
                        }
                    }
                }
            }
            // Move A: Smooth singleton bumps `v, v-1, v` or `v, v+1, v` via Kraft-preserving swap
            let mut max_freq_sym = [usize::MAX; 16];
            for idx in 0..hlit {
                let l = ll[idx] as usize;
                if l >= 2 && l <= 15 {
                    let prev = max_freq_sym[l];
                    if prev == usize::MAX || ll_freq[idx] > ll_freq[prev] {
                        max_freq_sym[l] = idx;
                    }
                }
            }
            for i in 1..hlit.saturating_sub(1) {
                let cur_l = ll[i] as usize;
                let v = ll[i - 1] as usize;
                if v >= 2 && v == ll[i + 1] as usize && cur_l != v && cur_l > 0 {
                    if cur_l + 1 == v {
                        // Lengthen i: v-1 -> v, shorten highest-freq symbol at length v -> v-1
                        let partner = max_freq_sym[v];
                        if partner != usize::MAX && partner != i && partner != i - 1 && partner != i + 1 {
                            let df = ll_freq[i] as i64 - ll_freq[partner] as i64;
                            if df <= 20 {
                                let new_data = (cur_data as i64 + df) as u64;
                                ll[i] = v as u8;
                                ll[partner] = (v - 1) as u8;
                                let hdr = build_dynamic_header(&ll, &dl, false);
                                let tot = 3 + hdr.header_bits + new_data;
                                if tot < best_bits {
                                    best_bits = tot;
                                    cur_data = new_data;
                                    best = DynTree { ll_lens: ll, d_lens: dl, hdr, total_bits: tot };
                                    improved_rle = true;
                                    pass_imp = true;
                                    max_freq_sym[v] = usize::MAX;
                                    for idx in 0..hlit {
                                        if ll[idx] as usize == v {
                                            let prev = max_freq_sym[v];
                                            if prev == usize::MAX || ll_freq[idx] > ll_freq[prev] {
                                                max_freq_sym[v] = idx;
                                            }
                                        }
                                    }
                                } else {
                                    ll[i] = cur_l as u8;
                                    ll[partner] = v as u8;
                                }
                            }
                        }
                    } else if cur_l == v + 1 && v < 15 {
                        // Shorten i: v+1 -> v, lengthen lowest-freq symbol at length v -> v+1
                        let partner = min_freq_sym[v];
                        if partner != usize::MAX && partner != i && partner != i - 1 && partner != i + 1 {
                            let df = ll_freq[partner] as i64 - ll_freq[i] as i64;
                            if df <= 20 {
                                let new_data = (cur_data as i64 + df) as u64;
                                ll[i] = v as u8;
                                ll[partner] = (v + 1) as u8;
                                let hdr = build_dynamic_header(&ll, &dl, false);
                                let tot = 3 + hdr.header_bits + new_data;
                                if tot < best_bits {
                                    best_bits = tot;
                                    cur_data = new_data;
                                    best = DynTree { ll_lens: ll, d_lens: dl, hdr, total_bits: tot };
                                    improved_rle = true;
                                    pass_imp = true;
                                    min_freq_sym[v] = usize::MAX;
                                    for idx in 0..hlit {
                                        if ll[idx] as usize == v {
                                            let prev = min_freq_sym[v];
                                            if prev == usize::MAX || ll_freq[idx] < ll_freq[prev] {
                                                min_freq_sym[v] = idx;
                                            }
                                        }
                                    }
                                } else {
                                    ll[i] = cur_l as u8;
                                    ll[partner] = v as u8;
                                }
                            }
                        }
                    }
                }
            }
            if !pass_imp {
                break;
            }
        }
    }
    let _ = improved_rle;
    best
}

fn fixed_block_bits(tbl: &Tables, ll_freq: &[u32; 286], d_freq: &[u32; 30], extra: u64) -> u64 {
    let mut bits = 3 + extra;
    for i in 0..286 {
        if ll_freq[i] > 0 {
            bits += ll_freq[i] as u64 * tbl.fixed_ll_lens[i] as u64;
        }
    }
    for i in 0..30 {
        if d_freq[i] > 0 {
            bits += d_freq[i] as u64 * 5;
        }
    }
    bits
}

fn stats_from_range(
    tbl: &Tables,
    lit_lens: &[u16],
    dists: &[u16],
    start: usize,
    end: usize,
) -> ([u32; 286], [u32; 30], u64) {
    let mut ll_freq = [0u32; 286];
    let mut d_freq = [0u32; 30];
    let mut extra = 0u64;
    for i in start..end {
        let d = dists[i] as usize;
        if d == 0 {
            ll_freq[lit_lens[i] as usize] += 1;
        } else {
            let l = lit_lens[i] as usize;
            ll_freq[tbl.len_sym[l] as usize] += 1;
            let ds = tbl.dist_sym[d] as usize;
            d_freq[ds] += 1;
            extra += (tbl.len_ex[l] + DIST_EXTRA[ds]) as u64;
        }
    }
    ll_freq[256] = 1;
    (ll_freq, d_freq, extra)
}

// ============================================================================
// BitWriter
// ============================================================================

struct BitWriter<'a> {
    out: &'a mut [u8],
    pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
    overflow: bool,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut [u8]) -> Self {
        Self {
            out,
            pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
            overflow: false,
        }
    }

    #[inline(always)]
    fn write_bits(&mut self, val: u32, n: u32) {
        self.bit_buf |= val << self.bit_cnt;
        self.bit_cnt += n;
        while self.bit_cnt >= 8 {
            if self.pos < self.out.len() {
                self.out[self.pos] = (self.bit_buf & 0xff) as u8;
                self.pos += 1;
            } else {
                self.overflow = true;
            }
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
        }
    }

    fn align_byte(&mut self) {
        if self.bit_cnt > 0 {
            if self.pos < self.out.len() {
                self.out[self.pos] = (self.bit_buf & 0xff) as u8;
                self.pos += 1;
            } else {
                self.overflow = true;
            }
            self.bit_buf = 0;
            self.bit_cnt = 0;
        }
    }

    #[inline(always)]
    fn write_byte(&mut self, b: u8) {
        if self.pos < self.out.len() {
            self.out[self.pos] = b;
            self.pos += 1;
        } else {
            self.overflow = true;
        }
    }

    fn write_bytes(&mut self, data: &[u8]) {
        if self.pos + data.len() <= self.out.len() {
            self.out[self.pos..self.pos + data.len()].copy_from_slice(data);
            self.pos += data.len();
        } else {
            self.overflow = true;
        }
    }
}

fn write_block_range(
    tbl: &Tables,
    bw: &mut BitWriter<'_>,
    is_final: bool,
    raw_slice: &[u8],
    lit_lens: &[u16],
    dists: &[u16],
) {
    let raw_len = raw_slice.len();
    let (ll_freq, d_freq, extra) = stats_from_range(tbl, lit_lens, dists, 0, lit_lens.len());
    let dyn_tree = build_best_dynamic_tree(&ll_freq, &d_freq, extra, false);
    let fix_bits = fixed_block_bits(tbl, &ll_freq, &d_freq, extra);
    let stored_bits = 50 + (raw_len as u64) * 8;

    if stored_bits < dyn_tree.total_bits && stored_bits < fix_bits && raw_len <= 65535 {
        bw.write_bits(if is_final { 1 } else { 0 }, 1);
        bw.write_bits(0, 2);
        bw.align_byte();
        bw.write_byte((raw_len & 0xff) as u8);
        bw.write_byte(((raw_len >> 8) & 0xff) as u8);
        bw.write_byte(((!raw_len) & 0xff) as u8);
        bw.write_byte((((!raw_len) >> 8) & 0xff) as u8);
        bw.write_bytes(raw_slice);
        return;
    }

    let mut ll_codes = [0u16; 288];
    let mut ll_lens_arr = [0u8; 288];
    let mut d_codes = [0u16; 30];
    let d_lens_arr: [u8; 30];

    if fix_bits < dyn_tree.total_bits {
        bw.write_bits(if is_final { 1 } else { 0 }, 1);
        bw.write_bits(1, 2);
        ll_codes = tbl.fixed_ll_codes;
        ll_lens_arr = tbl.fixed_ll_lens;
        d_codes = tbl.fixed_d_codes;
        d_lens_arr = tbl.fixed_d_lens;
    } else {
        bw.write_bits(if is_final { 1 } else { 0 }, 1);
        bw.write_bits(2, 2);
        let hdr = &dyn_tree.hdr;
        bw.write_bits((hdr.hlit - 257) as u32, 5);
        bw.write_bits((hdr.hdist - 1) as u32, 5);
        bw.write_bits((hdr.hclen - 4) as u32, 4);
        for i in 0..hdr.hclen {
            bw.write_bits(hdr.cl_lens[CL_ORDER[i]] as u32, 3);
        }
        let mut cl_codes = [0u16; 19];
        build_canonical_codes(&hdr.cl_lens, &mut cl_codes);
        for i in 0..hdr.syms.len() {
            let s = hdr.syms[i] as usize;
            bw.write_bits(cl_codes[s] as u32, hdr.cl_lens[s] as u32);
            if s == 16 {
                bw.write_bits(hdr.ex[i] as u32, 2);
            } else if s == 17 {
                bw.write_bits(hdr.ex[i] as u32, 3);
            } else if s == 18 {
                bw.write_bits(hdr.ex[i] as u32, 7);
            }
        }
        ll_lens_arr[..286].copy_from_slice(&dyn_tree.ll_lens);
        d_lens_arr = dyn_tree.d_lens;
        build_canonical_codes(&ll_lens_arr[..286], &mut ll_codes[..286]);
        build_canonical_codes(&d_lens_arr, &mut d_codes);
    }

    for i in 0..lit_lens.len() {
        let d = dists[i] as usize;
        if d == 0 {
            let sym = lit_lens[i] as usize;
            bw.write_bits(ll_codes[sym] as u32, ll_lens_arr[sym] as u32);
        } else {
            let l = lit_lens[i] as usize;
            let lsym = tbl.len_sym[l] as usize;
            bw.write_bits(ll_codes[lsym] as u32, ll_lens_arr[lsym] as u32);
            let lex = tbl.len_ex[l];
            if lex > 0 {
                bw.write_bits((l - LEN_BASE[lsym - 257] as usize) as u32, lex as u32);
            }
            let dsym = tbl.dist_sym[d] as usize;
            bw.write_bits(d_codes[dsym] as u32, d_lens_arr[dsym] as u32);
            let dex = DIST_EXTRA[dsym];
            if dex > 0 {
                bw.write_bits((d - DIST_BASE[dsym] as usize) as u32, dex as u32);
            }
        }
    }
    bw.write_bits(ll_codes[256] as u32, ll_lens_arr[256] as u32);
}

// ============================================================================
// Run Precomputation, Fast Lazy LZ77 & Golden-Section Block Splitting
// ============================================================================

struct Runs {
    run_left: Vec<u16>,
    run_bpp: Option<Vec<u16>>,
    run_row: Option<Vec<u16>>,
}

fn compute_runs(data: &[u8], bpp: usize, row_stride: usize) -> Runs {
    let n = data.len();
    let mut run_left = vec![0u16; n];
    if n > 0 {
        run_left[n - 1] = 1;
        for i in (0..n - 1).rev() {
            if data[i] == data[i + 1] {
                let v = run_left[i + 1] as u32 + 1;
                run_left[i] = v.min(65535) as u16;
            } else {
                run_left[i] = 1;
            }
        }
    }
    let run_bpp = if bpp > 1 {
        let mut rb = vec![0u16; n];
        for i in (bpp..n).rev() {
            if data[i] == data[i - bpp] {
                let v = if i + 1 < n { rb[i + 1] as u32 + 1 } else { 1 };
                rb[i] = v.min(65535) as u16;
            }
        }
        Some(rb)
    } else {
        None
    };
    let run_row = if row_stride > 1 && row_stride <= WIN_SIZE {
        let mut rr = vec![0u16; n];
        for i in (row_stride..n).rev() {
            if data[i] == data[i - row_stride] {
                let v = if i + 1 < n { rr[i + 1] as u32 + 1 } else { 1 };
                rr[i] = v.min(65535) as u16;
            }
        }
        Some(rr)
    } else {
        None
    };
    Runs {
        run_left,
        run_bpp,
        run_row,
    }
}

struct LzTokens {
    lit_lens: Vec<u16>,
    dists: Vec<u16>,
}

fn fast_lazy_lz77(
    data: &[u8],
    runs: &Runs,
    row_stride: usize,
    bpp: usize,
    max_chain: usize,
) -> LzTokens {
    let n = data.len();
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; WIN_SIZE];
    let cap = (n >> 2).max(4096);
    let mut lit_lens = Vec::with_capacity(cap);
    let mut dists = Vec::with_capacity(cap);

    let run_left = &runs.run_left;
    let run_bpp = runs.run_bpp.as_deref();
    let run_row = runs.run_row.as_deref();

    let insert = |p: usize, head: &mut [i32], prev: &mut [i32]| {
        if p + 2 >= n {
            return;
        }
        let h = hash3(data, p);
        prev[p & WIN_MASK] = head[h];
        head[h] = p as i32;
    };

    let find_best = |p: usize, head: &[i32], prev: &[i32]| -> (usize, usize) {
        let max_len = (n - p).min(258);
        if max_len < 3 {
            return (0, 0);
        }
        if p > 0 && data[p] == data[p - 1] && (run_left[p] as usize) >= max_len {
            return (max_len, 1);
        }
        if let Some(rb) = run_bpp {
            if (rb[p] as usize) >= max_len {
                return (max_len, bpp);
            }
        }
        let mut best_l = 2usize;
        let mut best_d = 0usize;
        if p > 0 && data[p] == data[p - 1] {
            let rl = (run_left[p] as usize).min(max_len);
            if rl > best_l {
                best_l = rl;
                best_d = 1;
            }
        }
        if let Some(rb) = run_bpp {
            let v = (rb[p] as usize).min(max_len);
            if v > best_l {
                best_l = v;
                best_d = bpp;
            }
        }
        if let Some(rr) = run_row {
            let v = (rr[p] as usize).min(max_len);
            if v > best_l {
                best_l = v;
                best_d = row_stride;
            }
        }
        if best_l == max_len {
            return (best_l, best_d);
        }
        let limit = p.saturating_sub(WIN_SIZE) as i32;
        let b0 = data[p];
        let b1 = data[p + 1];
        let b2 = data[p + 2];
        let mut cand = head[hash3(data, p)];
        let mut chain = max_chain;
        while cand >= limit && chain > 0 && best_l < max_len {
            chain -= 1;
            let c = cand as usize;
            if data[c + best_l] == data[p + best_l]
                && data[c] == b0
                && data[c + 1] == b1
                && data[c + 2] == b2
            {
                let l = match_len_fast(data, c, p, 3, max_len);
                if l > best_l {
                    best_l = l;
                    best_d = p - c;
                    if best_l == max_len {
                        break;
                    }
                }
            }
            let nxt = prev[c & WIN_MASK];
            if nxt >= cand {
                break;
            }
            cand = nxt;
        }
        if best_l >= 3 {
            (best_l, best_d)
        } else {
            (0, 0)
        }
    };

    let mut p = 0usize;
    while p < n {
        if p > 0 && data[p] == data[p - 1] && run_left[p] >= 258 {
            lit_lens.push(258);
            dists.push(1);
            if run_left[p] < 516 {
                for k in 0..258 {
                    insert(p + k, &mut head, &mut prev);
                }
            } else {
                insert(p, &mut head, &mut prev);
            }
            p += 258;
            continue;
        }
        if let Some(rb) = run_bpp {
            if rb[p] >= 258 && run_left[p] < 258 {
                lit_lens.push(258);
                dists.push(bpp as u16);
                if rb[p] < 516 {
                    for k in 0..258 {
                        insert(p + k, &mut head, &mut prev);
                    }
                } else {
                    for k in 0..bpp {
                        insert(p + k, &mut head, &mut prev);
                    }
                }
                p += 258;
                continue;
            }
        }
        let (l1, d1) = find_best(p, &head, &prev);
        insert(p, &mut head, &mut prev);
        if l1 >= 3 && l1 < 128 && p + 1 < n {
            let (l2, _) = find_best(p + 1, &head, &prev);
            if l2 > l1 + 1 {
                lit_lens.push(data[p] as u16);
                dists.push(0);
                p += 1;
                continue;
            }
        }
        if l1 >= 3 {
            lit_lens.push(l1 as u16);
            dists.push(d1 as u16);
            let end = p + l1;
            for q in p + 1..end {
                insert(q, &mut head, &mut prev);
            }
            p = end;
        } else {
            lit_lens.push(data[p] as u16);
            dists.push(0);
            p += 1;
        }
    }
    LzTokens { lit_lens, dists }
}

fn entropy_from_freqs(
    tbl: &Tables,
    ll_freq: &[u32; 286],
    d_freq: &[u32; 30],
    ll_tot: u32,
    d_tot: u32,
    extra: u64,
) -> f64 {
    let mut bits = extra as f64 + 64.0;
    let log_ll = tbl.fast_log2(ll_tot);
    let mut act_ll = 0u32;
    for &c in ll_freq {
        if c > 0 {
            act_ll += 1;
            let b = (log_ll - tbl.fast_log2(c)).max(1.0);
            bits += (c as f64) * b;
        }
    }
    let mut act_d = 0u32;
    if d_tot > 0 {
        let log_d = tbl.fast_log2(d_tot);
        for &c in d_freq {
            if c > 0 {
                act_d += 1;
                let b = (log_d - tbl.fast_log2(c)).max(1.0);
                bits += (c as f64) * b;
            }
        }
    }
    bits + (act_ll as f64) * 2.2 + (act_d as f64) * 2.8
}

fn exact_range_bits(
    tbl: &Tables,
    lit_lens: &[u16],
    dists: &[u16],
    l: usize,
    r: usize,
    fast: bool,
) -> u64 {
    let (ll_freq, d_freq, extra) = stats_from_range(tbl, lit_lens, dists, l, r);
    let dyn_b = build_best_dynamic_tree(&ll_freq, &d_freq, extra, fast).total_bits;
    let fix_b = fixed_block_bits(tbl, &ll_freq, &d_freq, extra);
    dyn_b.min(fix_b)
}

/// O(r - l) single-pass prefix-sum golden-zoom split finder (~50x faster than rescanning per probe!)
fn find_best_split_fast(
    tbl: &Tables,
    lit_lens: &[u16],
    dists: &[u16],
    l: usize,
    r: usize,
    npts_max: usize,
) -> (usize, u64) {
    if r <= l + 128 {
        return (0, u64::MAX);
    }
    let (tot_ll, tot_d, tot_extra) = stats_from_range(tbl, lit_lens, dists, l, r);
    let tot_ll_cnt: u32 = tot_ll.iter().sum();
    let tot_d_cnt: u32 = tot_d.iter().sum();

    // Helper: evaluate evenly spaced points in [lo..hi] in ONE linear scan from `lo` to `hi`,
    // starting from prefix counts `(cur_ll, cur_d, cur_ll_cnt, cur_d_cnt, cur_extra)` at `lo`.
    let scan_window = |lo: usize,
                       hi: usize,
                       base_ll: &[u32; 286],
                       base_d: &[u32; 30],
                       base_ll_cnt: u32,
                       base_d_cnt: u32,
                       base_extra: u64|
     -> (usize, f64, usize) {
        const MAX_NPTS: usize = 47;
        let npts_use = npts_max.min(MAX_NPTS).max(7);
        let span = hi - lo;
        let step = (span / (npts_use + 1)).max(1);
        let mut pts = [0usize; MAX_NPTS];
        let mut n_pts = 0usize;
        for k in 1..=npts_use {
            let mid = lo + k * step;
            if mid > l + 64 && mid < r - 64 && mid < hi {
                if n_pts == 0 || mid > pts[n_pts - 1] {
                    pts[n_pts] = mid;
                    n_pts += 1;
                }
            }
        }
        if n_pts == 0 {
            return (0, f64::INFINITY, step);
        }
        let mut left_ll = *base_ll;
        let mut left_d = *base_d;
        let mut left_ll_cnt = base_ll_cnt;
        let mut left_d_cnt = base_d_cnt;
        let mut left_ex = base_extra;
        let mut right_ll = [0u32; 286];
        let mut right_d = [0u32; 30];

        let mut best_p = 0usize;
        let mut best_c = f64::INFINITY;
        let mut pos = lo;

        for pi in 0..n_pts {
            let target = pts[pi];
            while pos < target {
                let d = dists[pos] as usize;
                if d == 0 {
                    left_ll[lit_lens[pos] as usize] += 1;
                    left_ll_cnt += 1;
                } else {
                    let len = lit_lens[pos] as usize;
                    left_ll[tbl.len_sym[len] as usize] += 1;
                    left_ll_cnt += 1;
                    let ds = tbl.dist_sym[d] as usize;
                    left_d[ds] += 1;
                    left_d_cnt += 1;
                    left_ex += (tbl.len_ex[len] + DIST_EXTRA[ds]) as u64;
                }
                pos += 1;
            }
            for s in 0..286 {
                right_ll[s] = tot_ll[s] - left_ll[s];
            }
            right_ll[256] = 1;
            for s in 0..30 {
                right_d[s] = tot_d[s] - left_d[s];
            }
            let right_ll_cnt = tot_ll_cnt - left_ll_cnt + 1;
            let right_d_cnt = tot_d_cnt - left_d_cnt;
            let right_ex = tot_extra - left_ex;

            let c = entropy_from_freqs(tbl, &left_ll, &left_d, left_ll_cnt, left_d_cnt, left_ex)
                + entropy_from_freqs(tbl, &right_ll, &right_d, right_ll_cnt, right_d_cnt, right_ex);
            if c < best_c {
                best_c = c;
                best_p = target;
            }
        }
        (best_p, best_c, step)
    };

    let mut cur_lo = l;
    let mut cur_hi = r;
    let mut base_ll = [0u32; 286];
    let mut base_d = [0u32; 30];
    base_ll[256] = 1;
    let mut base_ll_cnt = 1u32;
    let mut base_d_cnt = 0u32;
    let mut base_ex = 0u64;

    let mut best_pos = 0usize;
    let mut best_cost = f64::INFINITY;
    let mut last_step = 64usize;

    for _round in 0..5 {
        if cur_hi <= cur_lo + 64 {
            break;
        }
        let (p, c, step) = scan_window(cur_lo, cur_hi, &base_ll, &base_d, base_ll_cnt, base_d_cnt, base_ex);
        last_step = step.max(24);
        if p == 0 {
            break;
        }
        if c < best_cost {
            best_cost = c;
            best_pos = p;
        }
        let next_lo = p.saturating_sub(step).max(cur_lo);
        let next_hi = (p + step).min(cur_hi);
        // Advance prefix base from cur_lo to next_lo once!
        for pos in cur_lo..next_lo {
            let d = dists[pos] as usize;
            if d == 0 {
                base_ll[lit_lens[pos] as usize] += 1;
                base_ll_cnt += 1;
            } else {
                let len = lit_lens[pos] as usize;
                base_ll[tbl.len_sym[len] as usize] += 1;
                base_ll_cnt += 1;
                let ds = tbl.dist_sym[d] as usize;
                base_d[ds] += 1;
                base_d_cnt += 1;
                base_ex += (tbl.len_ex[len] + DIST_EXTRA[ds]) as u64;
            }
        }
        cur_lo = next_lo;
        cur_hi = next_hi;
        if step < 12 {
            break;
        }
    }

    if best_pos == 0 {
        return (0, u64::MAX);
    }
    let mut exact_pos = best_pos;
    let mut exact_cost = exact_range_bits(tbl, lit_lens, dists, l, best_pos, true)
        + exact_range_bits(tbl, lit_lens, dists, best_pos, r, true);
    let delta = last_step.min(96) as isize;
    let half_d = (delta / 2).max(1);
    for &off in &[-delta, -half_d, half_d, delta] {
        let p = (best_pos as isize + off) as usize;
        if p > l + 64 && p < r - 64 {
            let c = exact_range_bits(tbl, lit_lens, dists, l, p, true)
                + exact_range_bits(tbl, lit_lens, dists, p, r, true);
            if c < exact_cost {
                exact_cost = c;
                exact_pos = p;
            }
        }
    }
    (exact_pos, exact_cost)
}

fn extract_filter_regime_cuts(
    data: &[u8],
    b_start: usize,
    b_end: usize,
    row_stride: usize,
    pos_arr: &[u32],
    count: usize,
) -> Vec<usize> {
    let mut cuts = Vec::new();
    if row_stride <= 1 || b_end <= b_start + row_stride * 16 || count < 128 || pos_arr.len() != count {
        return cuts;
    }
    let y0 = (b_start + row_stride - 1) / row_stride;
    let y1 = b_end / row_stride;
    if y1 <= y0 + 16 {
        return cuts;
    }
    let mut runs: Vec<(usize, usize, u8)> = Vec::new();
    let mut ry = y0;
    while ry < y1 {
        let f = data[ry * row_stride];
        let mut end = ry + 1;
        while end < y1 && data[end * row_stride] == f {
            end += 1;
        }
        runs.push((ry, end, f));
        ry = end;
    }
    // Collect exact transition rows bordering any macro filter run (>= 8 rows)
    let mut boundary_rows: Vec<(usize, usize)> = Vec::new(); // (row, priority_run_len)
    for (ri, &(a, b, _)) in runs.iter().enumerate() {
        let rlen = b - a;
        if rlen >= 8 {
            if a > y0 {
                let prev_len = if ri > 0 { runs[ri - 1].1 - runs[ri - 1].0 } else { 0 };
                boundary_rows.push((a, rlen.max(prev_len)));
            }
            if b < y1 {
                let next_len = if ri + 1 < runs.len() { runs[ri + 1].1 - runs[ri + 1].0 } else { 0 };
                boundary_rows.push((b, rlen.max(next_len)));
            }
        }
    }
    if boundary_rows.is_empty() {
        return cuts;
    }
    boundary_rows.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    boundary_rows.dedup_by_key(|x| x.0);
    if boundary_rows.len() > 36 {
        boundary_rows.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        boundary_rows.truncate(36);
        boundary_rows.sort_unstable_by_key(|x| x.0);
    }
    for (row, _) in boundary_rows {
        let off = row * row_stride;
        if off > b_start && off < b_end {
            let tidx = pos_arr.partition_point(|&p| (p as usize) < off);
            if tidx > 64 && tidx + 64 < count {
                cuts.push(tidx);
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Run-Aware Optimal O(M^2) DP Block Splitter + Greedy Binary Splitter:
/// Combines macro PNG filter-regime transition cuts, greedy splits, and uniform grid cuts,
/// then solves exact shortest-path multi-cut DP (isolating A|B|A sandwiched regimes!) and
/// keeps the strictly lowest-bit partition verified by exact dynamic/fixed Huffman costing!
fn split_token_stream_dp(
    tbl: &Tables,
    data: &[u8],
    b_start: usize,
    b_end: usize,
    row_stride: usize,
    lit_lens: &[u16],
    dists: &[u16],
    pos_arr: &[u32],
    max_blocks: usize,
    min_tokens: usize,
    fast_eval: bool,
) -> Vec<usize> {
    let count = lit_lens.len();
    if count < min_tokens.max(128) || max_blocks <= 1 {
        return vec![0, count];
    }

    let greedy = split_token_stream(tbl, lit_lens, dists, max_blocks, min_tokens);
    if count < 256 {
        return greedy;
    }
    let eval_part = |sp: &[usize]| -> u64 {
        let mut tot = 0u64;
        for i in 0..sp.len() - 1 {
            tot += exact_range_bits(tbl, lit_lens, dists, sp[i], sp[i + 1], fast_eval);
        }
        tot
    };
    let mut best_splits = greedy.clone();
    let mut best_bits = eval_part(&best_splits);

    // Build candidate cut set (filter transitions + greedy cuts + uniform grid)
    let mut raw_cuts = Vec::with_capacity(88);
    raw_cuts.push(0usize);
    raw_cuts.push(count);
    for &g in &greedy {
        raw_cuts.push(g);
    }
    let reg_cuts = extract_filter_regime_cuts(data, b_start, b_end, row_stride, pos_arr, count);
    for &rc in &reg_cuts {
        raw_cuts.push(rc);
    }
    let grid_n = if fast_eval { 18usize } else { 28usize };
    let step = (count / (grid_n + 1)).max(64);
    for k in 1..=grid_n {
        let p = k * step;
        if p > 64 && p + 64 < count {
            raw_cuts.push(p);
        }
    }
    raw_cuts.sort_unstable();
    let mut cuts = Vec::with_capacity(raw_cuts.len());
    cuts.push(0usize);
    let min_gap = 48usize;
    for &c in &raw_cuts[1..] {
        if c == count {
            break;
        }
        if c >= *cuts.last().unwrap() + min_gap && c + min_gap <= count {
            cuts.push(c);
        }
    }
    cuts.push(count);
    if cuts.len() > 82 {
        // Prioritize filter-regime & greedy cuts, drop excess uniform grid cuts
        let mut keep = Vec::with_capacity(82);
        keep.push(0usize);
        keep.push(count);
        keep.extend_from_slice(&reg_cuts);
        keep.extend_from_slice(&greedy);
        keep.sort_unstable();
        keep.dedup();
        if keep.len() < 80 {
            let rem = 80 - keep.len();
            let st = (count / (rem + 1)).max(64);
            for k in 1..=rem {
                keep.push(k * st);
            }
        }
        keep.sort_unstable();
        cuts.clear();
        cuts.push(0usize);
        for &c in &keep {
            if c > 0 && c < count && c >= *cuts.last().unwrap() + min_gap && c + min_gap <= count {
                cuts.push(c);
                if cuts.len() >= 81 {
                    break;
                }
            }
        }
        cuts.push(count);
    }

    let m = cuts.len() - 1;
    if m < 2 {
        return best_splits;
    }

    // Build prefix frequency arrays in ONE linear pass O(count)
    let mut pref_ll = vec![[0u32; 286]; m + 1];
    let mut pref_d = vec![[0u32; 30]; m + 1];
    let mut pref_ll_tot = vec![0u32; m + 1];
    let mut pref_d_tot = vec![0u32; m + 1];
    let mut pref_ex = vec![0u64; m + 1];

    let mut cur_ll = [0u32; 286];
    let mut cur_d = [0u32; 30];
    let mut cur_ll_tot = 0u32;
    let mut cur_d_tot = 0u32;
    let mut cur_ex = 0u64;
    let mut pos = 0usize;
    for ci in 1..=m {
        let target = cuts[ci];
        while pos < target {
            let d = dists[pos] as usize;
            if d == 0 {
                cur_ll[lit_lens[pos] as usize] += 1;
                cur_ll_tot += 1;
            } else {
                let l = lit_lens[pos] as usize;
                cur_ll[tbl.len_sym[l] as usize] += 1;
                cur_ll_tot += 1;
                let ds = tbl.dist_sym[d] as usize;
                cur_d[ds] += 1;
                cur_d_tot += 1;
                cur_ex += (tbl.len_ex[l] + DIST_EXTRA[ds]) as u64;
            }
            pos += 1;
        }
        pref_ll[ci] = cur_ll;
        pref_d[ci] = cur_d;
        pref_ll_tot[ci] = cur_ll_tot;
        pref_d_tot[ci] = cur_d_tot;
        pref_ex[ci] = cur_ex;
    }

    // Precompute upper-triangular segment entropy costs once O(M^2 * 316) (~1ms for M=70!)
    let mut seg_cost = vec![0.0f64; (m + 1) * (m + 1)];
    let mut s_ll = [0u32; 286];
    let mut s_d = [0u32; 30];
    for i in 0..m {
        for j in (i + 1)..=m {
            for k in 0..286 {
                s_ll[k] = pref_ll[j][k] - pref_ll[i][k];
            }
            s_ll[256] = 1;
            for k in 0..30 {
                s_d[k] = pref_d[j][k] - pref_d[i][k];
            }
            let lt = pref_ll_tot[j] - pref_ll_tot[i] + 1;
            let dt = pref_d_tot[j] - pref_d_tot[i];
            let ex = pref_ex[j] - pref_ex[i];
            seg_cost[i * (m + 1) + j] = entropy_from_freqs(tbl, &s_ll, &s_d, lt, dt, ex);
        }
    }

    // Well-calibrated header-bias DP sweep (2 biases on small/medium streams, 1 on dense), exact Huffman merge cleanup!
    let biases: &[f64] = if !fast_eval && count <= 35_000 {
        &[60.0, 160.0]
    } else {
        &[120.0]
    };
    for &bias in biases {
        let mut dp = vec![f64::INFINITY; m + 1];
        let mut prev = vec![0usize; m + 1];
        dp[0] = 0.0;
        for j in 1..=m {
            let mut best_v = f64::INFINITY;
            let mut best_i = 0usize;
            for i in 0..j {
                let v = dp[i] + seg_cost[i * (m + 1) + j] + if i > 0 { bias } else { 0.0 };
                if v < best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            dp[j] = best_v;
            prev[j] = best_i;
        }
        let mut cand = Vec::new();
        let mut cur = m;
        while cur > 0 {
            cand.push(cuts[cur]);
            cur = prev[cur];
        }
        cand.push(0usize);
        cand.reverse();

        // Exact adjacent-block merge cleanup + enforce max_blocks
        let mut seg_bits: Vec<u64> = (0..cand.len() - 1)
            .map(|k| exact_range_bits(tbl, lit_lens, dists, cand[k], cand[k + 1], fast_eval))
            .collect();
        loop {
            if seg_bits.len() <= 1 {
                break;
            }
            let mut best_idx = usize::MAX;
            let mut best_delta = i64::MAX;
            let mut best_joined = 0u64;
            for k in 0..seg_bits.len() - 1 {
                let joined = exact_range_bits(tbl, lit_lens, dists, cand[k], cand[k + 2], fast_eval);
                let delta = joined as i64 - (seg_bits[k] + seg_bits[k + 1]) as i64;
                if delta < best_delta {
                    best_delta = delta;
                    best_idx = k;
                    best_joined = joined;
                }
            }
            if best_idx != usize::MAX && (best_delta <= 0 || seg_bits.len() > max_blocks) {
                cand.remove(best_idx + 1);
                seg_bits[best_idx] = best_joined;
                seg_bits.remove(best_idx + 1);
            } else {
                break;
            }
        }
        let tot_bits: u64 = seg_bits.iter().sum();
        if tot_bits < best_bits {
            best_bits = tot_bits;
            best_splits = cand;
        }
    }

    best_splits
}

fn split_token_stream(
    tbl: &Tables,
    lit_lens: &[u16],
    dists: &[u16],
    max_blocks: usize,
    min_tokens: usize,
) -> Vec<usize> {
    let count = lit_lens.len();
    let mut splits = vec![0usize, count];
    let mut done = Vec::with_capacity(max_blocks);
    for _ in 1..max_blocks {
        let mut best_seg = usize::MAX;
        let mut best_len = 0usize;
        for i in 0..splits.len() - 1 {
            let a = splits[i];
            let b = splits[i + 1];
            if done.contains(&a) {
                continue;
            }
            if b - a > best_len {
                best_len = b - a;
                best_seg = i;
            }
        }
        if best_seg == usize::MAX || best_len < min_tokens {
            break;
        }
        let a = splits[best_seg];
        let b = splits[best_seg + 1];
        let orig_cost = exact_range_bits(tbl, lit_lens, dists, a, b, false);
        let (pos, _) = find_best_split_fast(tbl, lit_lens, dists, a, b, 47);
        if pos > a && pos < b {
            let exact_split_cost = exact_range_bits(tbl, lit_lens, dists, a, pos, false)
                + exact_range_bits(tbl, lit_lens, dists, pos, b, false);
            if exact_split_cost + 8 < orig_cost {
                splits.insert(best_seg + 1, pos);
                continue;
            }
        }
        done.push(a);
    }
    splits
}

// ============================================================================
// Exact Pareto Match Precomputation (Dual Hash + Multi-Row Probes)
// ============================================================================

struct MatchPool {
    off: Vec<u32>,
    m_len: Vec<u16>,
    m_dist: Vec<u16>,
}

#[inline(always)]
fn match_len_fast(data: &[u8], c: usize, p: usize, start: usize, max_len: usize) -> usize {
    let mut l = start;
    let ptr = data.as_ptr();
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            use core::arch::wasm32::{u8x16_bitmask, u8x16_eq, v128, v128_load};
            while l + 16 <= max_len {
                let va = v128_load(ptr.add(c + l) as *const v128);
                let vb = v128_load(ptr.add(p + l) as *const v128);
                let mask = u8x16_bitmask(u8x16_eq(va, vb));
                if mask != 0xffff {
                    return l + ((!mask) as u32).trailing_zeros() as usize;
                }
                l += 16;
            }
        }
        while l + 4 <= max_len {
            let a = (ptr.add(c + l) as *const u32).read_unaligned();
            let b = (ptr.add(p + l) as *const u32).read_unaligned();
            if a != b {
                return l + ((a ^ b).trailing_zeros() >> 3) as usize;
            }
            l += 4;
        }
        while l < max_len && *ptr.add(c + l) == *ptr.add(p + l) {
            l += 1;
        }
    }
    l
}

/// 64K Binary-Tree (BT3) Exact Pareto Match Finder (`min_dist[3..=258]`):
fn precompute_block_matches(
    tbl: &Tables,
    data: &[u8],
    runs: &Runs,
    b_start: usize,
    b_end: usize,
    bt_head: &mut [i32],
    bt_son: &mut [i32],
    row_stride: usize,
    bpp: usize,
    max_depth: usize,
    pool: &mut MatchPool,
) {
    let b_len = b_end - b_start;
    pool.off.clear();
    pool.off.resize(b_len + 1, 0);
    pool.m_len.clear();
    pool.m_dist.clear();

    let run_left = &runs.run_left;
    let run_bpp = runs.run_bpp.as_deref();
    let run_row = runs.run_row.as_deref();
    let _ = tbl;

    let mut min_dist = [0u16; 259];
    let mut tmp_len = [0u16; 260];
    let mut tmp_dist = [0u16; 260];
    let ptr = data.as_ptr();
    let bt_depth = if max_depth <= 64 {
        max_depth
    } else if bpp == 1 {
        max_depth.max(2048)
    } else {
        max_depth.max(768)
    };

    for i in 0..b_len {
        pool.off[i] = pool.m_len.len() as u32;
        let p = b_start + i;
        let rem = b_end - p;
        let max_len = rem.min(258);

        // Fast path 1: period-1 RLE full match (dist=1 is globally optimal)
        if p > 0 && unsafe { *ptr.add(p) == *ptr.add(p - 1) } && (run_left[p] as usize) >= max_len && max_len >= 3 {
            pool.m_len.push(max_len as u16);
            pool.m_dist.push(1);
            continue;
        }

        // Fast path 2: period-bpp full match (dist=bpp has 0 extra bits)
        if let Some(rb) = run_bpp {
            if (rb[p] as usize) >= max_len && max_len >= 3 {
                if p > 0 && unsafe { *ptr.add(p) == *ptr.add(p - 1) } && run_left[p] >= 3 {
                    let rl = (run_left[p] as usize).min(max_len);
                    pool.m_len.push(rl as u16);
                    pool.m_dist.push(1);
                }
                pool.m_len.push(max_len as u16);
                pool.m_dist.push(bpp as u16);
                continue;
            }
        }

        if max_len < 3 {
            continue;
        }

        let limit = p.saturating_sub(WIN_SIZE) as i32;
        let b0 = unsafe { *ptr.add(p) };
        let my_run = run_left[p] as usize;
        let my_bpp_run = run_bpp.map_or(0, |rb| rb[p] as usize);
        let row_len_above = run_row.map_or(0, |rr| (rr[p] as usize).min(max_len));

        // Seed exact Pareto min_dist[3..=max_found] from O(1) structural matches (dist=1, 2, bpp, row_stride)
        let mut max_found = 2usize;
        if p > 0 && unsafe { *ptr.add(p - 1) == b0 } && my_run >= 3 {
            let rl = my_run.min(max_len);
            for k in 3..=rl {
                min_dist[k] = 1;
            }
            max_found = rl;
        }
        if bpp > 2 && p >= 2 && unsafe { *ptr.add(p - 2) == b0 && *ptr.add(p - 1) == *ptr.add(p + 1) } {
            let l2 = match_len_fast(data, p - 2, p, 2, max_len);
            if l2 >= 3 {
                if l2 > max_found {
                    for k in max_found + 1..=l2 {
                        min_dist[k] = 2;
                    }
                    max_found = l2;
                }
                let mut k = l2;
                while k >= 3 && 2u16 < min_dist[k] {
                    min_dist[k] = 2;
                    k -= 1;
                }
            }
        }
        if my_bpp_run >= 3 {
            let v = my_bpp_run.min(max_len);
            if v > max_found {
                for k in max_found + 1..=v {
                    min_dist[k] = bpp as u16;
                }
                max_found = v;
            }
            let mut k = v;
            while k >= 3 && (bpp as u16) < min_dist[k] {
                min_dist[k] = bpp as u16;
                k -= 1;
            }
        }
        if row_len_above >= 3 {
            let rs = row_stride as u16;
            if row_len_above > max_found {
                for k in max_found + 1..=row_len_above {
                    min_dist[k] = rs;
                }
                max_found = row_len_above;
            }
            let mut k = row_len_above;
            while k >= 3 && rs < min_dist[k] {
                min_dist[k] = rs;
                k -= 1;
            }
        }

        // Full BT3 search & insert
        let mut bt_exhausted = true;
        {
            let h = hash3(data, p);
            let mut cur = unsafe { *bt_head.get_unchecked(h) };
            unsafe { *bt_head.get_unchecked_mut(h) = p as i32; }
            let mut ptr0 = (p & BT_MASK) * 2 + 1;
            let mut ptr1 = (p & BT_MASK) * 2;
            let mut len0 = 0usize;
            let mut len1 = 0usize;
            let mut depth = bt_depth;
            let mut bt_best = 2usize;
            let mut finished_exact = false;

            while cur >= limit && depth > 0 {
                depth -= 1;
                let c = cur as usize;
                let dist = p - c;
                let c_slot = (c & BT_MASK) * 2;
                let start_l = len0.min(len1);
                let l = unsafe {
                    if *ptr.add(c + start_l) != *ptr.add(p + start_l) {
                        start_l
                    } else if start_l + 1 < max_len && *ptr.add(c + start_l + 1) != *ptr.add(p + start_l + 1) {
                        start_l + 1
                    } else if start_l + 2 < max_len && *ptr.add(c + start_l + 2) != *ptr.add(p + start_l + 2) {
                        start_l + 2
                    } else {
                        match_len_fast(data, c, p, (start_l + 3).min(max_len), max_len)
                    }
                };
                if l > bt_best {
                    bt_best = l;
                    let d16 = dist as u16;
                    if l > max_found {
                        for k in max_found + 1..=l {
                            min_dist[k] = d16;
                        }
                        max_found = l;
                    }
                    let mut k = l;
                    while k >= 3 && d16 < min_dist[k] {
                        min_dist[k] = d16;
                        k -= 1;
                    }
                    if l == max_len {
                        unsafe {
                            *bt_son.get_unchecked_mut(ptr1) = *bt_son.get_unchecked(c_slot);
                            *bt_son.get_unchecked_mut(ptr0) = *bt_son.get_unchecked(c_slot + 1);
                        }
                        finished_exact = true;
                        break;
                    }
                }
                if unsafe { *ptr.add(c + l) < *ptr.add(p + l) } {
                    unsafe {
                        *bt_son.get_unchecked_mut(ptr1) = cur;
                        ptr1 = c_slot + 1;
                        cur = *bt_son.get_unchecked(ptr1);
                    }
                    len1 = l;
                } else {
                    unsafe {
                        *bt_son.get_unchecked_mut(ptr0) = cur;
                        ptr0 = c_slot;
                        cur = *bt_son.get_unchecked(ptr0);
                    }
                    len0 = l;
                }
            }
            if !finished_exact {
                if cur >= limit && depth == 0 {
                    bt_exhausted = false;
                }
                unsafe {
                    *bt_son.get_unchecked_mut(ptr0) = -1;
                    *bt_son.get_unchecked_mut(ptr1) = -1;
                }
            }
        }

        // Extra O(1) multi-row probes (2..=6 * row_stride) when BT3 truncated
        if !bt_exhausted && row_stride > 1 {
            let b1 = unsafe { *ptr.add(p + 1) };
            let b2 = unsafe { *ptr.add(p + 2) };
            for mult in 2usize..=6usize {
                let rd = row_stride * mult;
                if rd > WIN_SIZE {
                    break;
                }
                if p >= rd && (max_found < max_len || (rd as u16) < min_dist[max_found]) {
                    let rc = p - rd;
                    if unsafe { *ptr.add(rc) == b0 && *ptr.add(rc + 1) == b1 && *ptr.add(rc + 2) == b2 } {
                        let l = match_len_fast(data, rc, p, 3, max_len);
                        let d16 = rd as u16;
                        if l > max_found {
                            for k in max_found + 1..=l {
                                min_dist[k] = d16;
                            }
                            max_found = l;
                        }
                        let mut k = l;
                        while k >= 3 && d16 < min_dist[k] {
                            min_dist[k] = d16;
                            k -= 1;
                        }
                    }
                }
            }
        }

        if max_found >= 3 {
            let mut steps = 0usize;
            for k in 3..=max_found {
                if k == max_found || min_dist[k] != min_dist[k + 1] {
                    tmp_len[steps] = k as u16;
                    tmp_dist[steps] = min_dist[k];
                    steps += 1;
                }
            }
            pool.m_len.extend_from_slice(&tmp_len[..steps]);
            pool.m_dist.extend_from_slice(&tmp_dist[..steps]);
        }
    }
    pool.off[b_len] = pool.m_len.len() as u32;
}

// ============================================================================
// Optimal Forward DP & Multi-Iteration Zopfli Optimization
// ============================================================================

struct ParseOut {
    lit_lens: Vec<u16>,
    dists: Vec<u16>,
    pos_arr: Vec<u32>,
}

fn optimal_parse_block(
    tbl: &Tables,
    data: &[u8],
    runs: &Runs,
    bpp: usize,
    row_stride: usize,
    b_start: usize,
    pool: &MatchPool,
    ll_cost: &[f32; 286],
    d_cost: &[f32; 30],
    hot_dists: &[(u16, f32)],
    dp_cost: &mut [f32],
    dp_len: &mut [u16],
    dp_dist: &mut [u16],
    sub_start: usize,
    sub_end: usize,
    exact_mode: u8,
) -> ParseOut {
    let mut len_full = [0f32; 259];
    for l in 3..=258 {
        len_full[l] = ll_cost[tbl.len_sym[l] as usize] + tbl.len_ex[l] as f32;
    }
    let mut dist_full = [0f32; 30];
    let mut pref_max_dc = [0f32; 30];
    let mut cur_max_dc = 0f32;
    for s in 0..30 {
        dist_full[s] = d_cost[s] + DIST_EXTRA[s] as f32;
        pref_max_dc[s] = cur_max_dc;
        if dist_full[s] > cur_max_dc {
            cur_max_dc = dist_full[s];
        }
    }

    let bpp_dsym = tbl.dist_sym[bpp] as usize;
    let bpp_dcost = dist_full[bpp_dsym];
    let bpp_nonmono = bpp > 1 && bpp_dcost + 0.05 < pref_max_dc[bpp_dsym];
    let (row_dcost, row_nonmono) = if row_stride > 1 && row_stride <= WIN_SIZE {
        let rds = tbl.dist_sym[row_stride] as usize;
        (dist_full[rds], dist_full[rds] + 0.05 < pref_max_dc[rds])
    } else {
        (1e9f32, false)
    };
    let cost_258_1 = len_full[258] + dist_full[0];
    let cost_258_bpp = len_full[258] + bpp_dcost;

    let sub_len = sub_end - sub_start;
    assert!(
        dp_cost.len() > sub_len
            && dp_len.len() > sub_len
            && dp_dist.len() > sub_len
            && b_start + sub_end <= data.len()
            && sub_end < pool.off.len()
            && b_start + sub_end <= runs.run_left.len()
    );
    for x in &mut dp_cost[..=sub_len] {
        *x = 1e30f32;
    }
    dp_cost[0] = 0.0;

    let run_left = &runs.run_left;
    let run_bpp = runs.run_bpp.as_deref();
    let run_row = runs.run_row.as_deref();
    let off = &pool.off;
    let m_len = &pool.m_len;
    let m_dist = &pool.m_dist;
    let dist_sym = &tbl.dist_sym;

    let dense_lim: usize = if exact_mode == 1 {
        32
    } else if exact_mode == 3 {
        4
    } else if bpp == 1 {
        16
    } else {
        12
    };
    let tail_pad: usize = if exact_mode == 1 {
        3
    } else if exact_mode == 3 {
        0
    } else if bpp == 1 {
        2
    } else {
        1
    };
    let run_skip: usize = if exact_mode == 0 || exact_mode == 3 {
        260
    } else {
        516
    };
    let stops: &[usize] = if exact_mode == 1 {
        &FINAL_STOPS
    } else if exact_mode == 3 {
        &[]
    } else {
        &WARMUP_STOPS
    };

    for rel in 0..sub_len {
        let c = unsafe { *dp_cost.get_unchecked(rel) };
        if c >= 1e29f32 {
            continue;
        }
        let i = sub_start + rel;
        let p = b_start + i;
        let rem = sub_len - rel;
        let dp_byte = unsafe { *data.get_unchecked(p) };
        let rl_cur = unsafe { *run_left.get_unchecked(p) } as usize;

        if p > 0 && dp_byte == unsafe { *data.get_unchecked(p - 1) } && rl_cur >= run_skip && rem >= run_skip {
            let mc = c + cost_258_1;
            let ni = rel + 258;
            unsafe {
                if mc < *dp_cost.get_unchecked(ni) {
                    *dp_cost.get_unchecked_mut(ni) = mc;
                    *dp_len.get_unchecked_mut(ni) = 258;
                    *dp_dist.get_unchecked_mut(ni) = 1;
                }
            }
            continue;
        }
        if let Some(rb) = run_bpp {
            let rbp = unsafe { *rb.get_unchecked(p) } as usize;
            if rbp >= run_skip && rl_cur < 3 && rem >= run_skip {
                let mc = c + cost_258_bpp;
                let ni = rel + 258;
                unsafe {
                    if mc < *dp_cost.get_unchecked(ni) {
                        *dp_cost.get_unchecked_mut(ni) = mc;
                        *dp_len.get_unchecked_mut(ni) = 258;
                        *dp_dist.get_unchecked_mut(ni) = bpp as u16;
                    }
                }
                continue;
            }
            let bl = rbp.min(rem).min(258);
            if bl >= 3 {
                let base_b = c + bpp_dcost;
                if bpp_nonmono {
                    let de = if exact_mode == 2 { bl.min(24) } else { dense_lim.min(bl) };
                    for l in 3..=de {
                        let ni = rel + l;
                        unsafe {
                            let mc = base_b + *len_full.get_unchecked(l);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = l as u16;
                                *dp_dist.get_unchecked_mut(ni) = bpp as u16;
                            }
                        }
                    }
                    let ts = bl.saturating_sub(tail_pad.max(2)).max(de + 1);
                    for l in ts..=bl {
                        let ni = rel + l;
                        unsafe {
                            let mc = base_b + *len_full.get_unchecked(l);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = l as u16;
                                *dp_dist.get_unchecked_mut(ni) = bpp as u16;
                            }
                        }
                    }
                } else {
                    let mc = base_b + unsafe { *len_full.get_unchecked(bl) };
                    let ni = rel + bl;
                    unsafe {
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = bl as u16;
                            *dp_dist.get_unchecked_mut(ni) = bpp as u16;
                        }
                    }
                }
            }
        }

        let lc = c + unsafe { *ll_cost.get_unchecked(dp_byte as usize) };
        unsafe {
            if lc < *dp_cost.get_unchecked(rel + 1) {
                *dp_cost.get_unchecked_mut(rel + 1) = lc;
                *dp_len.get_unchecked_mut(rel + 1) = 1;
                *dp_dist.get_unchecked_mut(rel + 1) = 0;
            }
        }

        if let Some(rr) = run_row {
            let rl = (unsafe { *rr.get_unchecked(p) } as usize).min(rem).min(258);
            if rl >= 3 {
                let base_r = c + row_dcost;
                if row_nonmono {
                    let de = if exact_mode == 2 { rl.min(24) } else { dense_lim.min(rl) };
                    for l in 3..=de {
                        let ni = rel + l;
                        unsafe {
                            let mc = base_r + *len_full.get_unchecked(l);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = l as u16;
                                *dp_dist.get_unchecked_mut(ni) = row_stride as u16;
                            }
                        }
                    }
                    let ts = rl.saturating_sub(tail_pad.max(2)).max(de + 1);
                    for l in ts..=rl {
                        let ni = rel + l;
                        unsafe {
                            let mc = base_r + *len_full.get_unchecked(l);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = l as u16;
                                *dp_dist.get_unchecked_mut(ni) = row_stride as u16;
                            }
                        }
                    }
                } else {
                    let mc = base_r + unsafe { *len_full.get_unchecked(rl) };
                    let ni = rel + rl;
                    unsafe {
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = rl as u16;
                            *dp_dist.get_unchecked_mut(ni) = row_stride as u16;
                        }
                    }
                }
            }
        }

        // Fast image-adaptive frequent-distance probes (recovers non-monotone cheap dist_sym matches!)
        if !hot_dists.is_empty() && rem >= 3 {
            let ptr = data.as_ptr();
            let b1 = unsafe { *ptr.add(p + 1) };
            let b2 = unsafe { *ptr.add(p + 2) };
            let max_l = rem.min(258);
            for &(hd, hdc) in hot_dists {
                let d = hd as usize;
                if p >= d
                    && unsafe {
                        *ptr.add(p - d) == dp_byte
                            && *ptr.add(p - d + 1) == b1
                            && *ptr.add(p - d + 2) == b2
                    }
                {
                    let hl = match_len_fast(data, p - d, p, 3, max_l);
                    let base_h = c + hdc;
                    let de = hl.min(12);
                    for l in 3..=de {
                        let ni = rel + l;
                        unsafe {
                            let mc = base_h + *len_full.get_unchecked(l);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = l as u16;
                                *dp_dist.get_unchecked_mut(ni) = hd;
                            }
                        }
                    }
                    if hl > de {
                        let ni = rel + hl;
                        unsafe {
                            let mc = base_h + *len_full.get_unchecked(hl);
                            if mc < *dp_cost.get_unchecked(ni) {
                                *dp_cost.get_unchecked_mut(ni) = mc;
                                *dp_len.get_unchecked_mut(ni) = hl as u16;
                                *dp_dist.get_unchecked_mut(ni) = hd;
                            }
                        }
                    }
                }
            }
        }

        let s0 = unsafe { *off.get_unchecked(i) } as usize;
        let s1 = unsafe { *off.get_unchecked(i + 1) } as usize;
        let mut prev_l = 2usize;
        let mut prev_max_dc = 0.0f32;
        for s in s0..s1 {
            let end_l = (unsafe { *m_len.get_unchecked(s) } as usize).min(rem);
            if end_l < 3 {
                continue;
            }
            let dist = unsafe { *m_dist.get_unchecked(s) };
            let dc = unsafe { *dist_full.get_unchecked(*dist_sym.get_unchecked(dist as usize) as usize) };
            let base = c + dc;
            let start_l = if prev_l >= 3 && dc >= prev_max_dc { prev_l + 1 } else { 3 };
            if start_l > end_l {
                continue;
            }
            if exact_mode == 2 || end_l <= dense_lim {
                for l in start_l..=end_l {
                    let ni = rel + l;
                    unsafe {
                        let mc = base + *len_full.get_unchecked(l);
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = l as u16;
                            *dp_dist.get_unchecked_mut(ni) = dist;
                        }
                    }
                }
            } else {
                let dense_end = dense_lim.min(end_l);
                for l in start_l..=dense_end {
                    let ni = rel + l;
                    unsafe {
                        let mc = base + *len_full.get_unchecked(l);
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = l as u16;
                            *dp_dist.get_unchecked_mut(ni) = dist;
                        }
                    }
                }
                let lo_stop = start_l.max(dense_lim + 1);
                let hi_stop = end_l.saturating_sub(tail_pad + 1);
                for &l in stops {
                    if l < lo_stop {
                        continue;
                    }
                    if l > hi_stop {
                        break;
                    }
                    let ni = rel + l;
                    unsafe {
                        let mc = base + *len_full.get_unchecked(l);
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = l as u16;
                            *dp_dist.get_unchecked_mut(ni) = dist;
                        }
                    }
                }
                let tail_start = end_l.saturating_sub(tail_pad).max(start_l);
                for l in tail_start..=end_l {
                    let ni = rel + l;
                    unsafe {
                        let mc = base + *len_full.get_unchecked(l);
                        if mc < *dp_cost.get_unchecked(ni) {
                            *dp_cost.get_unchecked_mut(ni) = mc;
                            *dp_len.get_unchecked_mut(ni) = l as u16;
                            *dp_dist.get_unchecked_mut(ni) = dist;
                        }
                    }
                }
            }
            if start_l == 3 && end_l >= prev_l {
                prev_max_dc = dc;
            } else if dc > prev_max_dc {
                prev_max_dc = dc;
            }
            if end_l > prev_l {
                prev_l = end_l;
            }
        }
    }

    let mut tok_cnt = 0usize;
    let mut rel = sub_len;
    while rel > 0 {
        tok_cnt += 1;
        rel -= dp_len[rel] as usize;
    }
    let mut lit_lens = vec![0u16; tok_cnt];
    let mut dists = vec![0u16; tok_cnt];
    let mut pos_arr = vec![0u32; tok_cnt];
    let mut idx = tok_cnt;
    let mut rel = sub_len;
    while rel > 0 {
        idx -= 1;
        let l = dp_len[rel] as usize;
        let d = dp_dist[rel];
        pos_arr[idx] = (b_start + sub_start + rel - l) as u32;
        if d == 0 {
            lit_lens[idx] = data[b_start + sub_start + rel - 1] as u16;
            dists[idx] = 0;
        } else {
            lit_lens[idx] = l as u16;
            dists[idx] = d;
        }
        rel -= l;
    }
    ParseOut {
        lit_lens,
        dists,
        pos_arr,
    }
}

fn entropy_costs_from_stats(
    ll_stats: &[f64; 286],
    d_stats: &[f64; 30],
    ll_cost: &mut [f32; 286],
    d_cost: &mut [f32; 30],
) {
    let ll_sum: f64 = ll_stats.iter().sum();
    let log_ll = if ll_sum > 0.0 { ll_sum.log2() } else { 1.0 };
    for i in 0..286 {
        let v = ll_stats[i];
        ll_cost[i] = if v > 0.0 {
            (log_ll - v.log2()).max(1.0) as f32
        } else {
            (log_ll + 1.0) as f32
        };
    }
    let d_sum: f64 = d_stats.iter().sum();
    let log_d = if d_sum > 0.0 { d_sum.log2() } else { 1.0 };
    for i in 0..30 {
        let v = d_stats[i];
        d_cost[i] = if v > 0.0 {
            (log_d - v.log2()).max(1.0) as f32
        } else {
            (log_d + 1.0) as f32
        };
    }
}

fn zopfli_optimize_range(
    tbl: &Tables,
    data: &[u8],
    runs: &Runs,
    bpp: usize,
    row_stride: usize,
    b_start: usize,
    pool: &MatchPool,
    iterations: usize,
    dp_cost: &mut [f32],
    dp_len: &mut [u16],
    dp_dist: &mut [u16],
    sub_start: usize,
    sub_end: usize,
    init_tokens: Option<(&[u16], &[u16], &[u32])>,
    allow_full_exact: bool,
) -> ParseOut {
    let mut ll_cost = [0f32; 286];
    let mut d_cost = [0f32; 30];
    let mut ll_smooth = [0f64; 286];
    let mut d_smooth = [0f64; 30];
    let mut best_ll_f = [0u32; 286];
    let mut best_d_f = [0u32; 30];

    let mut best_bits = u64::MAX;
    let mut best_out = ParseOut {
        lit_lens: Vec::new(),
        dists: Vec::new(),
        pos_arr: Vec::new(),
    };
    let mut best_dyn: Option<DynTree> = None;

    if let Some((init_ll, init_d, init_pos)) = init_tokens {
        let (ll_f, d_f, extra) = stats_from_range(tbl, init_ll, init_d, 0, init_ll.len());
        let dyn_t = build_best_dynamic_tree(&ll_f, &d_f, extra, false);
        let fix_b = fixed_block_bits(tbl, &ll_f, &d_f, extra);
        best_bits = dyn_t.total_bits.min(fix_b);
        best_out.lit_lens = init_ll.to_vec();
        best_out.dists = init_d.to_vec();
        best_out.pos_arr = init_pos.to_vec();
        best_dyn = Some(dyn_t);
        best_ll_f = ll_f;
        best_d_f = d_f;
        for i in 0..286 {
            ll_smooth[i] = ll_f[i] as f64;
        }
        for i in 0..30 {
            d_smooth[i] = d_f[i] as f64;
        }
        entropy_costs_from_stats(&ll_smooth, &d_smooth, &mut ll_cost, &mut d_cost);
    } else {
        // Raw-literal-entropy init + bpp-aware match cost
        let mut raw_freq = [0u32; 256];
        let sub_len = sub_end - sub_start;
        for i in sub_start..sub_end {
            raw_freq[data[b_start + i] as usize] += 1;
        }
        let log_r = (sub_len.max(2) as f64).log2();
        for i in 0..256 {
            ll_cost[i] = if raw_freq[i] > 0 {
                (log_r - (raw_freq[i] as f64).log2()).max(1.0) as f32
            } else {
                (log_r + 1.0) as f32
            };
        }
        ll_cost[256] = log_r as f32;
        let init_m = if bpp == 1 { 8.0f32 } else { 9.5f32 };
        for i in 257..286 {
            ll_cost[i] = init_m;
        }
        for i in 0..30 {
            d_cost[i] = 4.5f32;
        }
    }

    let mut last_bits = best_bits;
    let mut rng: u32 = 0x9e3779b9u32
        ^ ((b_start + sub_start + 1) as u32)
        ^ (((sub_end - sub_start) as u32).wrapping_mul(2654435761));
    let mut next_rand = || -> u32 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };

    let final_mode: u8 = if allow_full_exact && (sub_end - sub_start) <= 3_800_000 {
        2
    } else {
        1
    };

    // Helper: extract up to 8 top frequent non-monotone distances from a match-heavy block (>50% matches)
    let build_hot_dists = |dists_src: &[u16], dc_arr: &[f32; 30]| -> Vec<(u16, f32)> {
        let mut match_cnt = 0usize;
        for &d in dists_src {
            if d > 0 {
                match_cnt += 1;
            }
        }
        if match_cnt * 2 <= dists_src.len() {
            return Vec::new();
        }
        let mut dfull = [0f32; 30];
        let mut pref_max = [0f32; 30];
        let mut cur_max = 0f32;
        for s in 0..30 {
            dfull[s] = dc_arr[s] + DIST_EXTRA[s] as f32;
            pref_max[s] = cur_max;
            if dfull[s] > cur_max {
                cur_max = dfull[s];
            }
        }
        let mut nz: Vec<u16> = dists_src
            .iter()
            .copied()
            .filter(|&d| d > 1 && d as usize != bpp && d as usize != row_stride)
            .collect();
        nz.sort_unstable();
        let mut freq_d: Vec<(u32, u16)> = Vec::new();
        let mut idx = 0usize;
        while idx < nz.len() {
            let v = nz[idx];
            let mut end = idx + 1;
            while end < nz.len() && nz[end] == v {
                end += 1;
            }
            let cnt = (end - idx) as u32;
            if cnt >= 12 {
                let ds = tbl.dist_sym[v as usize] as usize;
                if dfull[ds] + 0.4 < pref_max[ds] {
                    freq_d.push((cnt, v));
                }
            }
            idx = end;
        }
        // Also seed 2D structural neighbors (row_stride +/- bpp, 2*row_stride, 3*row_stride) if non-monotone cheap
        if row_stride > bpp && row_stride + bpp <= WIN_SIZE {
            for cand in [
                row_stride - bpp,
                row_stride + bpp,
                row_stride * 2,
                row_stride * 3,
            ] {
                if cand > 1 && cand <= WIN_SIZE && cand != bpp && cand != row_stride {
                    let ds = tbl.dist_sym[cand] as usize;
                    if dfull[ds] + 0.4 < pref_max[ds] && !freq_d.iter().any(|x| x.1 == cand as u16) {
                        freq_d.push((1, cand as u16));
                    }
                }
            }
        }
        freq_d.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        freq_d.truncate(5);
        freq_d
            .into_iter()
            .map(|(_, d)| (d, dfull[tbl.dist_sym[d as usize] as usize]))
            .collect()
    };

    let mut same_streak = 0u32;
    let mut iter = 0usize;
    let mut hot_dists: Vec<(u16, f32)> = Vec::new();
    while iter < iterations {
        let exact_mode: u8 = if iter + 1 == iterations && allow_full_exact {
            final_mode
        } else if iter + 2 == iterations && allow_full_exact {
            1
        } else {
            0
        };
        if allow_full_exact && iter + 1 == iterations && !best_out.dists.is_empty() {
            hot_dists = build_hot_dists(&best_out.dists, &d_cost);
        } else {
            hot_dists.clear();
        }
        let parsed = optimal_parse_block(
            tbl,
            data,
            runs,
            bpp,
            row_stride,
            b_start,
            pool,
            &ll_cost,
            &d_cost,
            &hot_dists,
            dp_cost,
            dp_len,
            dp_dist,
            sub_start,
            sub_end,
            exact_mode,
        );
        let (ll_f, d_f, extra) =
            stats_from_range(tbl, &parsed.lit_lens, &parsed.dists, 0, parsed.lit_lens.len());
        let dyn_t = build_best_dynamic_tree(&ll_f, &d_f, extra, false);
        let fix_b = fixed_block_bits(tbl, &ll_f, &d_f, extra);
        let bits = dyn_t.total_bits.min(fix_b);

        if bits < best_bits {
            best_bits = bits;
            best_out = parsed;
            best_dyn = Some(dyn_t);
            best_ll_f = ll_f;
            best_d_f = d_f;
            same_streak = 0;
        } else if bits == last_bits {
            same_streak += 1;
        } else {
            same_streak = 0;
        }

        if iter + 1 == iterations {
            break;
        }

        // Deterministic early-convergence jump directly to the final 2 exact passes!
        let (streak_lim, min_iter) = if iterations >= 11 { (3u32, 6usize) } else { (2u32, 4usize) };
        if same_streak >= streak_lim && iter >= min_iter && iter + 3 < iterations {
            iter = iterations - 3;
        }

        if iter < 2 || (iter < 3 && init_tokens.is_none()) {
            for i in 0..286 {
                ll_smooth[i] = ll_f[i] as f64;
            }
            for i in 0..30 {
                d_smooth[i] = d_f[i] as f64;
            }
        } else if bits >= last_bits && iter + 3 < iterations {
            // True Zopfli perturbation centered on best_stats to escape local minima!
            for i in 0..286 {
                let mut v = (best_ll_f[i] as f64) * 0.55 + ll_smooth[i] * 0.45;
                if (next_rand() & 15) == 0 {
                    v = best_ll_f[(next_rand() as usize) % 286] as f64;
                }
                ll_smooth[i] = v;
            }
            for i in 0..30 {
                let mut v = (best_d_f[i] as f64) * 0.55 + d_smooth[i] * 0.45;
                if (next_rand() & 15) == 0 {
                    v = best_d_f[(next_rand() as usize) % 30] as f64;
                }
                d_smooth[i] = v;
            }
        } else {
            for i in 0..286 {
                ll_smooth[i] = ll_smooth[i] * 0.35 + (ll_f[i] as f64) * 0.65;
            }
            for i in 0..30 {
                d_smooth[i] = d_smooth[i] * 0.35 + (d_f[i] as f64) * 0.65;
            }
        }
        last_bits = bits;

        // Last 2 iterations: exact integer Huffman bit lengths from best_dyn for exact convergence!
        if iter + 2 >= iterations {
            if let Some(ref bd) = best_dyn {
                for i in 0..286 {
                    ll_cost[i] = if bd.ll_lens[i] > 0 {
                        bd.ll_lens[i] as f32
                    } else {
                        16.0
                    };
                }
                for i in 0..30 {
                    d_cost[i] = if bd.d_lens[i] > 0 {
                        bd.d_lens[i] as f32
                    } else {
                        16.0
                    };
                }
                iter += 1;
                continue;
            }
        }
        entropy_costs_from_stats(&ll_smooth, &d_smooth, &mut ll_cost, &mut d_cost);
        iter += 1;
    }

    best_out
}

struct FinalBlock {
    lit_lens: Vec<u16>,
    dists: Vec<u16>,
    pos_arr: Vec<u32>,
    raw_start: usize,
    raw_end: usize,
}

fn merge_adjacent_blocks(tbl: &Tables, mut merged: Vec<FinalBlock>, dirty: &mut Vec<bool>) -> Vec<FinalBlock> {
    let mut merge_guard = 0usize;
    loop {
        if merged.len() <= 1 || merge_guard >= 32 {
            break;
        }
        merge_guard += 1;
        let mut did_merge = false;
        let mut next: Vec<FinalBlock> = Vec::with_capacity(merged.len());
        let mut next_dirty: Vec<bool> = Vec::with_capacity(merged.len());
        let mut i = 0;
        while i < merged.len() {
            if i + 1 < merged.len() {
                let a = &merged[i];
                let b = &merged[i + 1];
                let (la, da, ea) = stats_from_range(tbl, &a.lit_lens, &a.dists, 0, a.lit_lens.len());
                let (lb, db, eb) = stats_from_range(tbl, &b.lit_lens, &b.dists, 0, b.lit_lens.len());
                let ca = build_best_dynamic_tree(&la, &da, ea, false)
                    .total_bits
                    .min(fixed_block_bits(tbl, &la, &da, ea));
                let cb = build_best_dynamic_tree(&lb, &db, eb, false)
                    .total_bits
                    .min(fixed_block_bits(tbl, &lb, &db, eb));
                let mut j_ll = [0u32; 286];
                let mut j_d = [0u32; 30];
                for k in 0..286 {
                    j_ll[k] = la[k] + lb[k];
                }
                j_ll[256] = 1;
                for k in 0..30 {
                    j_d[k] = da[k] + db[k];
                }
                let cj = build_best_dynamic_tree(&j_ll, &j_d, ea + eb, false)
                    .total_bits
                    .min(fixed_block_bits(tbl, &j_ll, &j_d, ea + eb));
                if cj <= ca + cb {
                    let mut nll = Vec::with_capacity(a.lit_lens.len() + b.lit_lens.len());
                    nll.extend_from_slice(&a.lit_lens);
                    nll.extend_from_slice(&b.lit_lens);
                    let mut nd = Vec::with_capacity(a.dists.len() + b.dists.len());
                    nd.extend_from_slice(&a.dists);
                    nd.extend_from_slice(&b.dists);
                    let mut np = Vec::with_capacity(a.pos_arr.len() + b.pos_arr.len());
                    np.extend_from_slice(&a.pos_arr);
                    np.extend_from_slice(&b.pos_arr);
                    next.push(FinalBlock {
                        lit_lens: nll,
                        dists: nd,
                        pos_arr: np,
                        raw_start: a.raw_start,
                        raw_end: b.raw_end,
                    });
                    next_dirty.push(true);
                    i += 2;
                    did_merge = true;
                    continue;
                }
            }
            next.push(FinalBlock {
                lit_lens: std::mem::take(&mut merged[i].lit_lens),
                dists: std::mem::take(&mut merged[i].dists),
                pos_arr: std::mem::take(&mut merged[i].pos_arr),
                raw_start: merged[i].raw_start,
                raw_end: merged[i].raw_end,
            });
            next_dirty.push(dirty.get(i).copied().unwrap_or(false));
            i += 1;
        }
        merged = next;
        *dirty = next_dirty;
        if !did_merge {
            break;
        }
    }
    merged
}

fn shift_adjacent_boundaries(tbl: &Tables, merged: &mut [FinalBlock], dirty: &mut [bool]) {
    if merged.len() <= 1 {
        return;
    }
    for i in 0..merged.len() - 1 {
        let a_len = merged[i].lit_lens.len();
        let b_len = merged[i + 1].lit_lens.len();
        if a_len < 128
            || b_len < 128
            || merged[i].pos_arr.len() != a_len
            || merged[i + 1].pos_arr.len() != b_len
        {
            continue;
        }
        let mut j_ll = Vec::with_capacity(a_len + b_len);
        j_ll.extend_from_slice(&merged[i].lit_lens);
        j_ll.extend_from_slice(&merged[i + 1].lit_lens);
        let mut j_d = Vec::with_capacity(a_len + b_len);
        j_d.extend_from_slice(&merged[i].dists);
        j_d.extend_from_slice(&merged[i + 1].dists);
        let mut j_pos = Vec::with_capacity(a_len + b_len);
        j_pos.extend_from_slice(&merged[i].pos_arr);
        j_pos.extend_from_slice(&merged[i + 1].pos_arr);

        let (mut la, mut da, mut ea) = stats_from_range(tbl, &j_ll, &j_d, 0, a_len);
        let (mut lb, mut db, mut eb) = stats_from_range(tbl, &j_ll, &j_d, a_len, a_len + b_len);
        let eval_pair = |la: &[u32; 286],
                         da: &[u32; 30],
                         ea: u64,
                         lb: &[u32; 286],
                         db: &[u32; 30],
                         eb: u64,
                         fast: bool|
         -> u64 {
            build_best_dynamic_tree(la, da, ea, fast)
                .total_bits
                .min(fixed_block_bits(tbl, la, da, ea))
                + build_best_dynamic_tree(lb, db, eb, fast)
                    .total_bits
                    .min(fixed_block_bits(tbl, lb, db, eb))
        };
        let mut best_split = a_len;
        let mut best_cost = eval_pair(&la, &da, ea, &lb, &db, eb, true);

        for &step in &[512isize, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            let fast = step > 4;
            if step == 4 {
                best_cost = eval_pair(&la, &da, ea, &lb, &db, eb, false);
            }
            for &dir in &[-1isize, 1] {
                let cand_i = best_split as isize + dir * step;
                if cand_i < 64 || (cand_i as usize) + 64 > a_len + b_len {
                    continue;
                }
                let cand = cand_i as usize;
                let mut nla = la;
                let mut nda = da;
                let mut nea = ea;
                let mut nlb = lb;
                let mut ndb = db;
                let mut neb = eb;
                if cand > best_split {
                    for idx in best_split..cand {
                        let d = j_d[idx] as usize;
                        if d == 0 {
                            let s = j_ll[idx] as usize;
                            nla[s] += 1;
                            nlb[s] -= 1;
                        } else {
                            let l = j_ll[idx] as usize;
                            let ls = tbl.len_sym[l] as usize;
                            let ds = tbl.dist_sym[d] as usize;
                            let ex = (tbl.len_ex[l] + DIST_EXTRA[ds]) as u64;
                            nla[ls] += 1;
                            nlb[ls] -= 1;
                            nda[ds] += 1;
                            ndb[ds] -= 1;
                            nea += ex;
                            neb -= ex;
                        }
                    }
                } else {
                    for idx in cand..best_split {
                        let d = j_d[idx] as usize;
                        if d == 0 {
                            let s = j_ll[idx] as usize;
                            nla[s] -= 1;
                            nlb[s] += 1;
                        } else {
                            let l = j_ll[idx] as usize;
                            let ls = tbl.len_sym[l] as usize;
                            let ds = tbl.dist_sym[d] as usize;
                            let ex = (tbl.len_ex[l] + DIST_EXTRA[ds]) as u64;
                            nla[ls] -= 1;
                            nlb[ls] += 1;
                            nda[ds] -= 1;
                            ndb[ds] += 1;
                            nea -= ex;
                            neb += ex;
                        }
                    }
                }
                let c = eval_pair(&nla, &nda, nea, &nlb, &ndb, neb, fast);
                if c < best_cost {
                    best_cost = c;
                    best_split = cand;
                    la = nla;
                    da = nda;
                    ea = nea;
                    lb = nlb;
                    db = ndb;
                    eb = neb;
                }
            }
        }
        if best_split != a_len {
            let new_raw_mid = j_pos[best_split] as usize;
            merged[i].lit_lens = j_ll[..best_split].to_vec();
            merged[i].dists = j_d[..best_split].to_vec();
            merged[i].pos_arr = j_pos[..best_split].to_vec();
            merged[i].raw_end = new_raw_mid;
            merged[i + 1].lit_lens = j_ll[best_split..].to_vec();
            merged[i + 1].dists = j_d[best_split..].to_vec();
            merged[i + 1].pos_arr = j_pos[best_split..].to_vec();
            merged[i + 1].raw_start = new_raw_mid;
            let moved = if best_split > a_len {
                best_split - a_len
            } else {
                a_len - best_split
            };
            if moved >= 32 {
                if i < dirty.len() {
                    dirty[i] = true;
                }
                if i + 1 < dirty.len() {
                    dirty[i + 1] = true;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn deflate_zopfli(
    in_ptr: *const u8,
    in_len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
    iterations: u32,
    max_chain: u32,
    max_blocks: u32,
    row_stride: u32,
    bpp: u32,
) -> usize {
    if in_ptr.is_null() || out_ptr.is_null() || out_cap < 16 {
        return 0;
    }
    let data = unsafe { slice::from_raw_parts(in_ptr, in_len) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_cap) };

    let tbl = get_tables();
    let iters = (iterations as usize).max(1);
    let chain = (max_chain as usize).max(8);
    let mblocks = (max_blocks as usize).max(1);
    let rstride = row_stride as usize;
    let bpp_u = (bpp as usize).max(1);

    let runs = compute_runs(data, bpp_u, rstride);

    // True Zopfli 2-Phase Optimal Block Splitting:
    // Single mega-chunk up to 3.8 MB (covers ALL 1280x720 RGBA images with ZERO artificial mega-seams!).
    let mut mega_bounds = Vec::new();
    mega_bounds.push(0usize);
    let max_mega = if rstride > 1 {
        ((3_800_000 / rstride).max(1)) * rstride
    } else {
        3_800_000usize
    };
    let total_rows = if rstride > 1 { in_len / rstride } else { 0 };
    let mut cur_m = 0usize;
    while cur_m < in_len {
        let mut nxt = (cur_m + max_mega).min(in_len);
        if in_len - nxt < 65536 {
            mega_bounds.push(in_len);
            break;
        }
        if rstride > 1 && total_rows > 64 {
            let target_row = nxt / rstride;
            let lo_r = target_row.saturating_sub(48).max(cur_m / rstride + 16);
            let hi_r = (target_row + 48).min(total_rows.saturating_sub(16));
            let mut best_r = target_row;
            let mut best_dist = usize::MAX;
            for r in lo_r..=hi_r {
                if r > 0 && r < total_rows && data[r * rstride] != data[(r - 1) * rstride] {
                    // Check if at least one side has a contiguous run >= 8 rows
                    let f_prev = data[(r - 1) * rstride];
                    let mut p_len = 1usize;
                    while p_len < 8 && r > p_len && data[(r - 1 - p_len) * rstride] == f_prev {
                        p_len += 1;
                    }
                    let f_cur = data[r * rstride];
                    let mut n_len = 1usize;
                    while n_len < 8 && r + n_len < total_rows && data[(r + n_len) * rstride] == f_cur {
                        n_len += 1;
                    }
                    if p_len >= 8 || n_len >= 8 {
                        let d = if r >= target_row { r - target_row } else { target_row - r };
                        if d < best_dist {
                            best_dist = d;
                            best_r = r;
                        }
                    }
                }
            }
            if best_dist < usize::MAX {
                nxt = best_r * rstride;
            }
        }
        mega_bounds.push(nxt);
        cur_m = nxt;
    }
    let num_megas = mega_bounds.len() - 1;
    let mut max_b = 0usize;
    for b in 0..num_megas {
        let len = mega_bounds[b + 1] - mega_bounds[b];
        if len > max_b {
            max_b = len;
        }
    }

    let mut dp_cost = vec![0f32; max_b + 1];
    let mut dp_len = vec![0u16; max_b + 1];
    let mut dp_dist = vec![0u16; max_b + 1];
    let mut pool = MatchPool {
        off: Vec::with_capacity(max_b + 1),
        m_len: Vec::with_capacity((max_b * 2).max(4096)),
        m_dist: Vec::with_capacity((max_b * 2).max(4096)),
    };

    let mut bt_head = vec![-1i32; HASH_SIZE];
    let mut bt_son = vec![-1i32; BT_SIZE * 2];

    let mut final_blocks: Vec<FinalBlock> = Vec::with_capacity(mblocks + 8);
    let per_mega_blocks = ((mblocks + num_megas - 1) / num_megas).max(1);

    for b in 0..num_megas {
        let b_start = mega_bounds[b];
        let b_end = mega_bounds[b + 1];
        precompute_block_matches(
            tbl, data, &runs, b_start, b_end, &mut bt_head, &mut bt_son,
            rstride, bpp_u, chain, &mut pool,
        );

        let mut mega_blocks: Vec<FinalBlock> = Vec::with_capacity(per_mega_blocks + 4);
        if per_mega_blocks > 1 && b_end - b_start > 8192 {
            // Phase 1: fast global optimal parse (3 iters on small <=1MB images for cleaner splits, 2 on large)
            let p1_iters = if b_end - b_start <= 1_000_000 { 3 } else { 2 };
            let init = zopfli_optimize_range(
                tbl, data, &runs, bpp_u, rstride, b_start, &pool, p1_iters, &mut dp_cost, &mut dp_len,
                &mut dp_dist, 0, b_end - b_start, None, false,
            );
            let splits = split_token_stream_dp(
                tbl, data, b_start, b_end, rstride, &init.lit_lens, &init.dists, &init.pos_arr,
                per_mega_blocks, 96, false,
            );
            let sub_iters = iters.max(3);

            for s in 0..splits.len() - 1 {
                let ts = splits[s];
                let te = splits[s + 1];
                let raw_start = init.pos_arr[ts] as usize;
                let raw_end = if te < init.lit_lens.len() {
                    init.pos_arr[te] as usize
                } else {
                    b_end
                };
                // Phase 2: Full Zopfli iterations independently on EACH split sub-block
                let res = zopfli_optimize_range(
                    tbl,
                    data,
                    &runs,
                    bpp_u,
                    rstride,
                    b_start,
                    &pool,
                    sub_iters,
                    &mut dp_cost,
                    &mut dp_len,
                    &mut dp_dist,
                    raw_start - b_start,
                    raw_end - b_start,
                    Some((&init.lit_lens[ts..te], &init.dists[ts..te], &init.pos_arr[ts..te])),
                    true,
                );
                // Phase 2b: Monotonic post-convergence sub-split check + 2-iter re-opt!
                if res.lit_lens.len() >= 384 {
                    let sub2 = split_token_stream(tbl, &res.lit_lens, &res.dists, 6, 96);
                    if sub2.len() > 2 {
                        let orig_res_bits = exact_range_bits(tbl, &res.lit_lens, &res.dists, 0, res.lit_lens.len(), false);
                        let mut cand_sub: Vec<FinalBlock> = Vec::with_capacity(sub2.len() - 1);
                        let mut cand_bits = 0u64;
                        for k in 0..sub2.len() - 1 {
                            let s2 = sub2[k];
                            let e2 = sub2[k + 1];
                            let rs2 = res.pos_arr[s2] as usize;
                            let re2 = if e2 < res.lit_lens.len() {
                                res.pos_arr[e2] as usize
                            } else {
                                raw_end
                            };
                            let reopt = zopfli_optimize_range(
                                tbl,
                                data,
                                &runs,
                                bpp_u,
                                rstride,
                                b_start,
                                &pool,
                                2,
                                &mut dp_cost,
                                &mut dp_len,
                                &mut dp_dist,
                                rs2 - b_start,
                                re2 - b_start,
                                Some((&res.lit_lens[s2..e2], &res.dists[s2..e2], &res.pos_arr[s2..e2])),
                                true,
                            );
                            cand_bits += exact_range_bits(tbl, &reopt.lit_lens, &reopt.dists, 0, reopt.lit_lens.len(), false);
                            cand_sub.push(FinalBlock {
                                lit_lens: reopt.lit_lens,
                                dists: reopt.dists,
                                pos_arr: reopt.pos_arr,
                                raw_start: rs2,
                                raw_end: re2,
                            });
                        }
                        if cand_bits < orig_res_bits {
                            mega_blocks.extend(cand_sub);
                            continue;
                        }
                    }
                }
                mega_blocks.push(FinalBlock {
                    lit_lens: res.lit_lens,
                    dists: res.dists,
                    pos_arr: res.pos_arr,
                    raw_start,
                    raw_end,
                });
            }
        } else {
            let res = zopfli_optimize_range(
                tbl, data, &runs, bpp_u, rstride, b_start, &pool, iters, &mut dp_cost, &mut dp_len,
                &mut dp_dist, 0, b_end - b_start, None, true,
            );
            mega_blocks.push(FinalBlock {
                lit_lens: res.lit_lens,
                dists: res.dists,
                pos_arr: res.pos_arr,
                raw_start: b_start,
                raw_end: b_end,
            });
        }

        // In-mega-chunk Stage 3 merge + Stage 4 boundary shift + Stage 4c 2-iter co-opt while `pool` is hot!
        if mega_blocks.len() > 1 {
            let mut dirty = vec![false; mega_blocks.len()];
            mega_blocks = merge_adjacent_blocks(tbl, mega_blocks, &mut dirty);
            shift_adjacent_boundaries(tbl, &mut mega_blocks, &mut dirty);
            mega_blocks = merge_adjacent_blocks(tbl, mega_blocks, &mut dirty);
            if iters >= 6 {
                for (bi, blk) in mega_blocks.iter_mut().enumerate() {
                    if dirty[bi] && blk.raw_end > blk.raw_start && blk.raw_end - blk.raw_start <= 280_000 {
                        let reopt = zopfli_optimize_range(
                            tbl,
                            data,
                            &runs,
                            bpp_u,
                            rstride,
                            b_start,
                            &pool,
                            2,
                            &mut dp_cost,
                            &mut dp_len,
                            &mut dp_dist,
                            blk.raw_start - b_start,
                            blk.raw_end - b_start,
                            Some((&blk.lit_lens, &blk.dists, &blk.pos_arr)),
                            true,
                        );
                        blk.lit_lens = reopt.lit_lens;
                        blk.dists = reopt.dists;
                        blk.pos_arr = reopt.pos_arr;
                    }
                }
                mega_blocks = merge_adjacent_blocks(tbl, mega_blocks, &mut dirty);
            }
        }
        final_blocks.extend(mega_blocks);
    }

    // Cross-mega seam merge & boundary polish!
    let mut merged = final_blocks;
    if num_megas > 1 && merged.len() > 1 {
        let mut dirty = vec![false; merged.len()];
        merged = merge_adjacent_blocks(tbl, merged, &mut dirty);
        shift_adjacent_boundaries(tbl, &mut merged, &mut dirty);
        merged = merge_adjacent_blocks(tbl, merged, &mut dirty);
    }

    let mut bw = BitWriter::new(out);
    bw.write_byte(0x78);
    bw.write_byte(0xda);
    for (idx, blk) in merged.iter().enumerate() {
        write_block_range(
            tbl,
            &mut bw,
            idx + 1 == merged.len(),
            &data[blk.raw_start..blk.raw_end],
            &blk.lit_lens,
            &blk.dists,
        );
    }
    bw.align_byte();
    let a32 = adler32(data);
    bw.write_byte(((a32 >> 24) & 0xff) as u8);
    bw.write_byte(((a32 >> 16) & 0xff) as u8);
    bw.write_byte(((a32 >> 8) & 0xff) as u8);
    bw.write_byte((a32 & 0xff) as u8);

    if bw.overflow {
        0
    } else {
        bw.pos
    }
}

/// Fast hash3 Pareto match precomputer for scout_zopfli_size (zero BT3 decapitation, ~80ms on 3 MB!).
fn precompute_scout_matches(
    data: &[u8],
    runs: &Runs,
    b_start: usize,
    b_end: usize,
    head: &mut [i32],
    prev: &mut [i32],
    row_stride: usize,
    bpp: usize,
    max_chain: usize,
    pool: &mut MatchPool,
) {
    let b_len = b_end - b_start;
    pool.off.clear();
    pool.off.resize(b_len + 1, 0);
    pool.m_len.clear();
    pool.m_dist.clear();
    let run_left = &runs.run_left;
    let run_bpp = runs.run_bpp.as_deref();
    let run_row = runs.run_row.as_deref();
    let ptr = data.as_ptr();
    let mut min_dist = [0u16; 259];
    let mut tmp_len = [0u16; 260];
    let mut tmp_dist = [0u16; 260];

    for i in 0..b_len {
        pool.off[i] = pool.m_len.len() as u32;
        let p = b_start + i;
        let max_len = (b_end - p).min(258);
        if max_len < 3 {
            continue;
        }
        if p > 0 && unsafe { *ptr.add(p) == *ptr.add(p - 1) } && (run_left[p] as usize) >= max_len {
            pool.m_len.push(max_len as u16);
            pool.m_dist.push(1);
            continue;
        }
        if let Some(rb) = run_bpp {
            if (rb[p] as usize) >= max_len {
                pool.m_len.push(max_len as u16);
                pool.m_dist.push(bpp as u16);
                continue;
            }
        }
        let b0 = unsafe { *ptr.add(p) };
        let b1 = unsafe { *ptr.add(p + 1) };
        let b2 = unsafe { *ptr.add(p + 2) };
        let mut max_found = 2usize;
        if p > 0 && unsafe { *ptr.add(p - 1) == b0 } && run_left[p] >= 3 {
            let rl = (run_left[p] as usize).min(max_len);
            for k in 3..=rl { min_dist[k] = 1; }
            max_found = rl;
        }
        if let Some(rb) = run_bpp {
            let v = (rb[p] as usize).min(max_len);
            if v > max_found {
                for k in max_found + 1..=v { min_dist[k] = bpp as u16; }
                max_found = v;
            }
        }
        if let Some(rr) = run_row {
            let v = (rr[p] as usize).min(max_len);
            if v > max_found {
                for k in max_found + 1..=v { min_dist[k] = row_stride as u16; }
                max_found = v;
            }
        }
        let h = hash3(data, p);
        let limit = p.saturating_sub(WIN_SIZE) as i32;
        let mut cand = head[h];
        let mut ch = max_chain;
        let mut h_best = 2usize;
        while cand >= limit && ch > 0 && h_best < max_len {
            ch -= 1;
            let c = cand as usize;
            if unsafe {
                *ptr.add(c + h_best) == *ptr.add(p + h_best)
                    && *ptr.add(c) == b0
                    && *ptr.add(c + 1) == b1
                    && *ptr.add(c + 2) == b2
            } {
                let l = match_len_fast(data, c, p, 3, max_len);
                if l > h_best {
                    h_best = l;
                    let d16 = (p - c) as u16;
                    if l > max_found {
                        for k in max_found + 1..=l { min_dist[k] = d16; }
                        max_found = l;
                    }
                    let mut k = l;
                    while k >= 3 && d16 < min_dist[k] {
                        min_dist[k] = d16;
                        k -= 1;
                    }
                    if h_best >= 64 {
                        break;
                    }
                }
            }
            let nxt = prev[c & WIN_MASK];
            if nxt >= cand { break; }
            cand = nxt;
        }
        prev[p & WIN_MASK] = head[h];
        head[h] = p as i32;

        if max_found >= 3 {
            let mut steps = 0usize;
            for k in 3..=max_found {
                if k == max_found || min_dist[k] != min_dist[k + 1] {
                    tmp_len[steps] = k as u16;
                    tmp_dist[steps] = min_dist[k];
                    steps += 1;
                }
            }
            pool.m_len.extend_from_slice(&tmp_len[..steps]);
            pool.m_dist.extend_from_slice(&tmp_dist[..steps]);
        }
    }
    pool.off[b_len] = pool.m_len.len() as u32;
}

/// Ultra-fast Hash3 + Optimal-DP + Run-Aware DP-Split Zopfli scout (~160ms on 3 MB!).
/// Eliminates greedy-zlib vs Zopfli rank inversions on dense multi-regime images (e.g. Photo|UI|Photo)!
#[no_mangle]
pub extern "C" fn scout_zopfli_size(
    in_ptr: *const u8,
    in_len: usize,
    row_stride: u32,
    bpp: u32,
) -> u32 {
    if in_ptr.is_null() || in_len == 0 {
        return 0;
    }
    let data = unsafe { slice::from_raw_parts(in_ptr, in_len) };
    let tbl = get_tables();
    let rstride = row_stride as usize;
    let bpp_u = (bpp as usize).max(1);
    let runs = compute_runs(data, bpp_u, rstride);

    let max_mega = if rstride > 1 {
        ((1_900_000 / rstride).max(1)) * rstride
    } else {
        1_900_000usize
    };
    let mut mega_bounds = vec![0usize];
    let mut cur_m = 0usize;
    while cur_m < in_len {
        let nxt = (cur_m + max_mega).min(in_len);
        if in_len - nxt < 65536 {
            mega_bounds.push(in_len);
            break;
        }
        mega_bounds.push(nxt);
        cur_m = nxt;
    }
    let num_megas = mega_bounds.len() - 1;
    let max_b = (0..num_megas)
        .map(|b| mega_bounds[b + 1] - mega_bounds[b])
        .max()
        .unwrap_or(0);

    let mut dp_cost = vec![0f32; max_b + 1];
    let mut dp_len = vec![0u16; max_b + 1];
    let mut dp_dist = vec![0u16; max_b + 1];
    let mut pool = MatchPool {
        off: Vec::with_capacity(max_b + 1),
        m_len: Vec::with_capacity((max_b * 2).max(4096)),
        m_dist: Vec::with_capacity((max_b * 2).max(4096)),
    };
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; WIN_SIZE];

    let mut total_bits = 0u64;
    for b in 0..num_megas {
        let b_start = mega_bounds[b];
        let b_end = mega_bounds[b + 1];
        precompute_scout_matches(
            data, &runs, b_start, b_end, &mut head, &mut prev, rstride,
            bpp_u, 10, &mut pool,
        );
        // Fast exact_mode=0 global init (2 passes) -> Run-Aware DP split -> 1 local exact_mode=0 re-parse per block!
        let mut ll_cost = [0f32; 286];
        let mut d_cost = [4.5f32; 30];
        let mut raw_freq = [0u32; 256];
        for &byte in &data[b_start..b_end] {
            raw_freq[byte as usize] += 1;
        }
        let log_r = ((b_end - b_start).max(2) as f64).log2();
        for i in 0..256 {
            ll_cost[i] = if raw_freq[i] > 0 {
                (log_r - (raw_freq[i] as f64).log2()).max(1.0) as f32
            } else {
                (log_r + 1.0) as f32
            };
        }
        ll_cost[256] = log_r as f32;
        for i in 257..286 {
            ll_cost[i] = if bpp_u == 1 { 8.0 } else { 9.5 };
        }
        let p0 = optimal_parse_block(
            tbl, data, &runs, bpp_u, rstride, b_start, &pool,
            &ll_cost, &d_cost, &[], &mut dp_cost, &mut dp_len, &mut dp_dist,
            0, b_end - b_start, 0,
        );
        let (ll_f0, d_f0, _) = stats_from_range(tbl, &p0.lit_lens, &p0.dists, 0, p0.lit_lens.len());
        let mut sm_ll = [0.0f64; 286];
        let mut sm_d = [0.0f64; 30];
        for i in 0..286 { sm_ll[i] = ll_f0[i] as f64; }
        for i in 0..30 { sm_d[i] = d_f0[i] as f64; }
        entropy_costs_from_stats(&sm_ll, &sm_d, &mut ll_cost, &mut d_cost);
        let init = optimal_parse_block(
            tbl, data, &runs, bpp_u, rstride, b_start, &pool,
            &ll_cost, &d_cost, &[], &mut dp_cost, &mut dp_len, &mut dp_dist,
            0, b_end - b_start, 0,
        );
        let splits = split_token_stream_dp(
            tbl, data, b_start, b_end, rstride, &init.lit_lens, &init.dists, &init.pos_arr,
            12, 96, true,
        );
        for s in 0..splits.len() - 1 {
            let ts = splits[s];
            let te = splits[s + 1];
            let rs = init.pos_arr[ts] as usize;
            let re = if te < init.lit_lens.len() {
                init.pos_arr[te] as usize
            } else {
                b_end
            };
            let (ll_f, d_f, _) = stats_from_range(tbl, &init.lit_lens, &init.dists, ts, te);
            for i in 0..286 { sm_ll[i] = ll_f[i] as f64; }
            for i in 0..30 { sm_d[i] = d_f[i] as f64; }
            entropy_costs_from_stats(&sm_ll, &sm_d, &mut ll_cost, &mut d_cost);
            let p2 = optimal_parse_block(
                tbl, data, &runs, bpp_u, rstride, b_start, &pool,
                &ll_cost, &d_cost, &[], &mut dp_cost, &mut dp_len, &mut dp_dist,
                rs - b_start, re - b_start, 0,
            );
            let b1 = exact_range_bits(tbl, &init.lit_lens, &init.dists, ts, te, true);
            let b2 = exact_range_bits(tbl, &p2.lit_lens, &p2.dists, 0, p2.lit_lens.len(), true);
            total_bits += b1.min(b2);
        }
    }
    ((total_bits + 7) / 8 + 6) as u32
}

/// Ultra-fast zero-syscall native WASM Zopfli-aligned filter/palette scorer!
/// Uses exact same run_left/run_bpp/run_row + RLE/bpp jump hashes + lazy LZ77 + optimal block splitting
/// + dynamic Huffman bit counting as `deflate_zopfli`, in ~15-25ms per image (zero gVisor syscalls!).
#[no_mangle]
pub extern "C" fn fast_zopfli_screen(
    in_ptr: *const u8,
    in_len: usize,
    row_stride: u32,
    bpp: u32,
) -> u32 {
    if in_ptr.is_null() || in_len == 0 {
        return 0;
    }
    let data = unsafe { slice::from_raw_parts(in_ptr, in_len) };
    let tbl = get_tables();
    let rstride = row_stride as usize;
    let bpp_u = (bpp as usize).max(1);
    let runs = compute_runs(data, bpp_u, rstride);

    let n = data.len();
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; WIN_SIZE];
    let mut head2 = vec![-1i32; HASH_SIZE];
    let mut prev2 = vec![-1i32; WIN_SIZE];
    let cap = (n >> 2).max(4096);
    let mut lit_lens = Vec::with_capacity(cap);
    let mut dists = Vec::with_capacity(cap);

    let run_left = &runs.run_left;
    let run_bpp = runs.run_bpp.as_deref();
    let run_row = runs.run_row.as_deref();
    let ptr = data.as_ptr();

    let insert = |p: usize, head: &mut [i32], prev: &mut [i32], head2: &mut [i32], prev2: &mut [i32]| {
        if p + 2 < n {
            let h = hash3(data, p);
            prev[p & WIN_MASK] = head[h];
            head[h] = p as i32;
        }
        let r = run_left[p] as usize;
        if bpp_u == 1 && r >= 3 && r < 258 && p + r < n {
            let h2 = hash_rle(r, data[p], data[p + r]);
            prev2[p & WIN_MASK] = head2[h2];
            head2[h2] = p as i32;
        } else if let Some(rb) = run_bpp {
            let br = rb[p] as usize;
            if br >= bpp_u * 2 && br < 258 && p + br < n {
                let h2 = hash_bpp_run(br, data, p, bpp_u, data[p + br]);
                prev2[p & WIN_MASK] = head2[h2];
                head2[h2] = p as i32;
            }
        }
    };

    let find_best = |p: usize, head: &[i32], prev: &[i32], head2: &[i32], prev2: &[i32]| -> (usize, usize) {
        let max_len = (n - p).min(258);
        if max_len < 3 {
            return (0, 0);
        }
        if p > 0 && data[p] == data[p - 1] && (run_left[p] as usize) >= max_len {
            return (max_len, 1);
        }
        if let Some(rb) = run_bpp {
            if (rb[p] as usize) >= max_len {
                return (max_len, bpp_u);
            }
        }
        let mut best_l = 2usize;
        let mut best_d = 0usize;
        if p > 0 && data[p] == data[p - 1] {
            let rl = (run_left[p] as usize).min(max_len);
            if rl > best_l {
                best_l = rl;
                best_d = 1;
            }
        }
        if let Some(rb) = run_bpp {
            let v = (rb[p] as usize).min(max_len);
            if v > best_l {
                best_l = v;
                best_d = bpp_u;
            }
        }
        if let Some(rr) = run_row {
            let v = (rr[p] as usize).min(max_len);
            if v > best_l {
                best_l = v;
                best_d = rstride;
            }
        }
        if best_l == max_len {
            return (best_l, best_d);
        }
        let limit = p.saturating_sub(WIN_SIZE) as i32;
        let b0 = data[p];
        let b1 = data[p + 1];
        let b2 = data[p + 2];
        let mut cand = head[hash3(data, p)];
        let mut chain = 16usize;
        while cand >= limit && chain > 0 && best_l < max_len {
            chain -= 1;
            let c = cand as usize;
            let hit = unsafe {
                *ptr.add(c + best_l) == *ptr.add(p + best_l)
                    && *ptr.add(c) == b0
                    && *ptr.add(c + 1) == b1
                    && *ptr.add(c + 2) == b2
            };
            if hit {
                let l = match_len_fast(data, c, p, 3, max_len);
                if l > best_l {
                    best_l = l;
                    best_d = p - c;
                    if best_l >= 64 || best_l == max_len {
                        break;
                    }
                }
            }
            let nxt = unsafe { *prev.get_unchecked(c & WIN_MASK) };
            if nxt >= cand {
                break;
            }
            cand = nxt;
        }
        let my_run = run_left[p] as usize;
        if bpp_u == 1 && my_run >= 3 && my_run < max_len && p + my_run < n && best_l < max_len {
            let h2 = hash_rle(my_run, b0, data[p + my_run]);
            let mut c2 = head2[h2];
            let mut ch2 = 12usize;
            while c2 >= limit && ch2 > 0 && best_l < max_len {
                ch2 -= 1;
                let c = c2 as usize;
                if (run_left[c] as usize) == my_run
                    && unsafe {
                        *ptr.add(c) == b0
                            && *ptr.add(c + my_run) == *ptr.add(p + my_run)
                            && *ptr.add(c + best_l) == *ptr.add(p + best_l)
                    }
                {
                    let l = match_len_fast(data, c, p, my_run + 1, max_len);
                    if l > best_l {
                        best_l = l;
                        best_d = p - c;
                        if best_l >= 96 || best_l == max_len {
                            break;
                        }
                    }
                }
                let nxt = unsafe { *prev2.get_unchecked(c & WIN_MASK) };
                if nxt >= c2 {
                    break;
                }
                c2 = nxt;
            }
        } else if let Some(rb) = run_bpp {
            let br = rb[p] as usize;
            if br >= bpp_u * 2 && br < max_len && p + br < n && best_l < max_len {
                let h2 = hash_bpp_run(br, data, p, bpp_u, data[p + br]);
                let mut c2 = head2[h2];
                let mut ch2 = 12usize;
                while c2 >= limit && ch2 > 0 && best_l < max_len {
                    ch2 -= 1;
                    let c = c2 as usize;
                    if (rb[c] as usize) == br
                        && unsafe {
                            *ptr.add(c) == b0
                                && *ptr.add(c + 1) == b1
                                && *ptr.add(c + br) == *ptr.add(p + br)
                                && *ptr.add(c + best_l) == *ptr.add(p + best_l)
                        }
                    {
                        let l = match_len_fast(data, c, p, br + 1, max_len);
                        if l > best_l {
                            best_l = l;
                            best_d = p - c;
                            if best_l >= 96 || best_l == max_len {
                                break;
                            }
                        }
                    }
                    let nxt = unsafe { *prev2.get_unchecked(c & WIN_MASK) };
                    if nxt >= c2 {
                        break;
                    }
                    c2 = nxt;
                }
            }
        }
        if best_l >= 3 {
            (best_l, best_d)
        } else {
            (0, 0)
        }
    };

    let mut p = 0usize;
    while p < n {
        if p > 0 && data[p] == data[p - 1] && run_left[p] >= 258 {
            lit_lens.push(258);
            dists.push(1);
            if run_left[p] < 516 {
                for k in 0..258 {
                    insert(p + k, &mut head, &mut prev, &mut head2, &mut prev2);
                }
            } else {
                insert(p, &mut head, &mut prev, &mut head2, &mut prev2);
            }
            p += 258;
            continue;
        }
        if let Some(rb) = run_bpp {
            if rb[p] >= 258 && run_left[p] < 258 {
                lit_lens.push(258);
                dists.push(bpp_u as u16);
                if rb[p] < 516 {
                    for k in 0..258 {
                        insert(p + k, &mut head, &mut prev, &mut head2, &mut prev2);
                    }
                } else {
                    for k in 0..bpp_u {
                        insert(p + k, &mut head, &mut prev, &mut head2, &mut prev2);
                    }
                }
                p += 258;
                continue;
            }
        }
        let (l1, d1) = find_best(p, &head, &prev, &head2, &prev2);
        insert(p, &mut head, &mut prev, &mut head2, &mut prev2);
        if l1 == 3 && d1 > 2048 {
            lit_lens.push(data[p] as u16);
            dists.push(0);
            p += 1;
            continue;
        }
        if l1 >= 3 && l1 < 48 && p + 1 < n {
            let (l2, d2) = find_best(p + 1, &head, &prev, &head2, &prev2);
            if l2 > l1 + 1 && !(l2 == 3 && d2 > 2048) {
                lit_lens.push(data[p] as u16);
                dists.push(0);
                p += 1;
                continue;
            }
        }
        if l1 >= 3 {
            lit_lens.push(l1 as u16);
            dists.push(d1 as u16);
            let end = p + l1;
            if l1 <= 32 {
                for q in p + 1..end {
                    insert(q, &mut head, &mut prev, &mut head2, &mut prev2);
                }
            } else {
                for q in p + 1..p + 6 {
                    insert(q, &mut head, &mut prev, &mut head2, &mut prev2);
                }
            }
            p = end;
        } else {
            lit_lens.push(data[p] as u16);
            dists.push(0);
            p += 1;
        }
    }

    let count = lit_lens.len();
    let mut splits = vec![0usize, count];
    let mut done = Vec::with_capacity(8);
    for _ in 1..8usize {
        let mut best_seg = usize::MAX;
        let mut best_len = 0usize;
        for i in 0..splits.len() - 1 {
            let a = splits[i];
            let b = splits[i + 1];
            if done.contains(&a) {
                continue;
            }
            if b - a > best_len {
                best_len = b - a;
                best_seg = i;
            }
        }
        if best_seg == usize::MAX || best_len < 512 {
            break;
        }
        let a = splits[best_seg];
        let b = splits[best_seg + 1];
        let orig_cost = exact_range_bits(tbl, &lit_lens, &dists, a, b, true);
        let (pos, cost) = find_best_split_fast(tbl, &lit_lens, &dists, a, b, 15);
        if pos > a && pos < b && cost + 16 < orig_cost {
            splits.insert(best_seg + 1, pos);
        } else {
            done.push(a);
        }
    }
    let mut total_bits = 0u64;
    for i in 0..splits.len() - 1 {
        total_bits += exact_range_bits(tbl, &lit_lens, &dists, splits[i], splits[i + 1], true);
    }
    ((total_bits + 7) / 8 + 6) as u32
}

// ============================================================================
// Fast Native WASM LZ Literal Counts (for pal-lzfreq)
// ============================================================================

#[no_mangle]
pub extern "C" fn fast_lz_lit_counts(
    in_ptr: *const u8,
    in_len: usize,
    row_stride: u32,
    out_counts_ptr: *mut u32,
) {
    if in_ptr.is_null() || out_counts_ptr.is_null() {
        return;
    }
    let data = unsafe { slice::from_raw_parts(in_ptr, in_len) };
    let counts = unsafe { slice::from_raw_parts_mut(out_counts_ptr, 256) };
    for x in counts.iter_mut() {
        *x = 0;
    }
    let runs = compute_runs(data, 1, row_stride as usize);
    let lz = fast_lazy_lz77(data, &runs, row_stride as usize, 1, 24);
    for i in 0..lz.lit_lens.len() {
        if lz.dists[i] == 0 {
            counts[lz.lit_lens[i] as usize] += 1;
        }
    }
}

// ============================================================================
// Native WASM Zero-Syscall Optimal PNG Filter Selectors!
// ============================================================================

#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let ai = a as i32;
    let bi = b as i32;
    let ci = c as i32;
    let p = ai + bi - ci;
    let pa = (p - ai).abs();
    let pb = (p - bi).abs();
    let pc = (p - ci).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn filter_row_into(ft: u8, cur: &[u8], prev: Option<&[u8]>, bpp: usize, dst: &mut [u8]) {
    let n = cur.len();
    match ft {
        0 => dst.copy_from_slice(cur),
        1 => {
            dst[..bpp].copy_from_slice(&cur[..bpp]);
            for i in bpp..n {
                dst[i] = cur[i].wrapping_sub(cur[i - bpp]);
            }
        }
        2 => match prev {
            None => dst.copy_from_slice(cur),
            Some(pr) => {
                for i in 0..n {
                    dst[i] = cur[i].wrapping_sub(pr[i]);
                }
            }
        },
        3 => match prev {
            None => {
                dst[..bpp].copy_from_slice(&cur[..bpp]);
                for i in bpp..n {
                    dst[i] = cur[i].wrapping_sub(cur[i - bpp] >> 1);
                }
            }
            Some(pr) => {
                for i in 0..bpp {
                    dst[i] = cur[i].wrapping_sub(pr[i] >> 1);
                }
                for i in bpp..n {
                    dst[i] = cur[i].wrapping_sub(((cur[i - bpp] as u16 + pr[i] as u16) >> 1) as u8);
                }
            }
        },
        _ => match prev {
            None => {
                dst[..bpp].copy_from_slice(&cur[..bpp]);
                for i in bpp..n {
                    dst[i] = cur[i].wrapping_sub(cur[i - bpp]);
                }
            }
            Some(pr) => {
                for i in 0..bpp {
                    dst[i] = cur[i].wrapping_sub(pr[i]);
                }
                for i in bpp..n {
                    dst[i] = cur[i].wrapping_sub(paeth(cur[i - bpp], pr[i], pr[i - bpp]));
                }
            }
        },
    }
}

fn build_row_bank_rust(tbl: &Tables, samples: &[u8], w: usize, h: usize, bpp: usize) -> Vec<Vec<u8>> {
    let stride = w * bpp;
    let row_len = stride + 1;
    let total = h * row_len;
    // 0..4 fixed, 5=pairs, 6=minsum, 7=true ECT incremental delta-H entropy
    let mut bank: Vec<Vec<u8>> = (0..8).map(|_| vec![0u8; total]).collect();
    let mut seen = vec![0u32; 65536];
    let mut stamp = 0u32;
    let mut global_hist = [0u32; 256];
    let mut global_tot = 0u32;
    let mut row_cnt = [0u32; 256];

    for y in 0..h {
        let cur = &samples[y * stride..(y + 1) * stride];
        let prev = if y == 0 {
            None
        } else {
            Some(&samples[(y - 1) * stride..y * stride])
        };
        let at = y * row_len;

        let mut best_p = 0usize;
        let mut best_pc = u32::MAX;
        let mut best_m = 0usize;
        let mut best_mc = u32::MAX;
        let mut best_e = 0usize;
        let mut best_ec = f64::INFINITY;

        for t in 0..5 {
            bank[t][at] = t as u8;
            filter_row_into(t as u8, cur, prev, bpp, &mut bank[t][at + 1..at + row_len]);
            let row = &bank[t][at + 1..at + row_len];

            // Unique pairs
            stamp += 1;
            let mut pc = 0u32;
            for i in 1..stride {
                let key = ((row[i - 1] as usize) << 8) | (row[i] as usize);
                if seen[key] != stamp {
                    seen[key] = stamp;
                    pc += 1;
                }
            }
            if pc < best_pc {
                best_pc = pc;
                best_p = t;
            }

            // Minsum
            let mut mc = 0u32;
            for &b in row {
                mc += if b < 128 { b as u32 } else { 256 - b as u32 };
            }
            if mc < best_mc {
                best_mc = mc;
                best_m = t;
            }

            // True ECT incremental delta-H entropy
            for x in &mut row_cnt {
                *x = 0;
            }
            row_cnt[t] += 1;
            for &b in row {
                row_cnt[b as usize] += 1;
            }
            let mut ec = 0.0f64;
            for i in 0..256 {
                let c = row_cnt[i];
                if c > 0 {
                    let h0 = global_hist[i];
                    let h1 = h0 + c;
                    let term1 = (h1 as f64) * tbl.fast_log2(h1);
                    let term0 = if h0 > 0 { (h0 as f64) * tbl.fast_log2(h0) } else { 0.0 };
                    ec -= term1 - term0;
                }
            }
            if ec < best_ec {
                best_ec = ec;
                best_e = t;
            }
        }

        for i in 0..row_len {
            bank[5][at + i] = bank[best_p][at + i];
            bank[6][at + i] = bank[best_m][at + i];
            bank[7][at + i] = bank[best_e][at + i];
            global_hist[bank[best_e][at + i] as usize] += 1;
        }
        global_tot += row_len as u32;
        if global_tot > 32768 {
            let mut s = 0u32;
            for i in 0..256 {
                global_hist[i] >>= 1;
                s += global_hist[i];
            }
            global_tot = s;
        }
    }
    bank
}

/// Fast 32 KiB Sliding-Window LZ77 + Adaptive Entropy Per-Row Filter Optimizer
/// (with direct row_stride vertical match check + SIMD128 match_len_fast!)
fn filter_lz_row_rust(
    tbl: &Tables,
    bank: &[Vec<u8>],
    w: usize,
    h: usize,
    bpp: usize,
    out: &mut [u8],
) {
    let row_len = w * bpp + 1;
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; WIN_SIZE];
    let mut row_head = vec![0usize; HASH_SIZE];
    let mut row_stamp = vec![0u32; HASH_SIZE];
    let mut stamp = 0u32;
    let mut hist = [0u32; 256];
    let mut hist_tot = 0u32;
    let mut lit_cost = [0f64; 256];

    let mut dist_cost = [0f64; 30];
    for s in 0..30 {
        dist_cost[s] = 4.2 + DIST_EXTRA[s] as f64;
    }

    for y in 0..h {
        let at = y * row_len;
        let log_den = tbl.fast_log2(hist_tot + 256);
        for i in 0..256 {
            lit_cost[i] = log_den - tbl.fast_log2(hist[i] + 1);
        }

        let mut best_t = 0usize;
        let mut best_bits = f64::INFINITY;

        for t in 0..5 {
            let row = &bank[t][at..at + row_len];
            stamp += 1;
            let mut bits = lit_cost[t];
            let mut x = 1usize;
            while x < row_len {
                let b0 = row[x];
                let rem = row_len - x;
                if rem >= 3 {
                    let max_l = rem.min(258);
                    let mut best_l = 2usize;
                    let mut best_d = 0usize;
                    if x > 1 && b0 == row[x - 1] {
                        let mut l = 1;
                        while l < max_l && row[x + l] == b0 {
                            l += 1;
                        }
                        if l > best_l {
                            best_l = l;
                            best_d = 1;
                        }
                    }
                    if bpp > 1 && x > bpp && b0 == row[x - bpp] && best_l < max_l {
                        let mut l = 1;
                        while l < max_l && row[x + l] == row[x - bpp + l] {
                            l += 1;
                        }
                        if l > best_l {
                            best_l = l;
                            best_d = bpp;
                        }
                    }
                    if best_l < 64 && best_l < max_l {
                        let h3 = hash3(row, x);
                        if row_stamp[h3] == stamp {
                            let rx = row_head[h3];
                            if row[rx] == b0 && row[rx + 1] == row[x + 1] && row[rx + 2] == row[x + 2] {
                                let l = match_len_fast(row, rx, x, 3, max_l);
                                if l > best_l {
                                    best_l = l;
                                    best_d = x - rx;
                                }
                            }
                        }
                        if at > 0 && best_l < 32 && best_l < max_l {
                            let cur_limit = (at + x).saturating_sub(WIN_SIZE) as i32;
                            let b1 = row[x + 1];
                            let b2 = row[x + 2];
                            let mut cand = head[h3];
                            let mut probes = 8usize;
                            while cand >= cur_limit && probes > 0 && best_l < max_l {
                                probes -= 1;
                                let c = cand as usize;
                                let lim = (at - c).min(max_l);
                                if lim > best_l
                                    && out[c + best_l] == row[x + best_l]
                                    && out[c] == b0
                                    && out[c + 1] == b1
                                    && out[c + 2] == b2
                                {
                                    let mut l = 3;
                                    while l < lim && out[c + l] == row[x + l] {
                                        l += 1;
                                    }
                                    if l > best_l {
                                        best_l = l;
                                        best_d = (at + x) - c;
                                        if best_l >= 32 {
                                            break;
                                        }
                                    }
                                }
                                let nxt = prev[c & WIN_MASK];
                                if nxt >= cand {
                                    break;
                                }
                                cand = nxt;
                            }
                        }
                        row_stamp[h3] = stamp;
                        row_head[h3] = x;
                    }
                    if best_l >= 3 {
                        let m_bits = 6.2
                            + tbl.len_ex[best_l] as f64
                            + dist_cost[tbl.dist_sym[best_d] as usize];
                        let lit_sum = if best_l <= 5 {
                            let mut s = 0.0;
                            for k in 0..best_l {
                                s += lit_cost[row[x + k] as usize];
                            }
                            s
                        } else {
                            1e9
                        };
                        if m_bits < lit_sum {
                            bits += m_bits;
                            let end = x + best_l;
                            if best_l <= 10 {
                                for q in x + 1..end {
                                    if q + 2 < row_len {
                                        let hq = hash3(row, q);
                                        row_stamp[hq] = stamp;
                                        row_head[hq] = q;
                                    }
                                }
                            }
                            x = end;
                            continue;
                        }
                    }
                }
                bits += lit_cost[b0 as usize];
                x += 1;
            }
            if bits < best_bits {
                best_bits = bits;
                best_t = t;
            }
        }

        out[at..at + row_len].copy_from_slice(&bank[best_t][at..at + row_len]);
        for x in 0..row_len {
            let p = at + x;
            hist[out[p] as usize] += 1;
            if x + 2 < row_len {
                let h3 = hash3(out, p);
                prev[p & WIN_MASK] = head[h3];
                head[h3] = p as i32;
            }
        }
        hist_tot += row_len as u32;
        if hist_tot > 65536 {
            let mut s = 0u32;
            for i in 0..256 {
                hist[i] >>= 1;
                s += hist[i];
            }
            hist_tot = s;
        }
    }
}

struct FastLzEval {
    ctx_head: Vec<i32>,
    ctx_stamp: Vec<u32>,
    ctx_prev: Vec<i32>,
    band_head: Vec<i32>,
    band_stamp: Vec<u32>,
    band_prev: Vec<i32>,
    c_stamp: u32,
    b_stamp: u32,
}

impl FastLzEval {
    fn new() -> Self {
        Self {
            ctx_head: vec![-1i32; HASH_SIZE],
            ctx_stamp: vec![0u32; HASH_SIZE],
            ctx_prev: vec![-1i32; WIN_SIZE],
            band_head: vec![-1i32; HASH_SIZE],
            band_stamp: vec![0u32; HASH_SIZE],
            band_prev: vec![-1i32; WIN_SIZE],
            c_stamp: 0,
            b_stamp: 0,
        }
    }

    fn index_ctx(&mut self, ctx: &[u8]) {
        self.c_stamp += 1;
        let cs = self.c_stamp;
        for p in 0..ctx.len().saturating_sub(2) {
            let h = hash3(ctx, p);
            self.ctx_prev[p & WIN_MASK] = if self.ctx_stamp[h] == cs {
                self.ctx_head[h]
            } else {
                -1
            };
            self.ctx_stamp[h] = cs;
            self.ctx_head[h] = p as i32;
        }
    }

    fn parse_slice(
        &mut self,
        tbl: &Tables,
        ctx: &[u8],
        src: &[u8],
        row_len: usize,
        bpp: usize,
        ll_out: &mut [u32; 286],
        d_out: &mut [u32; 30],
        has_ctx: bool,
        lit_aware: bool,
    ) -> (u64, u32, u32) {
        self.b_stamp += 1;
        let bs = self.b_stamp;
        let cs = self.c_stamp;
        let src_len = src.len();
        let ctx_len = ctx.len();
        let mut extra = 0u64;
        let mut ll_tot = 0u32;
        let mut d_tot = 0u32;

        let mut lit_bits = [8.0f32; 256];
        if lit_aware && src_len > 0 {
            let mut cnt = [0u32; 256];
            for &b in src {
                cnt[b as usize] += 1;
            }
            let log_s = tbl.fast_log2(src_len.max(2) as u32) as f32;
            for i in 0..256 {
                if cnt[i] > 0 {
                    lit_bits[i] = (log_s - tbl.fast_log2(cnt[i]) as f32).clamp(1.0, 11.0);
                } else {
                    lit_bits[i] = 11.0;
                }
            }
        }

        let mut p = 0usize;
        while p < src_len {
            let rem = src_len - p;
            let b0 = src[p];
            if rem >= 3 {
                let max_l = rem.min(258);
                let mut best_l = 2usize;
                let mut best_d = 0usize;

                let prev_b = if p > 0 {
                    Some(src[p - 1])
                } else if ctx_len > 0 {
                    Some(ctx[ctx_len - 1])
                } else {
                    None
                };
                if prev_b == Some(b0) {
                    let mut l = 1;
                    while l < max_l && src[p + l] == b0 {
                        l += 1;
                    }
                    if l > best_l {
                        best_l = l;
                        best_d = 1;
                    }
                }
                if bpp > 1 && p >= bpp && src[p - bpp] == b0 && best_l < max_l {
                    let mut l = 1;
                    while l < max_l && src[p + l] == src[p - bpp + l] {
                        l += 1;
                    }
                    if l > best_l {
                        best_l = l;
                        best_d = bpp;
                    }
                }
                if p >= row_len && src[p - row_len] == b0 && best_l < max_l {
                    let l = match_len_fast(src, p - row_len, p, 1, max_l);
                    if l > best_l {
                        best_l = l;
                        best_d = row_len;
                    }
                }
                if p >= row_len * 2 && row_len * 2 <= WIN_SIZE && src[p - row_len * 2] == b0 && best_l < max_l {
                    let l = match_len_fast(src, p - row_len * 2, p, 1, max_l);
                    if l > best_l {
                        best_l = l;
                        best_d = row_len * 2;
                    }
                }

                let h3 = hash3(src, p);
                if best_l < 64 && best_l < max_l {
                    let b1 = src[p + 1];
                    let b2 = src[p + 2];
                    if self.band_stamp[h3] == bs {
                        let mut cand = self.band_head[h3];
                        let b_lim = p.saturating_sub(WIN_SIZE) as i32;
                        let mut probes = if lit_aware { 5usize } else { 10usize };
                        while cand >= b_lim && probes > 0 && best_l < max_l {
                            probes -= 1;
                            let c = cand as usize;
                            if src[c + best_l] == src[p + best_l]
                                && src[c] == b0
                                && src[c + 1] == b1
                                && src[c + 2] == b2
                            {
                                let l = match_len_fast(src, c, p, 3, max_l);
                                if l > best_l {
                                    best_l = l;
                                    best_d = p - c;
                                    if best_l >= 64 {
                                        break;
                                    }
                                }
                            }
                            let nxt = self.band_prev[c & WIN_MASK];
                            if nxt >= cand {
                                break;
                            }
                            cand = nxt;
                        }
                    }
                    if has_ctx && best_l < 64 && best_l < max_l && self.ctx_stamp[h3] == cs {
                        let mut cand = self.ctx_head[h3];
                        let min_ctx = (p + ctx_len).saturating_sub(WIN_SIZE) as i32;
                        let mut probes = 6usize;
                        while cand >= min_ctx && probes > 0 && best_l < max_l {
                            probes -= 1;
                            let c = cand as usize;
                            let lim = (ctx_len - c).min(max_l);
                            if lim > best_l
                                && ctx[c + best_l] == src[p + best_l]
                                && ctx[c] == b0
                                && ctx[c + 1] == b1
                                && ctx[c + 2] == b2
                            {
                                let mut l = 3;
                                while l < lim && ctx[c + l] == src[p + l] {
                                    l += 1;
                                }
                                if l > best_l {
                                    best_l = l;
                                    best_d = (ctx_len + p) - c;
                                    if best_l >= 64 {
                                        break;
                                    }
                                }
                            }
                            let nxt = self.ctx_prev[c & WIN_MASK];
                            if nxt >= cand {
                                break;
                            }
                            cand = nxt;
                        }
                    }
                }

                self.band_prev[p & WIN_MASK] = if self.band_stamp[h3] == bs {
                    self.band_head[h3]
                } else {
                    -1
                };
                self.band_stamp[h3] = bs;
                self.band_head[h3] = p as i32;

                if lit_aware && best_l >= 3 && best_l <= 7 {
                    let m_bits = 5.5f32
                        + tbl.len_ex[best_l] as f32
                        + 4.0f32
                        + DIST_EXTRA[tbl.dist_sym[best_d] as usize] as f32;
                    let mut l_bits = 0.0f32;
                    for k in 0..best_l {
                        l_bits += lit_bits[src[p + k] as usize];
                    }
                    if m_bits >= l_bits {
                        best_l = 0;
                    }
                }
                if best_l >= 3 {
                    ll_out[tbl.len_sym[best_l] as usize] += 1;
                    ll_tot += 1;
                    let ds = tbl.dist_sym[best_d] as usize;
                    d_out[ds] += 1;
                    d_tot += 1;
                    extra += (tbl.len_ex[best_l] + DIST_EXTRA[ds]) as u64;
                    let end = p + best_l;
                    if best_l <= 24 {
                        for q in p + 1..end {
                            if q + 2 < src_len {
                                let hq = hash3(src, q);
                                self.band_prev[q & WIN_MASK] = if self.band_stamp[hq] == bs {
                                    self.band_head[hq]
                                } else {
                                    -1
                                };
                                self.band_stamp[hq] = bs;
                                self.band_head[hq] = q as i32;
                            }
                        }
                    }
                    p = end;
                    continue;
                }
            }
            ll_out[b0 as usize] += 1;
            ll_tot += 1;
            p += 1;
        }
        (extra, ll_tot, d_tot)
    }
}

fn entropy_cost_bits(
    tbl: &Tables,
    ll: &[u32; 286],
    d: &[u32; 30],
    ll_tot: u32,
    d_tot: u32,
    extra: u64,
) -> f64 {
    let mut bits = extra as f64 + 64.0;
    let log_ll = tbl.fast_log2(ll_tot + 1);
    let mut act_ll = 0u32;
    for &c in ll {
        if c > 0 {
            act_ll += 1;
            bits += (c as f64) * (log_ll - tbl.fast_log2(c)).max(1.0);
        }
    }
    let mut act_d = 0u32;
    if d_tot > 0 {
        let log_d = tbl.fast_log2(d_tot);
        for &c in d {
            if c > 0 {
                act_d += 1;
                bits += (c as f64) * (log_d - tbl.fast_log2(c)).max(1.0);
            }
        }
    }
    bits + (act_ll as f64) * 2.2 + (act_d as f64) * 2.8
}

/// Native WASM zero-syscall hybrid banded filter selector with exact row boundary refinement & bidir segment check!
/// When `sep_blocks` is true, scores each band/segment with its OWN independent dynamic Huffman table
/// (using preceding context strictly for LZ77 matching) + a realistic block-switch penalty — discovering
/// clean alternating multi-regime filters (e.g. Photo | UI gutter | Photo) cleanly!
fn filter_band_rust(
    tbl: &Tables,
    bank: &[Vec<u8>],
    w: usize,
    h: usize,
    bpp: usize,
    band_h: usize,
    sep_blocks: bool,
    out: &mut [u8],
) {
    let row_len = w * bpp + 1;
    let num_modes = bank.len();
    let max_win = (24576 / row_len).max(1) * row_len;
    let num_bands = (h + band_h - 1) / band_h;
    let mut band_modes = vec![0usize; num_bands];

    let mut eval = FastLzEval::new();
    let mut base_ll = [0u32; 286];
    let mut base_d = [0u32; 30];
    let mut ll = [0u32; 286];
    let mut d = [0u32; 30];

    for bi in 0..num_bands {
        let y0 = bi * band_h;
        let bh = band_h.min(h - y0);
        let at = y0 * row_len;
        let b_bytes = bh * row_len;
        let w_len = at.min(max_win);
        let ctx = &out[at - w_len..at];

        for x in &mut base_ll {
            *x = 0;
        }
        for x in &mut base_d {
            *x = 0;
        }
        base_ll[256] = 1;
        let (b_ex, b_ll, b_d) = if w_len > 0 {
            eval.index_ctx(ctx);
            if sep_blocks {
                (0u64, 1u32, 0u32)
            } else {
                let (e, l, dt) =
                    eval.parse_slice(tbl, &[], ctx, row_len, bpp, &mut base_ll, &mut base_d, false, false);
                (e, 1 + l, dt)
            }
        } else {
            (0, 1, 0)
        };

        let prev_mode = if bi > 0 { band_modes[bi - 1] } else { usize::MAX };
        let mut best_score = f64::INFINITY;
        let mut best_mode = 0usize;
        for m in 0..num_modes {
            ll = base_ll;
            d = base_d;
            let (e, l, dt) = eval.parse_slice(
                tbl,
                ctx,
                &bank[m][at..at + b_bytes],
                row_len,
                bpp,
                &mut ll,
                &mut d,
                w_len > 0,
                sep_blocks,
            );
            let mut bits = entropy_cost_bits(tbl, &ll, &d, b_ll + l, b_d + dt, b_ex + e);
            if sep_blocks && prev_mode != usize::MAX && m != prev_mode {
                bits += 260.0;
            }
            if bits < best_score {
                best_score = bits;
                best_mode = m;
            }
        }
        band_modes[bi] = best_mode;
        out[at..at + b_bytes].copy_from_slice(&bank[best_mode][at..at + b_bytes]);
    }

    // Merge adjacent bands with identical modes into segments
    let mut segs: Vec<(usize, usize, usize)> = Vec::new();
    for bi in 0..num_bands {
        let y0 = bi * band_h;
        let y1 = ((bi + 1) * band_h).min(h);
        let m = band_modes[bi];
        if let Some(last) = segs.last_mut() {
            if last.2 == m {
                last.1 = y1;
                continue;
            }
        }
        segs.push((y0, y1, m));
    }

    // Coarse-to-fine row-level boundary refinement with 8 KiB preceding context!
    let half_span = if sep_blocks { 14usize } else { band_h.min(28) };
    let ctx_max = (8192 / row_len).max(1) * row_len;
    let mut span_work = vec![0u8; ctx_max + (half_span * 2 + 2) * row_len];

    for si in 0..segs.len().saturating_sub(1) {
        let (a_y0, y_mid, mode_a) = segs[si];
        let (_, b_y1, mode_b) = segs[si + 1];
        let y_start = a_y0.max(y_mid.saturating_sub(half_span));
        let y_end = b_y1.min(y_mid + half_span);
        let span_h = y_end - y_start;
        if span_h < 6 {
            continue;
        }
        let at_start = y_start * row_len;
        let ctx_len = at_start.min(ctx_max);
        if ctx_len > 0 {
            span_work[..ctx_len].copy_from_slice(&out[at_start - ctx_len..at_start]);
        }
        let total_span = ctx_len + span_h * row_len;

        let mut eval_split = |split_y: usize| -> f64 {
            let k = split_y - y_start;
            if k > 0 {
                span_work[ctx_len..ctx_len + k * row_len]
                    .copy_from_slice(&bank[mode_a][at_start..at_start + k * row_len]);
            }
            if k < span_h {
                span_work[ctx_len + k * row_len..total_span]
                    .copy_from_slice(&bank[mode_b][at_start + k * row_len..y_end * row_len]);
            }
            if sep_blocks {
                let split_off = ctx_len + k * row_len;
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e1, l1, d1) = eval.parse_slice(tbl, &[], &span_work[..split_off], row_len, bpp, &mut ll, &mut d, false, true);
                let c1 = entropy_cost_bits(tbl, &ll, &d, 1 + l1, d1, e1);
                let c_len2 = split_off.min(8192);
                let ctx2 = &span_work[split_off - c_len2..split_off];
                if c_len2 > 0 { eval.index_ctx(ctx2); }
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e2, l2, d2) = eval.parse_slice(tbl, ctx2, &span_work[split_off..total_span], row_len, bpp, &mut ll, &mut d, c_len2 > 0, true);
                c1 + entropy_cost_bits(tbl, &ll, &d, 1 + l2, d2, e2)
            } else {
                for x in &mut ll {
                    *x = 0;
                }
                for x in &mut d {
                    *x = 0;
                }
                ll[256] = 1;
                let (e, l, dt) = eval.parse_slice(
                    tbl,
                    &[],
                    &span_work[..total_span],
                    row_len,
                    bpp,
                    &mut ll,
                    &mut d,
                    false,
                    false,
                );
                entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e)
            }
        };

        let mut best_y = y_mid;
        let mut best_c = eval_split(y_mid);
        let mut y = y_start + 4;
        while y + 4 <= y_end {
            if y != y_mid {
                let c = eval_split(y);
                if c < best_c {
                    best_c = c;
                    best_y = y;
                }
            }
            y += 4;
        }
        let center = best_y;
        for dy in [-2isize, -1, 1, 2] {
            let cy = (center as isize + dy) as usize;
            if cy > y_start && cy < y_end {
                let c = eval_split(cy);
                if c < best_c {
                    best_c = c;
                    best_y = cy;
                }
            }
        }
        if best_y != y_mid {
            let k = best_y - y_start;
            if k > 0 {
                out[at_start..at_start + k * row_len]
                    .copy_from_slice(&bank[mode_a][at_start..at_start + k * row_len]);
            }
            if k < span_h {
                out[at_start + k * row_len..y_end * row_len]
                    .copy_from_slice(&bank[mode_b][at_start + k * row_len..y_end * row_len]);
            }
            segs[si].1 = best_y;
            segs[si + 1].0 = best_y;
        }
    }

    // Bidirectional segment-mode re-check!
    if segs.len() >= 2 {
        let pad_rows = (12288 / row_len).clamp(4, 12);
        for si in 0..segs.len() {
            let (sy0, sy1, smode) = segs[si];
            if sy1 <= sy0 {
                continue;
            }
            if sep_blocks {
                let at = sy0 * row_len;
                let seg_bytes = (sy1 - sy0) * row_len;
                if seg_bytes > 520 * 1024 {
                    continue;
                }
                let w_len = at.min(max_win);
                let ctx = &out[at - w_len..at];
                if w_len > 0 {
                    eval.index_ctx(ctx);
                }
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e0, l0, d0) = eval.parse_slice(
                    tbl, ctx, &bank[smode][at..at + seg_bytes], row_len, bpp, &mut ll, &mut d, w_len > 0, true,
                );
                let mut best_m = smode;
                let mut best_c = entropy_cost_bits(tbl, &ll, &d, 1 + l0, d0, e0);
                for m in 0..num_modes {
                    if m == smode {
                        continue;
                    }
                    for x in &mut ll { *x = 0; }
                    for x in &mut d { *x = 0; }
                    ll[256] = 1;
                    let (e, l, dt) = eval.parse_slice(
                        tbl, ctx, &bank[m][at..at + seg_bytes], row_len, bpp, &mut ll, &mut d, w_len > 0, true,
                    );
                    let c = entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
                    if c + 24.0 < best_c {
                        best_c = c;
                        best_m = m;
                    }
                }
                if best_m != smode {
                    segs[si].2 = best_m;
                    out[at..at + seg_bytes]
                        .copy_from_slice(&bank[best_m][at..at + seg_bytes]);
                }
                continue;
            }
            let w_start = sy0.saturating_sub(pad_rows);
            let w_end = (sy1 + pad_rows).min(h);
            let w_bytes = (w_end - w_start) * row_len;
            if w_bytes > 520 * 1024 {
                continue;
            }
            let mut buf = out[w_start * row_len..w_end * row_len].to_vec();
            let seg_off = (sy0 - w_start) * row_len;
            let seg_bytes = (sy1 - sy0) * row_len;

            for x in &mut ll {
                *x = 0;
            }
            for x in &mut d {
                *x = 0;
            }
            ll[256] = 1;
            let (e0, l0, d0) = eval.parse_slice(tbl, &[], &buf, row_len, bpp, &mut ll, &mut d, false, false);
            let mut best_m = smode;
            let mut best_c = entropy_cost_bits(tbl, &ll, &d, 1 + l0, d0, e0);

            for m in 0..num_modes {
                if m == smode {
                    continue;
                }
                buf[seg_off..seg_off + seg_bytes]
                    .copy_from_slice(&bank[m][sy0 * row_len..sy1 * row_len]);
                for x in &mut ll {
                    *x = 0;
                }
                for x in &mut d {
                    *x = 0;
                }
                ll[256] = 1;
                let (e, l, dt) =
                    eval.parse_slice(tbl, &[], &buf, row_len, bpp, &mut ll, &mut d, false, false);
                let c = entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
                if c + 24.0 < best_c {
                    best_c = c;
                    best_m = m;
                }
            }
            if best_m != smode {
                segs[si].2 = best_m;
                out[sy0 * row_len..sy1 * row_len]
                    .copy_from_slice(&bank[best_m][sy0 * row_len..sy1 * row_len]);
            }
        }
    }
}

/// Exact-Up filter overlay: converts non-uniform exact-duplicate rows (cur == prev && !all_same_pixel)
/// to Filter 2 (Up -> all 0x00 bytes, dist=1 with 0 extra bits!).
/// When `leave_last` is true (3/4-bpp), only flips consecutive runs >= 2 and leaves the last row unfiltered
/// so the 32 KiB LZ77 window stays refreshed with raw pixel data for subsequent rows.
fn apply_exact_up_rust(
    samples: &[u8],
    base: &[u8],
    w: usize,
    h: usize,
    bpp: usize,
    leave_last: bool,
    out: &mut [u8],
) {
    let stride = w * bpp;
    let row_len = stride + 1;
    out.copy_from_slice(base);
    let mut y = 1usize;
    while y < h {
        let c_off = y * stride;
        let p_off = (y - 1) * stride;
        if samples[c_off..c_off + stride] == samples[p_off..p_off + stride] {
            // Check if row is uniform single-pixel (already dist=1 or dist=bpp with 0 extra bits)
            let mut unif = true;
            for i in bpp..stride {
                if samples[c_off + i] != samples[c_off + i - bpp] {
                    unif = false;
                    break;
                }
            }
            let mut end = y + 1;
            while end < h
                && samples[end * stride..(end + 1) * stride]
                    == samples[(end - 1) * stride..end * stride]
            {
                end += 1;
            }
            if !unif {
                let run_len = end - y;
                if !leave_last {
                    for r in y..end {
                        let at = r * row_len;
                        out[at] = 2;
                        for b in &mut out[at + 1..at + row_len] {
                            *b = 0;
                        }
                    }
                } else if run_len >= 2 {
                    for r in y..end - 1 {
                        let at = r * row_len;
                        out[at] = 2;
                        for b in &mut out[at + 1..at + row_len] {
                            *b = 0;
                        }
                    }
                }
            }
            y = end;
        } else {
            y += 1;
        }
    }
}

/// Steady-State + 2-Pass Marginal-Context & Exact-Up-Aware Viterbi DP Row Filter Optimizer!
/// Pass 1: Steady-state 32 KiB LZ77 (with BOTH RLE-run & period-bpp jump hashes!) + per-quarter exact Huffman.
///         Non-uniform exact-duplicate rows get a low transition penalty into Filter 2 (Exact-Up)!
/// Pass 2: Computes REAL per-quarter Huffman lengths of Pass 1's mixed winner, then re-scores every row's
///         marginal bit cost under the mixed stream's Huffman table and re-solves Viterbi DP!
fn filter_viterbi_rust(
    tbl: &Tables,
    bank: &[Vec<u8>],
    w: usize,
    h: usize,
    bpp: usize,
    out_best: &mut [u8],
    out_macro: &mut [u8],
) -> bool {
    let stride = w * bpp;
    let row_len = stride + 1;
    let img_bytes = h * row_len;
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; WIN_SIZE];
    let mut head2 = vec![-1i32; HASH_SIZE];
    let mut prev2 = vec![-1i32; WIN_SIZE];
    let mut row_cost = vec![[0.0f64; 5]; h];

    let mut tok_row = Vec::with_capacity((img_bytes >> 3).max(4096));
    let mut tok_ll = Vec::with_capacity((img_bytes >> 3).max(4096));
    let mut tok_ds = Vec::with_capacity((img_bytes >> 3).max(4096));
    let mut tok_ex = Vec::with_capacity((img_bytes >> 3).max(4096));

    for t in 0..5 {
        let src = &bank[t];
        for x in &mut head {
            *x = -1;
        }
        for x in &mut head2 {
            *x = -1;
        }
        tok_row.clear();
        tok_ll.clear();
        tok_ds.clear();
        tok_ex.clear();

        let mut reg_ll = [[0u32; 286]; 4];
        let mut reg_d = [[0u32; 30]; 4];

        let mut p = 0usize;
        while p < img_bytes {
            let y = p / row_len;
            let reg = ((y * 4) / h.max(1)).min(3);
            let rem = img_bytes - p;
            let b0 = src[p];
            if rem >= 3 {
                let max_l = rem.min(258);
                let mut best_l = 2usize;
                let mut best_d = 0usize;
                let mut bpp_run = 0usize;

                if p > 0 && src[p - 1] == b0 {
                    let mut l = 1;
                    while l < max_l && src[p + l] == b0 {
                        l += 1;
                    }
                    if l > best_l {
                        best_l = l;
                        best_d = 1;
                    }
                }
                if bpp > 1 && p >= bpp && src[p - bpp] == b0 && best_l < max_l {
                    let l = match_len_fast(src, p - bpp, p, 1, max_l);
                    bpp_run = l;
                    if l > best_l {
                        best_l = l;
                        best_d = bpp;
                    }
                }
                for mult in 1usize..=4usize {
                    let rd = row_len * mult;
                    if rd > WIN_SIZE || best_l == max_l {
                        break;
                    }
                    if p >= rd && src[p - rd] == b0 {
                        let l = match_len_fast(src, p - rd, p, 1, max_l);
                        if l > best_l {
                            best_l = l;
                            best_d = rd;
                        }
                    }
                }
                let h3 = hash3(src, p);
                if best_l < 64 && best_l < max_l {
                    let b1 = src[p + 1];
                    let b2 = src[p + 2];
                    let limit = p.saturating_sub(WIN_SIZE) as i32;
                    let mut cand = head[h3];
                    let mut probes = 8usize;
                    while cand >= limit && probes > 0 && best_l < max_l {
                        probes -= 1;
                        let c = cand as usize;
                        if src[c + best_l] == src[p + best_l]
                            && src[c] == b0
                            && src[c + 1] == b1
                            && src[c + 2] == b2
                        {
                            let l = match_len_fast(src, c, p, 3, max_l);
                            if l > best_l {
                                best_l = l;
                                best_d = p - c;
                                if best_l >= 64 {
                                    break;
                                }
                            }
                        }
                        let nxt = prev[c & WIN_MASK];
                        if nxt >= cand {
                            break;
                        }
                        cand = nxt;
                    }
                }
                // Period-bpp solid-pixel jump hash (eliminates solid-bar hash clogging on 3/4-bpp!)
                if bpp > 1
                    && bpp_run >= bpp * 2
                    && bpp_run < max_l
                    && p + bpp_run < img_bytes
                    && best_l < max_l
                {
                    let h2 = hash_bpp_run(bpp_run, src, p, bpp, src[p + bpp_run]);
                    let limit = p.saturating_sub(WIN_SIZE) as i32;
                    let mut cand = head2[h2];
                    let mut probes = 8usize;
                    while cand >= limit && probes > 0 && best_l < max_l {
                        probes -= 1;
                        let c = cand as usize;
                        if c + bpp_run < img_bytes
                            && src[c] == b0
                            && src[c + bpp_run] == src[p + bpp_run]
                            && src[c + best_l] == src[p + best_l]
                        {
                            let l = match_len_fast(src, c, p, 1, max_l);
                            if l > best_l {
                                best_l = l;
                                best_d = p - c;
                                if best_l >= 96 {
                                    break;
                                }
                            }
                        }
                        let nxt = prev2[c & WIN_MASK];
                        if nxt >= cand {
                            break;
                        }
                        cand = nxt;
                    }
                    prev2[p & WIN_MASK] = head2[h2];
                    head2[h2] = p as i32;
                }
                prev[p & WIN_MASK] = head[h3];
                head[h3] = p as i32;

                if best_l >= 3 {
                    let ls = tbl.len_sym[best_l];
                    let ds = tbl.dist_sym[best_d];
                    let ex = tbl.len_ex[best_l] + DIST_EXTRA[ds as usize];
                    reg_ll[reg][ls as usize] += 1;
                    reg_d[reg][ds as usize] += 1;
                    tok_row.push(y as u16);
                    tok_ll.push(ls);
                    tok_ds.push(ds);
                    tok_ex.push(ex);
                    let end = p + best_l;
                    if best_l <= 16 {
                        for q in p + 1..end {
                            if q + 2 < img_bytes {
                                let hq = hash3(src, q);
                                prev[q & WIN_MASK] = head[hq];
                                head[hq] = q as i32;
                            }
                        }
                    }
                    p = end;
                    continue;
                }
            }
            reg_ll[reg][b0 as usize] += 1;
            tok_row.push(y as u16);
            tok_ll.push(b0 as u16);
            tok_ds.push(255u8);
            tok_ex.push(0u8);
            p += 1;
        }

        // Exact per-quarter Huffman lengths -> exact per-row DEFLATE bit costs for steady-state filter t!
        let mut ll_lens = [[0u8; 286]; 4];
        let mut d_lens = [[0u8; 30]; 4];
        for r in 0..4 {
            reg_ll[r][256] = 1;
            package_merge(&reg_ll[r], 15, &mut ll_lens[r]);
            package_merge(&reg_d[r], 15, &mut d_lens[r]);
        }
        for k in 0..tok_ll.len() {
            let y = tok_row[k] as usize;
            let reg = ((y * 4) / h.max(1)).min(3);
            let mut bits = ll_lens[reg][tok_ll[k] as usize] as f64;
            let ds = tok_ds[k];
            if ds != 255 {
                bits += d_lens[reg][ds as usize] as f64 + tok_ex[k] as f64;
            }
            row_cost[y][t] += bits;
        }
    }

    let scale = 1.0 + 0.45 * (bpp.saturating_sub(1) as f64);
    let penalties = [
        120.0 * scale,
        360.0 * scale,
        1200.0 * scale,
        2600.0 * scale,
    ];
    let mut dp = vec![[0.0f64; 5]; h];
    let mut back = vec![[0u8; 5]; h];
    let mut seq = vec![0usize; h];
    let mut prev_seq = vec![usize::MAX; h];
    let mut best_seq = vec![usize::MAX; h];
    let mut macro_seq = vec![usize::MAX; h];
    let mut cand_buf = vec![0u8; img_bytes];
    let mut eval = FastLzEval::new();
    let mut ll = [0u32; 286];
    let mut d = [0u32; 30];
    let mut best_bits = f64::INFINITY;
    let mut best_macro_bits = f64::INFINITY;
    let q_rows = (h / 4).max(1);

    for (pi, &pen) in penalties.iter().enumerate() {
        for t in 0..5 {
            dp[0][t] = row_cost[0][t];
        }
        for y in 1..h {
            for t in 0..5 {
                let mut best_prev = 0usize;
                let mut best_v = f64::INFINITY;
                for p_t in 0..5 {
                    let v = dp[y - 1][p_t] + if p_t == t { 0.0 } else { pen };
                    if v < best_v {
                        best_v = v;
                        best_prev = p_t;
                    }
                }
                dp[y][t] = row_cost[y][t] + best_v;
                back[y][t] = best_prev as u8;
            }
        }
        let mut cur_t = 0usize;
        let mut best_end = f64::INFINITY;
        for t in 0..5 {
            if dp[h - 1][t] < best_end {
                best_end = dp[h - 1][t];
                cur_t = t;
            }
        }
        for y in (0..h).rev() {
            seq[y] = cur_t;
            cur_t = back[y][cur_t] as usize;
        }
        if seq == prev_seq && pi != 3 {
            continue;
        }
        prev_seq.copy_from_slice(&seq);
        for y in 0..h {
            let at = y * row_len;
            cand_buf[at..at + row_len].copy_from_slice(&bank[seq[y]][at..at + row_len]);
        }

        if pi < 3 {
            // Classic 4-quarter blocked FastLzEval scoring for fine/medium Viterbi (proven best on i003/i004!)
            let mut tot_bits = 0.0f64;
            for qi in 0..4 {
                let ry0 = qi * q_rows;
                let ry1 = if qi == 3 { h } else { ((qi + 1) * q_rows).min(h) };
                if ry0 >= ry1 {
                    continue;
                }
                let s_off = ry0 * row_len;
                let e_off = ry1 * row_len;
                let c_len = s_off.min(16384);
                let ctx = &cand_buf[s_off - c_len..s_off];
                if c_len > 0 {
                    eval.index_ctx(ctx);
                }
                for x in &mut ll {
                    *x = 0;
                }
                for x in &mut d {
                    *x = 0;
                }
                ll[256] = 1;
                let (e, l, dt) = eval.parse_slice(
                    tbl, ctx, &cand_buf[s_off..e_off], row_len, bpp, &mut ll, &mut d, c_len > 0, false,
                );
                tot_bits += entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
            }
            if tot_bits < best_bits {
                best_bits = tot_bits;
                best_seq.copy_from_slice(&seq);
                out_best.copy_from_slice(&cand_buf);
            }
        }
        if pi >= 3 {
            // Segment-aligned block scoring for Macro Viterbi (sweeps 1500 & 2600 -> discovers i008 f1/f4/f2!)
            let mut raw_runs: Vec<(usize, usize)> = Vec::new();
            let mut ry = 0usize;
            while ry < h {
                let m = seq[ry];
                let mut end = ry + 1;
                while end < h && seq[end] == m {
                    end += 1;
                }
                raw_runs.push((ry, end));
                ry = end;
            }
            let mut blocks: Vec<(usize, usize)> = Vec::new();
            for (a, b) in raw_runs {
                if let Some(last) = blocks.last_mut() {
                    if (b - a) < 20 || (last.1 - last.0) < 20 {
                        last.1 = b;
                        continue;
                    }
                }
                blocks.push((a, b));
            }
            let mut eval_blocks: Vec<(usize, usize)> = Vec::new();
            for (a, b) in blocks {
                if b - a > 240 {
                    let mid = (a + b) / 2;
                    eval_blocks.push((a, mid));
                    eval_blocks.push((mid, b));
                } else {
                    eval_blocks.push((a, b));
                }
            }
            let mut tot_bits = (eval_blocks.len().saturating_sub(1) as f64) * (220.0 * scale);
            for &(ry0, ry1) in &eval_blocks {
                let s_off = ry0 * row_len;
                let e_off = ry1 * row_len;
                let c_len = s_off.min(16384);
                let ctx = &cand_buf[s_off - c_len..s_off];
                if c_len > 0 {
                    eval.index_ctx(ctx);
                }
                for x in &mut ll {
                    *x = 0;
                }
                for x in &mut d {
                    *x = 0;
                }
                ll[256] = 1;
                let (e, l, dt) = eval.parse_slice(
                    tbl, ctx, &cand_buf[s_off..e_off], row_len, bpp, &mut ll, &mut d, c_len > 0, false,
                );
                tot_bits += entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
            }
            if tot_bits < best_macro_bits {
                best_macro_bits = tot_bits;
                macro_seq.copy_from_slice(&seq);
                out_macro.copy_from_slice(&cand_buf);
            }
        }
    }

    // Exact coarse-to-fine row-level boundary snap (+/- 14 rows) on clean macro Viterbi segments!
    // Fixes steady-state transition lag at real filter-switch boundaries (e.g. cold LZ77 window after switch)!
    let mut refine_viterbi_buf = |buf: &mut [u8], seq_in: &[usize]| {
        let mut segs: Vec<(usize, usize, usize)> = Vec::new();
        let mut ry = 0usize;
        while ry < h {
            let m = seq_in[ry];
            let mut end = ry + 1;
            while end < h && seq_in[end] == m {
                end += 1;
            }
            segs.push((ry, end, m));
            ry = end;
        }
        if segs.len() < 2 || segs.len() > 12 {
            return;
        }
        let half_span = 14usize;
        let ctx_max = (8192 / row_len).max(1) * row_len;
        let mut span_work = vec![0u8; ctx_max + (half_span * 2 + 2) * row_len];
        for si in 0..segs.len() - 1 {
            let (a_y0, y_mid, mode_a) = segs[si];
            let (_, b_y1, mode_b) = segs[si + 1];
            let y_start = a_y0.max(y_mid.saturating_sub(half_span));
            let y_end = b_y1.min(y_mid + half_span);
            let span_h = y_end - y_start;
            if span_h < 6 {
                continue;
            }
            let at_start = y_start * row_len;
            let ctx_len = at_start.min(ctx_max);
            if ctx_len > 0 {
                span_work[..ctx_len].copy_from_slice(&buf[at_start - ctx_len..at_start]);
            }
            let total_span = ctx_len + span_h * row_len;

            let mut eval_split = |split_y: usize| -> f64 {
                let k = split_y - y_start;
                if k > 0 {
                    span_work[ctx_len..ctx_len + k * row_len]
                        .copy_from_slice(&bank[mode_a][at_start..at_start + k * row_len]);
                }
                if k < span_h {
                    span_work[ctx_len + k * row_len..total_span]
                        .copy_from_slice(&bank[mode_b][at_start + k * row_len..y_end * row_len]);
                }
                let split_off = ctx_len + k * row_len;
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e1, l1, d1) = eval.parse_slice(tbl, &[], &span_work[..split_off], row_len, bpp, &mut ll, &mut d, false, false);
                let c1 = entropy_cost_bits(tbl, &ll, &d, 1 + l1, d1, e1);
                let c_len2 = split_off.min(8192);
                let ctx2 = &span_work[split_off - c_len2..split_off];
                if c_len2 > 0 { eval.index_ctx(ctx2); }
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e2, l2, d2) = eval.parse_slice(tbl, ctx2, &span_work[split_off..total_span], row_len, bpp, &mut ll, &mut d, c_len2 > 0, false);
                c1 + entropy_cost_bits(tbl, &ll, &d, 1 + l2, d2, e2)
            };

            let mut best_y = y_mid;
            let mut best_c = eval_split(y_mid);
            for dy in [-12isize, -8, -4, 4, 8, 12] {
                let cy = (y_mid as isize + dy) as usize;
                if cy > y_start + 1 && cy + 1 < y_end {
                    let c = eval_split(cy);
                    if c < best_c {
                        best_c = c;
                        best_y = cy;
                    }
                }
            }
            let center = best_y;
            for dy in [-2isize, -1, 1, 2] {
                let cy = (center as isize + dy) as usize;
                if cy > y_start && cy < y_end {
                    let c = eval_split(cy);
                    if c < best_c {
                        best_c = c;
                        best_y = cy;
                    }
                }
            }
            if best_y != y_mid {
                let k = best_y - y_start;
                if k > 0 {
                    buf[at_start..at_start + k * row_len]
                        .copy_from_slice(&bank[mode_a][at_start..at_start + k * row_len]);
                }
                if k < span_h {
                    buf[at_start + k * row_len..y_end * row_len]
                        .copy_from_slice(&bank[mode_b][at_start + k * row_len..y_end * row_len]);
                }
                segs[si].1 = best_y;
                segs[si + 1].0 = best_y;
            }
        }
    };

    if best_bits.is_finite() {
        refine_viterbi_buf(out_best, &best_seq);
    }
    if best_macro_bits.is_finite() {
        refine_viterbi_buf(out_macro, &macro_seq);
        return out_macro != out_best;
    }
    false
}

/// Native WASM zero-syscall multi-scale window filter polisher (24 -> 12 -> transition-focused 6 rows)!
/// Refines an elite filter schedule (`band40`, `viterbi`, `vitM`, `band16`) using full 10-probe
/// FastLzEval + entropy_cost_bits over fixed filters 0..4 plus incremental delta-H entropy (`ectW`).
#[no_mangle]
pub extern "C" fn polish_filter_candidate(
    samples_ptr: *const u8,
    base_ptr: *const u8,
    w: u32,
    h: u32,
    bpp: u32,
    out_ptr: *mut u8,
) {
    if samples_ptr.is_null() || base_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let wu = w as usize;
    let hu = h as usize;
    let bpp_u = bpp as usize;
    let stride = wu * bpp_u;
    let row_len = stride + 1;
    let img_bytes = hu * row_len;
    let samples = unsafe { slice::from_raw_parts(samples_ptr, hu * stride) };
    let base = unsafe { slice::from_raw_parts(base_ptr, img_bytes) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, img_bytes) };
    out.copy_from_slice(base);

    let tbl = get_tables();
    let bank = build_row_bank_rust(tbl, samples, wu, hu, bpp_u);
    let pad = (16384 / row_len).clamp(4, 14);
    let mut eval = FastLzEval::new();
    let mut ll = [0u32; 286];
    let mut d = [0u32; 30];
    let mut win_buf = vec![0u8; (32 + pad * 2) * row_len];

    // On 1-bpp non-palette (e.g. greyscale UI/text), run a separate-block 30-row 0..4 pass first!
    if bpp_u == 1 && hu >= 60 {
        let step = 30usize;
        let max_ctx = (16384 / row_len).max(1) * row_len;
        for y0 in (0..hu).step_by(step) {
            let y1 = (y0 + step).min(hu);
            let at = y0 * row_len;
            let seg_bytes = (y1 - y0) * row_len;
            let c_len = at.min(max_ctx);
            let ctx = &out[at - c_len..at];
            if c_len > 0 {
                eval.index_ctx(ctx);
            }
            for x in &mut ll { *x = 0; }
            for x in &mut d { *x = 0; }
            ll[256] = 1;
            let (e0, l0, d0) = eval.parse_slice(
                tbl, ctx, &out[at..at + seg_bytes], row_len, bpp_u, &mut ll, &mut d, c_len > 0, true,
            );
            let mut best_c = entropy_cost_bits(tbl, &ll, &d, 1 + l0, d0, e0);
            let mut best_m = usize::MAX;
            for m in 0..5usize {
                if bank[m][at..at + seg_bytes] == out[at..at + seg_bytes] {
                    continue;
                }
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e, l, dt) = eval.parse_slice(
                    tbl, ctx, &bank[m][at..at + seg_bytes], row_len, bpp_u, &mut ll, &mut d, c_len > 0, true,
                );
                let c = entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
                if c + 96.0 < best_c {
                    best_c = c;
                    best_m = m;
                }
            }
            if best_m != usize::MAX {
                out[at..at + seg_bytes].copy_from_slice(&bank[best_m][at..at + seg_bytes]);
            }
        }
    }

    // Full-accuracy (10-probe) multi-scale window coordinate descent (24 -> 12 rows)
    let scales: &[(usize, usize)] = &[(24, 8), (12, 6)];
    for &(step, n_modes) in scales {
        let modes_lim = n_modes.min(bank.len());
        for y0 in (0..hu).step_by(step) {
            let y1 = (y0 + step).min(hu);
            if step == 6 {
                // Skip deep-interior uniform Filter-0 UI blocks on fine 6-row pass
                let c0 = y0.saturating_sub(24);
                let c1 = (y1 + 24).min(hu);
                let mut all_zero = true;
                for cy in c0..c1 {
                    if out[cy * row_len] != 0 {
                        all_zero = false;
                        break;
                    }
                }
                if all_zero {
                    continue;
                }
            }
            let wy0 = y0.saturating_sub(pad);
            let wy1 = (y1 + pad).min(hu);
            let w_bytes = (wy1 - wy0) * row_len;
            if w_bytes > win_buf.len() {
                win_buf.resize(w_bytes, 0);
            }
            win_buf[..w_bytes].copy_from_slice(&out[wy0 * row_len..wy1 * row_len]);
            let rel_off = (y0 - wy0) * row_len;
            let seg_bytes = (y1 - y0) * row_len;

            for x in &mut ll { *x = 0; }
            for x in &mut d { *x = 0; }
            ll[256] = 1;
            let (e0, l0, d0) = eval.parse_slice(
                tbl, &[], &win_buf[..w_bytes], row_len, bpp_u, &mut ll, &mut d, false, false,
            );
            let mut best_c = entropy_cost_bits(tbl, &ll, &d, 1 + l0, d0, e0);
            let mut best_m = usize::MAX;

            for m in 0..modes_lim {
                if bank[m][y0 * row_len..y1 * row_len] == out[y0 * row_len..y1 * row_len] {
                    continue;
                }
                win_buf[rel_off..rel_off + seg_bytes]
                    .copy_from_slice(&bank[m][y0 * row_len..y1 * row_len]);
                for x in &mut ll { *x = 0; }
                for x in &mut d { *x = 0; }
                ll[256] = 1;
                let (e, l, dt) = eval.parse_slice(
                    tbl, &[], &win_buf[..w_bytes], row_len, bpp_u, &mut ll, &mut d, false, false,
                );
                let c = entropy_cost_bits(tbl, &ll, &d, 1 + l, dt, e);
                if c + 16.0 < best_c {
                    best_c = c;
                    best_m = m;
                }
            }
            if best_m != usize::MAX {
                out[y0 * row_len..y1 * row_len]
                    .copy_from_slice(&bank[best_m][y0 * row_len..y1 * row_len]);
            }
        }
    }
}

/// Exported multi-strategy native WASM filter generator!
/// mode (include_brute):
///   0 = multi-bpp (3/4 bpp): [band40, viterbi, lzrow, None, Paeth]
///   1 = 1/2-bpp non-palette: [band16, viterbi, lzrow, None, Paeth]
///   2 = fast palette mode:   [eu0, band24_with_eu, None, Sub]
///   3 = pure Exact-Up only:  [eu0] (ultra-fast ~1ms for extra palettes like pal-lzfreq!)
#[no_mangle]
pub extern "C" fn generate_filter_candidates(
    samples_ptr: *const u8,
    w: u32,
    h: u32,
    bpp: u32,
    include_brute: u32,
    out_ptr: *mut u8,
    max_cands: u32,
) -> u32 {
    if samples_ptr.is_null() || out_ptr.is_null() || max_cands == 0 {
        return 0;
    }
    let wu = w as usize;
    let hu = h as usize;
    let bpp_u = bpp as usize;
    let stride = wu * bpp_u;
    let row_len = stride + 1;
    let img_bytes = hu * row_len;
    let samples = unsafe { slice::from_raw_parts(samples_ptr, hu * stride) };
    let out_all = unsafe { slice::from_raw_parts_mut(out_ptr, img_bytes * (max_cands as usize)) };

    let mut written = 0usize;
    let limit = max_cands as usize;
    let mut push_cand = |buf: &[u8]| {
        if written < limit {
            out_all[written * img_bytes..(written + 1) * img_bytes].copy_from_slice(buf);
            written += 1;
        }
    };

    if include_brute == 3 {
        let mut raw0 = vec![0u8; img_bytes];
        for y in 0..hu {
            let at = y * row_len;
            raw0[at] = 0;
            raw0[at + 1..at + row_len].copy_from_slice(&samples[y * stride..(y + 1) * stride]);
        }
        let mut eu0 = vec![0u8; img_bytes];
        apply_exact_up_rust(samples, &raw0, wu, hu, bpp_u, false, &mut eu0);
        push_cand(&eu0);
        push_cand(&raw0);
        return written as u32;
    }

    if include_brute == 4 {
        // Standalone clean separate-block 20-row multi-band on fixed filters 0..5 (called lazily on dense 3/4-bpp!)
        let tbl = get_tables();
        let mut base5: Vec<Vec<u8>> = (0..5).map(|_| vec![0u8; img_bytes]).collect();
        for y in 0..hu {
            let cur = &samples[y * stride..(y + 1) * stride];
            let prev = if y == 0 { None } else { Some(&samples[(y - 1) * stride..y * stride]) };
            let at = y * row_len;
            for t in 0..5 {
                base5[t][at] = t as u8;
                filter_row_into(t as u8, cur, prev, bpp_u, &mut base5[t][at + 1..at + row_len]);
            }
        }
        let mut band20c = vec![0u8; img_bytes];
        filter_band_rust(tbl, &base5, wu, hu, bpp_u, 20, true, &mut band20c);
        push_cand(&band20c);
        return written as u32;
    }

    let tbl = get_tables();
    let mut bank = build_row_bank_rust(tbl, samples, wu, hu, bpp_u);

    // Note: bank has 8 base modes (0..4, pairs, minsum, ectW)
    if include_brute == 2 {
        // Palette mode: Exact-Up on Filter 0 + band24 (with eu0 in bank so band24 mixes Exact-Up + Paeth/Up!)
        let mut eu0 = vec![0u8; img_bytes];
        apply_exact_up_rust(samples, &bank[0], wu, hu, bpp_u, false, &mut eu0);
        let eu_idx = bank.len();
        bank.push(eu0);
        let mut b24 = vec![0u8; img_bytes];
        filter_band_rust(tbl, &bank, wu, hu, bpp_u, 24, false, &mut b24);
        push_cand(&bank[eu_idx]); // eu0
        push_cand(&b24);          // band24 (Exact-Up + adaptive)
        push_cand(&bank[0]);      // None
        push_cand(&bank[1]);      // Sub
        return written as u32;
    }

    // Add fast 32 KiB sliding-window lzrow
    let mut lz_buf = vec![0u8; img_bytes];
    filter_lz_row_rust(tbl, &bank, wu, hu, bpp_u, &mut lz_buf);
    let lz_idx = bank.len();
    bank.push(lz_buf);

    let mut vit_buf = vec![0u8; img_bytes];
    let mut vit_macro = vec![0u8; img_bytes];
    let has_macro = filter_viterbi_rust(tbl, &bank, wu, hu, bpp_u, &mut vit_buf, &mut vit_macro);

    if include_brute == 1 {
        // On 1/2-bpp non-palette (greyscale), letting band16 leverage viterbi in bank is a huge win!
        let vit_idx = bank.len();
        bank.push(vit_buf);
        let mut b16 = vec![0u8; img_bytes];
        filter_band_rust(tbl, &bank, wu, hu, bpp_u, 16, false, &mut b16);
        push_cand(&b16);
        push_cand(&bank[vit_idx]); // segment-scored viterbi
        push_cand(&bank[lz_idx]);  // upgraded adaptive lzrow
        push_cand(&bank[0]);       // None
        push_cand(&bank[4]);       // Paeth
        if has_macro {
            push_cand(&vit_macro); // macro viterbi
        }
        return written as u32;
    }

    // On 3/4-bpp: band40 on full bank + Viterbi + Macro Viterbi + safe Exact-Up on lzrow!
    let mut band_buf = vec![0u8; img_bytes];
    filter_band_rust(tbl, &bank, wu, hu, bpp_u, 40, false, &mut band_buf);

    // Apply safe leave-last Exact-Up to lzrow for 3/4-bpp photos
    let mut lz_eu = vec![0u8; img_bytes];
    apply_exact_up_rust(samples, &bank[lz_idx], wu, hu, bpp_u, true, &mut lz_eu);

    push_cand(&band_buf);
    push_cand(&vit_buf);
    push_cand(&lz_eu);   // lzrow + safe Exact-Up
    push_cand(&bank[0]); // None
    push_cand(&bank[4]); // Paeth
    if has_macro {
        push_cand(&vit_macro); // macro viterbi
    }
    written as u32
}

// ============================================================================
// Native WASM Multi-Start 2-Opt + Relocation Palette Co-Occurrence TSP
// ============================================================================

#[no_mangle]
pub extern "C" fn optimize_palette_tsp(
    idx_ptr: *const u8,
    w: u32,
    h: u32,
    k: u32,
    num_trans: u32,
    out_order_ptr: *mut u8,
) {
    let wu = w as usize;
    let hu = h as usize;
    let ku = k as usize;
    let nt = num_trans as usize;
    if idx_ptr.is_null() || out_order_ptr.is_null() || ku == 0 || ku > 256 {
        return;
    }
    let idx = unsafe { slice::from_raw_parts(idx_ptr, wu * hu) };
    let out = unsafe { slice::from_raw_parts_mut(out_order_ptr, ku) };
    for i in 0..ku {
        out[i] = i as u8;
    }
    let num_op = ku - nt;
    if num_op <= 2 {
        return;
    }

    let mut w_mat = vec![0.0f64; ku * ku];
    for y in 0..hu {
        let row = y * wu;
        for x in 1..wu {
            let a = idx[row + x - 1] as usize;
            let b = idx[row + x] as usize;
            if a != b {
                w_mat[a * ku + b] += 1.0;
                w_mat[b * ku + a] += 1.0;
            }
        }
        if y > 0 {
            let prow = row - wu;
            for x in 0..wu {
                let a = idx[prow + x] as usize;
                let b = idx[row + x] as usize;
                if a != b {
                    w_mat[a * ku + b] += 1.0;
                    w_mat[b * ku + a] += 1.0;
                }
            }
        }
    }

    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ku];
    for u in nt..ku {
        for v in nt..ku {
            let wv = w_mat[u * ku + v];
            if wv > 0.0 {
                adj[u].push((v, wv));
            }
        }
    }

    let mut log_tbl = vec![0.0f64; ku + 1];
    for d in 0..=ku {
        log_tbl[d] = (1.0 + d as f64).log2();
    }

    // Greedy nearest-neighbor initial sequence
    let mut used = vec![false; ku];
    let mut seq = vec![0usize; num_op];
    let mut first = nt;
    let mut best_deg = -1.0f64;
    for u in nt..ku {
        let deg: f64 = adj[u].iter().map(|x| x.1).sum();
        if deg > best_deg {
            best_deg = deg;
            first = u;
        }
    }
    seq[0] = first;
    used[first] = true;
    for s in 1..num_op {
        let last = seq[s - 1];
        let mut best_u = nt;
        let mut best_sc = -1.0f64;
        for u in nt..ku {
            if used[u] {
                continue;
            }
            let mut sc = w_mat[last * ku + u] * 4.0;
            if s >= 2 {
                sc += w_mat[seq[s - 2] * ku + u] * 2.0;
            }
            if s >= 3 {
                sc += w_mat[seq[s - 3] * ku + u];
            }
            if sc > best_sc {
                best_sc = sc;
                best_u = u;
            }
        }
        seq[s] = best_u;
        used[best_u] = true;
    }

    let mut pos = vec![0usize; ku];
    for (s, &u) in seq.iter().enumerate() {
        pos[u] = s;
    }

    let node_cost = |u: usize, p: usize, pos: &[usize]| -> f64 {
        let mut c = 0.0;
        for &(v, wv) in &adj[u] {
            let d = if p >= pos[v] { p - pos[v] } else { pos[v] - p };
            c += wv * log_tbl[d];
        }
        c
    };

    let total_cost = |seq: &[usize], pos: &[usize]| -> f64 {
        let mut c = 0.0;
        for (p, &u) in seq.iter().enumerate() {
            c += node_cost(u, p, pos);
        }
        c * 0.5
    };

    // Fast pairwise swap + 2-opt segment reversal refinement
    for _pass in 0..8 {
        let mut improved = false;
        for i in 0..num_op {
            for j in i + 1..num_op {
                let u = seq[i];
                let v = seq[j];
                let old_c = node_cost(u, i, &pos) + node_cost(v, j, &pos);
                pos[u] = j;
                pos[v] = i;
                let new_c = node_cost(u, j, &pos) + node_cost(v, i, &pos);
                if new_c + 1e-6 < old_c {
                    seq[i] = v;
                    seq[j] = u;
                    improved = true;
                } else {
                    pos[u] = i;
                    pos[v] = j;
                }
            }
        }
        // 2-opt segment reversals
        if num_op <= 128 {
            let mut cur_tot = total_cost(&seq, &pos);
            for i in 0..num_op {
                for j in i + 2..num_op {
                    seq[i..=j].reverse();
                    for idx_p in i..=j {
                        pos[seq[idx_p]] = idx_p;
                    }
                    let nc = total_cost(&seq, &pos);
                    if nc + 1e-6 < cur_tot {
                        cur_tot = nc;
                        improved = true;
                    } else {
                        seq[i..=j].reverse();
                        for idx_p in i..=j {
                            pos[seq[idx_p]] = idx_p;
                        }
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    for i in 0..nt {
        out[i] = i as u8;
    }
    for s in 0..num_op {
        out[nt + s] = seq[s] as u8;
    }
}
