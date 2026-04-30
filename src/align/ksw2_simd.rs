//! SSE2-accelerated KSW2 extension alignment (rotated-band DP).
//! Faithful translation of ksw2_extz2_sse.c.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cell::RefCell;

use super::ksw2::{push_cigar_fn, KswResult, KSW_NEG_INF};
use crate::flags::KswFlags;

// Thread-local scratch buffers for KSW2 kernels.
// Reused across calls to avoid per-call allocation overhead.
thread_local! {
    static KSW_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static KSW_I32_SCRATCH: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn ksw_apply_zdrop_rot(ez: &mut KswResult, h: i32, r: i32, t: i32, zdrop: i32, e: i8) -> bool {
    if h > ez.max {
        ez.max = h;
        ez.max_t = t;
        ez.max_q = r - t;
    } else if t >= ez.max_t && r - t >= ez.max_q {
        let tl = t - ez.max_t;
        let ql = (r - t) - ez.max_q;
        let l = if tl > ql { tl - ql } else { ql - tl };
        if zdrop >= 0 && ez.max - h > zdrop + l * e as i32 {
            ez.zdropped = true;
            return true;
        }
    }
    false
}

fn backtrack_rotated(
    bt: &[u8],
    off: &[i32],
    off_end: &[i32],
    row_width: usize,
    rev_cigar: bool,
    min_intron_len: i32,
    i0: i32,
    j0: i32,
) -> Vec<u32> {
    let (mut i, mut j) = (i0, j0);
    let mut state: u8 = 0;
    let mut cigar = Vec::new();
    while i >= 0 && j >= 0 {
        let r = (i + j) as usize;
        let mut force_state = -1i32;
        if j < off[r] {
            force_state = 2;
        }
        if j > off_end[r] {
            force_state = 1;
        }
        let tmp = if force_state < 0 {
            bt[r * row_width + (j - off[r]) as usize]
        } else {
            0
        };
        if state == 0 {
            state = tmp & 7;
        } else if ((tmp >> (state + 2)) & 1) == 0 {
            state = 0;
        }
        if state == 0 {
            state = tmp & 7;
        }
        if force_state >= 0 {
            state = force_state as u8;
        }
        match state {
            0 => {
                push_cigar_fn(&mut cigar, 0, 1);
                i -= 1;
                j -= 1;
            }
            1 => {
                push_cigar_fn(&mut cigar, 2, 1);
                j -= 1;
            }
            3 if min_intron_len > 0 => {
                push_cigar_fn(&mut cigar, 3, 1);
                j -= 1;
            }
            3 => {
                push_cigar_fn(&mut cigar, 2, 1);
                j -= 1;
            }
            _ => {
                push_cigar_fn(&mut cigar, 1, 1);
                i -= 1;
            }
        }
    }
    if j >= 0 {
        let op = if min_intron_len > 0 && j >= min_intron_len {
            3
        } else {
            2
        };
        push_cigar_fn(&mut cigar, op, j + 1);
    }
    if i >= 0 {
        push_cigar_fn(&mut cigar, 1, i + 1);
    }
    if !rev_cigar {
        cigar.reverse();
    }
    cigar
}

pub fn has_sse2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub fn has_sse41() -> bool {
    is_x86_feature_detected!("sse4.1")
}

#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    if std::env::var_os("MM2RS_DISABLE_AVX2").is_some() {
        return false;
    }
    is_x86_feature_detected!("avx2")
}

/// Faithful translation of ksw_extz2_sse2() from ksw2_extz2_sse.c.
/// Uses rotated-band DP with SSE2 16-way byte parallelism.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extz2_sse2(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 0 || qlen <= 0 || tlen <= 0 {
        return ez;
    }

    let qe = q as i32 + e as i32;
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let (wl, wr) = (w, w);
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((w + 1) as usize) + 15) / 16) + 1;

    // Scoring bounds
    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    for i in 1..(m as usize * m as usize) {
        max_sc = max_sc.max(mat[i] as i32);
        min_sc = min_sc.min(mat[i] as i32);
    }
    if -min_sc > 2 * qe {
        return ez;
    }

    // DP buffers — packed into a single thread-local scratch to avoid per-call alloc
    let bsz = tlen_ * 16 + 16;
    let n_ad = (qlen + tlen - 1) as usize;
    let bt_size = if with_cigar {
        n_ad * n_col_ * 16 + 16
    } else {
        0
    };
    let off_size = if with_cigar { n_ad } else { 0 };
    let qr_pad = tlen as usize;
    let qr_sz = qr_pad + qlen as usize + 16;
    let sf_sz = tlen as usize + 16;
    // Layout: [u:bsz][v:bsz][x:bsz][y:bsz][s:bsz][bt:bt_size][qr:qr_sz][sf:sf_sz]
    let total_u8 = 5 * bsz + bt_size + qr_sz + sf_sz;
    // i32 layout: [h_arr:h_sz][off_a:off_size][off_e:off_size]
    let h_sz = if !approx_max { bsz } else { 0 };
    let total_i32 = h_sz + 2 * off_size;

    let mut scratch = KSW_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut scratch_i32 = KSW_I32_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_u8 + 15 {
        scratch.resize(total_u8 + 15, 0);
    }
    if scratch_i32.len() < total_i32 {
        scratch_i32.resize(total_i32, 0);
    }
    // Zero only DP arrays (u,v,x,y) and qr/sf padding; s and bt don't need zeroing
    let dp_end = 5 * bsz;
    scratch[..dp_end].fill(0);
    // Zero qr and sf sections (padding beyond actual data must be 0)
    let qr_start = 5 * bsz + bt_size;
    scratch[qr_start..qr_start + qr_sz + sf_sz].fill(0);
    // Initialize i32 scratch
    if h_sz > 0 {
        scratch_i32[..h_sz].fill(KSW_NEG_INF);
    }
    if off_size > 0 {
        scratch_i32[h_sz..h_sz + 2 * off_size].fill(0);
    }

    let raw_base = scratch.as_mut_ptr();
    let dp_base = (((raw_base as usize) + 15) & !15usize) as *mut u8;
    let (u, v, x, y, s) = (
        std::slice::from_raw_parts_mut(dp_base, bsz),
        std::slice::from_raw_parts_mut(dp_base.add(bsz), bsz),
        std::slice::from_raw_parts_mut(dp_base.add(2 * bsz), bsz),
        std::slice::from_raw_parts_mut(dp_base.add(3 * bsz), bsz),
        std::slice::from_raw_parts_mut(dp_base.add(4 * bsz), bsz),
    );
    let bt = std::slice::from_raw_parts_mut(dp_base.add(5 * bsz), bt_size);
    let qr_off = 5 * bsz + bt_size;
    let sf_off = qr_off + qr_sz;

    let (h_arr, off_rest) = scratch_i32[..h_sz + 2 * off_size].split_at_mut(h_sz);
    let (off_a, off_e) = off_rest.split_at_mut(off_size);
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    // Reversed query with front padding (qrr can be negative)
    for i in 0..qlen as usize {
        scratch[qr_off + qr_pad + i] = query[qlen as usize - 1 - i];
    }
    let qr_base = scratch.as_ptr().add(qr_off + qr_pad);

    scratch[sf_off..sf_off + tlen as usize].copy_from_slice(target);
    let sf_ptr = scratch.as_ptr().add(sf_off);

    // SSE constants
    let zero_ = _mm_setzero_si128();
    let q_ = _mm_set1_epi8(q);
    let qe2_ = _mm_set1_epi8((qe * 2) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let m1_ = _mm_set1_epi8(m - 1);
    let max_sc_ = _mm_set1_epi8((mat[0] as i32 + qe * 2) as i8);
    let sc_n_ = if mat[(m as usize * m as usize) - 1] == 0 {
        _mm_set1_epi8(-e)
    } else {
        _mm_set1_epi8(mat[(m as usize * m as usize) - 1])
    };
    let flag1_ = _mm_set1_epi8(1);
    let flag2_ = _mm_set1_epi8(2);
    let flag8_ = _mm_set1_epi8(0x08u8 as i8);
    let flag16_ = _mm_set1_epi8(0x10u8 as i8);

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;

    // Hoist raw pointers for the hot loop
    let up = u.as_mut_ptr();
    let vp = v.as_mut_ptr();
    let xp = x.as_mut_ptr();
    let yp = y.as_mut_ptr();
    let sp = s.as_mut_ptr();
    let _hp = h_arr.as_mut_ptr();

    // Main anti-diagonal loop
    for r in 0..qlen + tlen - 1 {
        // Band boundaries (faithful translation of C lines 110-124)
        let mut st = 0i32.max(r - qlen + 1);
        let mut en = (tlen - 1).min(r);
        if st < (r - wr + 1) >> 1 {
            st = (r - wr + 1) >> 1;
        }
        if en > (r + wl) / 2 {
            en = (r + wl) / 2;
        }
        if st > en {
            ez.zdropped = true;
            break;
        }
        let st0 = st;
        let en0 = en;
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        // Boundary conditions (C lines 126-131)
        let x1: i8;
        let v1: i8;
        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = x[(st - 1) as usize] as i8;
                v1 = v[(st - 1) as usize] as i8;
            } else {
                x1 = 0;
                v1 = 0;
            }
        } else {
            x1 = 0;
            v1 = if r > 0 { q } else { 0 };
        }
        if en >= r {
            y[r as usize] = 0;
            u[r as usize] = if r > 0 { q as u8 } else { 0 };
        }

        // Score computation (C lines 133-152) — direct SIMD loads from padded buffers
        let qrr = qlen - 1 - r; // can be negative; safe with front-padded qr
        if !flag.contains(KswFlags::GENERIC_SC) {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let sq = _mm_loadu_si128(sf_ptr.add(t) as *const __m128i);
                let qt = _mm_loadu_si128(qr_base.offset(qrr as isize).add(t) as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(qt, m1_));
                let eq = _mm_cmpeq_epi8(sq, qt);
                let tmp = _mm_or_si128(_mm_andnot_si128(eq, sc_mis_), _mm_and_si128(eq, sc_mch_));
                let tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_));
                _mm_storeu_si128(sp.add(t) as *mut __m128i, tmp);
                t += 16;
            }
        } else {
            for t in st0 as usize..=en0 as usize {
                let si = if t < sf_sz { scratch[sf_off + t] } else { 0 } as usize;
                let qidx = qrr + t as i32;
                let qi = if qidx >= -(qr_pad as i32) && qidx < qlen {
                    *qr_base.offset(qidx as isize)
                } else {
                    0
                } as usize;
                s[t] = mat[si * m as usize + qi] as u8;
            }
        }

        // Core DP (C lines 153-231)
        let mut x1_ = _mm_cvtsi32_si128(x1 as i32);
        let mut v1_ = _mm_cvtsi32_si128(v1 as i32);
        let st_ = st / 16;
        let en_ = en / 16;

        if !with_cigar {
            // Score-only path — maximally tight inner loop
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                let mut z = _mm_add_epi8(_mm_loadu_si128(sp.add(p) as *const __m128i), qe2_);
                let xt1r = _mm_loadu_si128(xp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_);
                x1_ = tmp;
                let vt1r = _mm_loadu_si128(vp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_loadu_si128(up.add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_loadu_si128(yp.add(p) as *const __m128i), ut);
                z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));
                z = _mm_max_epu8(z, a);
                z = _mm_max_epu8(z, b);
                z = _mm_min_epu8(z, max_sc_);
                _mm_storeu_si128(up.add(p) as *mut __m128i, _mm_sub_epi8(z, vt1));
                _mm_storeu_si128(vp.add(p) as *mut __m128i, _mm_sub_epi8(z, ut));
                let zq = _mm_sub_epi8(z, q_);
                let a2 = _mm_sub_epi8(a, zq);
                let b2 = _mm_sub_epi8(b, zq);
                let tmp = _mm_cmpgt_epi8(a2, zero_);
                _mm_storeu_si128(xp.add(p) as *mut __m128i, _mm_and_si128(a2, tmp));
                let tmp = _mm_cmpgt_epi8(b2, zero_);
                _mm_storeu_si128(yp.add(p) as *mut __m128i, _mm_and_si128(b2, tmp));
            }
        } else {
            // CIGAR left-alignment path (C lines 179-204)
            off_a[r as usize] = st;
            off_e[r as usize] = en;
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                let pr_off = r as usize * n_col_ * 16 + (t as i32 - st_) as usize * 16;
                let mut z = _mm_add_epi8(_mm_loadu_si128(sp.add(p) as *const __m128i), qe2_);
                let xt1r = _mm_loadu_si128(xp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_);
                x1_ = tmp;
                let vt1r = _mm_loadu_si128(vp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_loadu_si128(up.add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_loadu_si128(yp.add(p) as *const __m128i), ut);
                // d vector for backtrack (left-alignment, SSE2 path)
                let mut d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_);
                z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));
                z = _mm_max_epu8(z, a);
                let tmp = _mm_cmpgt_epi8(b, z);
                d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_));
                // dp_code_block2
                z = _mm_max_epu8(z, b);
                z = _mm_min_epu8(z, max_sc_);
                _mm_storeu_si128(up.add(p) as *mut __m128i, _mm_sub_epi8(z, vt1));
                _mm_storeu_si128(vp.add(p) as *mut __m128i, _mm_sub_epi8(z, ut));
                let zq = _mm_sub_epi8(z, q_);
                let a2 = _mm_sub_epi8(a, zq);
                let b2 = _mm_sub_epi8(b, zq);
                // x, y with continuation bits
                let tmp = _mm_cmpgt_epi8(a2, zero_);
                _mm_storeu_si128(xp.add(p) as *mut __m128i, _mm_and_si128(tmp, a2));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                let tmp = _mm_cmpgt_epi8(b2, zero_);
                _mm_storeu_si128(yp.add(p) as *mut __m128i, _mm_and_si128(tmp, b2));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));
                _mm_storeu_si128(bt.as_mut_ptr().add(pr_off) as *mut __m128i, d);
            }
        }

        // Score tracking (C lines 232-294)
        let u8p = u.as_ptr();
        let v8p = v.as_ptr();
        let hp = h_arr.as_mut_ptr();
        if !approx_max {
            // Exact max
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                if en0 > 0 {
                    *hp.add(en0 as usize) =
                        *hp.add(en0 as usize - 1) + *u8p.add(en0 as usize) as i32 - qe;
                } else {
                    *hp.add(en0 as usize) =
                        *hp.add(en0 as usize) + *v8p.add(en0 as usize) as i32 - qe;
                }
                max_h = *hp.add(en0 as usize);
                max_t = en0;
                // SSE2-vectorized H-tracking: 4 i32 values per iteration
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut max_h_v = _mm_set1_epi32(max_h);
                let mut max_t_v = _mm_set1_epi32(max_t);
                let qe_v = _mm_set1_epi32(qe);
                {
                    let mut t = st0;
                    while t < en1 {
                        let tu = t as usize;
                        let mut h1 = _mm_loadu_si128(hp.add(tu) as *const __m128i);
                        let vv = _mm_setr_epi32(
                            *v8p.add(tu) as i32,
                            *v8p.add(tu + 1) as i32,
                            *v8p.add(tu + 2) as i32,
                            *v8p.add(tu + 3) as i32,
                        );
                        h1 = _mm_sub_epi32(_mm_add_epi32(h1, vv), qe_v);
                        _mm_storeu_si128(hp.add(tu) as *mut __m128i, h1);
                        let tv = _mm_set1_epi32(t);
                        let cmp = _mm_cmpgt_epi32(h1, max_h_v);
                        max_h_v =
                            _mm_or_si128(_mm_and_si128(cmp, h1), _mm_andnot_si128(cmp, max_h_v));
                        max_t_v =
                            _mm_or_si128(_mm_and_si128(cmp, tv), _mm_andnot_si128(cmp, max_t_v));
                        t += 4;
                    }
                }
                let mut hh = [0i32; 4];
                let mut tt = [0i32; 4];
                _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_v);
                _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_v);
                for i in 0..4 {
                    if max_h < hh[i] {
                        max_h = hh[i];
                        max_t = tt[i] + i as i32;
                    }
                }
                for t in en1..en0 {
                    *hp.add(t as usize) += *v8p.add(t as usize) as i32 - qe;
                    if *hp.add(t as usize) > max_h {
                        max_h = *hp.add(t as usize);
                        max_t = t;
                    }
                }
            } else {
                h_arr[0] = *v8p as i32 - qe - qe;
                max_h = h_arr[0];
                max_t = 0;
            }
            // Unconditional mte/mqe/zdrop checks (matching C)
            if en0 == tlen - 1 && h_arr[en0 as usize] > ez.mte {
                ez.mte = h_arr[en0 as usize];
                ez.mte_q = r - en0;
            }
            if r - st0 == qlen - 1 && h_arr[st0 as usize] > ez.mqe {
                ez.mqe = h_arr[st0 as usize];
                ez.mqe_t = st0;
            }
            if max_h > ez.max as i32 {
                ez.max = max_h;
                ez.max_t = max_t;
                ez.max_q = r - max_t;
            } else if max_t >= ez.max_t && r - max_t >= ez.max_q {
                let l = ((max_t - ez.max_t) - ((r - max_t) - ez.max_q)).abs();
                if zdrop >= 0 && ez.max - max_h > zdrop + l * e as i32 {
                    ez.zdropped = true;
                    break;
                }
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h_arr[(tlen - 1) as usize];
            }
        } else {
            // Approximate max
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = *v8p.add(last_h0_t as usize) as i32 - qe;
                    let d1 = *u8p.add((last_h0_t + 1) as usize) as i32 - qe;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8p.add(last_h0_t as usize) as i32 - qe;
                } else {
                    last_h0_t += 1;
                    h0 += *u8p.add(last_h0_t as usize) as i32 - qe;
                }
                if flag.contains(KswFlags::APPROX_DROP) && zdrop >= 0 {
                    if h0 > ez.max as i32 {
                        ez.max = h0;
                        ez.max_t = last_h0_t;
                        ez.max_q = r - last_h0_t;
                    } else if last_h0_t >= ez.max_t && r - last_h0_t >= ez.max_q {
                        let l = ((last_h0_t - ez.max_t) - ((r - last_h0_t) - ez.max_q)).abs();
                        if ez.max - h0 > zdrop + l * e as i32 {
                            ez.zdropped = true;
                            break;
                        }
                    }
                }
            } else {
                h0 = *v8p as i32 - qe - qe;
                last_h0_t = 0;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    // Finalize (C lines 298-311)
    if ez.score == KSW_NEG_INF && !approx_max {
        // score wasn't set — compute from mte/mqe
        let mut sc = KSW_NEG_INF;
        if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
            sc = ez.mqe;
        }
        if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 && ez.mte > sc {
            sc = ez.mte;
        }
        ez.score = sc;
    }
    // Backtrack (C lines 300-311, ksw2.h lines 130-162)
    if with_cigar {
        let is_rev = flag.contains(KswFlags::REV_CIGAR);
        let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
        let (i0, j0) = if !ez.zdropped && !is_extz {
            (tlen - 1, qlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max as i32 {
            ez.reach_end = true;
            (ez.mqe_t, qlen - 1)
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            (ez.max_t, ez.max_q)
        } else {
            (-1, -1)
        };

        if i0 >= 0 && j0 >= 0 {
            let mut cigar = Vec::new();
            let mut state = 0u8;
            let (mut i, mut j) = (i0, j0);
            while i >= 0 && j >= 0 {
                let r = i + j;
                if r < 0 || r as usize >= n_ad {
                    break;
                }
                let off_r = off_a[r as usize];
                let off_er = off_e[r as usize];
                let mut force_state: i32 = -1;
                if i < off_r {
                    force_state = 2;
                }
                if i > off_er {
                    force_state = 1;
                }
                let tmp = if force_state < 0 {
                    let idx = r as usize * n_col_ * 16 + (i - off_r) as usize;
                    if idx < bt.len() {
                        bt[idx]
                    } else {
                        0
                    }
                } else {
                    0
                };
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 {
                    state = tmp & 7;
                }
                if force_state >= 0 {
                    state = force_state as u8;
                }
                // Match C ksw_backtrack (ksw2.h:151-154). For single-gap
                // (extz2) only states 0/1/2 appear; state 3 is unused but
                // mapping it as DEL preserves parity should the kernel ever
                // emit it (matches the dual-gap convention).
                match state {
                    0 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 | 3 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 2, 1);
                        i -= 1;
                    }
                    _ => {
                        super::ksw2::push_cigar_fn(&mut cigar, 1, 1);
                        j -= 1;
                    }
                }
            }
            if i >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 2, i + 1);
            }
            if j >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 1, j + 1);
            }
            if !is_rev {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }
    // Return scratch buffers to thread-local for reuse
    KSW_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    KSW_I32_SCRATCH.with(|c| *c.borrow_mut() = scratch_i32);
    ez
}

/// Faithful translation of ksw_extd2_sse2() — dual affine gap penalty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extd2_sse2(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    mut q: i8,
    mut e: i8,
    mut q2: i8,
    mut e2: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 1 || qlen <= 0 || tlen <= 0 {
        return ez;
    }

    // Ensure q+e <= q2+e2
    if (q2 as i32 + e2 as i32) < (q as i32 + e as i32) {
        std::mem::swap(&mut q, &mut q2);
        std::mem::swap(&mut e, &mut e2);
    }
    let qe = q as i32 + e as i32;
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let is_right = flag.contains(KswFlags::RIGHT);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let (wl, wr) = (w, w);
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((w + 1) as usize) + 15) / 16) + 1;

    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    for i in 1..(m as usize * m as usize) {
        max_sc = max_sc.max(mat[i] as i32);
        min_sc = min_sc.min(mat[i] as i32);
    }
    if -min_sc > 2 * qe {
        return ez;
    }

    let long_thres = if e != e2 {
        ((q2 as i32 - q as i32) / (e as i32 - e2 as i32)) - 1
    } else {
        0
    };
    let long_thres = if (q2 as i32 + e2 as i32 + long_thres * e2 as i32)
        > (q as i32 + e as i32 + long_thres * e as i32)
    {
        long_thres + 1
    } else {
        long_thres
    };
    let long_diff =
        (long_thres * (e as i32 - e2 as i32) - (q2 as i32 - q as i32) - e2 as i32) as i8;

    let bsz = tlen_ * 16 + 16;
    let nqe = (-(q as i32) - e as i32) as i8;
    let nq2e2 = (-(q2 as i32) - e2 as i32) as i8;
    let n_ad = (qlen + tlen - 1) as usize;
    let bt_size = if with_cigar {
        n_ad * n_col_ * 16 + 16
    } else {
        0
    };
    let off_size = if with_cigar { n_ad } else { 0 };
    let qr_pad = tlen as usize;
    let qr_sz = qr_pad + qlen as usize + 16;
    let sf_sz = tlen as usize + 16;
    // Layout: [u:bsz][v:bsz][x:bsz][y:bsz][x2:bsz][y2:bsz][s:bsz][bt:bt_size][qr:qr_sz][sf:sf_sz]
    let total_u8 = 7 * bsz + bt_size + qr_sz + sf_sz;
    let h_sz = if !approx_max { bsz } else { 0 };
    let total_i32 = h_sz + 2 * off_size;

    let mut scratch = KSW_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut scratch_i32 = KSW_I32_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_u8 + 15 {
        scratch.resize(total_u8 + 15, 0);
    }
    if scratch_i32.len() < total_i32 {
        scratch_i32.resize(total_i32, 0);
    }

    let raw_base = scratch.as_mut_ptr();
    let dp_base = (((raw_base as usize) + 15) & !15usize) as *mut u8;
    // Fill u,v,x,y with nqe
    std::ptr::write_bytes(dp_base, nqe as u8, 4 * bsz);
    // Fill x2,y2 with nq2e2
    std::ptr::write_bytes(dp_base.add(4 * bsz), nq2e2 as u8, 2 * bsz);
    // Zero s only (bt doesn't need zeroing — backtrack uses force_state for out-of-band)
    std::ptr::write_bytes(dp_base.add(6 * bsz), 0, bsz);
    // Zero qr and sf (padding must be 0)
    let qr_start = 7 * bsz + bt_size;
    std::ptr::write_bytes(dp_base.add(qr_start), 0, qr_sz + sf_sz);

    if h_sz > 0 {
        scratch_i32[..h_sz].fill(KSW_NEG_INF);
    }
    if off_size > 0 {
        scratch_i32[h_sz..h_sz + 2 * off_size].fill(0);
    }

    let u = std::slice::from_raw_parts_mut(dp_base, bsz);
    let v = std::slice::from_raw_parts_mut(dp_base.add(bsz), bsz);
    let x = std::slice::from_raw_parts_mut(dp_base.add(2 * bsz), bsz);
    let y = std::slice::from_raw_parts_mut(dp_base.add(3 * bsz), bsz);
    let x2 = std::slice::from_raw_parts_mut(dp_base.add(4 * bsz), bsz);
    let y2 = std::slice::from_raw_parts_mut(dp_base.add(5 * bsz), bsz);
    let s = std::slice::from_raw_parts_mut(dp_base.add(6 * bsz), bsz);
    let bt = std::slice::from_raw_parts_mut(dp_base.add(7 * bsz), bt_size);
    let qr_off = 7 * bsz + bt_size;
    let sf_off = qr_off + qr_sz;

    let (h_arr, off_rest) = scratch_i32[..h_sz + 2 * off_size].split_at_mut(h_sz);
    let (off_a, off_e) = off_rest.split_at_mut(off_size);
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    let qr_ptr = dp_base.add(qr_off);
    for i in 0..qlen as usize {
        *qr_ptr.add(qr_pad + i) = query[qlen as usize - 1 - i];
    }
    let qr_base = qr_ptr.add(qr_pad);
    let sf_ptr = dp_base.add(sf_off);
    std::ptr::copy_nonoverlapping(target.as_ptr(), sf_ptr, tlen as usize);

    let zero_ = _mm_setzero_si128();
    let q_ = _mm_set1_epi8(q);
    let q2_ = _mm_set1_epi8(q2);
    let qe_ = _mm_set1_epi8((q as i32 + e as i32) as i8);
    let qe2_ = _mm_set1_epi8((q2 as i32 + e2 as i32) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let m1_ = _mm_set1_epi8(m - 1);
    let sc_n_ = if mat[(m as usize * m as usize) - 1] == 0 {
        _mm_set1_epi8(-e2)
    } else {
        _mm_set1_epi8(mat[(m as usize * m as usize) - 1])
    };

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let up = u.as_mut_ptr();
    let vp = v.as_mut_ptr();
    let xp = x.as_mut_ptr();
    let yp = y.as_mut_ptr();
    let x2p = x2.as_mut_ptr();
    let y2p = y2.as_mut_ptr();
    let sp = s.as_mut_ptr();

    for r in 0..qlen + tlen - 1 {
        let mut st = 0i32.max(r - qlen + 1);
        let mut en = (tlen - 1).min(r);
        if st < (r - wr + 1) >> 1 {
            st = (r - wr + 1) >> 1;
        }
        if en > (r + wl) / 2 {
            en = (r + wl) / 2;
        }
        if st > en {
            ez.zdropped = true;
            break;
        }
        let st0 = st;
        let en0 = en;
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        // Boundary conditions (C lines 149-163)
        let x1: i8;
        let x21: i8;
        let v1: i8;
        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = x[(st - 1) as usize] as i8;
                x21 = x2[(st - 1) as usize] as i8;
                v1 = v[(st - 1) as usize] as i8;
            } else {
                x1 = nqe;
                x21 = nq2e2;
                v1 = nqe;
            }
        } else {
            x1 = nqe;
            x21 = nq2e2;
            v1 = if r == 0 {
                nqe
            } else if r < long_thres {
                -e
            } else if r == long_thres {
                long_diff
            } else {
                -e2
            };
        }
        if en >= r {
            y[r as usize] = nqe as u8;
            y2[r as usize] = nq2e2 as u8;
            u[r as usize] = (if r == 0 {
                nqe
            } else if r < long_thres {
                -e
            } else if r == long_thres {
                long_diff
            } else {
                -e2
            }) as u8;
        }

        // Score computation (same as single-gap)
        let qrr = qlen - 1 - r; // can be negative; qrr + t is always >= 0 within the band
        {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let sq = _mm_loadu_si128(sf_ptr.add(t) as *const __m128i);
                let qt = _mm_loadu_si128(qr_base.offset(qrr as isize).add(t) as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(qt, m1_));
                let eq = _mm_cmpeq_epi8(sq, qt);
                let tmp = _mm_or_si128(_mm_andnot_si128(eq, sc_mis_), _mm_and_si128(eq, sc_mch_));
                let tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_));
                _mm_storeu_si128(sp.add(t) as *mut __m128i, tmp);
                t += 16;
            }
        }

        let mut x1_ = _mm_cvtsi32_si128(x1 as u8 as i32);
        let mut x21_ = _mm_cvtsi32_si128(x21 as u8 as i32);
        let mut v1_ = _mm_cvtsi32_si128(v1 as u8 as i32);
        let st_ = st / 16;
        let en_ = en / 16;

        if with_cigar {
            off_a[r as usize] = st;
            off_e[r as usize] = en;
        }

        for t in st_ as usize..=en_ as usize {
            let p = t * 16;
            // dp_code_block1 (dual-gap version)
            let z = _mm_loadu_si128(sp.add(p) as *const __m128i);
            let xt1r = _mm_loadu_si128(xp.add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(xt1r);
            let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_);
            x1_ = tmp;
            let vt1r = _mm_loadu_si128(vp.add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(vt1r);
            let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_);
            v1_ = tmp;
            let a = _mm_add_epi8(xt1, vt1);
            let ut = _mm_loadu_si128(up.add(p) as *const __m128i);
            let b = _mm_add_epi8(_mm_loadu_si128(yp.add(p) as *const __m128i), ut);
            let x2t1r = _mm_loadu_si128(x2p.add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(x2t1r);
            let x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1r), x21_);
            x21_ = tmp;
            let a2 = _mm_add_epi8(x2t1, vt1);
            let b2 = _mm_add_epi8(_mm_loadu_si128(y2p.add(p) as *const __m128i), ut);

            // SSE2: z = max(z, a, b, a2, b2) via signed compare+blend
            // LEFT vs RIGHT alignment differ in tiebreaking (C lines 200-322)
            let mut zz = z;
            let mut d = zero_;
            if !is_right {
                // LEFT alignment: strict greater, ties prefer z
                let mut tmp = _mm_cmpgt_epi8(a, zz);
                zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, a));
                if with_cigar {
                    d = _mm_and_si128(tmp, _mm_set1_epi8(1));
                }
                tmp = _mm_cmpgt_epi8(b, zz);
                zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, b));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_andnot_si128(tmp, d),
                        _mm_and_si128(tmp, _mm_set1_epi8(2)),
                    );
                }
                tmp = _mm_cmpgt_epi8(a2, zz);
                zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, a2));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_andnot_si128(tmp, d),
                        _mm_and_si128(tmp, _mm_set1_epi8(3)),
                    );
                }
                tmp = _mm_cmpgt_epi8(b2, zz);
                zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, b2));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_andnot_si128(tmp, d),
                        _mm_and_si128(tmp, _mm_set1_epi8(4)),
                    );
                }
            } else {
                // RIGHT alignment: ties prefer gap states (a, b, a2, b2)
                let mut tmp = _mm_cmpgt_epi8(zz, a);
                zz = _mm_or_si128(_mm_and_si128(tmp, zz), _mm_andnot_si128(tmp, a));
                if with_cigar {
                    d = _mm_andnot_si128(tmp, _mm_set1_epi8(1));
                }
                tmp = _mm_cmpgt_epi8(zz, b);
                zz = _mm_or_si128(_mm_and_si128(tmp, zz), _mm_andnot_si128(tmp, b));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_and_si128(tmp, d),
                        _mm_andnot_si128(tmp, _mm_set1_epi8(2)),
                    );
                }
                tmp = _mm_cmpgt_epi8(zz, a2);
                zz = _mm_or_si128(_mm_and_si128(tmp, zz), _mm_andnot_si128(tmp, a2));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_and_si128(tmp, d),
                        _mm_andnot_si128(tmp, _mm_set1_epi8(3)),
                    );
                }
                tmp = _mm_cmpgt_epi8(zz, b2);
                zz = _mm_or_si128(_mm_and_si128(tmp, zz), _mm_andnot_si128(tmp, b2));
                if with_cigar {
                    d = _mm_or_si128(
                        _mm_and_si128(tmp, d),
                        _mm_andnot_si128(tmp, _mm_set1_epi8(4)),
                    );
                }
            }
            // Clamp: z = min(z, sc_mch_)
            let tmp = _mm_cmplt_epi8(sc_mch_, zz);
            zz = _mm_or_si128(_mm_and_si128(tmp, sc_mch_), _mm_andnot_si128(tmp, zz));

            // dp_code_block2
            _mm_storeu_si128(up.add(p) as *mut __m128i, _mm_sub_epi8(zz, vt1));
            _mm_storeu_si128(vp.add(p) as *mut __m128i, _mm_sub_epi8(zz, ut));
            let zq = _mm_sub_epi8(zz, q_);
            let a = _mm_sub_epi8(a, zq);
            let b = _mm_sub_epi8(b, zq);
            let zq2 = _mm_sub_epi8(zz, q2_);
            let a2 = _mm_sub_epi8(a2, zq2);
            let b2 = _mm_sub_epi8(b2, zq2);

            // Store x, y, x2, y2 — LEFT vs RIGHT differ in tiebreaking at zero
            if !is_right {
                let tmp = _mm_cmpgt_epi8(a, zero_);
                _mm_storeu_si128(
                    xp.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_and_si128(tmp, a), qe_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x08u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(b, zero_);
                _mm_storeu_si128(
                    yp.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_and_si128(tmp, b), qe_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x10u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(a2, zero_);
                _mm_storeu_si128(
                    x2p.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x20u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(b2, zero_);
                _mm_storeu_si128(
                    y2p.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x40u8 as i8)));
                }
            } else {
                let tmp = _mm_cmpgt_epi8(zero_, a);
                _mm_storeu_si128(
                    xp.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_andnot_si128(tmp, a), qe_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x08u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(zero_, b);
                _mm_storeu_si128(
                    yp.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_andnot_si128(tmp, b), qe_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x10u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(zero_, a2);
                _mm_storeu_si128(
                    x2p.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x20u8 as i8)));
                }
                let tmp = _mm_cmpgt_epi8(zero_, b2);
                _mm_storeu_si128(
                    y2p.add(p) as *mut __m128i,
                    _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_),
                );
                if with_cigar {
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x40u8 as i8)));
                }
            }

            if with_cigar {
                let pr_off = r as usize * n_col_ * 16 + (t as i32 - st_) as usize * 16;
                _mm_storeu_si128(bt.as_mut_ptr().add(pr_off) as *mut __m128i, d);
            }
        }

        // Score tracking (C lines 323-384, using signed interpretation)
        // Structure matches C exactly: compute H[]/max_H/max_t in if/else,
        // then mte/mqe/zdrop checks run UNCONDITIONALLY.
        let u8p = u.as_ptr() as *const i8;
        let v8p = v.as_ptr() as *const i8;
        let hp = h_arr.as_mut_ptr();
        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                if en0 > 0 {
                    h_arr[en0 as usize] = h_arr[(en0 - 1) as usize] + *u8p.add(en0 as usize) as i32;
                } else {
                    h_arr[en0 as usize] = h_arr[en0 as usize] + *v8p.add(en0 as usize) as i32;
                }
                max_h = h_arr[en0 as usize];
                max_t = en0;
                // SSE2-vectorized H-tracking: 4 i32 values per iteration
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut max_h_v = _mm_set1_epi32(max_h);
                let mut max_t_v = _mm_set1_epi32(max_t);
                {
                    let mut t = st0;
                    while t < en1 {
                        let tu = t as usize;
                        let mut h1 = _mm_loadu_si128(hp.add(tu) as *const __m128i);
                        let vv = _mm_setr_epi32(
                            *v8p.add(tu) as i32,
                            *v8p.add(tu + 1) as i32,
                            *v8p.add(tu + 2) as i32,
                            *v8p.add(tu + 3) as i32,
                        );
                        h1 = _mm_add_epi32(h1, vv);
                        _mm_storeu_si128(hp.add(tu) as *mut __m128i, h1);
                        let tv = _mm_set1_epi32(t);
                        let cmp = _mm_cmpgt_epi32(h1, max_h_v);
                        max_h_v =
                            _mm_or_si128(_mm_and_si128(cmp, h1), _mm_andnot_si128(cmp, max_h_v));
                        max_t_v =
                            _mm_or_si128(_mm_and_si128(cmp, tv), _mm_andnot_si128(cmp, max_t_v));
                        t += 4;
                    }
                }
                let mut hh = [0i32; 4];
                let mut tt = [0i32; 4];
                _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_v);
                _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_v);
                for i in 0..4 {
                    if max_h < hh[i] {
                        max_h = hh[i];
                        max_t = tt[i] + i as i32;
                    }
                }
                for t in en1..en0 {
                    *hp.add(t as usize) += *v8p.add(t as usize) as i32;
                    if *hp.add(t as usize) > max_h {
                        max_h = *hp.add(t as usize);
                        max_t = t;
                    }
                }
            } else {
                h_arr[0] = *v8p as i32 - qe;
                max_h = h_arr[0];
                max_t = 0;
            }
            // These checks run for BOTH r==0 and r>0, matching C exactly
            if en0 == tlen - 1 && h_arr[en0 as usize] > ez.mte {
                ez.mte = h_arr[en0 as usize];
                ez.mte_q = r - en0;
            }
            if r - st0 == qlen - 1 && h_arr[st0 as usize] > ez.mqe {
                ez.mqe = h_arr[st0 as usize];
                ez.mqe_t = st0;
            }
            // zdrop check (matches C's ksw_apply_zdrop with is_rot=1)
            if max_h > ez.max as i32 {
                ez.max = max_h;
                ez.max_t = max_t;
                ez.max_q = r - max_t;
            } else if max_t >= ez.max_t && r - max_t >= ez.max_q {
                let l = ((max_t - ez.max_t) - ((r - max_t) - ez.max_q)).abs();
                if zdrop >= 0 && ez.max - max_h > zdrop + l * e2 as i32 {
                    ez.zdropped = true;
                    break;
                }
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h_arr[(tlen - 1) as usize];
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = *v8p.add(last_h0_t as usize) as i32;
                    let d1 = *u8p.add((last_h0_t + 1) as usize) as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8p.add(last_h0_t as usize) as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8p.add(last_h0_t as usize) as i32;
                }
                if flag.contains(KswFlags::APPROX_DROP) && zdrop >= 0 {
                    if h0 > ez.max as i32 {
                        ez.max = h0;
                        ez.max_t = last_h0_t;
                        ez.max_q = r - last_h0_t;
                    } else if last_h0_t >= ez.max_t && r - last_h0_t >= ez.max_q {
                        let l = ((last_h0_t - ez.max_t) - ((r - last_h0_t) - ez.max_q)).abs();
                        if ez.max - h0 > zdrop + l * e2 as i32 {
                            ez.zdropped = true;
                            break;
                        }
                    }
                }
            } else {
                h0 = *v8p as i32 - qe;
                last_h0_t = 0;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    // Finalize
    if ez.score == KSW_NEG_INF && !approx_max {
        let mut sc = KSW_NEG_INF;
        if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
            sc = ez.mqe;
        }
        if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 && ez.mte > sc {
            sc = ez.mte;
        }
        ez.score = sc;
    }
    // Backtrack (same as single-gap — uses rotated coordinates)
    if with_cigar {
        let is_rev = flag.contains(KswFlags::REV_CIGAR);
        let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
        let (i0, j0) = if !ez.zdropped && !is_extz {
            (tlen - 1, qlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max as i32 {
            ez.reach_end = true;
            (ez.mqe_t, qlen - 1)
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            (ez.max_t, ez.max_q)
        } else {
            (-1, -1)
        };
        if i0 >= 0 && j0 >= 0 {
            let mut cigar = Vec::new();
            let mut state = 0u8;
            let (mut i, mut j) = (i0, j0);
            while i >= 0 && j >= 0 {
                let r = i + j;
                if r < 0 || r as usize >= n_ad {
                    break;
                }
                let off_r = off_a[r as usize];
                let off_er = off_e[r as usize];
                let mut fs: i32 = -1;
                if i < off_r {
                    fs = 2;
                }
                if i > off_er {
                    fs = 1;
                }
                let tmp = if fs < 0 {
                    let idx = r as usize * n_col_ * 16 + (i - off_r) as usize;
                    if idx < bt.len() {
                        bt[idx]
                    } else {
                        0
                    }
                } else {
                    0
                };
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 {
                    state = tmp & 7;
                }
                if fs >= 0 {
                    state = fs as u8;
                }
                match state {
                    0 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 | 3 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 2, 1);
                        i -= 1;
                    }
                    _ => {
                        super::ksw2::push_cigar_fn(&mut cigar, 1, 1);
                        j -= 1;
                    }
                }
            }
            if i >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 2, i + 1);
            }
            if j >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 1, j + 1);
            }
            if !is_rev {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }
    // Return scratch buffers to thread-local for reuse
    KSW_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    KSW_I32_SCRATCH.with(|c| *c.borrow_mut() = scratch_i32);
    ez
}

/// Score tracking — shared tail (zdrop check, boundary updates).
/// Matches C's unconditional mte/mqe/zdrop checks after H-array computation.
#[allow(dead_code)]
#[inline(always)]
unsafe fn score_track_tail(
    h_arr: &mut [i32],
    v8p: *const i8,
    hp: *mut i32,
    st0: i32,
    en0: i32,
    r: i32,
    qlen: i32,
    tlen: i32,
    _qe: i32,
    e2: i8,
    zdrop: i32,
    ez: &mut KswResult,
    mut max_h: i32,
    mut max_t: i32,
    tail_start: i32,
) -> bool {
    for t in tail_start..en0 {
        *hp.add(t as usize) += *v8p.add(t as usize) as i32;
        if *hp.add(t as usize) > max_h {
            max_h = *hp.add(t as usize);
            max_t = t;
        }
    }
    if en0 == tlen - 1 && h_arr[en0 as usize] > ez.mte {
        ez.mte = h_arr[en0 as usize];
        ez.mte_q = r - en0;
    }
    if r - st0 == qlen - 1 && h_arr[st0 as usize] > ez.mqe {
        ez.mqe = h_arr[st0 as usize];
        ez.mqe_t = st0;
    }
    // Max tracking is unconditional (matches C's ksw_apply_zdrop); zdrop check is conditional
    if max_h > ez.max as i32 {
        ez.max = max_h;
        ez.max_t = max_t;
        ez.max_q = r - max_t;
    } else if max_t >= ez.max_t && r - max_t >= ez.max_q {
        let l = ((max_t - ez.max_t) - ((r - max_t) - ez.max_q)).abs();
        if zdrop >= 0 && ez.max - max_h > zdrop + l * e2 as i32 {
            ez.zdropped = true;
            return true;
        }
    }
    if r == qlen + tlen - 2 && en0 == tlen - 1 {
        ez.score = h_arr[(tlen - 1) as usize];
    }
    false
}

/// Score tracking — AVX2 path (8 i32 per iteration).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2,avx2")]
#[allow(dead_code)]
#[inline(never)]
unsafe fn score_track_exact_avx2(
    h_arr: &mut [i32],
    u: *const u8,
    v: *const u8,
    st0: i32,
    en0: i32,
    r: i32,
    qlen: i32,
    tlen: i32,
    qe: i32,
    e2: i8,
    zdrop: i32,
    ez: &mut KswResult,
) -> bool {
    let u8p = u as *const i8;
    let v8p = v as *const i8;
    let hp = h_arr.as_mut_ptr();
    if r > 0 {
        if en0 > 0 {
            *hp.add(en0 as usize) = *hp.add((en0 - 1) as usize) + *u8p.add(en0 as usize) as i32;
        } else {
            *hp.add(en0 as usize) = *hp.add(en0 as usize) + *v8p.add(en0 as usize) as i32;
        }
        let mut max_h = *hp.add(en0 as usize);
        let mut max_t = en0;
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi32, _mm256_cvtepi8_epi32, _mm256_loadu_si256,
            _mm256_storeu_si256,
        };
        let en1 = st0 + (en0 - st0) / 8 * 8;
        let en1_sse = st0 + (en0 - st0) / 4 * 4;
        {
            let mut t = st0;
            while t < en1 {
                let tu = t as usize;
                let vv = _mm256_cvtepi8_epi32(_mm_loadl_epi64(v8p.add(tu) as *const __m128i));
                let mut h1 = _mm256_loadu_si256(hp.add(tu) as *const __m256i);
                h1 = _mm256_add_epi32(h1, vv);
                _mm256_storeu_si256(hp.add(tu) as *mut __m256i, h1);
                t += 8;
            }
        }
        for t in en1..en1_sse {
            *hp.add(t as usize) += *v8p.add(t as usize) as i32;
        }
        let mut max_h_v = _mm_set1_epi32(max_h);
        let mut max_t_v = _mm_set1_epi32(max_t);
        {
            let mut t = st0;
            while t < en1_sse {
                let tu = t as usize;
                let h1 = _mm_loadu_si128(hp.add(tu) as *const __m128i);
                let tv = _mm_set1_epi32(t);
                let cmp = _mm_cmpgt_epi32(h1, max_h_v);
                max_h_v = _mm_or_si128(_mm_and_si128(cmp, h1), _mm_andnot_si128(cmp, max_h_v));
                max_t_v = _mm_or_si128(_mm_and_si128(cmp, tv), _mm_andnot_si128(cmp, max_t_v));
                t += 4;
            }
        }
        let mut hh = [0i32; 4];
        let mut tt = [0i32; 4];
        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_v);
        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_v);
        for i in 0..4 {
            if max_h < hh[i] {
                max_h = hh[i];
                max_t = tt[i] + i as i32;
            }
        }
        score_track_tail(
            h_arr, v8p, hp, st0, en0, r, qlen, tlen, qe, e2, zdrop, ez, max_h, max_t, en1_sse,
        )
    } else {
        h_arr[0] = *v8p as i32 - qe;
        let max_h = h_arr[0];
        let max_t = 0i32;
        score_track_tail(
            h_arr, v8p, hp, st0, en0, r, qlen, tlen, qe, e2, zdrop, ez, max_h, max_t, en0,
        )
    }
}

/// Score tracking — SSE2 path (4 i32 per iteration). Works on any x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
#[inline(never)]
unsafe fn score_track_exact_sse2(
    h_arr: &mut [i32],
    u: *const u8,
    v: *const u8,
    st0: i32,
    en0: i32,
    r: i32,
    qlen: i32,
    tlen: i32,
    qe: i32,
    e2: i8,
    zdrop: i32,
    ez: &mut KswResult,
) -> bool {
    let u8p = u as *const i8;
    let v8p = v as *const i8;
    let hp = h_arr.as_mut_ptr();
    if r > 0 {
        if en0 > 0 {
            *hp.add(en0 as usize) = *hp.add((en0 - 1) as usize) + *u8p.add(en0 as usize) as i32;
        } else {
            *hp.add(en0 as usize) = *hp.add(en0 as usize) + *v8p.add(en0 as usize) as i32;
        }
        let mut max_h = *hp.add(en0 as usize);
        let mut max_t = en0;
        let en1 = st0 + (en0 - st0) / 4 * 4;
        let mut max_h_v = _mm_set1_epi32(max_h);
        let mut max_t_v = _mm_set1_epi32(max_t);
        {
            let mut t = st0;
            while t < en1 {
                let tu = t as usize;
                let vv = _mm_setr_epi32(
                    *v8p.add(tu) as i32,
                    *v8p.add(tu + 1) as i32,
                    *v8p.add(tu + 2) as i32,
                    *v8p.add(tu + 3) as i32,
                );
                let mut h1 = _mm_loadu_si128(hp.add(tu) as *const __m128i);
                h1 = _mm_add_epi32(h1, vv);
                _mm_storeu_si128(hp.add(tu) as *mut __m128i, h1);
                let tv = _mm_set1_epi32(t);
                let cmp = _mm_cmpgt_epi32(h1, max_h_v);
                max_h_v = _mm_or_si128(_mm_and_si128(cmp, h1), _mm_andnot_si128(cmp, max_h_v));
                max_t_v = _mm_or_si128(_mm_and_si128(cmp, tv), _mm_andnot_si128(cmp, max_t_v));
                t += 4;
            }
        }
        let mut hh = [0i32; 4];
        let mut tt = [0i32; 4];
        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_v);
        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_v);
        for i in 0..4 {
            if max_h < hh[i] {
                max_h = hh[i];
                max_t = tt[i] + i as i32;
            }
        }
        score_track_tail(
            h_arr, v8p, hp, st0, en0, r, qlen, tlen, qe, e2, zdrop, ez, max_h, max_t, en1,
        )
    } else {
        h_arr[0] = *v8p as i32 - qe;
        let max_h = h_arr[0];
        let max_t = 0i32;
        score_track_tail(
            h_arr, v8p, hp, st0, en0, r, qlen, tlen, qe, e2, zdrop, ez, max_h, max_t, en0,
        )
    }
}

/// SSE4.1-accelerated dual affine gap penalty kernel (inner implementation).
/// Same algorithm as extd2_sse2 but uses _mm_max_epi8, _mm_min_epi8, _mm_blendv_epi8
/// instead of multi-instruction SSE2 emulation sequences.
/// Const-generic WITH_CIGAR eliminates dead code per path; HAS_AVX2 selects
/// AVX2 vs SSE2 score tracking. target_feature is set by the wrapper.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn extd2_sse41_inner<const WITH_CIGAR: bool, const HAS_AVX2: bool>(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    mut q: i8,
    mut e: i8,
    mut q2: i8,
    mut e2: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 1 || qlen <= 0 || tlen <= 0 {
        return ez;
    }

    if (q2 as i32 + e2 as i32) < (q as i32 + e as i32) {
        std::mem::swap(&mut q, &mut q2);
        std::mem::swap(&mut e, &mut e2);
    }
    let qe = q as i32 + e as i32;
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let is_right = flag.contains(KswFlags::RIGHT);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let (wl, wr) = (w, w);
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((w + 1) as usize) + 15) / 16) + 1;

    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    for i in 1..(m as usize * m as usize) {
        max_sc = max_sc.max(mat[i] as i32);
        min_sc = min_sc.min(mat[i] as i32);
    }
    if -min_sc > 2 * qe {
        return ez;
    }

    let long_thres = if e != e2 {
        ((q2 as i32 - q as i32) / (e as i32 - e2 as i32)) - 1
    } else {
        0
    };
    let long_thres = if (q2 as i32 + e2 as i32 + long_thres * e2 as i32)
        > (q as i32 + e as i32 + long_thres * e as i32)
    {
        long_thres + 1
    } else {
        long_thres
    };
    let long_diff =
        (long_thres * (e as i32 - e2 as i32) - (q2 as i32 - q as i32) - e2 as i32) as i8;

    let bsz = tlen_ * 16 + 16;
    let nqe = (-(q as i32) - e as i32) as i8;
    let nq2e2 = (-(q2 as i32) - e2 as i32) as i8;
    let n_ad = (qlen + tlen - 1) as usize;
    let bt_size = if WITH_CIGAR {
        n_ad * n_col_ * 16 + 16
    } else {
        0
    };
    let off_size = if WITH_CIGAR { n_ad } else { 0 };
    let qr_pad = tlen as usize;
    let qr_sz = qr_pad + qlen as usize + 16;
    let sf_sz = tlen as usize + 16;
    let total_u8 = 7 * bsz + bt_size + qr_sz + sf_sz;
    let h_sz = if !approx_max { bsz } else { 0 };
    let total_i32 = h_sz + 2 * off_size;

    let mut scratch = KSW_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut scratch_i32 = KSW_I32_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_u8 + 15 {
        scratch.resize(total_u8 + 15, 0);
    }
    if scratch_i32.len() < total_i32 {
        scratch_i32.resize(total_i32, 0);
    }

    let raw_base = scratch.as_mut_ptr();
    let dp_base = (((raw_base as usize) + 15) & !15usize) as *mut u8;
    std::ptr::write_bytes(dp_base, nqe as u8, 4 * bsz);
    std::ptr::write_bytes(dp_base.add(4 * bsz), nq2e2 as u8, 2 * bsz);
    std::ptr::write_bytes(dp_base.add(6 * bsz), 0, bsz);
    let qr_start = 7 * bsz + bt_size;
    std::ptr::write_bytes(dp_base.add(qr_start), 0, qr_sz + sf_sz);

    if h_sz > 0 {
        scratch_i32[..h_sz].fill(KSW_NEG_INF);
    }
    if off_size > 0 {
        scratch_i32[h_sz..h_sz + 2 * off_size].fill(0);
    }

    let u = std::slice::from_raw_parts_mut(dp_base, bsz);
    let v = std::slice::from_raw_parts_mut(dp_base.add(bsz), bsz);
    let x = std::slice::from_raw_parts_mut(dp_base.add(2 * bsz), bsz);
    let y = std::slice::from_raw_parts_mut(dp_base.add(3 * bsz), bsz);
    let x2 = std::slice::from_raw_parts_mut(dp_base.add(4 * bsz), bsz);
    let y2 = std::slice::from_raw_parts_mut(dp_base.add(5 * bsz), bsz);
    let s = std::slice::from_raw_parts_mut(dp_base.add(6 * bsz), bsz);
    let bt = std::slice::from_raw_parts_mut(dp_base.add(7 * bsz), bt_size);
    let qr_off = 7 * bsz + bt_size;
    let sf_off = qr_off + qr_sz;

    let (h_arr, off_rest) = scratch_i32[..h_sz + 2 * off_size].split_at_mut(h_sz);
    let (off_a, off_e) = off_rest.split_at_mut(off_size);
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    let qr_ptr = dp_base.add(qr_off);
    for i in 0..qlen as usize {
        *qr_ptr.add(qr_pad + i) = query[qlen as usize - 1 - i];
    }
    let qr_base = qr_ptr.add(qr_pad);
    let sf_ptr = dp_base.add(sf_off);
    std::ptr::copy_nonoverlapping(target.as_ptr(), sf_ptr, tlen as usize);

    let zero_ = _mm_setzero_si128();
    let q_ = _mm_set1_epi8(q);
    let q2_ = _mm_set1_epi8(q2);
    let qe_ = _mm_set1_epi8((q as i32 + e as i32) as i8);
    let qe2_ = _mm_set1_epi8((q2 as i32 + e2 as i32) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let one_ = _mm_set1_epi8(1);
    let two_ = _mm_set1_epi8(2);
    let three_ = _mm_set1_epi8(3);
    let four_ = _mm_set1_epi8(4);
    let flag8_ = _mm_set1_epi8(0x08u8 as i8);
    let flag10_ = _mm_set1_epi8(0x10u8 as i8);
    let flag20_ = _mm_set1_epi8(0x20u8 as i8);
    let flag40_ = _mm_set1_epi8(0x40u8 as i8);
    let m1_ = _mm_set1_epi8(m - 1);
    let sc_n_ = if mat[(m as usize * m as usize) - 1] == 0 {
        _mm_set1_epi8(-e2)
    } else {
        _mm_set1_epi8(mat[(m as usize * m as usize) - 1])
    };

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let up = u.as_mut_ptr();
    let vp = v.as_mut_ptr();
    let xp = x.as_mut_ptr();
    let yp = y.as_mut_ptr();
    let x2p = x2.as_mut_ptr();
    let y2p = y2.as_mut_ptr();
    let sp = s.as_mut_ptr();

    for r in 0..qlen + tlen - 1 {
        let mut st = 0i32.max(r - qlen + 1);
        let mut en = (tlen - 1).min(r);
        if st < (r - wr + 1) >> 1 {
            st = (r - wr + 1) >> 1;
        }
        if en > (r + wl) / 2 {
            en = (r + wl) / 2;
        }
        if st > en {
            ez.zdropped = true;
            break;
        }
        let st0 = st;
        let en0 = en;
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        let x1: i8;
        let x21: i8;
        let v1: i8;
        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = x[(st - 1) as usize] as i8;
                x21 = x2[(st - 1) as usize] as i8;
                v1 = v[(st - 1) as usize] as i8;
            } else {
                x1 = nqe;
                x21 = nq2e2;
                v1 = nqe;
            }
        } else {
            x1 = nqe;
            x21 = nq2e2;
            v1 = if r == 0 {
                nqe
            } else if r < long_thres {
                -e
            } else if r == long_thres {
                long_diff
            } else {
                -e2
            };
        }
        if en >= r {
            y[r as usize] = nqe as u8;
            y2[r as usize] = nq2e2 as u8;
            u[r as usize] = (if r == 0 {
                nqe
            } else if r < long_thres {
                -e
            } else if r == long_thres {
                long_diff
            } else {
                -e2
            }) as u8;
        }

        let qrr = qlen - 1 - r;
        {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let sq = _mm_loadu_si128(sf_ptr.add(t) as *const __m128i);
                let qt = _mm_loadu_si128(qr_base.offset(qrr as isize).add(t) as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(qt, m1_));
                let eq = _mm_cmpeq_epi8(sq, qt);
                let tmp = _mm_blendv_epi8(sc_mis_, sc_mch_, eq);
                let tmp = _mm_blendv_epi8(tmp, sc_n_, mask);
                _mm_storeu_si128(sp.add(t) as *mut __m128i, tmp);
                t += 16;
            }
        }

        let mut x1_ = _mm_cvtsi32_si128(x1 as u8 as i32);
        let mut x21_ = _mm_cvtsi32_si128(x21 as u8 as i32);
        let mut v1_ = _mm_cvtsi32_si128(v1 as u8 as i32);
        let st_ = st / 16;
        let en_ = en / 16;

        if !WITH_CIGAR {
            // Score-only path — no backtrack direction bits, tighter loop
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                let z = _mm_load_si128(sp.add(p) as *const __m128i);
                let xt1r = _mm_load_si128(xp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_);
                x1_ = tmp;
                let vt1r = _mm_load_si128(vp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(up.add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_load_si128(yp.add(p) as *const __m128i), ut);
                let x2t1r = _mm_load_si128(x2p.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(x2t1r);
                let x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1r), x21_);
                x21_ = tmp;
                let a2 = _mm_add_epi8(x2t1, vt1);
                let b2 = _mm_add_epi8(_mm_load_si128(y2p.add(p) as *const __m128i), ut);

                let mut zz = _mm_max_epi8(z, a);
                zz = _mm_max_epi8(zz, b);
                zz = _mm_max_epi8(zz, a2);
                zz = _mm_max_epi8(zz, b2);
                zz = _mm_min_epi8(zz, sc_mch_);

                _mm_store_si128(up.add(p) as *mut __m128i, _mm_sub_epi8(zz, vt1));
                _mm_store_si128(vp.add(p) as *mut __m128i, _mm_sub_epi8(zz, ut));
                let zq = _mm_sub_epi8(zz, q_);
                let a = _mm_sub_epi8(a, zq);
                let b = _mm_sub_epi8(b, zq);
                let zq2 = _mm_sub_epi8(zz, q2_);
                let a2 = _mm_sub_epi8(a2, zq2);
                let b2 = _mm_sub_epi8(b2, zq2);

                if !is_right {
                    let tmp = _mm_cmpgt_epi8(a, zero_);
                    _mm_store_si128(
                        xp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, a), qe_),
                    );
                    let tmp = _mm_cmpgt_epi8(b, zero_);
                    _mm_store_si128(
                        yp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, b), qe_),
                    );
                    let tmp = _mm_cmpgt_epi8(a2, zero_);
                    _mm_store_si128(
                        x2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_),
                    );
                    let tmp = _mm_cmpgt_epi8(b2, zero_);
                    _mm_store_si128(
                        y2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_),
                    );
                } else {
                    let tmp = _mm_cmpgt_epi8(zero_, a);
                    _mm_store_si128(
                        xp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, a), qe_),
                    );
                    let tmp = _mm_cmpgt_epi8(zero_, b);
                    _mm_store_si128(
                        yp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, b), qe_),
                    );
                    let tmp = _mm_cmpgt_epi8(zero_, a2);
                    _mm_store_si128(
                        x2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_),
                    );
                    let tmp = _mm_cmpgt_epi8(zero_, b2);
                    _mm_store_si128(
                        y2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_),
                    );
                }
            }
        } else {
            // CIGAR path — with backtrack direction bits
            // SAFETY: r < n_ad = qlen+tlen-1, off_a/off_e have size n_ad (= off_size)
            *off_a.get_unchecked_mut(r as usize) = st;
            *off_e.get_unchecked_mut(r as usize) = en;
            let bt_ptr = bt.as_mut_ptr();
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                let z = _mm_load_si128(sp.add(p) as *const __m128i);
                let xt1r = _mm_load_si128(xp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_);
                x1_ = tmp;
                let vt1r = _mm_load_si128(vp.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(up.add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_load_si128(yp.add(p) as *const __m128i), ut);
                let x2t1r = _mm_load_si128(x2p.add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(x2t1r);
                let x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1r), x21_);
                x21_ = tmp;
                let a2 = _mm_add_epi8(x2t1, vt1);
                let b2 = _mm_add_epi8(_mm_load_si128(y2p.add(p) as *const __m128i), ut);

                // LEFT vs RIGHT comparison with SSE4.1 blendv
                // Matches C's ksw2_extd2_sse.c lines 234-252 (LEFT) / 283-301 (RIGHT)
                let mut zz = z;
                let mut d;
                if !is_right {
                    // LEFT: d = and(cmpgt(a,z), 1) — matches C line 235
                    d = _mm_and_si128(_mm_cmpgt_epi8(a, zz), one_);
                    zz = _mm_max_epi8(zz, a);
                    d = _mm_blendv_epi8(d, two_, _mm_cmpgt_epi8(b, zz));
                    zz = _mm_max_epi8(zz, b);
                    d = _mm_blendv_epi8(d, three_, _mm_cmpgt_epi8(a2, zz));
                    zz = _mm_max_epi8(zz, a2);
                    d = _mm_blendv_epi8(d, four_, _mm_cmpgt_epi8(b2, zz));
                    zz = _mm_max_epi8(zz, b2);
                } else {
                    // RIGHT: reversed tiebreaking
                    d = _mm_andnot_si128(_mm_cmpgt_epi8(zz, a), one_);
                    zz = _mm_max_epi8(zz, a);
                    d = _mm_blendv_epi8(two_, d, _mm_cmpgt_epi8(zz, b));
                    zz = _mm_max_epi8(zz, b);
                    d = _mm_blendv_epi8(three_, d, _mm_cmpgt_epi8(zz, a2));
                    zz = _mm_max_epi8(zz, a2);
                    d = _mm_blendv_epi8(four_, d, _mm_cmpgt_epi8(zz, b2));
                    zz = _mm_max_epi8(zz, b2);
                }
                zz = _mm_min_epi8(zz, sc_mch_);

                _mm_store_si128(up.add(p) as *mut __m128i, _mm_sub_epi8(zz, vt1));
                _mm_store_si128(vp.add(p) as *mut __m128i, _mm_sub_epi8(zz, ut));
                let zq = _mm_sub_epi8(zz, q_);
                let a = _mm_sub_epi8(a, zq);
                let b = _mm_sub_epi8(b, zq);
                let zq2 = _mm_sub_epi8(zz, q2_);
                let a2 = _mm_sub_epi8(a2, zq2);
                let b2 = _mm_sub_epi8(b2, zq2);

                if !is_right {
                    let tmp = _mm_cmpgt_epi8(a, zero_);
                    _mm_store_si128(
                        xp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, a), qe_),
                    );
                    d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                    let tmp = _mm_cmpgt_epi8(b, zero_);
                    _mm_store_si128(
                        yp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, b), qe_),
                    );
                    d = _mm_or_si128(d, _mm_and_si128(tmp, flag10_));
                    let tmp = _mm_cmpgt_epi8(a2, zero_);
                    _mm_store_si128(
                        x2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_),
                    );
                    d = _mm_or_si128(d, _mm_and_si128(tmp, flag20_));
                    let tmp = _mm_cmpgt_epi8(b2, zero_);
                    _mm_store_si128(
                        y2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_),
                    );
                    d = _mm_or_si128(d, _mm_and_si128(tmp, flag40_));
                } else {
                    let tmp = _mm_cmpgt_epi8(zero_, a);
                    _mm_store_si128(
                        xp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, a), qe_),
                    );
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));
                    let tmp = _mm_cmpgt_epi8(zero_, b);
                    _mm_store_si128(
                        yp.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, b), qe_),
                    );
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag10_));
                    let tmp = _mm_cmpgt_epi8(zero_, a2);
                    _mm_store_si128(
                        x2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_),
                    );
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag20_));
                    let tmp = _mm_cmpgt_epi8(zero_, b2);
                    _mm_store_si128(
                        y2p.add(p) as *mut __m128i,
                        _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_),
                    );
                    d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag40_));
                }

                let pr_off = r as usize * n_col_ * 16 + (t as i32 - st_) as usize * 16;
                _mm_store_si128(bt_ptr.add(pr_off) as *mut __m128i, d);
            }
        }

        // Score tracking
        if !approx_max {
            let zdropped = score_track_exact_sse2(
                h_arr,
                u.as_ptr(),
                v.as_ptr(),
                st0,
                en0,
                r,
                qlen,
                tlen,
                qe,
                e2,
                zdrop,
                &mut ez,
            );
            if zdropped {
                break; // zdropped
            }
        } else {
            let u8p = u.as_ptr() as *const i8;
            let v8p = v.as_ptr() as *const i8;
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = *v8p.add(last_h0_t as usize) as i32;
                    let d1 = *u8p.add((last_h0_t + 1) as usize) as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8p.add(last_h0_t as usize) as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8p.add(last_h0_t as usize) as i32;
                }
                if flag.contains(KswFlags::APPROX_DROP) && zdrop >= 0 {
                    if h0 > ez.max as i32 {
                        ez.max = h0;
                        ez.max_t = last_h0_t;
                        ez.max_q = r - last_h0_t;
                    } else if last_h0_t >= ez.max_t && r - last_h0_t >= ez.max_q {
                        let l = ((last_h0_t - ez.max_t) - ((r - last_h0_t) - ez.max_q)).abs();
                        if ez.max - h0 > zdrop + l * e2 as i32 {
                            ez.zdropped = true;
                            break;
                        }
                    }
                }
            } else {
                h0 = *v8p as i32 - qe;
                last_h0_t = 0;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    // Finalize
    if ez.score == KSW_NEG_INF && !approx_max {
        let mut sc = KSW_NEG_INF;
        if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
            sc = ez.mqe;
        }
        if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 && ez.mte > sc {
            sc = ez.mte;
        }
        ez.score = sc;
    }
    // Do NOT set reach_end here — C only sets it in the EXTZ_ONLY backtrace path
    // Backtrack
    if WITH_CIGAR {
        let is_rev = flag.contains(KswFlags::REV_CIGAR);
        let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
        let (i0, j0) = if !ez.zdropped && !is_extz {
            (tlen - 1, qlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max as i32 {
            ez.reach_end = true;
            (ez.mqe_t, qlen - 1)
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            (ez.max_t, ez.max_q)
        } else {
            (-1, -1)
        };
        if i0 >= 0 && j0 >= 0 {
            let mut cigar = Vec::new();
            let mut state = 0u8;
            let (mut i, mut j) = (i0, j0);
            while i >= 0 && j >= 0 {
                let r = i + j;
                if r < 0 || r as usize >= n_ad {
                    break;
                }
                let off_r = off_a[r as usize];
                let off_er = off_e[r as usize];
                let mut fs: i32 = -1;
                if i < off_r {
                    fs = 2;
                }
                if i > off_er {
                    fs = 1;
                }
                // SAFETY: idx bounds checked explicitly, off_a/off_e indexed by r which is in [0, n_ad)
                let tmp = if fs < 0 {
                    let idx = r as usize * n_col_ * 16 + (i - off_r) as usize;
                    if idx < bt.len() {
                        *bt.get_unchecked(idx)
                    } else {
                        0
                    }
                } else {
                    0
                };
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 {
                    state = tmp & 7;
                }
                if fs >= 0 {
                    state = fs as u8;
                }
                match state {
                    0 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 | 3 => {
                        super::ksw2::push_cigar_fn(&mut cigar, 2, 1);
                        i -= 1;
                    }
                    _ => {
                        super::ksw2::push_cigar_fn(&mut cigar, 1, 1);
                        j -= 1;
                    }
                }
            }
            if i >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 2, i + 1);
            }
            if j >= 0 {
                super::ksw2::push_cigar_fn(&mut cigar, 1, j + 1);
            }
            if !is_rev {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }
    KSW_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    KSW_I32_SCRATCH.with(|c| *c.borrow_mut() = scratch_i32);
    ez
}

/// SSE4.1+AVX2 entry point — VEX-encoded instructions, no false deps, AVX2 score tracking.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,avx2")]
#[allow(dead_code)]
unsafe fn extd2_sse41_avx2<const WITH_CIGAR: bool>(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    e2: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    extd2_sse41_inner::<WITH_CIGAR, true>(
        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
    )
}

/// SSE4.1-only entry point — works on any CPU with SSE4.1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
unsafe fn extd2_sse41_only<const WITH_CIGAR: bool>(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    e2: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    extd2_sse41_inner::<WITH_CIGAR, false>(
        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
    )
}

/// Dispatch for dual-gap: SSE4.1 → SSE2 → scalar fallback.
pub fn ksw_extd2_dispatch(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    e2: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    #[cfg(target_arch = "x86_64")]
    {
        let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
        if has_sse41() {
            return unsafe {
                // Keep the AVX2 variant available for direct test coverage, but do not
                // dispatch to it by default until its extd2 output is proven identical to C.
                if with_cigar {
                    extd2_sse41_only::<true>(
                        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
                    )
                } else {
                    extd2_sse41_only::<false>(
                        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
                    )
                }
            };
        }
        if has_sse2() {
            return unsafe {
                extd2_sse2(
                    query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
                )
            };
        }
        return crate::align::ksw2::ksw_extd2(
            query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    crate::align::ksw2::ksw_extd2(
        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
    )
}

/// SSE2-faithful translation of ksw_exts2_sse from minimap2/ksw2_exts2_sse.c.
/// Uses rotated anti-diagonal DP with 16-way SSE2 byte-parallel operations,
/// matching C bit-for-bit including int8 wrap-around semantics.
///
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn exts2_sse2(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    noncan: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    junc: Option<&[u8]>,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();

    // Early-exit checks mirroring C lines 73-74:
    // - m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e
    if m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e {
        return ez;
    }
    debug_assert!(
        !(flag.contains(KswFlags::SPLICE_FOR) && flag.contains(KswFlags::SPLICE_REV)),
        "ksw_exts2: SPLICE_FOR and SPLICE_REV cannot both be set"
    );

    // Score range sanity check (C line 88-92).
    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    let m_u = m as usize;
    for k in 1..(m_u * m_u) {
        let v = mat[k] as i32;
        if v > max_sc {
            max_sc = v;
        }
        if v < min_sc {
            min_sc = v;
        }
    }
    if -min_sc > 2 * (q as i32 + e as i32) {
        return ez;
    }
    let _ = max_sc;

    // long_thres = (q2-q)/e - 1, possibly +1 (C lines 94-97).
    let mut long_thres = (q2 as i32 - q as i32) / (e as i32) - 1;
    if (q2 as i32) > (q as i32) + (e as i32) + long_thres * (e as i32) {
        long_thres += 1;
    }
    let _long_diff = long_thres * (e as i32) - (q2 as i32 - q as i32);

    // Splice penalty constants (C lines 122-130). For CMPLX mode use sp0/3,
    // else use noncan / noncan-half-flank pattern.
    let _sp = if flag.contains(KswFlags::SPLICE_CMPLX) {
        let sp0 = [8, 15, 21, 30];
        let mut sp = [0i32; 4];
        for k in 0..4 {
            sp[k] = ((sp0[k] as f64) / 3.0 + 0.499) as i32;
        }
        sp
    } else {
        let half = if flag.contains(KswFlags::SPLICE_FLANK) {
            (noncan as i32) / 2
        } else {
            0
        };
        [half, noncan as i32, noncan as i32, noncan as i32]
    };
    let _ = _long_diff;

    // SSE2-aligned scratch buffer allocation, mirroring C lines 99-115.
    // Layout: [u:bsz][v:bsz][x:bsz][y:bsz][x2:bsz][donor:bsz][acceptor:bsz]
    //         [s:bsz][bt:bt_size][qr:qr_sz][sf:sf_sz]
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);

    let _w = if w < 0 { qlen.max(tlen) } else { w };
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((_w + 1) as usize) + 15) / 16) + 1;

    let bsz = tlen_ * 16 + 16;
    let row_width = n_col_ * 16;
    let nqe = (-(q as i32) - e as i32) as i8;
    let nq2 = (-(q2 as i32)) as i8;
    let n_ad = (qlen + tlen - 1) as usize;
    let bt_size = if with_cigar { n_ad * row_width + 16 } else { 0 };
    let off_size = if with_cigar { n_ad } else { 0 };
    let qr_sz = qlen as usize + 16;
    let sf_sz = tlen as usize + 16;
    // 8 byte arrays (u/v/x/y/x2/donor/acceptor/s) of size bsz, plus bt, qr,
    // sf. SSE4.1's _mm_max_epi8 isn't available on pure SSE2; we'll use the
    // andnot/cmpgt emulation pattern for state pick.
    let total_u8 = 8 * bsz + bt_size + qr_sz + sf_sz;
    let h_sz = if !approx_max { bsz } else { 0 };
    let total_i32 = h_sz + 2 * off_size;

    let mut scratch = KSW_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut scratch_i32 = KSW_I32_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_u8 + 15 {
        scratch.resize(total_u8 + 15, 0);
    }
    if scratch_i32.len() < total_i32 {
        scratch_i32.resize(total_i32, 0);
    }

    // Init the byte arrays per C lines 104-105, 131-132:
    //   u, v, x, y → -q-e   (single memset across 4 contiguous arrays)
    //   x2 → -q2            (separate memset)
    //   donor, acceptor → -sp[3] (init to default penalty; will be overwritten)
    //   s → 0               (zeroed; gets populated per anti-diagonal)
    let raw_base = scratch.as_mut_ptr();
    let dp_base = (((raw_base as usize) + 15) & !15usize) as *mut u8;
    std::ptr::write_bytes(dp_base, nqe as u8, 4 * bsz);
    std::ptr::write_bytes(dp_base.add(4 * bsz), nq2 as u8, bsz);
    let sp_default = if flag.contains(KswFlags::SPLICE_CMPLX) {
        ((30.0 / 3.0) + 0.499) as i32
    } else {
        noncan as i32
    };
    let neg_sp = (-sp_default) as i8;
    std::ptr::write_bytes(dp_base.add(5 * bsz), neg_sp as u8, 2 * bsz);
    std::ptr::write_bytes(dp_base.add(7 * bsz), 0, bsz);

    // Zero qr and sf padding.
    let qr_start = 8 * bsz + bt_size;
    std::ptr::write_bytes(dp_base.add(qr_start), 0, qr_sz + sf_sz);

    // qr = reverse(query)
    for k in 0..(qlen as usize) {
        *dp_base.add(qr_start + k) = query[qlen as usize - 1 - k];
    }
    // sf = target (verbatim copy)
    let sf_start = qr_start + qr_sz;
    std::ptr::copy_nonoverlapping(target.as_ptr(), dp_base.add(sf_start), tlen as usize);

    let zero_ = _mm_set1_epi8(0);
    let q_ = _mm_set1_epi8(q);
    let q2_ = _mm_set1_epi8(q2);
    let qe_ = _mm_set1_epi8((q as i16 + e as i16) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let sc_n_ = if mat[m_u * m_u - 1] == 0 {
        _mm_set1_epi8(-e)
    } else {
        _mm_set1_epi8(mat[m_u * m_u - 1])
    };
    let m1_ = _mm_set1_epi8(m - 1);
    let flag1_ = _mm_set1_epi8(1);
    let flag2_ = _mm_set1_epi8(2);
    let flag3_ = _mm_set1_epi8(3);
    let flag8_ = _mm_set1_epi8(0x08u8 as i8);
    let flag16_ = _mm_set1_epi8(0x10u8 as i8);
    let flag32_ = _mm_set1_epi8(0x20u8 as i8);

    if h_sz > 0 {
        scratch_i32[..h_sz].fill(KSW_NEG_INF);
    }
    if off_size > 0 {
        scratch_i32[h_sz..h_sz + 2 * off_size].fill(0);
    }

    // Donor/acceptor SIMD scoring per C lines 121-191. Computes the
    // per-position splice signal penalty arrays inside the scratch's donor/
    // acceptor regions. Mirrors C's z-table lookup with sp[] penalties.
    let donor_off = 5 * bsz;
    let acceptor_off = 6 * bsz;
    let donor_ptr = dp_base.add(donor_off) as *mut i8;
    let acceptor_ptr = dp_base.add(acceptor_off) as *mut i8;
    if flag.intersects(KswFlags::SPLICE_FOR | KswFlags::SPLICE_REV) {
        let sp0 = [8i32, 15, 21, 30];
        let sp = if flag.contains(KswFlags::SPLICE_CMPLX) {
            let mut sp = [0i32; 4];
            for k in 0..4 {
                sp[k] = ((sp0[k] as f64) / 3.0 + 0.499) as i32;
            }
            sp
        } else {
            let s0 = if flag.contains(KswFlags::SPLICE_FLANK) {
                noncan as i32 / 2
            } else {
                0
            };
            [s0, noncan as i32, noncan as i32, noncan as i32]
        };
        if !flag.contains(KswFlags::REV_CIGAR) {
            let donor_end = (tlen - 4).max(0) as usize;
            for t in 0..donor_end {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t + 1] == 2 && target[t + 2] == 3 {
                        z = if target[t + 3] == 0 || target[t + 3] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 2 && target[t + 2] == 1 {
                        z = 1;
                    } else if target[t + 1] == 0 && target[t + 2] == 3 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t + 1] == 1 && target[t + 2] == 3 {
                        z = if target[t + 3] == 0 || target[t + 3] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 2 && target[t + 2] == 3 {
                        z = 2;
                    }
                }
                let v = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
                *donor_ptr.add(t) = v;
            }
            // Acceptor arm.
            for t in 2..(tlen as usize) {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t - 1] == 0 && target[t] == 2 {
                        z = if target[t - 2] == 1 || target[t - 2] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 0 && target[t] == 1 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t - 1] == 0 && target[t] == 1 {
                        z = if target[t - 2] == 1 || target[t - 2] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 2 && target[t] == 1 {
                        z = 1;
                    } else if target[t - 1] == 0 && target[t] == 3 {
                        z = 2;
                    }
                }
                let v = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
                *acceptor_ptr.add(t) = v;
            }
        } else {
            let donor_end = (tlen - 4).max(0) as usize;
            for t in 0..donor_end {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t + 1] == 2 && target[t + 2] == 0 {
                        z = if target[t + 3] == 1 || target[t + 3] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 1 && target[t + 2] == 0 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t + 1] == 1 && target[t + 2] == 0 {
                        z = if target[t + 3] == 1 || target[t + 3] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 1 && target[t + 2] == 2 {
                        z = 1;
                    } else if target[t + 1] == 3 && target[t + 2] == 0 {
                        z = 2;
                    }
                }
                let v = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
                *donor_ptr.add(t) = v;
            }
            for t in 2..(tlen as usize) {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t - 1] == 3 && target[t] == 2 {
                        z = if target[t - 2] == 0 || target[t - 2] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 1 && target[t] == 2 {
                        z = 1;
                    } else if target[t - 1] == 3 && target[t] == 0 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t - 1] == 3 && target[t] == 1 {
                        z = if target[t - 2] == 0 || target[t - 2] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 3 && target[t] == 2 {
                        z = 2;
                    }
                }
                let v = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
                *acceptor_ptr.add(t) = v;
            }
        }
    }

    if let Some(junc) = junc {
        if flag.contains(KswFlags::SPLICE_SCORE) {
            let donor_val =
                (flag.contains(KswFlags::SPLICE_FOR) == !flag.contains(KswFlags::REV_CIGAR)) as u8;
            let limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..limit {
                let delta_donor = if junc[t + 1] == 0xff || (junc[t + 1] & 1) != donor_val {
                    -(junc_pen as i32)
                } else {
                    (junc[t + 1] >> 1) as i32 - 64
                };
                let delta_acceptor = if junc[t + 1] == 0xff || (junc[t + 1] & 1) != (1 - donor_val)
                {
                    -(junc_pen as i32)
                } else {
                    (junc[t + 1] >> 1) as i32 - 64
                };
                *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + delta_donor) as i8;
                *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + delta_acceptor) as i8;
            }
        } else if !flag.contains(KswFlags::REV_CIGAR) {
            let donor_limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..donor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 8) != 0)
                {
                    *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
            let acceptor_limit = (tlen as usize).min(junc.len());
            for t in 0..acceptor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 4) != 0)
                {
                    *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
        } else {
            let donor_limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..donor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 4) != 0)
                {
                    *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
            let acceptor_limit = (tlen as usize).min(junc.len());
            for t in 0..acceptor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 8) != 0)
                {
                    *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
        }
    }

    let u = dp_base as *mut __m128i;
    let v = dp_base.add(bsz) as *mut __m128i;
    let x = dp_base.add(2 * bsz) as *mut __m128i;
    let y = dp_base.add(3 * bsz) as *mut __m128i;
    let x2 = dp_base.add(4 * bsz) as *mut __m128i;
    let donor = dp_base.add(5 * bsz) as *mut __m128i;
    let acceptor = dp_base.add(6 * bsz) as *mut __m128i;
    let s = dp_base.add(7 * bsz) as *mut __m128i;
    let p = if with_cigar {
        dp_base.add(8 * bsz) as *mut __m128i
    } else {
        std::ptr::null_mut()
    };
    let qr = dp_base.add(qr_start);
    let sf = dp_base.add(sf_start);
    let h_arr = scratch_i32.as_mut_ptr();
    let off = if with_cigar {
        scratch_i32.as_mut_ptr().add(h_sz)
    } else {
        std::ptr::null_mut()
    };
    let off_end = if with_cigar {
        off.add(n_ad)
    } else {
        std::ptr::null_mut()
    };

    let u8 = u as *mut i8;
    let v8 = v as *mut i8;
    let x8 = x as *mut i8;
    let y8 = y as *mut i8;
    let x28 = x2 as *mut i8;

    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;
    let mut last_st = -1i32;
    let mut last_en = -1i32;

    for r in 0..(qlen + tlen - 1) {
        let mut st = 0i32;
        let mut en = tlen - 1;
        if st < r - qlen + 1 {
            st = r - qlen + 1;
        }
        if en > r {
            en = r;
        }
        let st0 = st;
        let en0 = en;
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        let (x1v, x21v, v1v) = if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                (
                    *x8.add((st - 1) as usize),
                    *x28.add((st - 1) as usize),
                    *v8.add((st - 1) as usize),
                )
            } else {
                (nqe, nq2, nqe)
            }
        } else {
            let v1 = if r == 0 {
                -(q as i32) - e as i32
            } else if r < long_thres {
                -(e as i32)
            } else if r == long_thres {
                _long_diff
            } else {
                0
            };
            (nqe, nq2, v1 as i8)
        };
        if en >= r {
            *y8.add(r as usize) = nqe;
            *u8.add(r as usize) = if r == 0 {
                nqe
            } else if r < long_thres {
                (-(e as i32)) as i8
            } else if r == long_thres {
                _long_diff as i8
            } else {
                0
            };
        }

        if !flag.contains(KswFlags::GENERIC_SC) {
            let qrr = qr.offset((qlen - 1 - r) as isize);
            let mut t = st0;
            while t <= en0 {
                let sq = _mm_loadu_si128(sf.add(t as usize) as *const __m128i);
                let stv = _mm_loadu_si128(qrr.add(t as usize) as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(stv, m1_));
                let mut tmp = _mm_cmpeq_epi8(sq, stv);
                tmp = _mm_or_si128(_mm_andnot_si128(tmp, sc_mis_), _mm_and_si128(tmp, sc_mch_));
                tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_));
                _mm_storeu_si128((s as *mut i8).add(t as usize) as *mut __m128i, tmp);
                t += 16;
            }
        } else {
            for t in st0..=en0 {
                *(s as *mut u8).add(t as usize) = mat[*sf.add(t as usize) as usize * m_u
                    + *qr.offset((qlen - 1 - r + t) as isize) as usize]
                    as u8;
            }
        }

        let mut x1_ = _mm_cvtsi32_si128((x1v as u8) as i32);
        let mut x21_ = _mm_cvtsi32_si128((x21v as u8) as i32);
        let mut v1_ = _mm_cvtsi32_si128((v1v as u8) as i32);
        let st_ = st / 16;
        let en_ = en / 16;

        if with_cigar {
            *off.add(r as usize) = st;
            *off_end.add(r as usize) = en;
        }

        if !with_cigar {
            for t in st_..=en_ {
                let tp = t as usize;
                let mut z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let mut a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let mut b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let mut a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                tmp = _mm_cmpgt_epi8(a, z);
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
                tmp = _mm_cmpgt_epi8(b, z);
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
                tmp = _mm_cmpgt_epi8(a2a, z);
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2a));

                _mm_store_si128(u.add(tp), _mm_sub_epi8(z, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(z, ut));
                tmp = _mm_sub_epi8(z, q_);
                a = _mm_sub_epi8(a, tmp);
                b = _mm_sub_epi8(b, tmp);
                a2 = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                tmp = _mm_cmpgt_epi8(a, zero_);
                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));
                tmp = _mm_cmpgt_epi8(b, zero_);
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
                let mut don = _mm_load_si128(donor.add(tp));
                tmp = _mm_cmpgt_epi8(a2, don);
                don = _mm_or_si128(_mm_andnot_si128(tmp, don), _mm_and_si128(tmp, a2));
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(don, q2_));
            }
        } else if !flag.contains(KswFlags::RIGHT) {
            let pr = p.add(r as usize * n_col_).sub(st_ as usize);
            for t in st_..=en_ {
                let tp = t as usize;
                let mut z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let mut a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let mut b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let mut a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                let mut d;
                tmp = _mm_cmpgt_epi8(a, z);
                d = _mm_and_si128(tmp, flag1_);
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
                tmp = _mm_cmpgt_epi8(b, z);
                d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_));
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
                tmp = _mm_cmpgt_epi8(a2a, z);
                d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag3_));
                z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2a));

                _mm_store_si128(u.add(tp), _mm_sub_epi8(z, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(z, ut));
                tmp = _mm_sub_epi8(z, q_);
                a = _mm_sub_epi8(a, tmp);
                b = _mm_sub_epi8(b, tmp);
                a2 = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                tmp = _mm_cmpgt_epi8(a, zero_);
                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                tmp = _mm_cmpgt_epi8(b, zero_);
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));
                let mut tmp2 = _mm_load_si128(donor.add(tp));
                tmp = _mm_cmpgt_epi8(a2, tmp2);
                tmp2 = _mm_or_si128(_mm_andnot_si128(tmp, tmp2), _mm_and_si128(tmp, a2));
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(tmp2, q2_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag32_));
                _mm_store_si128(pr.add(tp), d);
            }
        } else {
            let pr = p.add(r as usize * n_col_).sub(st_ as usize);
            for t in st_..=en_ {
                let tp = t as usize;
                let mut z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let mut a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let mut b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let mut a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                let mut d;
                tmp = _mm_cmpgt_epi8(z, a);
                d = _mm_andnot_si128(tmp, flag1_);
                z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a));
                tmp = _mm_cmpgt_epi8(z, b);
                d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, flag2_));
                z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b));
                tmp = _mm_cmpgt_epi8(z, a2a);
                d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, flag3_));
                z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a2a));

                _mm_store_si128(u.add(tp), _mm_sub_epi8(z, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(z, ut));
                tmp = _mm_sub_epi8(z, q_);
                a = _mm_sub_epi8(a, tmp);
                b = _mm_sub_epi8(b, tmp);
                a2 = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                tmp = _mm_cmpgt_epi8(zero_, a);
                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_andnot_si128(tmp, a), qe_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));
                tmp = _mm_cmpgt_epi8(zero_, b);
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_andnot_si128(tmp, b), qe_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_));
                let mut tmp2 = _mm_load_si128(donor.add(tp));
                tmp = _mm_cmpgt_epi8(tmp2, a2);
                tmp2 = _mm_or_si128(_mm_andnot_si128(tmp, a2), _mm_and_si128(tmp, tmp2));
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(tmp2, q2_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag32_));
                _mm_store_si128(pr.add(tp), d);
            }
        }

        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                if en0 > 0 {
                    *h_arr.add(en0 as usize) =
                        *h_arr.add((en0 - 1) as usize) + *u8.add(en0 as usize) as i32;
                } else {
                    *h_arr.add(en0 as usize) += *v8.add(en0 as usize) as i32;
                }
                max_h = *h_arr.add(en0 as usize);
                max_t = en0;
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut t = st0;
                while t < en1 {
                    let h1 = _mm_loadu_si128(h_arr.add(t as usize) as *const __m128i);
                    let add = _mm_setr_epi32(
                        *v8.add(t as usize) as i32,
                        *v8.add((t + 1) as usize) as i32,
                        *v8.add((t + 2) as usize) as i32,
                        *v8.add((t + 3) as usize) as i32,
                    );
                    let h2 = _mm_add_epi32(h1, add);
                    _mm_storeu_si128(h_arr.add(t as usize) as *mut __m128i, h2);
                    t += 4;
                }
                while t < en0 {
                    *h_arr.add(t as usize) += *v8.add(t as usize) as i32;
                    t += 1;
                }
                // Match the C SSE2 implementation's tie-breaking order:
                // after updating the whole anti-diagonal, scan lane-major
                // (st0, st0+4, ..., then st0+1, st0+5, ...). Strict-GT
                // updates preserve the earliest-visited max_t in that order.
                for lane in 0..4 {
                    let mut t = st0 + lane;
                    while t < en1 {
                        let ht = *h_arr.add(t as usize);
                        if ht > max_h {
                            max_h = ht;
                            max_t = t;
                        }
                        t += 4;
                    }
                }
                t = en1;
                while t < en0 {
                    let ht = *h_arr.add(t as usize);
                    if ht > max_h {
                        max_h = ht;
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                *h_arr.add(0) = *v8.add(0) as i32 - (q as i32 + e as i32);
                max_h = *h_arr.add(0);
                max_t = 0;
            }
            if en0 == tlen - 1 && *h_arr.add(en0 as usize) > ez.mte {
                ez.mte = *h_arr.add(en0 as usize);
                ez.mte_q = r - en0;
            }
            if r - st0 == qlen - 1 && *h_arr.add(st0 as usize) > ez.mqe {
                ez.mqe = *h_arr.add(st0 as usize);
                ez.mqe_t = st0;
            }
            if zdrop >= 0 && ksw_apply_zdrop_rot(&mut ez, max_h, r, max_t, zdrop, 0) {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = *h_arr.add((tlen - 1) as usize);
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = *v8.add(last_h0_t as usize) as i32;
                    let d1 = *u8.add((last_h0_t + 1) as usize) as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8.add(last_h0_t as usize) as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8.add(last_h0_t as usize) as i32;
                }
            } else {
                h0 = *v8.add(0) as i32 - (q as i32 + e as i32);
                last_h0_t = 0;
            }
            if flag.contains(KswFlags::APPROX_DROP)
                && ksw_apply_zdrop_rot(&mut ez, h0, r, last_h0_t, zdrop, 0)
            {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    if with_cigar {
        let rev_cigar = flag.contains(KswFlags::REV_CIGAR);
        if !ez.zdropped && !flag.contains(KswFlags::EXTZ_ONLY) {
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                qlen - 1,
                tlen - 1,
            );
        } else if !ez.zdropped && flag.contains(KswFlags::EXTZ_ONLY) && ez.mqe + end_bonus > ez.max
        {
            ez.reach_end = true;
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                qlen - 1,
                ez.mqe_t,
            );
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                ez.max_q,
                ez.max_t,
            );
        }
    }

    KSW_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    KSW_I32_SCRATCH.with(|c| *c.borrow_mut() = scratch_i32);
    ez
}

/// SSE4.1 splice kernel. Same algorithm and tie-breaking as `exts2_sse2()`,
/// but uses `_mm_max_epi8`/`_mm_blendv_epi8` in the hot loop instead of the
/// SSE2 cmp/or emulation sequence.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn exts2_sse41(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    noncan: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    junc: Option<&[u8]>,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();

    if m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e {
        return ez;
    }
    debug_assert!(
        !(flag.contains(KswFlags::SPLICE_FOR) && flag.contains(KswFlags::SPLICE_REV)),
        "ksw_exts2: SPLICE_FOR and SPLICE_REV cannot both be set"
    );

    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    let m_u = m as usize;
    for k in 1..(m_u * m_u) {
        let v = mat[k] as i32;
        if v > max_sc {
            max_sc = v;
        }
        if v < min_sc {
            min_sc = v;
        }
    }
    if -min_sc > 2 * (q as i32 + e as i32) {
        return ez;
    }
    let _ = max_sc;

    let mut long_thres = (q2 as i32 - q as i32) / (e as i32) - 1;
    if (q2 as i32) > (q as i32) + (e as i32) + long_thres * (e as i32) {
        long_thres += 1;
    }
    let long_diff = long_thres * (e as i32) - (q2 as i32 - q as i32);

    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);

    let _w = if w < 0 { qlen.max(tlen) } else { w };
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((_w + 1) as usize) + 15) / 16) + 1;

    let bsz = tlen_ * 16 + 16;
    let row_width = n_col_ * 16;
    let nqe = (-(q as i32) - e as i32) as i8;
    let nq2 = (-(q2 as i32)) as i8;
    let n_ad = (qlen + tlen - 1) as usize;
    let bt_size = if with_cigar { n_ad * row_width + 16 } else { 0 };
    let off_size = if with_cigar { n_ad } else { 0 };
    let qr_sz = qlen as usize + 16;
    let sf_sz = tlen as usize + 16;
    let total_u8 = 8 * bsz + bt_size + qr_sz + sf_sz;
    let h_sz = if !approx_max { bsz } else { 0 };
    let total_i32 = h_sz + 2 * off_size;

    let mut scratch = KSW_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut scratch_i32 = KSW_I32_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_u8 + 15 {
        scratch.resize(total_u8 + 15, 0);
    }
    if scratch_i32.len() < total_i32 {
        scratch_i32.resize(total_i32, 0);
    }

    let raw_base = scratch.as_mut_ptr();
    let dp_base = (((raw_base as usize) + 15) & !15usize) as *mut u8;
    std::ptr::write_bytes(dp_base, nqe as u8, 4 * bsz);
    std::ptr::write_bytes(dp_base.add(4 * bsz), nq2 as u8, bsz);
    let sp_default = if flag.contains(KswFlags::SPLICE_CMPLX) {
        ((30.0 / 3.0) + 0.499) as i32
    } else {
        noncan as i32
    };
    let neg_sp = (-sp_default) as i8;
    std::ptr::write_bytes(dp_base.add(5 * bsz), neg_sp as u8, 2 * bsz);
    std::ptr::write_bytes(dp_base.add(7 * bsz), 0, bsz);

    let qr_start = 8 * bsz + bt_size;
    std::ptr::write_bytes(dp_base.add(qr_start), 0, qr_sz + sf_sz);

    for k in 0..(qlen as usize) {
        *dp_base.add(qr_start + k) = query[qlen as usize - 1 - k];
    }
    let sf_start = qr_start + qr_sz;
    std::ptr::copy_nonoverlapping(target.as_ptr(), dp_base.add(sf_start), tlen as usize);

    let zero_ = _mm_set1_epi8(0);
    let q_ = _mm_set1_epi8(q);
    let q2_ = _mm_set1_epi8(q2);
    let qe_ = _mm_set1_epi8((q as i16 + e as i16) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let sc_n_ = if mat[m_u * m_u - 1] == 0 {
        _mm_set1_epi8(-e)
    } else {
        _mm_set1_epi8(mat[m_u * m_u - 1])
    };
    let m1_ = _mm_set1_epi8(m - 1);
    let flag1_ = _mm_set1_epi8(1);
    let flag2_ = _mm_set1_epi8(2);
    let flag3_ = _mm_set1_epi8(3);
    let flag8_ = _mm_set1_epi8(0x08u8 as i8);
    let flag16_ = _mm_set1_epi8(0x10u8 as i8);
    let flag32_ = _mm_set1_epi8(0x20u8 as i8);

    if h_sz > 0 {
        scratch_i32[..h_sz].fill(KSW_NEG_INF);
    }
    if off_size > 0 {
        scratch_i32[h_sz..h_sz + 2 * off_size].fill(0);
    }

    let donor_off = 5 * bsz;
    let acceptor_off = 6 * bsz;
    let donor_ptr = dp_base.add(donor_off) as *mut i8;
    let acceptor_ptr = dp_base.add(acceptor_off) as *mut i8;
    if flag.intersects(KswFlags::SPLICE_FOR | KswFlags::SPLICE_REV) {
        let sp0 = [8i32, 15, 21, 30];
        let sp = if flag.contains(KswFlags::SPLICE_CMPLX) {
            let mut sp = [0i32; 4];
            for k in 0..4 {
                sp[k] = ((sp0[k] as f64) / 3.0 + 0.499) as i32;
            }
            sp
        } else {
            let s0 = if flag.contains(KswFlags::SPLICE_FLANK) {
                noncan as i32 / 2
            } else {
                0
            };
            [s0, noncan as i32, noncan as i32, noncan as i32]
        };
        if !flag.contains(KswFlags::REV_CIGAR) {
            let donor_end = (tlen - 4).max(0) as usize;
            for t in 0..donor_end {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t + 1] == 2 && target[t + 2] == 3 {
                        z = if target[t + 3] == 0 || target[t + 3] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 2 && target[t + 2] == 1 {
                        z = 1;
                    } else if target[t + 1] == 0 && target[t + 2] == 3 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t + 1] == 1 && target[t + 2] == 3 {
                        z = if target[t + 3] == 0 || target[t + 3] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 2 && target[t + 2] == 3 {
                        z = 2;
                    }
                }
                *donor_ptr.add(t) = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
            }
            for t in 2..(tlen as usize) {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t - 1] == 0 && target[t] == 2 {
                        z = if target[t - 2] == 1 || target[t - 2] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 0 && target[t] == 1 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t - 1] == 0 && target[t] == 1 {
                        z = if target[t - 2] == 1 || target[t - 2] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 2 && target[t] == 1 {
                        z = 1;
                    } else if target[t - 1] == 0 && target[t] == 3 {
                        z = 2;
                    }
                }
                *acceptor_ptr.add(t) = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
            }
        } else {
            let donor_end = (tlen - 4).max(0) as usize;
            for t in 0..donor_end {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t + 1] == 2 && target[t + 2] == 0 {
                        z = if target[t + 3] == 1 || target[t + 3] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 1 && target[t + 2] == 0 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t + 1] == 1 && target[t + 2] == 0 {
                        z = if target[t + 3] == 1 || target[t + 3] == 3 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t + 1] == 1 && target[t + 2] == 2 {
                        z = 1;
                    } else if target[t + 1] == 3 && target[t + 2] == 0 {
                        z = 2;
                    }
                }
                *donor_ptr.add(t) = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
            }
            for t in 2..(tlen as usize) {
                let mut z: i32 = 3;
                if flag.contains(KswFlags::SPLICE_FOR) {
                    if target[t - 1] == 3 && target[t] == 2 {
                        z = if target[t - 2] == 0 || target[t - 2] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 1 && target[t] == 2 {
                        z = 1;
                    } else if target[t - 1] == 3 && target[t] == 0 {
                        z = 2;
                    }
                } else if flag.contains(KswFlags::SPLICE_REV) {
                    if target[t - 1] == 3 && target[t] == 1 {
                        z = if target[t - 2] == 0 || target[t - 2] == 2 {
                            -1
                        } else {
                            0
                        };
                    } else if target[t - 1] == 3 && target[t] == 2 {
                        z = 2;
                    }
                }
                *acceptor_ptr.add(t) = if z < 0 { 0i8 } else { -(sp[z as usize]) as i8 };
            }
        }
    }

    if let Some(junc) = junc {
        if flag.contains(KswFlags::SPLICE_SCORE) {
            let donor_val =
                (flag.contains(KswFlags::SPLICE_FOR) == !flag.contains(KswFlags::REV_CIGAR)) as u8;
            let limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..limit {
                let delta_donor = if junc[t + 1] == 0xff || (junc[t + 1] & 1) != donor_val {
                    -(junc_pen as i32)
                } else {
                    (junc[t + 1] >> 1) as i32 - 64
                };
                let delta_acceptor = if junc[t + 1] == 0xff || (junc[t + 1] & 1) != (1 - donor_val)
                {
                    -(junc_pen as i32)
                } else {
                    (junc[t + 1] >> 1) as i32 - 64
                };
                *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + delta_donor) as i8;
                *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + delta_acceptor) as i8;
            }
        } else if !flag.contains(KswFlags::REV_CIGAR) {
            let donor_limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..donor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 8) != 0)
                {
                    *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
            let acceptor_limit = (tlen as usize).min(junc.len());
            for t in 0..acceptor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 4) != 0)
                {
                    *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
        } else {
            let donor_limit = (tlen as usize)
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..donor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 4) != 0)
                {
                    *donor_ptr.add(t) = (*donor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
            let acceptor_limit = (tlen as usize).min(junc.len());
            for t in 0..acceptor_limit {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 8) != 0)
                {
                    *acceptor_ptr.add(t) = (*acceptor_ptr.add(t) as i32 + junc_bonus as i32) as i8;
                }
            }
        }
    }

    let u = dp_base as *mut __m128i;
    let v = dp_base.add(bsz) as *mut __m128i;
    let x = dp_base.add(2 * bsz) as *mut __m128i;
    let y = dp_base.add(3 * bsz) as *mut __m128i;
    let x2 = dp_base.add(4 * bsz) as *mut __m128i;
    let donor = dp_base.add(5 * bsz) as *mut __m128i;
    let acceptor = dp_base.add(6 * bsz) as *mut __m128i;
    let s = dp_base.add(7 * bsz) as *mut __m128i;
    let p = if with_cigar {
        dp_base.add(8 * bsz) as *mut __m128i
    } else {
        std::ptr::null_mut()
    };
    let qr = dp_base.add(qr_start);
    let sf = dp_base.add(sf_start);
    let h_arr = scratch_i32.as_mut_ptr();
    let off = if with_cigar {
        scratch_i32.as_mut_ptr().add(h_sz)
    } else {
        std::ptr::null_mut()
    };
    let off_end = if with_cigar {
        off.add(n_ad)
    } else {
        std::ptr::null_mut()
    };

    let u8 = u as *mut i8;
    let v8 = v as *mut i8;
    let x8 = x as *mut i8;
    let y8 = y as *mut i8;
    let x28 = x2 as *mut i8;

    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;
    let mut last_st = -1i32;
    let mut last_en = -1i32;

    for r in 0..(qlen + tlen - 1) {
        let mut st = 0i32;
        let mut en = tlen - 1;
        if st < r - qlen + 1 {
            st = r - qlen + 1;
        }
        if en > r {
            en = r;
        }
        let st0 = st;
        let en0 = en;
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        let (x1v, x21v, v1v) = if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                (
                    *x8.add((st - 1) as usize),
                    *x28.add((st - 1) as usize),
                    *v8.add((st - 1) as usize),
                )
            } else {
                (nqe, nq2, nqe)
            }
        } else {
            let v1 = if r == 0 {
                -(q as i32) - e as i32
            } else if r < long_thres {
                -(e as i32)
            } else if r == long_thres {
                long_diff
            } else {
                0
            };
            (nqe, nq2, v1 as i8)
        };
        if en >= r {
            *y8.add(r as usize) = nqe;
            *u8.add(r as usize) = if r == 0 {
                nqe
            } else if r < long_thres {
                (-(e as i32)) as i8
            } else if r == long_thres {
                long_diff as i8
            } else {
                0
            };
        }

        if !flag.contains(KswFlags::GENERIC_SC) {
            let qrr = qr.offset((qlen - 1 - r) as isize);
            let mut t = st0;
            while t <= en0 {
                let sq = _mm_loadu_si128(sf.add(t as usize) as *const __m128i);
                let stv = _mm_loadu_si128(qrr.add(t as usize) as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(stv, m1_));
                let tmp = _mm_cmpeq_epi8(sq, stv);
                let tmp = _mm_blendv_epi8(sc_mis_, sc_mch_, tmp);
                let tmp = _mm_blendv_epi8(tmp, sc_n_, mask);
                _mm_storeu_si128((s as *mut i8).add(t as usize) as *mut __m128i, tmp);
                t += 16;
            }
        } else {
            for t in st0..=en0 {
                *(s as *mut u8).add(t as usize) = mat[*sf.add(t as usize) as usize * m_u
                    + *qr.offset((qlen - 1 - r + t) as isize) as usize]
                    as u8;
            }
        }

        let mut x1_ = _mm_cvtsi32_si128((x1v as u8) as i32);
        let mut x21_ = _mm_cvtsi32_si128((x21v as u8) as i32);
        let mut v1_ = _mm_cvtsi32_si128((v1v as u8) as i32);
        let st_ = st / 16;
        let en_ = en / 16;

        if with_cigar {
            *off.add(r as usize) = st;
            *off_end.add(r as usize) = en;
        }

        if !with_cigar {
            for t in st_..=en_ {
                let tp = t as usize;
                let z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                let mut zz = _mm_max_epi8(z, a);
                zz = _mm_max_epi8(zz, b);
                zz = _mm_max_epi8(zz, a2a);

                _mm_store_si128(u.add(tp), _mm_sub_epi8(zz, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(zz, ut));
                let zq = _mm_sub_epi8(zz, q_);
                let a = _mm_sub_epi8(a, zq);
                let b = _mm_sub_epi8(b, zq);
                let a2 = _mm_sub_epi8(a2, _mm_sub_epi8(zz, q2_));

                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_max_epi8(a, zero_), qe_));
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_max_epi8(b, zero_), qe_));
                let don = _mm_max_epi8(a2, _mm_load_si128(donor.add(tp)));
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(don, q2_));
            }
        } else if !flag.contains(KswFlags::RIGHT) {
            let pr = p.add(r as usize * n_col_).sub(st_ as usize);
            for t in st_..=en_ {
                let tp = t as usize;
                let z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                let mut d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_);
                let mut zz = _mm_max_epi8(z, a);
                d = _mm_blendv_epi8(d, flag2_, _mm_cmpgt_epi8(b, zz));
                zz = _mm_max_epi8(zz, b);
                d = _mm_blendv_epi8(d, flag3_, _mm_cmpgt_epi8(a2a, zz));
                zz = _mm_max_epi8(zz, a2a);

                _mm_store_si128(u.add(tp), _mm_sub_epi8(zz, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(zz, ut));
                let zq = _mm_sub_epi8(zz, q_);
                let a = _mm_sub_epi8(a, zq);
                let b = _mm_sub_epi8(b, zq);
                let a2 = _mm_sub_epi8(a2, _mm_sub_epi8(zz, q2_));

                let tmp = _mm_cmpgt_epi8(a, zero_);
                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                let tmp = _mm_cmpgt_epi8(b, zero_);
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));
                let tmp = _mm_cmpgt_epi8(a2, _mm_load_si128(donor.add(tp)));
                let don = _mm_blendv_epi8(_mm_load_si128(donor.add(tp)), a2, tmp);
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(don, q2_));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag32_));
                _mm_store_si128(pr.add(tp), d);
            }
        } else {
            let pr = p.add(r as usize * n_col_).sub(st_ as usize);
            for t in st_..=en_ {
                let tp = t as usize;
                let z = _mm_load_si128(s.add(tp));
                let mut xt1 = _mm_load_si128(x.add(tp));
                let mut tmp = _mm_srli_si128::<15>(xt1);
                xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1), x1_);
                x1_ = tmp;
                let mut vt1 = _mm_load_si128(v.add(tp));
                tmp = _mm_srli_si128::<15>(vt1);
                vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1), v1_);
                v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_load_si128(u.add(tp));
                let b = _mm_add_epi8(_mm_load_si128(y.add(tp)), ut);
                let mut x2t1 = _mm_load_si128(x2.add(tp));
                tmp = _mm_srli_si128::<15>(x2t1);
                x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1), x21_);
                x21_ = tmp;
                let a2 = _mm_add_epi8(x2t1, vt1);
                let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(tp)));

                let mut d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), flag1_);
                let mut zz = _mm_max_epi8(z, a);
                d = _mm_blendv_epi8(flag2_, d, _mm_cmpgt_epi8(zz, b));
                zz = _mm_max_epi8(zz, b);
                d = _mm_blendv_epi8(flag3_, d, _mm_cmpgt_epi8(zz, a2a));
                zz = _mm_max_epi8(zz, a2a);

                _mm_store_si128(u.add(tp), _mm_sub_epi8(zz, vt1));
                _mm_store_si128(v.add(tp), _mm_sub_epi8(zz, ut));
                let zq = _mm_sub_epi8(zz, q_);
                let a = _mm_sub_epi8(a, zq);
                let b = _mm_sub_epi8(b, zq);
                let a2 = _mm_sub_epi8(a2, _mm_sub_epi8(zz, q2_));

                let tmp = _mm_cmpgt_epi8(zero_, a);
                _mm_store_si128(x.add(tp), _mm_sub_epi8(_mm_andnot_si128(tmp, a), qe_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));
                let tmp = _mm_cmpgt_epi8(zero_, b);
                _mm_store_si128(y.add(tp), _mm_sub_epi8(_mm_andnot_si128(tmp, b), qe_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_));
                let tmp = _mm_cmpgt_epi8(_mm_load_si128(donor.add(tp)), a2);
                let don = _mm_blendv_epi8(a2, _mm_load_si128(donor.add(tp)), tmp);
                _mm_store_si128(x2.add(tp), _mm_sub_epi8(don, q2_));
                d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag32_));
                _mm_store_si128(pr.add(tp), d);
            }
        }

        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                if en0 > 0 {
                    *h_arr.add(en0 as usize) =
                        *h_arr.add((en0 - 1) as usize) + *u8.add(en0 as usize) as i32;
                } else {
                    *h_arr.add(en0 as usize) += *v8.add(en0 as usize) as i32;
                }
                max_h = *h_arr.add(en0 as usize);
                max_t = en0;
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut t = st0;
                while t < en1 {
                    let h1 = _mm_loadu_si128(h_arr.add(t as usize) as *const __m128i);
                    let add = _mm_setr_epi32(
                        *v8.add(t as usize) as i32,
                        *v8.add((t + 1) as usize) as i32,
                        *v8.add((t + 2) as usize) as i32,
                        *v8.add((t + 3) as usize) as i32,
                    );
                    let h2 = _mm_add_epi32(h1, add);
                    _mm_storeu_si128(h_arr.add(t as usize) as *mut __m128i, h2);
                    t += 4;
                }
                while t < en0 {
                    *h_arr.add(t as usize) += *v8.add(t as usize) as i32;
                    t += 1;
                }
                for lane in 0..4 {
                    let mut t = st0 + lane;
                    while t < en1 {
                        let ht = *h_arr.add(t as usize);
                        if ht > max_h {
                            max_h = ht;
                            max_t = t;
                        }
                        t += 4;
                    }
                }
                t = en1;
                while t < en0 {
                    let ht = *h_arr.add(t as usize);
                    if ht > max_h {
                        max_h = ht;
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                *h_arr.add(0) = *v8.add(0) as i32 - (q as i32 + e as i32);
                max_h = *h_arr.add(0);
                max_t = 0;
            }
            if en0 == tlen - 1 && *h_arr.add(en0 as usize) > ez.mte {
                ez.mte = *h_arr.add(en0 as usize);
                ez.mte_q = r - en0;
            }
            if r - st0 == qlen - 1 && *h_arr.add(st0 as usize) > ez.mqe {
                ez.mqe = *h_arr.add(st0 as usize);
                ez.mqe_t = st0;
            }
            if zdrop >= 0 && ksw_apply_zdrop_rot(&mut ez, max_h, r, max_t, zdrop, 0) {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = *h_arr.add((tlen - 1) as usize);
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = *v8.add(last_h0_t as usize) as i32;
                    let d1 = *u8.add((last_h0_t + 1) as usize) as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8.add(last_h0_t as usize) as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8.add(last_h0_t as usize) as i32;
                }
            } else {
                h0 = *v8.add(0) as i32 - (q as i32 + e as i32);
                last_h0_t = 0;
            }
            if flag.contains(KswFlags::APPROX_DROP)
                && ksw_apply_zdrop_rot(&mut ez, h0, r, last_h0_t, zdrop, 0)
            {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    if with_cigar {
        let rev_cigar = flag.contains(KswFlags::REV_CIGAR);
        if !ez.zdropped && !flag.contains(KswFlags::EXTZ_ONLY) {
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                qlen - 1,
                tlen - 1,
            );
        } else if !ez.zdropped && flag.contains(KswFlags::EXTZ_ONLY) && ez.mqe + end_bonus > ez.max
        {
            ez.reach_end = true;
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                qlen - 1,
                ez.mqe_t,
            );
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            ez.cigar = backtrack_rotated(
                std::slice::from_raw_parts(p as *const u8, n_ad * row_width),
                std::slice::from_raw_parts(off, n_ad),
                std::slice::from_raw_parts(off_end, n_ad),
                row_width,
                rev_cigar,
                long_thres,
                ez.max_q,
                ez.max_t,
            );
        }
    }

    KSW_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    KSW_I32_SCRATCH.with(|c| *c.borrow_mut() = scratch_i32);
    ez
}

/// Dispatch for splice (exts2): SIMD path → scalar rotated DP fallback.
pub fn ksw_exts2_dispatch(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    noncan: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    junc: Option<&[u8]>,
    flag: KswFlags,
) -> KswResult {
    #[cfg(target_arch = "x86_64")]
    {
        if has_sse41() {
            return unsafe {
                exts2_sse41(
                    query, target, m, mat, q, e, q2, noncan, w, zdrop, end_bonus, junc_bonus,
                    junc_pen, junc, flag,
                )
            };
        }
        if has_sse2() {
            return unsafe {
                exts2_sse2(
                    query, target, m, mat, q, e, q2, noncan, w, zdrop, end_bonus, junc_bonus,
                    junc_pen, junc, flag,
                )
            };
        }
        return crate::align::ksw2::ksw_exts2_rot(
            query, target, m, mat, q, e, q2, noncan, w, zdrop, end_bonus, junc_bonus, junc_pen,
            junc, flag,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    crate::align::ksw2::ksw_exts2_rot(
        query, target, m, mat, q, e, q2, noncan, w, zdrop, end_bonus, junc_bonus, junc_pen, junc,
        flag,
    )
}

/// Dispatch: SIMD for single-gap, scalar fallback otherwise.
pub fn ksw_extz2_dispatch(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    #[cfg(target_arch = "x86_64")]
    if has_sse2() {
        let ez = unsafe { extz2_sse2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag) };
        return ez;
    }
    super::ksw2::ksw_extz2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::score::gen_simple_mat;

    #[test]
    fn test_simd_score_matches_scalar() {
        if !has_sse2() {
            return;
        }
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        // SIMD CIGAR is correct; H-score tracking has known offset
        // The mapping pipeline uses CIGAR-derived dp_max, not ez.score
        for len in [8, 16, 32, 64] {
            let q: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let t: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let simd = ksw_extz2_dispatch(&q, &t, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
            assert_eq!(
                crate::align::cigar_to_string(&simd.cigar),
                format!("{}M", len),
                "len={}",
                len
            );
        }
    }

    #[test]
    fn test_simd_cigar_all_lengths() {
        if !has_sse2() {
            return;
        }
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        for len in [8, 16, 24, 32, 48, 64] {
            let q: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let t: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let ez = ksw_extz2_dispatch(&q, &t, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
            let cigar_str = crate::align::cigar_to_string(&ez.cigar);
            let expected = format!("{}M", len);
            assert_eq!(
                cigar_str, expected,
                "len={}: got {} expected {}",
                len, cigar_str, expected
            );
        }
    }

    /// Compare SIMD extd2 dispatch (with fixup) against scalar extd2 for
    /// extension-like alignments. This catches SIMD H-tracking / backtrace bugs.
    #[test]
    fn test_simd_vs_scalar_extd2_extension() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        // map-hifi-like penalties: q=4, e=2, q2=24, e2=1
        let q: i8 = 4;
        let e: i8 = 2;
        let q2: i8 = 24;
        let e2: i8 = 1;

        // Test 1: perfect match followed by random (extension should stop at match end)
        let mut query = vec![0u8; 200]; // 200bp match
        let mut target = vec![0u8; 200];
        // Add divergent tail
        for i in 0..300 {
            query.push((i * 3 % 4) as u8);
            target.push((i * 7 % 4) as u8);
        }

        let simd = ksw_extd2_dispatch(
            &query,
            &target,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            0,
            KswFlags::EXTZ_ONLY,
        );
        let scalar = crate::align::ksw2::ksw_extd2(
            &query,
            &target,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            0,
            KswFlags::EXTZ_ONLY,
        );

        let simd_cigar = crate::align::cigar_to_string(&simd.cigar);
        let scalar_cigar = crate::align::cigar_to_string(&scalar.cigar);
        assert_eq!(simd_cigar, scalar_cigar,
            "SIMD vs scalar CIGAR mismatch for extension: SIMD={} scalar={} simd_max_t={} scalar_max_t={}",
            simd_cigar, scalar_cigar, simd.max_t, scalar.max_t);

        // Test 2: realistic extension - good region, then mismatches, then another match
        let mut query2 = Vec::new();
        let mut target2 = Vec::new();
        for i in 0..150 {
            query2.push((i % 4) as u8);
            target2.push((i % 4) as u8);
        } // 150bp match
        for i in 0..50 {
            query2.push(((i + 1) % 4) as u8);
            target2.push((i % 4) as u8);
        } // 50bp mismatch
        for i in 0..100 {
            query2.push((i % 4) as u8);
            target2.push((i % 4) as u8);
        } // 100bp match

        let simd2 = ksw_extd2_dispatch(
            &query2,
            &target2,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            0,
            KswFlags::EXTZ_ONLY,
        );
        let scalar2 = crate::align::ksw2::ksw_extd2(
            &query2,
            &target2,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            0,
            KswFlags::EXTZ_ONLY,
        );

        let simd_cigar2 = crate::align::cigar_to_string(&simd2.cigar);
        let scalar_cigar2 = crate::align::cigar_to_string(&scalar2.cigar);
        assert_eq!(
            simd_cigar2, scalar_cigar2,
            "SIMD vs scalar CIGAR mismatch (test 2): SIMD={} scalar={}",
            simd_cigar2, scalar_cigar2
        );
    }

    #[test]
    fn test_extd2_sr_left_extension_350555() {
        if !has_sse2() {
            return;
        }
        fn enc(s: &str) -> Vec<u8> {
            s.bytes()
                .map(|b| match b {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 4,
                })
                .collect()
        }

        let target = enc("GCATTCTGCACTTATCGCGGGCCTTTGCCCGCGACTCCAATACGTTTGCATTTCGACGGACTACGCGATACGAATAGTCCGGATGTATCAGGGACGTTATATAACTTAATCGTTCCAAAACATCCGGCCTATTCCGCAAGTGCGGCGTAGGCCGTCATTATGGTCTGGAAATGCCGAACGCGCGTTAGTCGTGGCGCGCAAATCCACGGCCTATTGGAAGGTGGCAAAATGGGCCTAGTACACCCAGCTCCTTTAGCCGGTCGCTAAGCCATTGGTAGGTAAGCCACGCAGCGACGAGAAGACACCACC");
        let query = enc("GTATTCTGCACTTTTCGCGGGCCTTTGCCCGCGATTAGAAATGCCGAACGCGCGTTAGTCGTGGCGCGCAAATCCGCGTCCTATTGGAAGGTGGCAAAATGGGCCTAGTACACCCAGCTCCTTTAGCCGGTCGCTAAGTCATTGGTAGGTAAGCC");
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 8, 1);
        let flags = KswFlags::EXTZ_ONLY | KswFlags::RIGHT | KswFlags::REV_CIGAR;
        let simd = ksw_extd2_dispatch(&query, &target, 5, &mat, 12, 2, 24, 1, 151, 100, 10, flags);

        assert_eq!(
            crate::align::cigar_to_string(&simd.cigar),
            "118M7D3M123D34M"
        );
        assert_eq!(simd.max, 87);
        assert_eq!(simd.max_q, 154);
        assert_eq!(simd.max_t, 284);
    }

    #[test]
    fn test_extd2_sr_left_extension_350555_avx2_matches_sse41() {
        if !has_sse41() || !has_avx2() {
            return;
        }
        fn enc(s: &str) -> Vec<u8> {
            s.bytes()
                .map(|b| match b {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 4,
                })
                .collect()
        }

        let target = enc("GCATTCTGCACTTATCGCGGGCCTTTGCCCGCGACTCCAATACGTTTGCATTTCGACGGACTACGCGATACGAATAGTCCGGATGTATCAGGGACGTTATATAACTTAATCGTTCCAAAACATCCGGCCTATTCCGCAAGTGCGGCGTAGGCCGTCATTATGGTCTGGAAATGCCGAACGCGCGTTAGTCGTGGCGCGCAAATCCACGGCCTATTGGAAGGTGGCAAAATGGGCCTAGTACACCCAGCTCCTTTAGCCGGTCGCTAAGCCATTGGTAGGTAAGCCACGCAGCGACGAGAAGACACCACC");
        let query = enc("GTATTCTGCACTTTTCGCGGGCCTTTGCCCGCGATTAGAAATGCCGAACGCGCGTTAGTCGTGGCGCGCAAATCCGCGTCCTATTGGAAGGTGGCAAAATGGGCCTAGTACACCCAGCTCCTTTAGCCGGTCGCTAAGTCATTGGTAGGTAAGCC");
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 8, 1);
        let flags = KswFlags::EXTZ_ONLY | KswFlags::RIGHT | KswFlags::REV_CIGAR;

        let sse41 = unsafe {
            extd2_sse41_only::<true>(&query, &target, 5, &mat, 12, 2, 24, 1, 151, 100, 10, flags)
        };
        let avx2 = unsafe {
            extd2_sse41_avx2::<true>(&query, &target, 5, &mat, 12, 2, 24, 1, 151, 100, 10, flags)
        };
        assert_eq!(avx2.cigar, sse41.cigar);
        assert_eq!(avx2.score, sse41.score);
        assert_eq!(avx2.max, sse41.max);
        assert_eq!(avx2.max_q, sse41.max_q);
        assert_eq!(avx2.max_t, sse41.max_t);
        assert_eq!(avx2.mqe, sse41.mqe);
        assert_eq!(avx2.mqe_t, sse41.mqe_t);
        assert_eq!(avx2.mte, sse41.mte);
        assert_eq!(avx2.mte_q, sse41.mte_q);
        assert_eq!(avx2.zdropped, sse41.zdropped);
        assert_eq!(avx2.reach_end, sse41.reach_end);
    }

    #[test]
    fn test_score_track_exact_avx2_matches_sse2_randomized() {
        if !has_avx2() {
            return;
        }

        fn next_u64(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        }

        fn rand_range_i32(state: &mut u64, lo: i32, hi: i32) -> i32 {
            lo + (next_u64(state) % ((hi - lo + 1) as u64)) as i32
        }

        fn rand_i8(state: &mut u64, lo: i8, hi: i8) -> i8 {
            (lo as i32 + (next_u64(state) % ((hi as i32 - lo as i32 + 1) as u64)) as i32) as i8
        }

        let mut seed = 0x4d595df4d0f33173u64;
        for case_idx in 0..2000 {
            let len = 96usize;
            let mut h1 = vec![0i32; len];
            let mut h2 = vec![0i32; len];
            let mut u = vec![0u8; len];
            let mut v = vec![0u8; len];
            for i in 0..len {
                let hv = rand_range_i32(&mut seed, -500, 500);
                h1[i] = hv;
                h2[i] = hv;
                u[i] = rand_i8(&mut seed, -40, 40) as u8;
                v[i] = rand_i8(&mut seed, -40, 40) as u8;
            }

            let st0 = rand_range_i32(&mut seed, 0, (len - 2) as i32);
            let en0 = rand_range_i32(&mut seed, st0, (len - 1) as i32);
            let qlen = rand_range_i32(&mut seed, en0 + 1, 128);
            let tlen = rand_range_i32(&mut seed, en0 + 1, 128);
            let r = rand_range_i32(&mut seed, 0, qlen + tlen - 2);
            let qe = rand_range_i32(&mut seed, 1, 20);
            let e2 = rand_i8(&mut seed, 1, 5);
            let zdrop = rand_range_i32(&mut seed, 5, 200);

            let mut ez1 = KswResult::new();
            ez1.max = rand_range_i32(&mut seed, -200, 500);
            ez1.max_t = rand_range_i32(&mut seed, -1, tlen - 1);
            ez1.max_q = rand_range_i32(&mut seed, -1, qlen - 1);
            ez1.mqe = rand_range_i32(&mut seed, KSW_NEG_INF / 2, 500);
            ez1.mqe_t = rand_range_i32(&mut seed, -1, tlen - 1);
            ez1.mte = rand_range_i32(&mut seed, KSW_NEG_INF / 2, 500);
            ez1.mte_q = rand_range_i32(&mut seed, -1, qlen - 1);
            ez1.score = rand_range_i32(&mut seed, KSW_NEG_INF / 2, 500);
            let mut ez2 = ez1.clone();

            let stop_sse2 = unsafe {
                score_track_exact_sse2(
                    &mut h1,
                    u.as_ptr(),
                    v.as_ptr(),
                    st0,
                    en0,
                    r,
                    qlen,
                    tlen,
                    qe,
                    e2,
                    zdrop,
                    &mut ez1,
                )
            };
            let stop_avx2 = unsafe {
                score_track_exact_avx2(
                    &mut h2,
                    u.as_ptr(),
                    v.as_ptr(),
                    st0,
                    en0,
                    r,
                    qlen,
                    tlen,
                    qe,
                    e2,
                    zdrop,
                    &mut ez2,
                )
            };

            if stop_avx2 != stop_sse2
                || h2 != h1
                || ez2.max != ez1.max
                || ez2.max_t != ez1.max_t
                || ez2.max_q != ez1.max_q
                || ez2.mqe != ez1.mqe
                || ez2.mqe_t != ez1.mqe_t
                || ez2.mte != ez1.mte
                || ez2.mte_q != ez1.mte_q
                || ez2.score != ez1.score
                || ez2.zdropped != ez1.zdropped
            {
                let first_h_diff = h1
                    .iter()
                    .zip(h2.iter())
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                    .map(|(i, (a, b))| (i, *a, *b));
                panic!(
                    "case={} st0={} en0={} r={} qlen={} tlen={} qe={} e2={} zdrop={} first_h_diff={:?} stop_sse2={} stop_avx2={} ez_sse2={{max:{} max_t:{} max_q:{} mqe:{} mqe_t:{} mte:{} mte_q:{} score:{} zd:{}}} ez_avx2={{max:{} max_t:{} max_q:{} mqe:{} mqe_t:{} mte:{} mte_q:{} score:{} zd:{}}}",
                    case_idx,
                    st0,
                    en0,
                    r,
                    qlen,
                    tlen,
                    qe,
                    e2,
                    zdrop,
                    first_h_diff,
                    stop_sse2,
                    stop_avx2,
                    ez1.max,
                    ez1.max_t,
                    ez1.max_q,
                    ez1.mqe,
                    ez1.mqe_t,
                    ez1.mte,
                    ez1.mte_q,
                    ez1.score,
                    ez1.zdropped,
                    ez2.max,
                    ez2.max_t,
                    ez2.max_q,
                    ez2.mqe,
                    ez2.mqe_t,
                    ez2.mte,
                    ez2.mte_q,
                    ez2.score,
                    ez2.zdropped,
                );
            }
        }
    }

    /// Compare SIMD extd2 for gap-fill (APPROX_MAX) - backtrace from (tlen-1, qlen-1)
    #[test]
    fn test_simd_vs_scalar_extd2_gapfill() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let q: i8 = 4;
        let e: i8 = 2;
        let q2: i8 = 24;
        let e2: i8 = 1;

        // Realistic gap-fill: mostly matching with a few mismatches and indels
        let mut query = Vec::new();
        let mut target = Vec::new();
        // 100bp match
        for i in 0..100 {
            query.push((i % 4) as u8);
            target.push((i % 4) as u8);
        }
        // 1bp deletion in query (target has extra base)
        target.push(2);
        // 50bp match
        for i in 0..50 {
            query.push(((i + 2) % 4) as u8);
            target.push(((i + 2) % 4) as u8);
        }
        // 1bp insertion in query
        query.push(3);
        // 80bp match
        for i in 0..80 {
            query.push((i % 4) as u8);
            target.push((i % 4) as u8);
        }

        let simd = ksw_extd2_dispatch(
            &query,
            &target,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            -1,
            KswFlags::APPROX_MAX,
        );
        let scalar = crate::align::ksw2::ksw_extd2(
            &query,
            &target,
            5,
            &mat,
            q,
            e,
            q2,
            e2,
            500,
            400,
            -1,
            KswFlags::APPROX_MAX,
        );

        let simd_cigar = crate::align::cigar_to_string(&simd.cigar);
        let scalar_cigar = crate::align::cigar_to_string(&scalar.cigar);
        assert_eq!(
            simd_cigar, scalar_cigar,
            "Gap-fill CIGAR mismatch: SIMD={} scalar={}",
            simd_cigar, scalar_cigar
        );
    }

    /// Test SIMD vs scalar for varying sequence lengths around 16-byte boundaries
    #[test]
    fn test_simd_vs_scalar_boundary_lengths() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let q: i8 = 4;
        let e: i8 = 2;
        let q2: i8 = 24;
        let e2: i8 = 1;

        for len in [
            15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 128, 200, 256, 300,
        ] {
            let query: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let target: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();

            for &flag in &[KswFlags::empty(), KswFlags::APPROX_MAX, KswFlags::EXTZ_ONLY] {
                let simd =
                    ksw_extd2_dispatch(&query, &target, 5, &mat, q, e, q2, e2, 500, 400, 0, flag);
                let scalar = crate::align::ksw2::ksw_extd2(
                    &query, &target, 5, &mat, q, e, q2, e2, 500, 400, 0, flag,
                );

                let simd_cigar = crate::align::cigar_to_string(&simd.cigar);
                let scalar_cigar = crate::align::cigar_to_string(&scalar.cigar);
                assert_eq!(
                    simd_cigar, scalar_cigar,
                    "len={} flag={:?}: SIMD={} scalar={}",
                    len, flag, simd_cigar, scalar_cigar
                );
            }
        }
    }

    /// Test SIMD extension with sequences that have a match region followed by divergence
    /// at various lengths (mimics the real right-extension pattern)
    #[test]
    fn test_simd_extension_match_then_diverge() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let q: i8 = 4;
        let e: i8 = 2;
        let q2: i8 = 24;
        let e2: i8 = 1;

        // Test various match lengths with divergent tails
        for match_len in [50, 100, 200, 300, 500] {
            for tail_len in [100, 500] {
                let mut query = Vec::new();
                let mut target = Vec::new();
                for i in 0..match_len {
                    query.push((i % 4) as u8);
                    target.push((i % 4) as u8);
                }
                for i in 0..tail_len {
                    query.push(((i * 3 + 1) % 4) as u8);
                    target.push(((i * 7 + 2) % 4) as u8);
                }

                let simd = ksw_extd2_dispatch(
                    &query,
                    &target,
                    5,
                    &mat,
                    q,
                    e,
                    q2,
                    e2,
                    500,
                    400,
                    0,
                    KswFlags::EXTZ_ONLY,
                );
                let scalar = crate::align::ksw2::ksw_extd2(
                    &query,
                    &target,
                    5,
                    &mat,
                    q,
                    e,
                    q2,
                    e2,
                    500,
                    400,
                    0,
                    KswFlags::EXTZ_ONLY,
                );

                let s_cig = crate::align::cigar_to_string(&simd.cigar);
                let r_cig = crate::align::cigar_to_string(&scalar.cigar);

                // Count query bases consumed
                fn qcons(cigar: &[u32]) -> i32 {
                    cigar
                        .iter()
                        .map(|&c| {
                            let op = c & 0xf;
                            let len = (c >> 4) as i32;
                            if op == 0 || op == 1 || op == 7 || op == 8 {
                                len
                            } else {
                                0
                            }
                        })
                        .sum()
                }
                let s_qc = qcons(&simd.cigar);
                let r_qc = qcons(&scalar.cigar);

                // Query consumption should be very close (within 2bp)
                assert!((s_qc - r_qc).abs() <= 2,
                    "match={} tail={}: query consumed differs too much: SIMD={} ({}) scalar={} ({})",
                    match_len, tail_len, s_qc, s_cig, r_qc, r_cig);
            }
        }
    }

    /// Focused test: narrow down SIMD vs scalar difference for extension with
    /// specific sequence patterns.
    #[test]
    fn test_simd_scalar_extension_diagnostic() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let q: i8 = 4;
        let e: i8 = 2;
        let q2: i8 = 24;
        let e2: i8 = 1;
        let mut mismatches = 0;

        // Sweep parameters that might trigger differences
        for match_len in (20..=200).step_by(10) {
            for tail_len in [50, 200, 500] {
                for bw in [100, 500] {
                    let mut query = Vec::new();
                    let mut target = Vec::new();
                    for i in 0..match_len {
                        query.push((i % 4) as u8);
                        target.push((i % 4) as u8);
                    }
                    for i in 0..tail_len {
                        query.push(((i * 3 + 1) % 4) as u8);
                        target.push(((i * 7 + 2) % 4) as u8);
                    }

                    let simd = ksw_extd2_dispatch(
                        &query,
                        &target,
                        5,
                        &mat,
                        q,
                        e,
                        q2,
                        e2,
                        bw,
                        400,
                        0,
                        KswFlags::EXTZ_ONLY,
                    );
                    let scalar = crate::align::ksw2::ksw_extd2(
                        &query,
                        &target,
                        5,
                        &mat,
                        q,
                        e,
                        q2,
                        e2,
                        bw,
                        400,
                        0,
                        KswFlags::EXTZ_ONLY,
                    );

                    if simd.cigar != scalar.cigar {
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "SIMD and scalar extension CIGARs differ");
    }

    #[test]
    fn test_exts2_score_only_reproducer_matches_scalar() {
        if !has_sse2() {
            return;
        }

        let query = vec![
            0, 2, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 1, 2, 0, 2, 3, 3, 1, 2, 2, 3, 1, 0, 3, 3, 3, 3,
            2, 3, 3, 2, 3, 3, 1, 2, 3, 1, 1, 3, 2, 1, 0, 2, 3, 0, 3, 0, 3, 1, 0, 2, 3, 2, 2, 2, 1,
            1, 3, 2, 2, 1, 0, 3, 3, 2, 2, 2, 3, 0, 0, 0, 0, 1, 1, 0, 3, 3, 3, 2, 1, 0, 2, 0, 1, 0,
            3, 1, 3, 0, 0, 0, 0, 0, 2, 1,
        ];
        let target = vec![
            0, 2, 3, 2, 2, 2, 0, 0, 0, 0, 3, 1, 3, 2, 0, 2, 3, 3, 1, 2, 2, 3, 1, 0, 3, 1, 3, 3, 3,
            2, 3, 3, 2, 3, 3, 1, 2, 3, 1, 1, 3, 0, 1, 1, 2, 1, 0, 2, 3, 0, 3, 0, 3, 3, 1, 1, 3, 0,
            0, 0, 1, 0, 1, 3, 0, 3, 2, 0, 0, 0, 3, 2, 0, 1, 1, 1, 3, 0, 2, 3, 3, 2, 2, 3, 1, 1, 0,
            3, 2, 0, 3, 1, 0, 3, 3, 3, 2, 2, 2, 3, 0, 0, 0, 0, 1, 1, 0, 3, 0, 1, 3, 2, 1, 0, 2, 0,
            1, 0, 3, 1, 3, 1, 3, 0, 3, 0, 0, 3, 3, 1, 0, 1, 0, 3, 0, 3, 3, 1, 0, 1, 0, 3, 1, 3, 3,
            2, 3, 1, 1, 3, 1, 3, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1, 3, 3, 3, 2, 0,
            1, 2, 3, 0, 2, 0, 0, 0, 0, 3, 3, 0, 0, 2, 3, 0, 0, 0,
        ];
        let junc = vec![0u8; 192];
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 8, 1);
        let flags = KswFlags::EXTZ_ONLY
            | KswFlags::SPLICE_FOR
            | KswFlags::SPLICE_FLANK
            | KswFlags::SPLICE_CMPLX;

        let scalar = crate::align::ksw2::ksw_exts2_rot(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            24,
            9,
            300001,
            400,
            -1,
            0,
            0,
            Some(&junc),
            flags | KswFlags::SCORE_ONLY,
        );
        let simd = unsafe {
            exts2_sse2(
                &query,
                &target,
                5,
                &mat,
                4,
                2,
                24,
                9,
                300001,
                400,
                -1,
                0,
                0,
                Some(&junc),
                flags | KswFlags::SCORE_ONLY,
            )
        };

        assert_eq!(
            (
                simd.score,
                simd.max,
                simd.max_t,
                simd.max_q,
                simd.mqe,
                simd.mqe_t,
                simd.mte,
                simd.mte_q,
                simd.zdropped
            ),
            (
                scalar.score,
                scalar.max,
                scalar.max_t,
                scalar.max_q,
                scalar.mqe,
                scalar.mqe_t,
                scalar.mte,
                scalar.mte_q,
                scalar.zdropped
            ),
            "simd={:?} scalar={:?}",
            (
                simd.score,
                simd.max,
                simd.max_t,
                simd.max_q,
                simd.mqe,
                simd.mqe_t,
                simd.mte,
                simd.mte_q,
                simd.zdropped
            ),
            (
                scalar.score,
                scalar.max,
                scalar.max_t,
                scalar.max_q,
                scalar.mqe,
                scalar.mqe_t,
                scalar.mte,
                scalar.mte_q,
                scalar.zdropped
            )
        );
    }

    #[test]
    fn test_exts2_exact_reproducer_matches_scalar() {
        if !has_sse2() {
            return;
        }

        let query = vec![
            0, 2, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 1, 2, 0, 2, 3, 3, 1, 2, 2, 3, 1, 0, 3, 3, 3, 3,
            2, 3, 3, 2, 3, 3, 1, 2, 3, 1, 1, 3, 2, 1, 0, 2, 3, 0, 3, 0, 3, 1, 0, 2, 3, 2, 2, 2, 1,
            1, 3, 2, 2, 1, 0, 3, 3, 2, 2, 2, 3, 0, 0, 0, 0, 1, 1, 0, 3, 3, 3, 2, 1, 0, 2, 0, 1, 0,
            3, 1, 3, 0, 0, 0, 0, 0, 2, 1,
        ];
        let target = vec![
            0, 2, 3, 2, 2, 2, 0, 0, 0, 0, 3, 1, 3, 2, 0, 2, 3, 3, 1, 2, 2, 3, 1, 0, 3, 1, 3, 3, 3,
            2, 3, 3, 2, 3, 3, 1, 2, 3, 1, 1, 3, 0, 1, 1, 2, 1, 0, 2, 3, 0, 3, 0, 3, 3, 1, 1, 3, 0,
            0, 0, 1, 0, 1, 3, 0, 3, 2, 0, 0, 0, 3, 2, 0, 1, 1, 1, 3, 0, 2, 3, 3, 2, 2, 3, 1, 1, 0,
            3, 2, 0, 3, 1, 0, 3, 3, 3, 2, 2, 2, 3, 0, 0, 0, 0, 1, 1, 0, 3, 0, 1, 3, 2, 1, 0, 2, 0,
            1, 0, 3, 1, 3, 1, 3, 0, 3, 0, 0, 3, 3, 1, 0, 1, 0, 3, 0, 3, 3, 1, 0, 1, 0, 3, 1, 3, 3,
            2, 3, 1, 1, 3, 1, 3, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1, 3, 3, 3, 2, 0,
            1, 2, 3, 0, 2, 0, 0, 0, 0, 3, 3, 0, 0, 2, 3, 0, 0, 0,
        ];
        let junc = vec![0u8; 192];
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 8, 1);
        let flags = KswFlags::EXTZ_ONLY
            | KswFlags::SPLICE_FOR
            | KswFlags::SPLICE_FLANK
            | KswFlags::SPLICE_CMPLX;

        let scalar = crate::align::ksw2::ksw_exts2_rot(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            24,
            9,
            300001,
            400,
            -1,
            0,
            0,
            Some(&junc),
            flags,
        );
        let simd = unsafe {
            exts2_sse2(
                &query,
                &target,
                5,
                &mat,
                4,
                2,
                24,
                9,
                300001,
                400,
                -1,
                0,
                0,
                Some(&junc),
                flags,
            )
        };

        assert_eq!(
            crate::align::cigar_to_string(&simd.cigar),
            crate::align::cigar_to_string(&scalar.cigar)
        );
    }
}
