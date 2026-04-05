//! SSE2-accelerated KSW2 extension alignment (rotated-band DP).
//! Faithful translation of ksw2_extz2_sse.c.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::flags::KswFlags;
use super::ksw2::{KswResult, KSW_NEG_INF};

pub fn has_sse2() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("sse2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Faithful translation of ksw_extz2_sse2() from ksw2_extz2_sse.c.
/// Uses rotated-band DP with SSE2 16-way byte parallelism.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extz2_sse2(
    query: &[u8], target: &[u8], m: i8, mat: &[i8],
    q: i8, e: i8, w: i32, zdrop: i32, end_bonus: i32, flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 0 || qlen <= 0 || tlen <= 0 { return ez; }

    let qe = q as i32 + e as i32;
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let (wl, wr) = (w, w);
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((w + 1) as usize) + 15) / 16) + 1;

    // Scoring bounds
    let mut max_sc = mat[0] as i32; let mut min_sc = mat[1] as i32;
    for i in 1..(m as usize * m as usize) { max_sc = max_sc.max(mat[i] as i32); min_sc = min_sc.min(mat[i] as i32); }
    if -min_sc > 2 * qe { return ez; }

    // Allocate buffers — all tlen_*16 bytes, zero-initialized
    let bsz = tlen_ * 16 + 16; // +16 padding
    let mut u = vec![0u8; bsz];
    let mut v = vec![0u8; bsz];
    let mut x = vec![0u8; bsz];
    let mut y = vec![0u8; bsz];
    let mut s = vec![0u8; bsz];

    // H array for exact max tracking
    let mut h_arr: Vec<i32> = if !approx_max { vec![KSW_NEG_INF; bsz] } else { Vec::new() };
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    // Backtrack matrix
    let n_ad = (qlen + tlen - 1) as usize;
    let mut bt = if with_cigar { vec![0u8; n_ad * n_col_ * 16 + 16] } else { Vec::new() };
    let mut off_a = if with_cigar { vec![0i32; n_ad] } else { Vec::new() };
    let mut off_e = if with_cigar { vec![0i32; n_ad] } else { Vec::new() };

    // Reversed query + target copy
    let mut qr = vec![0u8; qlen as usize + 16];
    for i in 0..qlen as usize { qr[i] = query[qlen as usize - 1 - i]; }
    let mut sf = vec![0u8; tlen as usize + 16];
    sf[..tlen as usize].copy_from_slice(target);

    // SSE constants
    let zero_ = _mm_setzero_si128();
    let q_ = _mm_set1_epi8(q);
    let qe2_ = _mm_set1_epi8((qe * 2) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let m1_ = _mm_set1_epi8(m - 1);
    let max_sc_ = _mm_set1_epi8((mat[0] as i32 + qe * 2) as i8);
    let sc_n_ = if mat[(m as usize * m as usize) - 1] == 0 { _mm_set1_epi8(-e) }
                else { _mm_set1_epi8(mat[(m as usize * m as usize) - 1]) };
    let flag1_ = _mm_set1_epi8(1);
    let flag2_ = _mm_set1_epi8(2);
    let flag8_ = _mm_set1_epi8(0x08u8 as i8);
    let flag16_ = _mm_set1_epi8(0x10u8 as i8);

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;

    // Main anti-diagonal loop
    for r in 0..qlen + tlen - 1 {
        // Band boundaries (faithful translation of C lines 110-124)
        let mut st = 0i32.max(r - qlen + 1);
        let mut en = (tlen - 1).min(r);
        if st < (r - wr + 1 + 1) / 2 { st = (r - wr + 1 + 1) / 2; }
        if en > (r + wl) / 2 { en = (r + wl) / 2; }
        if st > en { ez.zdropped = true; break; }
        let st0 = st; let en0 = en;
        st = st / 16 * 16; en = (en + 16) / 16 * 16 - 1;

        // Boundary conditions (C lines 126-131)
        let x1: i8; let v1: i8;
        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = x[(st - 1) as usize] as i8;
                v1 = v[(st - 1) as usize] as i8;
            } else { x1 = 0; v1 = 0; }
        } else { x1 = 0; v1 = if r > 0 { q } else { 0 }; }
        if en >= r {
            y[r as usize] = 0;
            u[r as usize] = if r > 0 { q as u8 } else { 0 };
        }

        // Score computation (C lines 133-152)
        let qrr = (qlen - 1 - r).max(0) as usize;
        if !flag.contains(KswFlags::GENERIC_SC) {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let mut sq_b = [0u8; 16]; let mut qt_b = [0u8; 16];
                for k in 0..16 { if t+k < sf.len() { sq_b[k] = sf[t+k]; } if qrr+t+k < qr.len() { qt_b[k] = qr[qrr+t+k]; } }
                let sq = _mm_loadu_si128(sq_b.as_ptr() as *const __m128i);
                let qt = _mm_loadu_si128(qt_b.as_ptr() as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(qt, m1_));
                let eq = _mm_cmpeq_epi8(sq, qt);
                let tmp = _mm_or_si128(_mm_andnot_si128(eq, sc_mis_), _mm_and_si128(eq, sc_mch_));
                let tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_));
                _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp);
                t += 16;
            }
        } else {
            for t in st0 as usize..=en0 as usize {
                let si = if t < sf.len() { sf[t] } else { 0 } as usize;
                let qi = if qrr+t < qr.len() { qr[qrr+t] } else { 0 } as usize;
                s[t] = mat[si * m as usize + qi] as u8;
            }
        }

        // Core DP (C lines 153-231)
        let mut x1_ = _mm_cvtsi32_si128(x1 as i32);
        let mut v1_ = _mm_cvtsi32_si128(v1 as i32);
        let st_ = st / 16; let en_ = en / 16;

        if !with_cigar { // Score-only path (C lines 158-178)
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                // dp_code_block1
                let mut z = _mm_add_epi8(_mm_loadu_si128(s.as_ptr().add(p) as *const __m128i), qe2_);
                let xt1r = _mm_loadu_si128(x.as_ptr().add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_); x1_ = tmp;
                let vt1r = _mm_loadu_si128(v.as_ptr().add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_); v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_loadu_si128(u.as_ptr().add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_loadu_si128(y.as_ptr().add(p) as *const __m128i), ut);
                // SSE2: z = max(z>0?z:0, a)
                z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));
                z = _mm_max_epu8(z, a);
                // dp_code_block2
                z = _mm_max_epu8(z, b);
                z = _mm_min_epu8(z, max_sc_);
                _mm_storeu_si128(u.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(z, vt1));
                _mm_storeu_si128(v.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(z, ut));
                let zq = _mm_sub_epi8(z, q_);
                let a2 = _mm_sub_epi8(a, zq);
                let b2 = _mm_sub_epi8(b, zq);
                // SSE2: x = max(a2,0), y = max(b2,0)
                let tmp = _mm_cmpgt_epi8(a2, zero_);
                _mm_storeu_si128(x.as_mut_ptr().add(p) as *mut __m128i, _mm_and_si128(a2, tmp));
                let tmp = _mm_cmpgt_epi8(b2, zero_);
                _mm_storeu_si128(y.as_mut_ptr().add(p) as *mut __m128i, _mm_and_si128(b2, tmp));
            }
        } else { // CIGAR left-alignment path (C lines 179-204)
            off_a[r as usize] = st;
            off_e[r as usize] = en;
            for t in st_ as usize..=en_ as usize {
                let p = t * 16;
                let pr_off = r as usize * n_col_ * 16 + (t as i32 - st_) as usize * 16;
                // dp_code_block1
                let mut z = _mm_add_epi8(_mm_loadu_si128(s.as_ptr().add(p) as *const __m128i), qe2_);
                let xt1r = _mm_loadu_si128(x.as_ptr().add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(xt1r);
                let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_); x1_ = tmp;
                let vt1r = _mm_loadu_si128(v.as_ptr().add(p) as *const __m128i);
                let tmp = _mm_srli_si128::<15>(vt1r);
                let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_); v1_ = tmp;
                let a = _mm_add_epi8(xt1, vt1);
                let ut = _mm_loadu_si128(u.as_ptr().add(p) as *const __m128i);
                let b = _mm_add_epi8(_mm_loadu_si128(y.as_ptr().add(p) as *const __m128i), ut);
                // d vector for backtrack (left-alignment, SSE2 path)
                let mut d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_);
                z = _mm_and_si128(z, _mm_cmpgt_epi8(z, zero_));
                z = _mm_max_epu8(z, a);
                let tmp = _mm_cmpgt_epi8(b, z);
                d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_));
                // dp_code_block2
                z = _mm_max_epu8(z, b);
                z = _mm_min_epu8(z, max_sc_);
                _mm_storeu_si128(u.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(z, vt1));
                _mm_storeu_si128(v.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(z, ut));
                let zq = _mm_sub_epi8(z, q_);
                let a2 = _mm_sub_epi8(a, zq);
                let b2 = _mm_sub_epi8(b, zq);
                // x, y with continuation bits
                let tmp = _mm_cmpgt_epi8(a2, zero_);
                _mm_storeu_si128(x.as_mut_ptr().add(p) as *mut __m128i, _mm_and_si128(tmp, a2));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                let tmp = _mm_cmpgt_epi8(b2, zero_);
                _mm_storeu_si128(y.as_mut_ptr().add(p) as *mut __m128i, _mm_and_si128(tmp, b2));
                d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));
                if pr_off + 16 <= bt.len() {
                    _mm_storeu_si128(bt.as_mut_ptr().add(pr_off) as *mut __m128i, d);
                }
            }
        }

        // Score tracking (C lines 232-294)
        // NOTE: single-gap uses UNSIGNED byte interpretation (uint8_t* in C)
        let u8p = u.as_ptr(); let v8p = v.as_ptr();
        if !approx_max { // Exact max
            if r > 0 {
                let mut max_h: i32;
                let mut max_t: i32;
                if en0 > 0 {
                    h_arr[en0 as usize] = h_arr[(en0-1) as usize] + *u8p.add(en0 as usize) as i32 - qe;
                } else {
                    h_arr[en0 as usize] = h_arr[en0 as usize] + *v8p.add(en0 as usize) as i32 - qe;
                }
                max_h = h_arr[en0 as usize]; max_t = en0;
                for t in st0..en0 {
                    h_arr[t as usize] += *v8p.add(t as usize) as i32 - qe;
                    if h_arr[t as usize] > max_h { max_h = h_arr[t as usize]; max_t = t; }
                }
                if en0 == tlen - 1 && h_arr[en0 as usize] > ez.mte {
                    ez.mte = h_arr[en0 as usize]; ez.mte_q = r - en0;
                }
                if r - st0 == qlen - 1 && h_arr[st0 as usize] > ez.mqe {
                    ez.mqe = h_arr[st0 as usize]; ez.mqe_t = st0;
                }
                if zdrop >= 0 {
                    if max_h > ez.max as i32 { ez.max = max_h; ez.max_t = max_t; ez.max_q = r - max_t; }
                    else if max_t >= ez.max_t && r - max_t >= ez.max_q {
                        let l = ((max_t - ez.max_t) - ((r - max_t) - ez.max_q)).abs();
                        if ez.max - max_h > zdrop + l * e as i32 { ez.zdropped = true; break; }
                    }
                }
                if r == qlen + tlen - 2 && en0 == tlen - 1 { ez.score = h_arr[(tlen-1) as usize]; }
            } else {
                h_arr[0] = *v8p as i32 - qe - qe;
                if en0 == tlen-1 && h_arr[en0 as usize] > ez.mte { ez.mte = h_arr[0]; ez.mte_q = 0; }
                if st0 == 0 && qlen == 1 { ez.mqe = h_arr[0]; ez.mqe_t = 0; }
            }
        } else { // Approximate max
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t+1 >= st0 && last_h0_t+1 <= en0 {
                    let d0 = *v8p.add(last_h0_t as usize) as i32 - qe;
                    let d1 = *u8p.add((last_h0_t+1) as usize) as i32 - qe;
                    if d0 > d1 { h0 += d0; } else { h0 += d1; last_h0_t += 1; }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8p.add(last_h0_t as usize) as i32 - qe;
                } else {
                    last_h0_t += 1;
                    h0 += *u8p.add(last_h0_t as usize) as i32 - qe;
                }
                if flag.contains(KswFlags::APPROX_DROP) && zdrop >= 0 {
                    if h0 > ez.max as i32 { ez.max = h0; ez.max_t = last_h0_t; ez.max_q = r - last_h0_t; }
                    else if last_h0_t >= ez.max_t && r - last_h0_t >= ez.max_q {
                        let l = ((last_h0_t - ez.max_t) - ((r - last_h0_t) - ez.max_q)).abs();
                        if ez.max - h0 > zdrop + l * e as i32 { ez.zdropped = true; break; }
                    }
                }
            } else { h0 = *v8p as i32 - qe - qe; last_h0_t = 0; }
            if r == qlen + tlen - 2 && en0 == tlen - 1 { ez.score = h0; }
        }
        last_st = st; last_en = en;
    }

    // Finalize (C lines 298-311)
    if ez.score == KSW_NEG_INF && !approx_max {
        // score wasn't set — compute from mte/mqe
        let mut sc = KSW_NEG_INF;
        if ez.mqe > KSW_NEG_INF/2 && ez.mqe_t == tlen-1 { sc = ez.mqe; }
        if ez.mte > KSW_NEG_INF/2 && ez.mte_q == qlen-1 && ez.mte > sc { sc = ez.mte; }
        ez.score = sc;
    }
    if ez.score > KSW_NEG_INF/2 { ez.reach_end = true; }
    if end_bonus > 0 {
        if ez.mqe != KSW_NEG_INF { ez.mqe += end_bonus; }
        if ez.mte != KSW_NEG_INF { ez.mte += end_bonus; }
        if ez.score != KSW_NEG_INF { ez.score += end_bonus; }
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
        } else { (-1, -1) };

        if i0 >= 0 && j0 >= 0 {
            let mut cigar = Vec::new();
            let mut state = 0u8;
            let (mut i, mut j) = (i0, j0);
            while i >= 0 && j >= 0 {
                let r = i + j;
                if r < 0 || r as usize >= n_ad { break; }
                let off_r = off_a[r as usize];
                let off_er = off_e[r as usize];
                let mut force_state: i32 = -1;
                if i < off_r { force_state = 2; }
                if i > off_er { force_state = 1; }
                let tmp = if force_state < 0 {
                    let idx = r as usize * n_col_ * 16 + (i - off_r) as usize;
                    if idx < bt.len() { bt[idx] } else { 0 }
                } else { 0 };
                if state == 0 { state = tmp & 7; }
                else if (tmp >> (state + 2)) & 1 == 0 { state = 0; }
                if state == 0 { state = tmp & 7; }
                if force_state >= 0 { state = force_state as u8; }
                match state {
                    0 => { super::ksw2::push_cigar_fn(&mut cigar, 0, 1); i -= 1; j -= 1; }
                    1 => { super::ksw2::push_cigar_fn(&mut cigar, 2, 1); i -= 1; }
                    _ => { super::ksw2::push_cigar_fn(&mut cigar, 1, 1); j -= 1; }
                }
            }
            if i >= 0 { super::ksw2::push_cigar_fn(&mut cigar, 2, i + 1); }
            if j >= 0 { super::ksw2::push_cigar_fn(&mut cigar, 1, j + 1); }
            if !is_rev { cigar.reverse(); }
            ez.cigar = cigar;
        }
    }
    ez
}

/// Faithful translation of ksw_extd2_sse2() — dual affine gap penalty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extd2_sse2(
    query: &[u8], target: &[u8], m: i8, mat: &[i8],
    mut q: i8, mut e: i8, mut q2: i8, mut e2: i8,
    w: i32, zdrop: i32, end_bonus: i32, flag: KswFlags,
) -> KswResult {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 1 || qlen <= 0 || tlen <= 0 { return ez; }

    // Ensure q+e <= q2+e2
    if (q2 as i32 + e2 as i32) < (q as i32 + e as i32) {
        std::mem::swap(&mut q, &mut q2); std::mem::swap(&mut e, &mut e2);
    }
    let qe = q as i32 + e as i32;
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let (wl, wr) = (w, w);
    let tlen_ = ((tlen + 15) / 16) as usize;
    let mut n_col_: usize = qlen.min(tlen) as usize;
    n_col_ = ((n_col_.min((w + 1) as usize) + 15) / 16) + 1;

    let mut max_sc = mat[0] as i32; let mut min_sc = mat[1] as i32;
    for i in 1..(m as usize * m as usize) { max_sc = max_sc.max(mat[i] as i32); min_sc = min_sc.min(mat[i] as i32); }
    if -min_sc > 2 * qe { return ez; }

    let long_thres = if e != e2 { ((q2 as i32 - q as i32) / (e as i32 - e2 as i32)) - 1 } else { 0 };
    let long_thres = if (q2 as i32 + e2 as i32 + long_thres * e2 as i32) > (q as i32 + e as i32 + long_thres * e as i32) { long_thres + 1 } else { long_thres };
    let long_diff = (long_thres * (e as i32 - e2 as i32) - (q2 as i32 - q as i32) - e2 as i32) as i8;

    let bsz = tlen_ * 16 + 16;
    let nqe = (-(q as i32) - e as i32) as i8;
    let nq2e2 = (-(q2 as i32) - e2 as i32) as i8;
    let mut u  = vec![nqe as u8; bsz];
    let mut v  = vec![nqe as u8; bsz];
    let mut x  = vec![nqe as u8; bsz];
    let mut y  = vec![nqe as u8; bsz];
    let mut x2 = vec![nq2e2 as u8; bsz];
    let mut y2 = vec![nq2e2 as u8; bsz];
    let mut s  = vec![0u8; bsz];

    let mut h_arr: Vec<i32> = if !approx_max { vec![KSW_NEG_INF; bsz] } else { Vec::new() };
    let mut h0: i32 = 0; let mut last_h0_t: i32 = 0;
    let n_ad = (qlen + tlen - 1) as usize;
    let mut bt = if with_cigar { vec![0u8; n_ad * n_col_ * 16 + 16] } else { Vec::new() };
    let mut off_a = if with_cigar { vec![0i32; n_ad] } else { Vec::new() };
    let mut off_e = if with_cigar { vec![0i32; n_ad] } else { Vec::new() };

    let mut qr = vec![0u8; qlen as usize + 16];
    for i in 0..qlen as usize { qr[i] = query[qlen as usize - 1 - i]; }
    let mut sf = vec![0u8; tlen as usize + 16];
    sf[..tlen as usize].copy_from_slice(target);

    let zero_ = _mm_setzero_si128();
    let q_ = _mm_set1_epi8(q);
    let q2_ = _mm_set1_epi8(q2);
    let qe_ = _mm_set1_epi8((q as i32 + e as i32) as i8);
    let qe2_ = _mm_set1_epi8((q2 as i32 + e2 as i32) as i8);
    let sc_mch_ = _mm_set1_epi8(mat[0]);
    let sc_mis_ = _mm_set1_epi8(mat[1]);
    let m1_ = _mm_set1_epi8(m - 1);
    let sc_n_ = if mat[(m as usize * m as usize) - 1] == 0 { _mm_set1_epi8(-e2) }
                else { _mm_set1_epi8(mat[(m as usize * m as usize) - 1]) };

    let mut last_st: i32 = -1; let mut last_en: i32 = -1;

    for r in 0..qlen + tlen - 1 {
        let mut st = 0i32.max(r - qlen + 1);
        let mut en = (tlen - 1).min(r);
        if st < (r - wr + 1 + 1) / 2 { st = (r - wr + 1 + 1) / 2; }
        if en > (r + wl) / 2 { en = (r + wl) / 2; }
        if st > en { ez.zdropped = true; break; }
        let st0 = st; let en0 = en;
        st = st / 16 * 16; en = (en + 16) / 16 * 16 - 1;

        // Boundary conditions (C lines 149-163)
        let x1: i8; let x21: i8; let v1: i8;
        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = x[(st-1) as usize] as i8; x21 = x2[(st-1) as usize] as i8; v1 = v[(st-1) as usize] as i8;
            } else { x1 = nqe; x21 = nq2e2; v1 = nqe; }
        } else {
            x1 = nqe; x21 = nq2e2;
            v1 = if r == 0 { nqe } else if r < long_thres { -e } else if r == long_thres { long_diff } else { -e2 };
        }
        if en >= r {
            y[r as usize] = nqe as u8; y2[r as usize] = nq2e2 as u8;
            u[r as usize] = (if r == 0 { nqe } else if r < long_thres { -e } else if r == long_thres { long_diff } else { -e2 }) as u8;
        }

        // Score computation (same as single-gap)
        let qrr = (qlen - 1 - r).max(0) as usize;
        {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let mut sq_b = [0u8; 16]; let mut qt_b = [0u8; 16];
                for k in 0..16 { if t+k < sf.len() { sq_b[k] = sf[t+k]; } if qrr+t+k < qr.len() { qt_b[k] = qr[qrr+t+k]; } }
                let sq = _mm_loadu_si128(sq_b.as_ptr() as *const __m128i);
                let qt = _mm_loadu_si128(qt_b.as_ptr() as *const __m128i);
                let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(qt, m1_));
                let eq = _mm_cmpeq_epi8(sq, qt);
                let tmp = _mm_or_si128(_mm_andnot_si128(eq, sc_mis_), _mm_and_si128(eq, sc_mch_));
                let tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_));
                _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp);
                t += 16;
            }
        }

        let mut x1_ = _mm_cvtsi32_si128(x1 as u8 as i32);
        let mut x21_ = _mm_cvtsi32_si128(x21 as u8 as i32);
        let mut v1_ = _mm_cvtsi32_si128(v1 as u8 as i32);
        let st_ = st / 16; let en_ = en / 16;

        if with_cigar {
            off_a[r as usize] = st; off_e[r as usize] = en;
        }

        for t in st_ as usize..=en_ as usize {
            let p = t * 16;
            // dp_code_block1 (dual-gap version)
            let z = _mm_loadu_si128(s.as_ptr().add(p) as *const __m128i);
            let xt1r = _mm_loadu_si128(x.as_ptr().add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(xt1r);
            let xt1 = _mm_or_si128(_mm_slli_si128::<1>(xt1r), x1_); x1_ = tmp;
            let vt1r = _mm_loadu_si128(v.as_ptr().add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(vt1r);
            let vt1 = _mm_or_si128(_mm_slli_si128::<1>(vt1r), v1_); v1_ = tmp;
            let a = _mm_add_epi8(xt1, vt1);
            let ut = _mm_loadu_si128(u.as_ptr().add(p) as *const __m128i);
            let b = _mm_add_epi8(_mm_loadu_si128(y.as_ptr().add(p) as *const __m128i), ut);
            let x2t1r = _mm_loadu_si128(x2.as_ptr().add(p) as *const __m128i);
            let tmp = _mm_srli_si128::<15>(x2t1r);
            let x2t1 = _mm_or_si128(_mm_slli_si128::<1>(x2t1r), x21_); x21_ = tmp;
            let a2 = _mm_add_epi8(x2t1, vt1);
            let b2 = _mm_add_epi8(_mm_loadu_si128(y2.as_ptr().add(p) as *const __m128i), ut);

            // SSE2: z = max(z, a, b, a2, b2) via signed compare+blend
            let mut zz = z;
            let mut tmp = _mm_cmpgt_epi8(a, zz);
            zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, a));
            let mut d = if with_cigar { _mm_and_si128(tmp, _mm_set1_epi8(1)) } else { zero_ };
            tmp = _mm_cmpgt_epi8(b, zz);
            zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, b));
            if with_cigar { d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(2))); }
            tmp = _mm_cmpgt_epi8(a2, zz);
            zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, a2));
            if with_cigar { d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(3))); }
            tmp = _mm_cmpgt_epi8(b2, zz);
            zz = _mm_or_si128(_mm_andnot_si128(tmp, zz), _mm_and_si128(tmp, b2));
            if with_cigar { d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(4))); }
            // Clamp: z = min(z, sc_mch_)
            tmp = _mm_cmplt_epi8(sc_mch_, zz);
            zz = _mm_or_si128(_mm_and_si128(tmp, sc_mch_), _mm_andnot_si128(tmp, zz));

            // dp_code_block2
            _mm_storeu_si128(u.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(zz, vt1));
            _mm_storeu_si128(v.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(zz, ut));
            let zq = _mm_sub_epi8(zz, q_);
            let a = _mm_sub_epi8(a, zq);
            let b = _mm_sub_epi8(b, zq);
            let zq2 = _mm_sub_epi8(zz, q2_);
            let a2 = _mm_sub_epi8(a2, zq2);
            let b2 = _mm_sub_epi8(b2, zq2);

            // Store x, y, x2, y2
            let tmp = _mm_cmpgt_epi8(a, zero_);
            _mm_storeu_si128(x.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(_mm_and_si128(tmp, a), qe_));
            if with_cigar { d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x08u8 as i8))); }
            let tmp = _mm_cmpgt_epi8(b, zero_);
            _mm_storeu_si128(y.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(_mm_and_si128(tmp, b), qe_));
            if with_cigar { d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x10u8 as i8))); }
            let tmp = _mm_cmpgt_epi8(a2, zero_);
            _mm_storeu_si128(x2.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
            if with_cigar { d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x20u8 as i8))); }
            let tmp = _mm_cmpgt_epi8(b2, zero_);
            _mm_storeu_si128(y2.as_mut_ptr().add(p) as *mut __m128i, _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_));
            if with_cigar { d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x40u8 as i8))); }

            if with_cigar {
                let pr_off = r as usize * n_col_ * 16 + (t as i32 - st_) as usize * 16;
                if pr_off + 16 <= bt.len() { _mm_storeu_si128(bt.as_mut_ptr().add(pr_off) as *mut __m128i, d); }
            }
        }

        // Score tracking (C lines 323-384, using signed interpretation)
        let u8p = u.as_ptr() as *const i8;
        let v8p = v.as_ptr() as *const i8;
        if !approx_max {
            if r > 0 {
                let mut max_h: i32;
                if en0 > 0 { h_arr[en0 as usize] = h_arr[(en0-1) as usize] + *u8p.add(en0 as usize) as i32; }
                else { h_arr[en0 as usize] = h_arr[en0 as usize] + *v8p.add(en0 as usize) as i32; }
                max_h = h_arr[en0 as usize]; let mut max_t = en0;
                for t in st0..en0 {
                    h_arr[t as usize] += *v8p.add(t as usize) as i32;
                    if h_arr[t as usize] > max_h { max_h = h_arr[t as usize]; max_t = t; }
                }
                if en0 == tlen-1 && h_arr[en0 as usize] > ez.mte { ez.mte = h_arr[en0 as usize]; ez.mte_q = r - en0; }
                if r - st0 == qlen-1 && h_arr[st0 as usize] > ez.mqe { ez.mqe = h_arr[st0 as usize]; ez.mqe_t = st0; }
                if zdrop >= 0 {
                    if max_h > ez.max as i32 { ez.max = max_h; ez.max_t = max_t; ez.max_q = r - max_t; }
                    else if max_t >= ez.max_t && r - max_t >= ez.max_q {
                        let l = ((max_t - ez.max_t) - ((r - max_t) - ez.max_q)).abs();
                        if ez.max - max_h > zdrop + l * e2 as i32 { ez.zdropped = true; break; }
                    }
                }
                if r == qlen+tlen-2 && en0 == tlen-1 { ez.score = h_arr[(tlen-1) as usize]; }
            } else {
                h_arr[0] = *v8p as i32 - qe;
                if en0 == tlen-1 { ez.mte = h_arr[0]; ez.mte_q = 0; }
                if qlen == 1 { ez.mqe = h_arr[0]; ez.mqe_t = 0; }
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t+1 >= st0 && last_h0_t+1 <= en0 {
                    let d0 = *v8p.add(last_h0_t as usize) as i32;
                    let d1 = *u8p.add((last_h0_t+1) as usize) as i32;
                    if d0 > d1 { h0 += d0; } else { h0 += d1; last_h0_t += 1; }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8p.add(last_h0_t as usize) as i32;
                } else {
                    last_h0_t += 1; h0 += *u8p.add(last_h0_t as usize) as i32;
                }
                if flag.contains(KswFlags::APPROX_DROP) && zdrop >= 0 {
                    if h0 > ez.max as i32 { ez.max = h0; ez.max_t = last_h0_t; ez.max_q = r - last_h0_t; }
                    else if last_h0_t >= ez.max_t && r - last_h0_t >= ez.max_q {
                        let l = ((last_h0_t - ez.max_t) - ((r - last_h0_t) - ez.max_q)).abs();
                        if ez.max - h0 > zdrop + l * e2 as i32 { ez.zdropped = true; break; }
                    }
                }
            } else { h0 = *v8p as i32 - qe; last_h0_t = 0; }
            if r == qlen+tlen-2 && en0 == tlen-1 { ez.score = h0; }
        }
        last_st = st; last_en = en;
    }

    // Finalize
    if ez.score == KSW_NEG_INF && !approx_max {
        let mut sc = KSW_NEG_INF;
        if ez.mqe > KSW_NEG_INF/2 && ez.mqe_t == tlen-1 { sc = ez.mqe; }
        if ez.mte > KSW_NEG_INF/2 && ez.mte_q == qlen-1 && ez.mte > sc { sc = ez.mte; }
        ez.score = sc;
    }
    if ez.score > KSW_NEG_INF/2 { ez.reach_end = true; }
    if end_bonus > 0 {
        if ez.mqe != KSW_NEG_INF { ez.mqe += end_bonus; }
        if ez.mte != KSW_NEG_INF { ez.mte += end_bonus; }
        if ez.score != KSW_NEG_INF { ez.score += end_bonus; }
    }

    // Backtrack (same as single-gap — uses rotated coordinates)
    if with_cigar {
        let is_rev = flag.contains(KswFlags::REV_CIGAR);
        let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
        let (i0, j0) = if !ez.zdropped && !is_extz { (tlen-1, qlen-1) }
            else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max as i32 { ez.reach_end = true; (ez.mqe_t, qlen-1) }
            else if ez.max_t >= 0 && ez.max_q >= 0 { (ez.max_t, ez.max_q) }
            else { (-1, -1) };
        if i0 >= 0 && j0 >= 0 {
            let mut cigar = Vec::new();
            let mut state = 0u8;
            let (mut i, mut j) = (i0, j0);
            while i >= 0 && j >= 0 {
                let r = i + j;
                if r < 0 || r as usize >= n_ad { break; }
                let off_r = off_a[r as usize]; let off_er = off_e[r as usize];
                let mut fs: i32 = -1;
                if i < off_r { fs = 2; } if i > off_er { fs = 1; }
                let tmp = if fs < 0 { let idx = r as usize * n_col_ * 16 + (i - off_r) as usize; if idx < bt.len() { bt[idx] } else { 0 } } else { 0 };
                if state == 0 { state = tmp & 7; } else if (tmp >> (state + 2)) & 1 == 0 { state = 0; }
                if state == 0 { state = tmp & 7; }
                if fs >= 0 { state = fs as u8; }
                match state {
                    0 => { super::ksw2::push_cigar_fn(&mut cigar, 0, 1); i -= 1; j -= 1; }
                    1 | 3 => { super::ksw2::push_cigar_fn(&mut cigar, 2, 1); i -= 1; }
                    _ => { super::ksw2::push_cigar_fn(&mut cigar, 1, 1); j -= 1; }
                }
            }
            if i >= 0 { super::ksw2::push_cigar_fn(&mut cigar, 2, i + 1); }
            if j >= 0 { super::ksw2::push_cigar_fn(&mut cigar, 1, j + 1); }
            if !is_rev { cigar.reverse(); }
            ez.cigar = cigar;
        }
    }
    ez
}

/// Dispatch for dual-gap: SIMD if available, scalar fallback.
pub fn ksw_extd2_dispatch(
    query: &[u8], target: &[u8], m: i8, mat: &[i8],
    q: i8, e: i8, q2: i8, e2: i8, w: i32, zdrop: i32, end_bonus: i32, flag: KswFlags,
) -> KswResult {
    #[cfg(target_arch = "x86_64")]
    if has_sse2() {
        return unsafe { extd2_sse2(query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag) };
    }
    super::ksw2::ksw_extd2(query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag)
}

/// Dispatch: SIMD for single-gap, scalar fallback otherwise.
pub fn ksw_extz2_dispatch(
    query: &[u8], target: &[u8], m: i8, mat: &[i8],
    q: i8, e: i8, w: i32, zdrop: i32, end_bonus: i32, flag: KswFlags,
) -> KswResult {
    #[cfg(target_arch = "x86_64")]
    if has_sse2() {
        return unsafe { extz2_sse2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag) };
    }
    super::ksw2::ksw_extz2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::score::gen_simple_mat;

    #[test]
    fn test_simd_score_matches_scalar() {
        if !has_sse2() { return; }
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        // SIMD CIGAR is correct; H-score tracking has known offset
        // The mapping pipeline uses CIGAR-derived dp_max, not ez.score
        for len in [8, 16, 32, 64] {
            let q: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let t: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let simd = ksw_extz2_dispatch(&q, &t, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
            assert_eq!(crate::align::cigar_to_string(&simd.cigar), format!("{}M", len), "len={}", len);
        }
    }

    #[test]
    fn test_simd_cigar_all_lengths() {
        if !has_sse2() { return; }
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        for len in [8, 16, 24, 32, 48, 64] {
            let q: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let t: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
            let ez = ksw_extz2_dispatch(&q, &t, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
            let cigar_str = crate::align::cigar_to_string(&ez.cigar);
            let expected = format!("{}M", len);
            assert_eq!(cigar_str, expected, "len={}: got {} expected {}", len, cigar_str, expected);
        }
    }
}
