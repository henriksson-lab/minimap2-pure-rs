pub mod ksw2;
pub mod ksw2_simd;
pub mod score;

use std::cell::RefCell;

use crate::flags::KswFlags;
use crate::index::MmIdx;
use crate::options::MapOpt;
use crate::seq::SEQ_NT4_TABLE;
use crate::types::{AlignExtra, AlignReg, Cigar, Mm128};
pub use ksw2::KswResult;

// Thread-local scratch buffers for alignment to avoid per-call allocations.
thread_local! {
    static ALIGN_TSEQ_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ALIGN_REV_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ALIGN_QSEQ_FWD: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ALIGN_QSEQ_REV: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// High-level alignment interface dispatching to scalar or SIMD implementations.
///
/// Single-gap affine alignment.
pub fn align_pair(
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
    // SSE2 SIMD for single-gap affine (score-only and CIGAR)
    ksw2_simd::ksw_extz2_dispatch(query, target, m, mat, q, e, w, zdrop, end_bonus, flag)
}

/// High-level dual-gap affine alignment.
pub fn align_pair_dual(
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
    ksw2_simd::ksw_extd2_dispatch(
        query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag,
    )
}

/// Decode a BAM-style CIGAR u32 into (op, len).
#[inline]
pub fn cigar_decode(c: u32) -> (u32, u32) {
    (c & 0xf, c >> 4)
}

/// Format a CIGAR array as a string.
pub fn cigar_to_string(cigar: &[u32]) -> String {
    const OPS: &[u8] = b"MIDNSHP=XB";
    let mut s = String::new();
    for &c in cigar {
        let (op, len) = cigar_decode(c);
        s.push_str(&len.to_string());
        if (op as usize) < OPS.len() {
            s.push(OPS[op as usize] as char);
        }
    }
    s
}

/// Clean up CIGAR: remove zero-length ops, merge adjacent same-op entries,
/// remove leading/trailing insertions/deletions.
/// Clean up CIGAR: remove zero-length ops, merge adjacent same-op entries,
/// remove leading/trailing insertions/deletions.
/// Returns (q_shift, t_shift) for leading I/D that were removed.
fn fix_cigar(cigar: &mut Vec<u32>) -> (i32, i32) {
    if cigar.is_empty() {
        return (0, 0);
    }
    let mut q_shift = 0i32;
    let mut t_shift = 0i32;
    // Remove zero-length operations
    cigar.retain(|&c| c >> 4 != 0);
    // Merge adjacent same-op
    let mut merged = Vec::with_capacity(cigar.len());
    for &c in cigar.iter() {
        if let Some(last) = merged.last_mut() {
            let last_val: &mut u32 = last;
            if (*last_val & 0xf) == (c & 0xf) {
                *last_val += c >> 4 << 4;
                continue;
            }
        }
        merged.push(c);
    }
    // Remove leading I/D
    while !merged.is_empty() {
        let op = merged[0] & 0xf;
        let len = (merged[0] >> 4) as i32;
        if op == 1 {
            q_shift += len;
            merged.remove(0);
        } else if op == 2 {
            t_shift += len;
            merged.remove(0);
        } else {
            break;
        }
    }
    // Remove trailing I/D
    while !merged.is_empty() {
        let last_idx = merged.len() - 1;
        let op = merged[last_idx] & 0xf;
        if op == 1 || op == 2 {
            merged.pop();
        } else {
            break;
        }
    }
    *cigar = merged;
    (q_shift, t_shift)
}

/// Append CIGAR operations to a vector, merging adjacent same-op entries.
fn append_cigar(cigar: &mut Vec<u32>, new_ops: &[u32]) {
    for &c in new_ops {
        if c >> 4 == 0 {
            continue;
        }
        if let Some(last) = cigar.last_mut() {
            if (*last & 0xf) == (c & 0xf) {
                *last += c >> 4 << 4;
                continue;
            }
        }
        cigar.push(c);
    }
}

/// Helper: perform a single KSW2 alignment and return the result.
fn do_align(
    opt: &MapOpt,
    qsub: &[u8],
    tsub: &[u8],
    mat: &[i8],
    bw: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    if qsub.is_empty() || tsub.is_empty() {
        return KswResult::new();
    }
    if opt.flag.contains(crate::flags::MapFlags::SPLICE) {
        let mut splice_flag = flag;
        if opt.flag.contains(crate::flags::MapFlags::SPLICE_FOR)
            && !opt.flag.contains(crate::flags::MapFlags::SPLICE_REV)
        {
            splice_flag |= KswFlags::SPLICE_FOR;
        } else if opt.flag.contains(crate::flags::MapFlags::SPLICE_REV)
            && !opt.flag.contains(crate::flags::MapFlags::SPLICE_FOR)
        {
            splice_flag |= KswFlags::SPLICE_REV;
        } else {
            splice_flag |= KswFlags::SPLICE_FOR;
        }
        if opt.flag.contains(crate::flags::MapFlags::SPLICE_FLANK) {
            splice_flag |= KswFlags::SPLICE_FLANK;
        }
        if !opt.flag.contains(crate::flags::MapFlags::SPLICE_OLD) {
            splice_flag |= KswFlags::SPLICE_CMPLX;
        }
        if flag.contains(KswFlags::REV_CIGAR) {
            if splice_flag.contains(KswFlags::SPLICE_FOR) {
                splice_flag.remove(KswFlags::SPLICE_FOR);
                splice_flag |= KswFlags::SPLICE_REV;
            } else if splice_flag.contains(KswFlags::SPLICE_REV) {
                splice_flag.remove(KswFlags::SPLICE_REV);
                splice_flag |= KswFlags::SPLICE_FOR;
            }
        }
        return ksw2::ksw_exts2(
            qsub,
            tsub,
            5,
            mat,
            opt.q as i8,
            opt.e as i8,
            opt.q2 as i8,
            opt.noncan as i8,
            bw,
            zdrop,
            end_bonus,
            splice_flag,
        );
    }
    if opt.q == opt.q2 && opt.e == opt.e2 {
        align_pair(
            qsub,
            tsub,
            5,
            mat,
            opt.q as i8,
            opt.e as i8,
            bw,
            zdrop,
            end_bonus,
            flag,
        )
    } else {
        align_pair_dual(
            qsub,
            tsub,
            5,
            mat,
            opt.q as i8,
            opt.e as i8,
            opt.q2 as i8,
            opt.e2 as i8,
            bw,
            zdrop,
            end_bonus,
            flag,
        )
    }
}

/// Test for z-drop in a CIGAR alignment.
/// Returns 0 if no z-drop, 1 if regular z-drop > zdrop, 2 if potential inversion.
/// Matches mm_test_zdrop() from align.c.
fn test_zdrop(opt: &MapOpt, qseq: &[u8], tseq: &[u8], cigar: &[u32], mat: &[i8]) -> i32 {
    let mut score = 0i32;
    let mut max_score = i32::MIN;
    let mut max_i: i32 = -1;
    let mut max_j: i32 = -1;
    let mut max_zdrop = 0i32;
    let mut i = 0usize;
    let mut j = 0usize;
    // pos[0] = (max_t_start, max_t_end), pos[1] = (max_q_start, max_q_end)
    let mut pos = [[-1i32; 2]; 2];

    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as usize;
        if op == 0 {
            // M
            let safe_len = len
                .min(tseq.len().saturating_sub(i))
                .min(qseq.len().saturating_sub(j));
            for l in 0..safe_len {
                // SAFETY: l < safe_len ensures i+l < tseq.len() and j+l < qseq.len()
                // tseq/qseq values are < 5, so tseq[]*5+qseq[] < 25 <= mat.len()
                unsafe {
                    score += *mat.get_unchecked(
                        *tseq.get_unchecked(i + l) as usize * 5
                            + *qseq.get_unchecked(j + l) as usize,
                    ) as i32;
                }
                // update_max_zdrop
                let ci = (i + l) as i32;
                let cj = (j + l) as i32;
                if score < max_score {
                    let tl = ci - max_i;
                    let ql = cj - max_j;
                    let diff = (tl - ql).abs();
                    let z = max_score - score - diff * opt.e;
                    if z > max_zdrop {
                        max_zdrop = z;
                        pos[0][0] = max_i;
                        pos[0][1] = ci;
                        pos[1][0] = max_j;
                        pos[1][1] = cj;
                    }
                } else {
                    max_score = score;
                    max_i = ci;
                    max_j = cj;
                }
            }
            i += len;
            j += len;
        } else if op == 1 || op == 2 || op == 3 {
            // I, D, N
            score -= opt.q + opt.e * len as i32;
            if op == 1 {
                j += len;
            } else {
                i += len;
            }
            let ci = i as i32;
            let cj = j as i32;
            if score < max_score {
                let tl = ci - max_i;
                let ql = cj - max_j;
                let diff = (tl - ql).abs();
                let z = max_score - score - diff * opt.e;
                if z > max_zdrop {
                    max_zdrop = z;
                    pos[0][0] = max_i;
                    pos[0][1] = ci;
                    pos[1][0] = max_j;
                    pos[1][1] = cj;
                }
            } else {
                max_score = score;
                max_i = ci;
                max_j = cj;
            }
        }
    }

    // Test for potential inversion in the z-dropped region
    let q_len = pos[1][1] - pos[1][0];
    let t_len = pos[0][1] - pos[0][0];
    if !opt.flag.intersects(
        crate::flags::MapFlags::SPLICE
            | crate::flags::MapFlags::SR
            | crate::flags::MapFlags::FOR_ONLY
            | crate::flags::MapFlags::REV_ONLY,
    ) && max_zdrop > opt.zdrop_inv
        && q_len > 0
        && q_len < opt.max_gap
        && t_len > 0
        && t_len < opt.max_gap
    {
        // Check if reverse-complement of the q region aligns well to the t region
        let q_start = pos[1][0] as usize;
        let q_end = pos[1][1] as usize;
        let t_start = pos[0][0] as usize;
        if q_end <= qseq.len() && (t_start + t_len as usize) <= tseq.len() {
            let qseq_rc: Vec<u8> = qseq[q_start..q_end]
                .iter()
                .rev()
                .map(|&c| if c < 4 { 3 - c } else { 4 })
                .collect();
            // Use ksw_ll_i16 for proper local alignment score
            let (score, _qe, _te) = ksw2::ksw_ll_i16(
                &qseq_rc,
                &tseq[t_start..t_start + t_len as usize],
                5,
                mat,
                opt.q,
                opt.e,
            );
            if score >= opt.min_chain_score * opt.a && score >= opt.min_dp_max {
                return 2; // potential inversion
            }
        }
    }
    if max_zdrop > opt.zdrop {
        1
    } else {
        0
    }
}

/// Adjust anchor position to center on k-mer midpoint.
/// Matches mm_adjust_minier() from align.c.
#[inline]
fn adjust_minier(anchor: &Mm128, k_half: i32, is_hpc: bool) -> (i32, i32) {
    if is_hpc {
        let q_span = ((anchor.y >> 32) & 0xff) as i32;
        let r = ((anchor.x as i32) + 1 - q_span).max(0);
        let q = ((anchor.y as i32) + 1 - q_span).max(0);
        (r, q)
    } else {
        let r = ((anchor.x as i32) - k_half).max(0);
        let q = ((anchor.y as i32) - k_half).max(0);
        (r, q)
    }
}

/// Trim bad chain ends where anchors have large diagonal deviations.
/// Matches mm_fix_bad_ends() from align.c.
/// Returns (new_as, new_cnt) trimming the start and end of the chain.
fn fix_bad_ends(
    as1: usize,
    cnt1: usize,
    a: &[Mm128],
    bw: i32,
    min_match: i32,
    mlen: i32,
) -> (usize, usize) {
    if cnt1 < 3 {
        return (as1, cnt1);
    }
    let mut new_as = as1;
    let mut new_cnt;

    // Trim from the start
    let mut m = ((a[as1].y >> 32) & 0xff) as i32;
    let mut l = m;
    for i in 1..cnt1 - 1 {
        let q_span = ((a[as1 + i].y >> 32) & 0xff) as i32;
        if a[as1 + i].y & crate::flags::SEED_LONG_JOIN != 0 {
            break;
        }
        let lr = (a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32);
        let lq = (a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32);
        let min_d = lr.min(lq);
        let max_d = lr.max(lq);
        if max_d - min_d > l >> 1 {
            new_as = as1 + i;
        }
        l += min_d;
        m += min_d.min(q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 {
            break;
        }
    }
    new_cnt = as1 + cnt1 - new_as;

    // Trim from the end
    m = ((a[as1 + cnt1 - 1].y >> 32) & 0xff) as i32;
    l = m;
    let mut i = as1 + cnt1 - 2;
    while i > new_as {
        let q_span = ((a[i + 1].y >> 32) & 0xff) as i32;
        if a[i + 1].y & crate::flags::SEED_LONG_JOIN != 0 {
            break;
        }
        let lr = (a[i + 1].x as i32) - (a[i].x as i32);
        let lq = (a[i + 1].y as i32) - (a[i].y as i32);
        let min_d = lr.min(lq);
        let max_d = lr.max(lq);
        if max_d - min_d > l >> 1 {
            new_cnt = i + 1 - new_as;
        }
        l += min_d;
        m += min_d.min(q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 {
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    (new_as, new_cnt)
}

/// Filter bad seeds: mark seeds with large gaps for ignoring.
/// Collect indices of seeds with large diagonal gaps. Matches collect_long_gaps() from align.c.
fn collect_long_gaps(as1: usize, cnt1: usize, a: &[Mm128], min_gap: i32) -> Vec<usize> {
    let mut k = Vec::new();
    for i in 1..cnt1 {
        let gap = ((a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32))
            - ((a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32));
        if gap < -min_gap || gap > min_gap {
            k.push(i);
        } // strictly greater, matching C
    }
    if k.len() <= 1 {
        return Vec::new();
    } // C returns 0 (null) when n <= 1
    k
}

/// Filter bad seeds by finding clusters of large-gap seeds. Matches mm_filter_bad_seeds() from align.c.
fn filter_bad_seeds(
    as1: usize,
    cnt1: usize,
    a: &mut [Mm128],
    min_gap: i32,
    diff_thres: i32,
    max_ext_len: i32,
    max_ext_cnt: usize,
) {
    let k = collect_long_gaps(as1, cnt1, a, min_gap);
    if k.is_empty() {
        return;
    }
    let n = k.len();
    let mut max_diff_val = 0i32;
    let mut max_st: i32 = -1;
    let mut max_en: i32 = -1;
    let mut ki = 0;
    loop {
        if ki == n || ki as i32 >= max_en {
            if max_en > 0 {
                let start = k[max_st as usize];
                let end = k[max_en as usize];
                for i in start..end {
                    a[as1 + i].y |= crate::flags::SEED_IGNORE;
                }
            }
            max_diff_val = 0;
            max_st = -1;
            max_en = -1;
            if ki == n {
                break;
            }
        }
        let i = k[ki];
        let gap = ((a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32))
            - ((a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32));
        let mut n_ins = if gap > 0 { gap } else { 0 };
        let mut n_del = if gap < 0 { -gap } else { 0 };
        let qs = a[as1 + i - 1].y as i32;
        let rs = a[as1 + i - 1].x as i32;
        let mut cur_max_diff = 0i32;
        let mut cur_max_diff_l: i32 = -1;
        for l in (ki + 1)..n.min(ki + 1 + max_ext_cnt) {
            let j = k[l];
            if (a[as1 + j].y as i32) - qs > max_ext_len || (a[as1 + j].x as i32) - rs > max_ext_len
            {
                break;
            }
            let gap2 = ((a[as1 + j].y as i32) - (a[as1 + j - 1].y as i32))
                - ((a[as1 + j].x as i32) - (a[as1 + j - 1].x as i32));
            if gap2 > 0 {
                n_ins += gap2;
            } else {
                n_del += -gap2;
            }
            let diff = n_ins + n_del - (n_ins - n_del).abs();
            if cur_max_diff < diff {
                cur_max_diff = diff;
                cur_max_diff_l = l as i32;
            }
        }
        if cur_max_diff > diff_thres && cur_max_diff > max_diff_val {
            max_diff_val = cur_max_diff;
            max_st = ki as i32;
            max_en = cur_max_diff_l;
        }
        ki += 1;
    }
}

/// Filter bad seeds — alternate version. Matches mm_filter_bad_seeds_alt() from align.c.
/// Finds chains of consecutive close gaps and marks intermediate seeds as SEED_IGNORE,
/// end seeds as SEED_LONG_JOIN.
fn filter_bad_seeds_alt(as1: usize, cnt1: usize, a: &mut [Mm128], min_gap: i32, max_ext: i32) {
    let k = collect_long_gaps(as1, cnt1, a, min_gap);
    if k.is_empty() {
        return;
    }
    let n = k.len();
    let mut ki = 0;
    while ki < n {
        let i = k[ki];
        let gap1 = ((a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32))
            - ((a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32));
        let mut re1 = a[as1 + i].x as i32;
        let mut qe1 = a[as1 + i].y as i32;
        let mut gap1_abs = gap1.abs();
        let mut l = ki + 1;
        while l < n {
            let j = k[l];
            if (a[as1 + j].y as i32) - qe1 > max_ext || (a[as1 + j].x as i32) - re1 > max_ext {
                break;
            }
            let gap2 = ((a[as1 + j].y as i32) - (a[as1 + j - 1].y as i32))
                - ((a[as1 + j].x as i32) - (a[as1 + j - 1].x as i32));
            let q_span_pre = ((a[as1 + j - 1].y >> 32) & 0xff) as i32;
            let rs2 = (a[as1 + j - 1].x as i32) + q_span_pre;
            let qs2 = (a[as1 + j - 1].y as i32) + q_span_pre;
            let m = (rs2 - re1).min(qs2 - qe1);
            let gap2_abs = gap2.abs();
            if m > gap1_abs + gap2_abs {
                break;
            }
            re1 = a[as1 + j].x as i32;
            qe1 = a[as1 + j].y as i32;
            gap1_abs = gap2_abs;
            l += 1;
        }
        if l > ki + 1 {
            let end = k[l - 1];
            for j in k[ki]..end {
                a[as1 + j].y |= crate::flags::SEED_IGNORE;
            }
            a[as1 + end].y |= crate::flags::SEED_LONG_JOIN;
        }
        ki = l;
    }
}

/// Perform anchor-based gap-filling alignment for a single region.
///
/// Follows mm_align1() from align.c:
/// 1. Left extension from first anchor
/// 2. Gap filling between consecutive anchors
/// 3. Right extension from last anchor
///
/// Returns additional split regions created by z-drop.
fn align1(
    opt: &MapOpt,
    mi: &MmIdx,
    qlen: i32,
    qseq_fwd: &[u8],
    qseq_rev: &[u8],
    r: &mut AlignReg,
    a: &mut [Mm128],
    mat: &[i8],
) -> Vec<AlignReg> {
    let mut split_regs: Vec<AlignReg> = Vec::new();
    if r.cnt == 0 {
        return split_regs;
    }
    let raw_as = r.as_ as usize;
    let raw_cnt = r.cnt as usize;

    // Filter bad seeds and trim bad ends
    filter_bad_seeds(raw_as, raw_cnt, a, 10, 40, opt.max_gap >> 1, 10);
    filter_bad_seeds_alt(raw_as, raw_cnt, a, 30, opt.max_gap >> 1);
    let (as1, cnt1) = if !(opt.flag.contains(crate::flags::MapFlags::NO_END_FLT)) {
        fix_bad_ends(raw_as, raw_cnt, a, opt.bw, opt.min_chain_score * 2, r.mlen)
    } else {
        (raw_as, raw_cnt)
    };
    if cnt1 == 0 {
        return split_regs;
    }
    let rid = r.rid as u32;
    let rev = r.rev;
    let qseq = if rev { qseq_rev } else { qseq_fwd };
    let ref_len = mi.seqs[rid as usize].len as i32;

    let bw = (opt.bw as f32 * 1.5 + 1.0) as i32;
    let bw_long = ((opt.bw_long as f32 * 1.5 + 1.0) as i32).max(bw);

    // Get anchor coordinates using mm_adjust_minier logic
    // For non-HPC: shift position left by k/2 to center on k-mer
    let k_half = mi.k >> 1;
    let (rs, qs) = adjust_minier(
        &a[as1],
        k_half,
        mi.flag.contains(crate::flags::IdxFlags::HPC),
    );
    let (re, qe) = {
        let last = &a[as1 + cnt1 - 1];
        ((last.x as i32) + 1, (last.y as i32) + 1)
    };

    log::debug!(
        "align1: rid={} rev={} cnt={} rs={} re={} qs={} qe={} qlen={} qseq.len={} ref_len={}",
        rid,
        rev,
        cnt1,
        rs,
        re,
        qs,
        qe,
        qlen,
        qseq.len(),
        ref_len
    );

    // Compute flanking regions
    let flank_l = qs.min(rs).min(opt.max_gap);
    let rs0 = (rs - flank_l).max(0);
    let qs0 = (qs - flank_l).max(0);
    let flank_r = (qlen - qe).min(ref_len - re).min(opt.max_gap);
    let re0 = (re + flank_r).min(ref_len);
    let qe0 = (qe + flank_r).min(qlen);

    if re0 <= rs0 || qe0 <= qs0 {
        log::debug!(
            "align1: bail re0={} rs0={} qe0={} qs0={}",
            re0,
            rs0,
            qe0,
            qs0
        );
        return split_regs;
    }
    if qs0 < 0 || qe0 > qlen {
        log::debug!("align1: bail qs0={} qe0={} qlen={}", qs0, qe0, qlen);
        return split_regs;
    }
    if (qs0 as usize) >= qseq.len() || (qe0 as usize) > qseq.len() {
        log::debug!(
            "align1: bail qs0={} qe0={} qseq.len={}",
            qs0,
            qe0,
            qseq.len()
        );
        return split_regs;
    }

    let mut cigar_ops: Vec<u32> = Vec::new();
    let mut dp_score = 0i32;
    let mut tseq_buf = ALIGN_TSEQ_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let needed = (re0 - rs0 + 1) as usize;
    if tseq_buf.len() < needed {
        tseq_buf.resize(needed, 0);
    }
    let mut rs1 = rs;
    let mut qs1 = qs;

    // === LEFT EXTENSION ===
    if qs > qs0 && rs > rs0 {
        let tlen_ext = (rs - rs0) as usize;
        let qlen_ext = (qs - qs0) as usize;
        if tlen_ext > 0 && qlen_ext > 0 {
            mi.getseq(rid, rs0 as u32, rs as u32, &mut tseq_buf[..tlen_ext]);
            let mut rev_buf = ALIGN_REV_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
            let total_rev = tlen_ext + qlen_ext;
            if rev_buf.len() < total_rev {
                rev_buf.resize(total_rev, 0);
            }
            // Pack reversed query and target into rev_buf: [qrev | trev]
            for k in 0..qlen_ext {
                rev_buf[k] = qseq[qs as usize - 1 - k];
            }
            for k in 0..tlen_ext {
                rev_buf[qlen_ext + k] = tseq_buf[tlen_ext - 1 - k];
            }
            let ez = do_align(
                opt,
                &rev_buf[..qlen_ext],
                &rev_buf[qlen_ext..total_rev],
                mat,
                bw,
                opt.zdrop,
                opt.end_bonus,
                KswFlags::EXTZ_ONLY | KswFlags::RIGHT | KswFlags::REV_CIGAR,
            );
            if !ez.cigar.is_empty() {
                append_cigar(&mut cigar_ops, &ez.cigar);
                dp_score += ez.max;
            }
            if ez.reach_end {
                rs1 = rs - (ez.mqe_t + 1);
                qs1 = qs0;
            } else if ez.max_t >= 0 && ez.max_q >= 0 {
                rs1 = rs - (ez.max_t + 1);
                qs1 = qs - (ez.max_q + 1);
            }
            ALIGN_REV_BUF.with(|c| *c.borrow_mut() = rev_buf);
        }
    }

    // === GAP FILLING between anchors ===
    let is_hpc = mi.flag.contains(crate::flags::IdxFlags::HPC);
    let mut cur_rs = rs;
    let mut cur_qs = qs;
    let mut dropped = false;
    for i in 1..cnt1 {
        let ai = &a[as1 + i];
        // Skip tandem/ignored seeds (except last)
        if i != cnt1 - 1 && (ai.y & (crate::flags::SEED_IGNORE | crate::flags::SEED_TANDEM)) != 0 {
            continue;
        }
        // Compute next anchor position using mm_adjust_minier logic
        // C calls mm_adjust_minier for ALL anchors including the last.
        let (next_re, next_qe) = adjust_minier(ai, k_half, is_hpc);

        // Check if gap is large enough to align, or if it's the last anchor, or LONG_JOIN
        let has_long_join = ai.y & crate::flags::SEED_LONG_JOIN != 0;
        if i == cnt1 - 1
            || has_long_join
            || (next_qe - cur_qs >= opt.min_ksw_len && next_re - cur_rs >= opt.min_ksw_len)
        {
            let gap_tlen = (next_re - cur_rs).max(0) as usize;
            let gap_qlen = (next_qe - cur_qs).max(0) as usize;
            if gap_tlen > 0
                && gap_qlen > 0
                && cur_qs >= 0
                && cur_rs >= 0
                && (cur_qs as usize) < qseq.len()
                && (next_qe as usize) <= qseq.len()
                && cur_rs < ref_len
                && next_re <= ref_len
            {
                // Bandwidth: bw_long for all gaps; for LONG_JOIN, use max(gap_q, gap_t)
                // Matches C: bw1 = bw_long, then overridden for LONG_JOIN
                let use_bw = if has_long_join {
                    (gap_tlen as i32).max(gap_qlen as i32).max(bw_long)
                } else {
                    bw_long
                };
                if gap_tlen > tseq_buf.len() {
                    tseq_buf.resize(gap_tlen, 0);
                }
                mi.getseq(
                    rid,
                    cur_rs as u32,
                    next_re as u32,
                    &mut tseq_buf[..gap_tlen],
                );
                let qsub = &qseq[cur_qs as usize..next_qe as usize];
                // First pass: alignment with approximate z-drop
                let mut ez = do_align(
                    opt,
                    qsub,
                    &tseq_buf[..gap_tlen],
                    mat,
                    use_bw,
                    opt.zdrop,
                    -1,
                    KswFlags::APPROX_MAX,
                );

                // Test z-drop and potential inversion
                if !ez.cigar.is_empty() && !ez.zdropped {
                    let zdrop_code = test_zdrop(opt, qsub, &tseq_buf[..gap_tlen], &ez.cigar, mat);
                    if zdrop_code > 0 {
                        // Second pass: re-align with exact z-drop (use zdrop_inv for inversions)
                        // Note: APPROX_MAX is intentionally NOT set here (matching C's "lift approximate")
                        // so the exact H-tracking path is used, which enables proper z-drop detection.
                        let zdrop2 = if zdrop_code == 2 {
                            opt.zdrop_inv
                        } else {
                            opt.zdrop
                        };
                        ez = do_align(
                            opt,
                            qsub,
                            &tseq_buf[..gap_tlen],
                            mat,
                            use_bw,
                            zdrop2,
                            -1,
                            KswFlags::empty(),
                        );
                    }
                }

                log::debug!("  gap i={} cur_rs={} next_re={} cur_qs={} next_qe={} tlen={} qlen={} bw={} score={} zdropped={}",
                    i, cur_rs, next_re, cur_qs, next_qe, gap_tlen, gap_qlen, use_bw, ez.score, ez.zdropped);
                if !ez.cigar.is_empty() {
                    append_cigar(&mut cigar_ops, &ez.cigar);
                }
                if ez.zdropped {
                    dp_score += ez.max;
                    dropped = true;

                    let zdrop_re = cur_rs + ez.max_t + 1;
                    let zdrop_qe = cur_qs + ez.max_q + 1;

                    // Try to split at the first anchor after the z-drop point,
                    // matching minimap2's mm_align1().
                    let mut split_j = i as isize - 1;
                    while split_j >= 0 {
                        let anchor_x = a[as1 + split_j as usize].x as i32;
                        if anchor_x <= cur_rs + ez.max_t {
                            break;
                        }
                        split_j -= 1;
                    }
                    if split_j < 0 {
                        split_j = 0;
                    }
                    let split_n = as1 as i32 + split_j as i32 + 1 - r.as_;
                    let remaining_cnt = r.cnt - split_n;
                    if remaining_cnt >= opt.min_cnt {
                        if let Some(mut r2) = crate::hit::split_reg(
                            r,
                            split_n,
                            qlen,
                            a,
                            opt.flag.contains(crate::flags::MapFlags::QSTRAND),
                        ) {
                            // Check if zdrop was due to inversion
                            let zdrop_code =
                                test_zdrop(opt, qsub, &tseq_buf[..gap_tlen], &ez.cigar, mat);
                            if zdrop_code == 2 {
                                r2.split_inv = true;
                            }
                            split_regs.push(r2);
                        }
                    }
                    cur_rs = zdrop_re;
                    cur_qs = zdrop_qe;
                    break;
                }
                if ez.score > ksw2::KSW_NEG_INF / 2 {
                    dp_score += ez.score;
                }
            }
            cur_rs = next_re;
            cur_qs = next_qe;
        }
    }
    let mut re1 = cur_rs;
    let mut qe1 = cur_qs;

    // === RIGHT EXTENSION ===
    if !dropped && cur_qs < qe0 && cur_rs < re0 {
        let rext = (re0 - cur_rs) as usize;
        let qext = (qe0 - cur_qs) as usize;
        if rext > 0 && qext > 0 && (cur_qs as usize) < qseq.len() && (qe0 as usize) <= qseq.len() {
            if rext > tseq_buf.len() {
                tseq_buf.resize(rext, 0);
            }
            mi.getseq(rid, cur_rs as u32, re0 as u32, &mut tseq_buf[..rext]);
            let qsub = &qseq[cur_qs as usize..qe0 as usize];
            let ez = do_align(
                opt,
                qsub,
                &tseq_buf[..rext],
                mat,
                bw,
                opt.zdrop,
                opt.end_bonus,
                KswFlags::EXTZ_ONLY,
            );
            if !ez.cigar.is_empty() {
                append_cigar(&mut cigar_ops, &ez.cigar);
                dp_score += ez.max;
            }
            if ez.reach_end {
                re1 = cur_rs + ez.mqe_t + 1;
                qe1 = qe0;
            } else if ez.max_t >= 0 && ez.max_q >= 0 {
                re1 = cur_rs + ez.max_t + 1;
                qe1 = cur_qs + ez.max_q + 1;
            }
        }
    }

    log::debug!(
        "align1: after gap-fill, cigar_ops.len={} dp_score={} re1={} qe1={} rs1={} qs1={}",
        cigar_ops.len(),
        dp_score,
        re1,
        qe1,
        rs1,
        qs1
    );
    if cigar_ops.is_empty() {
        return split_regs;
    }

    // Fix CIGAR: remove leading/trailing I/D, merge ops
    let (q_shift, t_shift) = fix_cigar(&mut cigar_ops);
    rs1 += t_shift;
    qs1 += q_shift;
    if cigar_ops.is_empty() {
        return split_regs;
    }

    if opt.flag.contains(crate::flags::MapFlags::SPLICE) && !rev {
        if let Some(rescue) = rescue_exact_annotated_introns(opt, mi, rid, qseq, rs1, re1, rev) {
            rs1 = rescue.rs;
            re1 = rescue.re;
            qs1 = 0;
            qe1 = qlen;
            cigar_ops = rescue.cigar;
            dp_score = (qlen * opt.a) + rescue.bonus;
        } else if let Some(rescue) = rescue_exact_single_intron(opt, mi, rid, qseq, rs1, re1, rev) {
            rs1 = rescue.rs;
            re1 = rescue.re;
            qs1 = 0;
            qe1 = qlen;
            cigar_ops = rescue.cigar;
            dp_score = (qlen * opt.a) + rescue.bonus;
        }
    }

    // Compute stats from final alignment — reuse tseq_buf
    let final_tlen = (re1 - rs1).max(0) as usize;
    if final_tlen > tseq_buf.len() {
        tseq_buf.resize(final_tlen, 0);
    }
    if final_tlen > 0 {
        mi.getseq(rid, rs1 as u32, re1 as u32, &mut tseq_buf[..final_tlen]);
    }
    let final_qseq = if qs1 >= 0 && qe1 <= qlen && qs1 < qe1 && (qe1 as usize) <= qseq.len() {
        &qseq[qs1 as usize..qe1 as usize]
    } else {
        &[]
    };

    let (is_spliced, trans_strand, splice_bonus) =
        annotate_splice_cigar(opt, mi, rid, rs1, &mut cigar_ops, rev);
    let (mlen, blen, n_ambi, dp_max) = compute_cigar_stats(
        &cigar_ops,
        final_qseq,
        &tseq_buf[..final_tlen],
        mat,
        opt.q,
        opt.e,
    );

    // Convert M to =/X if EQX mode requested
    if opt.flag.contains(crate::flags::MapFlags::EQX) && !final_qseq.is_empty() {
        update_cigar_eqx(&mut cigar_ops, final_qseq, &tseq_buf[..final_tlen]);
    }
    // Return thread-local buffers for reuse
    ALIGN_TSEQ_BUF.with(|c| *c.borrow_mut() = tseq_buf);

    let extra = AlignExtra {
        dp_score: dp_score + splice_bonus,
        dp_max: dp_max.max(dp_score + splice_bonus),
        dp_max2: 0,
        dp_max0: dp_max.max(dp_score + splice_bonus),
        n_ambi,
        trans_strand,
        cigar: Cigar(cigar_ops),
    };

    // Walk CIGAR to determine actual consumed bases for coordinate consistency
    let mut q_cig = 0i32;
    let mut t_cig = 0i32;
    for &c in &extra.cigar.0 {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 7 | 8 => {
                q_cig += len;
                t_cig += len;
            }
            1 => {
                q_cig += len;
            }
            2 | 3 => {
                t_cig += len;
            }
            _ => {}
        }
    }
    r.rs = rs1.max(0);
    r.re = (rs1 + t_cig).min(ref_len);
    if !rev {
        r.qs = qs1.max(0);
        r.qe = (qs1 + q_cig).min(qlen);
    } else {
        r.qs = (qlen - (qs1 + q_cig)).max(0);
    r.qe = (qlen - qs1).min(qlen);
    }
    r.mlen = mlen;
    r.blen = blen;
    r.is_spliced = is_spliced;
    r.extra = Some(Box::new(extra));
    split_regs
}

struct SpliceRescue {
    rs: i32,
    re: i32,
    cigar: Vec<u32>,
    bonus: i32,
}

fn rescue_exact_annotated_introns(
    opt: &MapOpt,
    mi: &MmIdx,
    rid: u32,
    qseq: &[u8],
    current_rs: i32,
    current_re: i32,
    _rev: bool,
) -> Option<SpliceRescue> {
    let db = mi.junc_db.as_ref()?;
    let juncs = db.juncs.get(rid as usize)?;
    if juncs.is_empty() || qseq.len() < 50 {
        return None;
    }
    let ref_len = mi.seqs.get(rid as usize)?.len as i32;
    let flank = (qseq.len() as i32 + opt.max_gap_ref.max(opt.max_gap).max(6)).min(1_000_000);
    let win_st = (current_rs - flank).max(0);
    let win_en = (current_re + flank).min(ref_len);
    if win_en <= win_st {
        return None;
    }
    let win_len = (win_en - win_st) as usize;
    let mut tseq = vec![4u8; win_len];
    mi.getseq(rid, win_st as u32, win_en as u32, &mut tseq);

    let min_first_exon = opt.min_chain_score.max(15) as usize;
    let mut best: Option<SpliceRescue> = None;
    for ts_off in 0..win_len {
        if ts_off + min_first_exon > win_len || qseq.first() != tseq.get(ts_off) {
            continue;
        }
        let mut q = 0usize;
        let mut t = ts_off;
        let mut mrun = 0usize;
        let mut cigar = Vec::new();
        let mut n_introns = 0i32;
        while q < qseq.len() {
            if t < win_len && qseq[q] == tseq[t] {
                q += 1;
                t += 1;
                mrun += 1;
                continue;
            }
            let abs_t = win_st + t as i32;
            let Some(j) = juncs.iter().find(|j| j.st == abs_t && j.en > j.st) else {
                break;
            };
            if mrun > 0 {
                cigar.push((mrun as u32) << 4);
                mrun = 0;
            }
            cigar.push(((j.en - j.st) as u32) << 4 | 3);
            t = (j.en - win_st) as usize;
            n_introns += 1;
            if t > win_len {
                break;
            }
        }
        if q == qseq.len() {
            if mrun > 0 {
                cigar.push((mrun as u32) << 4);
            }
            let matched: i32 = cigar
                .iter()
                .filter(|&&c| (c & 0xf) == 0)
                .map(|&c| (c >> 4) as i32)
                .sum();
            if matched != qseq.len() as i32 || n_introns == 0 {
                continue;
            }
            let re = win_st + t as i32;
            let splice_bonus = n_introns * (opt.junc_bonus - (opt.q2 + 3));
            let rescue = SpliceRescue {
                rs: win_st + ts_off as i32,
                re,
                cigar,
                bonus: splice_bonus,
            };
            let replace = best
                .as_ref()
                .map(|b| rescue.rs < b.rs || (rescue.rs == b.rs && rescue.re > b.re))
                .unwrap_or(true);
            if replace {
                best = Some(rescue);
            }
        }
    }
    best
}

fn rescue_exact_single_intron(
    opt: &MapOpt,
    mi: &MmIdx,
    rid: u32,
    qseq: &[u8],
    current_rs: i32,
    current_re: i32,
    rev: bool,
) -> Option<SpliceRescue> {
    let qlen = qseq.len();
    if qlen < 50 || qlen > 100_000 {
        return None;
    }
    let ref_len = mi.seqs.get(rid as usize)?.len as i32;
    let max_intron = opt.max_gap_ref.max(opt.max_gap).max(6);
    let flank = (qlen as i32 + max_intron).min(1_000_000);
    let win_st = (current_rs - flank).max(0);
    let win_en = (current_re + flank).min(ref_len);
    let win_len = (win_en - win_st) as usize;
    if win_len < qlen + 6 {
        return None;
    }

    let mut tseq = vec![0u8; win_len];
    mi.getseq(rid, win_st as u32, win_en as u32, &mut tseq);

    let min_exon = opt.min_chain_score.max(15) as usize;
    let mut best: Option<(usize, usize, usize, u8, i32)> = None;
    for ts_off in 0..win_len {
        if ts_off + min_exon >= win_len {
            break;
        }
        let mut prefix = 0usize;
        while prefix < qlen && ts_off + prefix < win_len && qseq[prefix] == tseq[ts_off + prefix] {
            prefix += 1;
        }
        if prefix < min_exon {
            continue;
        }
        let max_split = prefix.min(qlen.saturating_sub(min_exon));
        for split in min_exon..=max_split {
            let donor = ts_off + split;
            let remaining = qlen - split;
            if remaining < min_exon {
                continue;
            }
            let min_acc = donor + 6;
            let max_acc = (donor + max_intron as usize).min(win_len.saturating_sub(remaining));
            if min_acc > max_acc {
                continue;
            }
            for acc in min_acc..=max_acc {
                if tseq[acc] != qseq[split] {
                    continue;
                }
                if tseq[acc..acc + remaining] != qseq[split..] {
                    continue;
                }
                let intron_len = (acc - donor) as i32;
                let abs_donor = win_st + donor as i32;
                let strand = annotated_junction_strand(mi, rid, abs_donor, abs_donor + intron_len)
                    .max(infer_splice_strand(opt, mi, rid, abs_donor, intron_len, rev));
                if strand == 0 && opt.flag.intersects(crate::flags::MapFlags::SPLICE_FOR | crate::flags::MapFlags::SPLICE_REV) {
                    continue;
                }
                let splice_cost = opt.q2 + 3;
                let bonus = if annotated_junction_strand(mi, rid, abs_donor, abs_donor + intron_len) != 0 {
                    opt.junc_bonus - splice_cost
                } else {
                    -splice_cost
                };
                let candidate = (ts_off, split, acc, strand, bonus);
                let replace = best
                    .as_ref()
                    .map(|b| split > b.1 || (split == b.1 && remaining > qlen - b.1))
                    .unwrap_or(true);
                if replace {
                    best = Some(candidate);
                }
            }
        }
    }

    let (ts_off, split, acc, _strand, bonus) = best?;
    let intron_len = acc - (ts_off + split);
    let suffix = qlen - split;
    let mut cigar = Vec::with_capacity(3);
    cigar.push(((split as u32) << 4) | 0);
    cigar.push(((intron_len as u32) << 4) | 3);
    cigar.push(((suffix as u32) << 4) | 0);
    Some(SpliceRescue {
        rs: win_st + ts_off as i32,
        re: win_st + (acc + suffix) as i32,
        cigar,
        bonus,
    })
}

fn annotate_splice_cigar(
    opt: &MapOpt,
    mi: &MmIdx,
    rid: u32,
    rs: i32,
    cigar: &mut [u32],
    rev: bool,
) -> (bool, u8, i32) {
    if !opt.flag.contains(crate::flags::MapFlags::SPLICE) {
        return (false, 0, 0);
    }
    let mut t_off = rs;
    let mut is_spliced = false;
    let mut trans_strand = 0u8;
    let mut bonus = 0i32;
    for c in cigar.iter_mut() {
        let op = *c & 0xf;
        let len = (*c >> 4) as i32;
        match op {
            0 | 7 | 8 => t_off += len,
            2 => {
                let annotated = annotated_junction_strand(mi, rid, t_off, t_off + len);
                if len >= 6 || annotated != 0 {
                    *c = (len as u32) << 4 | 3;
                    is_spliced = true;
                    let strand = if annotated != 0 {
                        annotated
                    } else {
                        infer_splice_strand(opt, mi, rid, t_off, len, rev)
                    };
                    if strand == 1 || strand == 2 {
                        trans_strand = strand;
                    }
                    if annotated != 0 {
                        bonus += opt.junc_bonus;
                    } else if strand == 0 {
                        bonus -= splice_signal_penalty(opt, mi, rid, t_off, len, rev);
                    }
                }
                t_off += len;
            }
            3 => {
                is_spliced = true;
                let strand = annotated_junction_strand(mi, rid, t_off, t_off + len);
                if strand == 1 || strand == 2 {
                    trans_strand = strand;
                    bonus += opt.junc_bonus;
                } else {
                    let inferred = infer_splice_strand(opt, mi, rid, t_off, len, rev);
                    if inferred == 1 || inferred == 2 {
                        trans_strand = inferred;
                    } else {
                        bonus -= splice_signal_penalty(opt, mi, rid, t_off, len, rev);
                    }
                }
                t_off += len;
            }
            _ => {}
        }
    }
    (is_spliced, trans_strand, bonus)
}

fn splice_signal_penalty(
    opt: &MapOpt,
    mi: &MmIdx,
    rid: u32,
    st: i32,
    len: i32,
    rev: bool,
) -> i32 {
    if opt.noncan <= 0 || len < 2 {
        return 0;
    }
    let mut donor = [4u8; 2];
    let mut acceptor = [4u8; 2];
    if mi.getseq(rid, st as u32, (st + 2) as u32, &mut donor) != 2
        || mi.getseq(rid, (st + len - 2) as u32, (st + len) as u32, &mut acceptor) != 2
    {
        return opt.noncan;
    }
    if rev {
        revcomp_pair(&mut donor);
        revcomp_pair(&mut acceptor);
        std::mem::swap(&mut donor, &mut acceptor);
    }
    let donor_ok = matches!((donor[0], donor[1]), (2, 3) | (2, 1) | (0, 3));
    let acceptor_ok = matches!((acceptor[0], acceptor[1]), (0, 2) | (0, 1));
    if donor_ok && acceptor_ok {
        0
    } else if opt.flag.contains(crate::flags::MapFlags::SPLICE_FLANK) && (donor_ok || acceptor_ok) {
        opt.noncan / 2
    } else {
        opt.noncan
    }
}

fn revcomp_pair(pair: &mut [u8; 2]) {
    pair.reverse();
    for b in pair {
        if *b < 4 {
            *b = 3 - *b;
        }
    }
}

fn annotated_junction_strand(mi: &MmIdx, rid: u32, st: i32, en: i32) -> u8 {
    let Some(ref db) = mi.junc_db else {
        return 0;
    };
    let Some(juncs) = db.juncs.get(rid as usize) else {
        return 0;
    };
    for j in juncs {
        if j.st == st && j.en == en {
            return match j.strand {
                1 => 1,
                2 => 2,
                _ => 0,
            };
        }
        if j.st > st {
            break;
        }
    }
    0
}

fn infer_splice_strand(opt: &MapOpt, mi: &MmIdx, rid: u32, st: i32, len: i32, rev: bool) -> u8 {
    if opt.flag.contains(crate::flags::MapFlags::SPLICE_FOR)
        && !opt.flag.contains(crate::flags::MapFlags::SPLICE_REV)
    {
        return if rev { 2 } else { 1 };
    }
    if opt.flag.contains(crate::flags::MapFlags::SPLICE_REV)
        && !opt.flag.contains(crate::flags::MapFlags::SPLICE_FOR)
    {
        return if rev { 1 } else { 2 };
    }
    let mut donor = [4u8; 2];
    let mut acceptor = [4u8; 2];
    if len >= 2
        && mi.getseq(rid, st as u32, (st + 2) as u32, &mut donor) == 2
        && mi.getseq(rid, (st + len - 2) as u32, (st + len) as u32, &mut acceptor) == 2
    {
        let plus = matches!(
            (donor[0], donor[1], acceptor[0], acceptor[1]),
            (2, 3, 0, 2) | (2, 1, 0, 2) | (0, 3, 0, 1)
        );
        let minus = matches!(
            (donor[0], donor[1], acceptor[0], acceptor[1]),
            (1, 3, 0, 1) | (1, 3, 2, 1) | (0, 1, 0, 3)
        );
        if plus {
            return if rev { 2 } else { 1 };
        }
        if minus {
            return if rev { 1 } else { 2 };
        }
    }
    0
}

/// Attempt inversion realignment between two split regions.
/// Matches mm_align1_inv() from align.c.
/// Returns an inversion AlignReg if successful.
pub fn align1_inv(
    opt: &MapOpt,
    mi: &MmIdx,
    qlen: i32,
    qseq_fwd: &[u8],
    qseq_rev: &[u8],
    r1: &AlignReg,
    r2: &AlignReg,
    mat: &[i8],
) -> Option<AlignReg> {
    if r1.split & 1 == 0 || r2.split & 2 == 0 {
        return None;
    }
    if r1.parent != r1.id && r1.parent != crate::types::PARENT_TMP_PRI {
        return None;
    }
    if r2.parent != r2.id && r2.parent != crate::types::PARENT_TMP_PRI {
        return None;
    }
    if r1.rid != r2.rid || r1.rev != r2.rev {
        return None;
    }

    let ql = if r1.rev { r1.qs - r2.qe } else { r2.qs - r1.qe };
    let tl = r2.rs - r1.re;
    if ql < opt.min_chain_score || ql > opt.max_gap {
        return None;
    }
    if tl < opt.min_chain_score || tl > opt.max_gap {
        return None;
    }

    // Get reference and query subsequences
    let mut tseq = vec![0u8; tl as usize];
    mi.getseq(r1.rid as u32, r1.re as u32, r2.rs as u32, &mut tseq);

    // Get reverse-complement query for the inversion gap
    let qseq_src = if r1.rev { qseq_fwd } else { qseq_rev };
    let q_start = if r1.rev {
        r2.qe as usize
    } else {
        (qlen - r2.qs) as usize
    };
    if q_start + ql as usize > qseq_src.len() {
        return None;
    }
    let qsub = &qseq_src[q_start..q_start + ql as usize];

    // Reverse both for LL alignment
    let qsub_rev: Vec<u8> = qsub.iter().rev().copied().collect();
    let tseq_rev: Vec<u8> = tseq.iter().rev().copied().collect();

    // Quick score check with ksw_ll_i16
    let (score, q_off_rev, t_off_rev) =
        ksw2::ksw_ll_i16(&qsub_rev, &tseq_rev, 5, mat, opt.q, opt.e);
    if score < opt.min_dp_max {
        return None;
    }

    // Convert reversed offsets back
    let q_off = ql - (q_off_rev + 1);
    let t_off = tl - (t_off_rev + 1);
    if q_off < 0 || t_off < 0 {
        return None;
    }

    // Full extension alignment on the remaining portion
    let bw = (opt.bw as f32 * 1.5) as i32;
    let ez = do_align(
        opt,
        &qsub[q_off as usize..],
        &tseq[t_off as usize..],
        mat,
        bw,
        opt.zdrop,
        -1,
        KswFlags::EXTZ_ONLY,
    );
    if ez.cigar.is_empty() {
        return None;
    }

    let mut r_inv = AlignReg::default();
    r_inv.id = -1;
    r_inv.parent = crate::types::PARENT_UNSET;
    r_inv.inv = true;
    r_inv.rev = !r1.rev;
    r_inv.rid = r1.rid;
    r_inv.div = -1.0;
    if !r_inv.rev {
        r_inv.qs = r2.qe + q_off;
        r_inv.qe = r_inv.qs + ez.max_q + 1;
    } else {
        r_inv.qe = r2.qs - q_off;
        r_inv.qs = r_inv.qe - (ez.max_q + 1);
    }
    r_inv.rs = r1.re + t_off;
    r_inv.re = r_inv.rs + ez.max_t + 1;

    let extra = AlignExtra {
        dp_score: ez.max,
        dp_max: ez.max,
        dp_max2: 0,
        dp_max0: ez.max,
        n_ambi: 0,
        trans_strand: 0,
        cigar: Cigar(ez.cigar),
    };
    r_inv.extra = Some(Box::new(extra));
    Some(r_inv)
}

/// Convert M operations to =/X based on actual base comparison.
/// Matches mm_update_cigar_eqx() from align.c.
fn update_cigar_eqx(cigar: &mut Vec<u32>, qseq: &[u8], tseq: &[u8]) {
    let mut new_cigar = Vec::new();
    let mut q_off = 0usize;
    let mut t_off = 0usize;
    for &c in cigar.iter() {
        let op = c & 0xf;
        let len = (c >> 4) as usize;
        if op == 0 {
            // M → split into runs of = and X
            let mut i = 0;
            while i < len {
                // Run of matches (=)
                let mut n_eq = 0;
                while i + n_eq < len
                    && q_off + i + n_eq < qseq.len()
                    && t_off + i + n_eq < tseq.len()
                    && qseq[q_off + i + n_eq] == tseq[t_off + i + n_eq]
                {
                    n_eq += 1;
                }
                if n_eq > 0 {
                    append_cigar(&mut new_cigar, &[((n_eq as u32) << 4) | 7]); // =
                    i += n_eq;
                }
                // Run of mismatches (X)
                let mut n_x = 0;
                while i + n_x < len
                    && q_off + i + n_x < qseq.len()
                    && t_off + i + n_x < tseq.len()
                    && qseq[q_off + i + n_x] != tseq[t_off + i + n_x]
                {
                    n_x += 1;
                }
                if n_x > 0 {
                    append_cigar(&mut new_cigar, &[((n_x as u32) << 4) | 8]); // X
                    i += n_x;
                }
                if n_eq == 0 && n_x == 0 {
                    break;
                } // safety
            }
            q_off += len;
            t_off += len;
        } else {
            append_cigar(&mut new_cigar, &[c]);
            match op {
                1 => q_off += len,     // I
                2 | 3 => t_off += len, // D or N
                _ => {
                    q_off += len;
                    t_off += len;
                }
            }
        }
    }
    *cigar = new_cigar;
}

/// Compute match/block lengths from CIGAR and sequences.
fn compute_cigar_stats(
    cigar: &[u32],
    qseq: &[u8],
    tseq: &[u8],
    mat: &[i8],
    q_pen: i32,
    e_pen: i32,
) -> (i32, i32, u32, i32) {
    let mut mlen = 0i32;
    let mut blen = 0i32;
    let mut n_ambi = 0u32;
    let mut score_cur = 0.0f64;
    let mut dp_max = 0.0f64;
    let mut q_off = 0usize;
    let mut t_off = 0usize;
    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as usize;
        match op {
            0 | 7 | 8 => {
                let safe_len = len
                    .min(qseq.len().saturating_sub(q_off))
                    .min(tseq.len().saturating_sub(t_off));
                for l in 0..safe_len {
                    // SAFETY: l < safe_len ensures q_off+l < qseq.len() and t_off+l < tseq.len()
                    let (qb, tb) = unsafe {
                        (
                            *qseq.get_unchecked(q_off + l),
                            *tseq.get_unchecked(t_off + l),
                        )
                    };
                    if qb > 3 || tb > 3 {
                        n_ambi += 1;
                    } else {
                        blen += 1;
                        if qb == tb {
                            mlen += 1;
                        }
                    }
                    // SAFETY: tb < 5, qb < 5 (encoded bases), so tb*5+qb < 25 <= mat.len()
                    score_cur +=
                        unsafe { *mat.get_unchecked(tb as usize * 5 + qb as usize) } as f64;
                    if score_cur < 0.0 {
                        score_cur = 0.0;
                    }
                    if score_cur > dp_max {
                        dp_max = score_cur;
                    }
                }
                q_off += len;
                t_off += len;
            }
            1 => {
                let safe_len = len.min(qseq.len().saturating_sub(q_off));
                let mut ambi = 0usize;
                for l in 0..safe_len {
                    if unsafe { *qseq.get_unchecked(q_off + l) } > 3 {
                        ambi += 1;
                    }
                }
                n_ambi += ambi as u32;
                blen += (safe_len - ambi) as i32;
                score_cur -=
                    q_pen as f64 + e_pen as f64 * crate::chain::mg_log2(1.0 + len as f32) as f64;
                if score_cur < 0.0 {
                    score_cur = 0.0;
                }
                q_off += len;
            }
            2 => {
                let safe_len = len.min(tseq.len().saturating_sub(t_off));
                let mut ambi = 0usize;
                for l in 0..safe_len {
                    if unsafe { *tseq.get_unchecked(t_off + l) } > 3 {
                        ambi += 1;
                    }
                }
                n_ambi += ambi as u32;
                blen += (safe_len - ambi) as i32;
                score_cur -=
                    q_pen as f64 + e_pen as f64 * crate::chain::mg_log2(1.0 + len as f32) as f64;
                if score_cur < 0.0 {
                    score_cur = 0.0;
                }
                t_off += len;
            }
            3 => {
                t_off += len;
            } // N_SKIP
            _ => {
                q_off += len;
                t_off += len;
            }
        }
    }
    (mlen, blen, n_ambi, (dp_max + 0.499) as i32)
}

/// Perform DP alignment for all regions.
///
/// Simplified version of mm_align_skeleton() from align.c.
pub fn align_skeleton(
    opt: &MapOpt,
    mi: &MmIdx,
    qlen: i32,
    qstr: &[u8], // ASCII query
    regs: &mut Vec<AlignReg>,
    a: &mut [Mm128],
) {
    if regs.is_empty() {
        return;
    }

    // Encode query (forward and reverse complement) using thread-local buffers
    let mut qseq_fwd = ALIGN_QSEQ_FWD.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut qseq_rev = ALIGN_QSEQ_REV.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let ql = qlen as usize;
    qseq_fwd.clear();
    qseq_fwd.resize(ql, 0);
    qseq_rev.clear();
    qseq_rev.resize(ql, 0);
    for i in 0..ql {
        let c = unsafe { *SEQ_NT4_TABLE.get_unchecked(*qstr.get_unchecked(i) as usize) };
        qseq_fwd[i] = c;
        qseq_rev[ql - 1 - i] = if c < 4 { 3 - c } else { 4 };
    }

    // Generate scoring matrix
    let mut mat = Vec::new();
    if opt.transition != 0 && opt.b != opt.transition {
        score::gen_ts_mat(5, &mut mat, opt.a, opt.b, opt.transition, opt.sc_ambi);
    } else {
        score::gen_simple_mat(5, &mut mat, opt.a, opt.b, opt.sc_ambi);
    }

    // Align each region, collecting any split regions.
    // Like C's mm_align_skeleton, split regions are inserted back into the work queue
    // so they can be further split (cascading z-drop splits).
    let mut work: Vec<AlignReg> = regs.drain(..).collect();
    let mut new_regs: Vec<AlignReg> = Vec::new();
    let mut wi = 0;
    while wi < work.len() {
        let mut r = work[wi].clone();
        let splits = align1(opt, mi, qlen, &qseq_fwd, &qseq_rev, &mut r, &mut *a, &mat);
        if opt.flag.contains(crate::flags::MapFlags::SPLICE) && mi.junc_db.is_some() {
            let ts = r
                .extra
                .as_ref()
                .map(|p| p.trans_strand as i32)
                .unwrap_or(0);
            crate::jump::jump_split(mi, opt, qlen, &qseq_fwd, &mut r, ts);
        }
        new_regs.push(r);
        // Insert splits right after current position so they're processed next
        for (j, sr) in splits.into_iter().enumerate() {
            work.insert(wi + 1 + j, sr);
        }
        wi += 1;
    }

    // Try inversion realignment between consecutive split regions
    if !(opt.flag.contains(crate::flags::MapFlags::NO_INV)) {
        let mut inv_regs: Vec<AlignReg> = Vec::new();
        for i in 1..new_regs.len() {
            if new_regs[i].split_inv {
                if let Some(inv) = align1_inv(
                    opt,
                    mi,
                    qlen,
                    &qseq_fwd,
                    &qseq_rev,
                    &new_regs[i - 1],
                    &new_regs[i],
                    &mat,
                ) {
                    inv_regs.push(inv);
                }
            }
        }
        new_regs.extend(inv_regs);
    }
    *regs = new_regs;

    // Filter and sort
    crate::hit::filter_regs(opt, qlen, regs);
    // Assembly ranking: recalibrate dp_max based on divergence
    if !(opt.flag.intersects(
        crate::flags::MapFlags::SR
            | crate::flags::MapFlags::SR_RNA
            | crate::flags::MapFlags::ALL_CHAINS,
    )) && opt.split_prefix.is_none()
        && qlen >= opt.rank_min_len
    {
        crate::hit::update_dp_max(qlen, regs, opt.rank_frac, opt.a, opt.b);
        crate::hit::filter_regs(opt, qlen, regs);
    }
    crate::hit::hit_sort(regs, opt.alt_drop);
    // Return thread-local buffers
    ALIGN_QSEQ_FWD.with(|c| *c.borrow_mut() = qseq_fwd);
    ALIGN_QSEQ_REV.with(|c| *c.borrow_mut() = qseq_rev);
}

#[cfg(test)]
mod tests {
    use super::*;
    use score::gen_simple_mat;

    #[test]
    fn test_align_pair() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = align_pair(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        // CIGAR must be correct; score may differ between SIMD/scalar implementations
        assert_eq!(cigar_to_string(&ez.cigar), "8M");
    }

    #[test]
    fn test_align_pair_dual() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = align_pair_dual(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            24,
            1,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert_eq!(ez.score, 16);
    }

    #[test]
    fn test_cigar_to_string() {
        let cigar = vec![
            (10u32 << 4) | 0, // 10M
            (2u32 << 4) | 1,  // 2I
            (5u32 << 4) | 0,  // 5M
        ];
        assert_eq!(cigar_to_string(&cigar), "10M2I5M");
    }

    #[test]
    fn test_update_cigar_eqx() {
        let qseq = [0u8, 1, 2, 3, 0, 0, 2, 3]; // ACGTAAGTGT
        let tseq = [0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let mut cigar = vec![(8u32 << 4) | 0]; // 8M
        update_cigar_eqx(&mut cigar, &qseq, &tseq);
        // Should be 5=1X2= (positions 5 differs: 0 vs 1)
        let s = cigar_to_string(&cigar);
        assert!(
            s.contains('=') && s.contains('X'),
            "EQX CIGAR should contain = and X, got: {}",
            s
        );
    }

    #[test]
    fn test_annotate_splice_cigar_converts_intronic_deletion() {
        let mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"ACGTGTAAAAAGACGT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        let mut opt = MapOpt::default();
        opt.flag |= crate::flags::MapFlags::SPLICE
            | crate::flags::MapFlags::SPLICE_FOR
            | crate::flags::MapFlags::SPLICE_REV;
        let mut cigar = vec![
            (4u32 << 4),
            (8u32 << 4) | 2,
            (4u32 << 4),
        ];

        let (is_spliced, trans_strand, _) =
            annotate_splice_cigar(&opt, &mi, 0, 0, &mut cigar, false);

        assert!(is_spliced);
        assert_eq!(trans_strand, 1);
        assert_eq!(cigar_to_string(&cigar), "4M8N4M");
    }

    #[test]
    fn test_annotate_splice_cigar_penalizes_noncanonical_skip() {
        let mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"ACGTAAAAAAAAACGT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        let mut opt = MapOpt::default();
        opt.flag |= crate::flags::MapFlags::SPLICE
            | crate::flags::MapFlags::SPLICE_FOR
            | crate::flags::MapFlags::SPLICE_REV;
        opt.noncan = 9;
        let mut cigar = vec![(4u32 << 4), (8u32 << 4) | 2, (4u32 << 4)];

        let (is_spliced, trans_strand, bonus) =
            annotate_splice_cigar(&opt, &mi, 0, 0, &mut cigar, false);

        assert!(is_spliced);
        assert_eq!(trans_strand, 0);
        assert_eq!(bonus, -9);
        assert_eq!(cigar_to_string(&cigar), "4M8N4M");
    }

    #[test]
    fn test_simd_dispatch() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = ksw2_simd::ksw_extz2_dispatch(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert_eq!(crate::align::cigar_to_string(&ez.cigar), "8M");
    }
}
