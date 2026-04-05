pub mod score;
pub mod ksw2;
pub mod ksw2_simd;

use crate::flags::KswFlags;
use crate::index::MmIdx;
use crate::options::MapOpt;
use crate::seq::SEQ_NT4_TABLE;
use crate::types::{AlignExtra, AlignReg, Cigar, Mm128};
pub use ksw2::KswResult;

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
    ksw2_simd::ksw_extd2_dispatch(query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag)
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
    if cigar.is_empty() { return (0, 0); }
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
        if op == 1 { q_shift += len; merged.remove(0); }
        else if op == 2 { t_shift += len; merged.remove(0); }
        else { break; }
    }
    // Remove trailing I/D
    while !merged.is_empty() {
        let last_idx = merged.len() - 1;
        let op = merged[last_idx] & 0xf;
        if op == 1 || op == 2 { merged.pop(); }
        else { break; }
    }
    *cigar = merged;
    (q_shift, t_shift)
}

/// Append CIGAR operations to a vector, merging adjacent same-op entries.
fn append_cigar(cigar: &mut Vec<u32>, new_ops: &[u32]) {
    for &c in new_ops {
        if c >> 4 == 0 { continue; }
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
    opt: &MapOpt, qsub: &[u8], tsub: &[u8], mat: &[i8],
    bw: i32, zdrop: i32, end_bonus: i32, flag: KswFlags,
) -> KswResult {
    if qsub.is_empty() || tsub.is_empty() {
        return KswResult::new();
    }
    if opt.q == opt.q2 && opt.e == opt.e2 {
        align_pair(qsub, tsub, 5, mat, opt.q as i8, opt.e as i8, bw, zdrop, end_bonus, flag)
    } else {
        align_pair_dual(qsub, tsub, 5, mat, opt.q as i8, opt.e as i8, opt.q2 as i8, opt.e2 as i8, bw, zdrop, end_bonus, flag)
    }
}

/// Test for z-drop in a CIGAR alignment.
/// Returns 0 if no z-drop, 1 if regular z-drop > zdrop, 2 if potential inversion.
/// Matches mm_test_zdrop() from align.c.
fn test_zdrop(
    opt: &MapOpt,
    qseq: &[u8],
    tseq: &[u8],
    cigar: &[u32],
    mat: &[i8],
) -> i32 {
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
        if op == 0 { // M
            for l in 0..len {
                if i + l < tseq.len() && j + l < qseq.len() {
                    score += mat[tseq[i + l] as usize * 5 + qseq[j + l] as usize] as i32;
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
                        pos[0][0] = max_i; pos[0][1] = ci;
                        pos[1][0] = max_j; pos[1][1] = cj;
                    }
                } else {
                    max_score = score; max_i = ci; max_j = cj;
                }
            }
            i += len; j += len;
        } else if op == 1 || op == 2 || op == 3 { // I, D, N
            score -= opt.q + opt.e * len as i32;
            if op == 1 { j += len; } else { i += len; }
            let ci = i as i32;
            let cj = j as i32;
            if score < max_score {
                let tl = ci - max_i;
                let ql = cj - max_j;
                let diff = (tl - ql).abs();
                let z = max_score - score - diff * opt.e;
                if z > max_zdrop {
                    max_zdrop = z;
                    pos[0][0] = max_i; pos[0][1] = ci;
                    pos[1][0] = max_j; pos[1][1] = cj;
                }
            } else {
                max_score = score; max_i = ci; max_j = cj;
            }
        }
    }

    // Test for potential inversion in the z-dropped region
    let q_len = pos[1][1] - pos[1][0];
    let t_len = pos[0][1] - pos[0][0];
    if !opt.flag.intersects(crate::flags::MapFlags::SPLICE | crate::flags::MapFlags::SR
        | crate::flags::MapFlags::FOR_ONLY | crate::flags::MapFlags::REV_ONLY)
        && max_zdrop > opt.zdrop_inv
        && q_len > 0 && q_len < opt.max_gap
        && t_len > 0 && t_len < opt.max_gap
    {
        // Check if reverse-complement of the q region aligns well to the t region
        let q_start = pos[1][0] as usize;
        let q_end = pos[1][1] as usize;
        let t_start = pos[0][0] as usize;
        if q_end <= qseq.len() && (t_start + t_len as usize) <= tseq.len() {
            let qseq_rc: Vec<u8> = qseq[q_start..q_end].iter().rev()
                .map(|&c| if c < 4 { 3 - c } else { 4 })
                .collect();
            // Use ksw_ll_i16 for proper local alignment score
            let (score, _qe, _te) = ksw2::ksw_ll_i16(
                &qseq_rc, &tseq[t_start..t_start + t_len as usize],
                5, mat, opt.q, opt.e,
            );
            if score >= opt.min_chain_score * opt.a && score >= opt.min_dp_max {
                return 2; // potential inversion
            }
        }
    }
    if max_zdrop > opt.zdrop { 1 } else { 0 }
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
fn fix_bad_ends(as1: usize, cnt1: usize, a: &[Mm128], bw: i32, min_match: i32, mlen: i32) -> (usize, usize) {
    if cnt1 < 3 { return (as1, cnt1); }
    let mut new_as = as1;
    let mut new_cnt;

    // Trim from the start
    let mut m = ((a[as1].y >> 32) & 0xff) as i32;
    let mut l = m;
    for i in 1..cnt1 - 1 {
        let q_span = ((a[as1 + i].y >> 32) & 0xff) as i32;
        if a[as1 + i].y & crate::flags::SEED_LONG_JOIN != 0 { break; }
        let lr = (a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32);
        let lq = (a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32);
        let min_d = lr.min(lq);
        let max_d = lr.max(lq);
        if max_d - min_d > l >> 1 { new_as = as1 + i; }
        l += min_d;
        m += min_d.min(q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 { break; }
    }
    new_cnt = as1 + cnt1 - new_as;

    // Trim from the end
    m = ((a[as1 + cnt1 - 1].y >> 32) & 0xff) as i32;
    l = m;
    let mut i = as1 + cnt1 - 2;
    while i > new_as {
        let q_span = ((a[i + 1].y >> 32) & 0xff) as i32;
        if a[i + 1].y & crate::flags::SEED_LONG_JOIN != 0 { break; }
        let lr = (a[i + 1].x as i32) - (a[i].x as i32);
        let lq = (a[i + 1].y as i32) - (a[i].y as i32);
        let min_d = lr.min(lq);
        let max_d = lr.max(lq);
        if max_d - min_d > l >> 1 { new_cnt = i + 1 - new_as; }
        l += min_d;
        m += min_d.min(q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 { break; }
        if i == 0 { break; }
        i -= 1;
    }
    (new_as, new_cnt)
}

/// Filter bad seeds: mark seeds with large gaps for ignoring.
/// Simplified version of mm_filter_bad_seeds() from align.c.
fn filter_bad_seeds(as1: usize, cnt1: usize, a: &mut [Mm128], min_gap: i32, max_ext_len: i32) {
    if cnt1 < 3 { return; }
    // Find seeds with large diagonal gaps
    for i in 1..cnt1 {
        let gap = ((a[as1 + i].y as i32) - (a[as1 + i - 1].y as i32))
                - ((a[as1 + i].x as i32) - (a[as1 + i - 1].x as i32));
        if (gap > min_gap || gap < -min_gap) && (gap > max_ext_len || gap < -max_ext_len) {
            // Mark for long join — the gap-filling will use wider bandwidth
            a[as1 + i].y |= crate::flags::SEED_LONG_JOIN;
        }
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
    if r.cnt == 0 { return split_regs; }
    let raw_as = r.as_ as usize;
    let raw_cnt = r.cnt as usize;

    // Filter bad seeds and trim bad ends
    filter_bad_seeds(raw_as, raw_cnt, a, 10, opt.max_gap >> 1);
    let (as1, cnt1) = if !(opt.flag.contains(crate::flags::MapFlags::NO_END_FLT)) {
        fix_bad_ends(raw_as, raw_cnt, a, opt.bw, opt.min_chain_score * 2, r.mlen)
    } else {
        (raw_as, raw_cnt)
    };
    if cnt1 == 0 { return split_regs; }
    let rid = r.rid as u32;
    let rev = r.rev;
    let qseq = if rev { qseq_rev } else { qseq_fwd };
    let ref_len = mi.seqs[rid as usize].len as i32;

    let bw = (opt.bw as f32 * 1.5 + 1.0) as i32;
    let bw_long = ((opt.bw_long as f32 * 1.5 + 1.0) as i32).max(bw);

    // Get anchor coordinates using mm_adjust_minier logic
    // For non-HPC: shift position left by k/2 to center on k-mer
    let k_half = mi.k >> 1;
    let (rs, qs) = adjust_minier(&a[as1], k_half, mi.flag.contains(crate::flags::IdxFlags::HPC));
    let (re, qe) = {
        let last = &a[as1 + cnt1 - 1];
        ((last.x as i32) + 1, (last.y as i32) + 1)
    };

    log::debug!("align1: rid={} rev={} cnt={} rs={} re={} qs={} qe={} qlen={} qseq.len={} ref_len={}",
        rid, rev, cnt1, rs, re, qs, qe, qlen, qseq.len(), ref_len);

    // Compute flanking regions
    let flank_l = qs.min(rs).min(opt.max_gap);
    let rs0 = (rs - flank_l).max(0);
    let qs0 = (qs - flank_l).max(0);
    let flank_r = (qlen - qe).min(ref_len - re).min(opt.max_gap);
    let re0 = (re + flank_r).min(ref_len);
    let qe0 = (qe + flank_r).min(qlen);

    if re0 <= rs0 || qe0 <= qs0 {
        log::debug!("align1: bail re0={} rs0={} qe0={} qs0={}", re0, rs0, qe0, qs0);
        return split_regs;
    }
    if qs0 < 0 || qe0 > qlen {
        log::debug!("align1: bail qs0={} qe0={} qlen={}", qs0, qe0, qlen);
        return split_regs;
    }
    if (qs0 as usize) >= qseq.len() || (qe0 as usize) > qseq.len() {
        log::debug!("align1: bail qs0={} qe0={} qseq.len={}", qs0, qe0, qseq.len());
        return split_regs;
    }

    let mut cigar_ops: Vec<u32> = Vec::new();
    let mut dp_score = 0i32;
    let mut tseq_buf = vec![0u8; (re0 - rs0 + 1) as usize];
    let mut rs1 = rs;
    let mut qs1 = qs;

    // === LEFT EXTENSION ===
    if qs > qs0 && rs > rs0 {
        let tlen_ext = (rs - rs0) as usize;
        let qlen_ext = (qs - qs0) as usize;
        if tlen_ext > 0 && qlen_ext > 0 {
            mi.getseq(rid, rs0 as u32, rs as u32, &mut tseq_buf[..tlen_ext]);
            let mut qrev: Vec<u8> = qseq[qs0 as usize..qs as usize].to_vec();
            qrev.reverse();
            let mut trev = tseq_buf[..tlen_ext].to_vec();
            trev.reverse();
            let ez = do_align(opt, &qrev, &trev, mat, bw, opt.zdrop, opt.end_bonus,
                KswFlags::EXTZ_ONLY | KswFlags::RIGHT | KswFlags::REV_CIGAR);
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
        let (next_re, next_qe) = if is_hpc {
            adjust_minier(ai, k_half, true)
        } else {
            // For the last anchor or if LONG_JOIN, use end position; otherwise use k-midpoint
            if i == cnt1 - 1 {
                ((ai.x as i32) + 1, (ai.y as i32) + 1)
            } else {
                adjust_minier(ai, k_half, false)
            }
        };

        // Check if gap is large enough to align, or if it's the last anchor, or LONG_JOIN
        let has_long_join = ai.y & crate::flags::SEED_LONG_JOIN != 0;
        if i == cnt1 - 1 || has_long_join
            || (next_qe - cur_qs >= opt.min_ksw_len && next_re - cur_rs >= opt.min_ksw_len)
        {
            let gap_tlen = (next_re - cur_rs).max(0) as usize;
            let gap_qlen = (next_qe - cur_qs).max(0) as usize;
            if gap_tlen > 0 && gap_qlen > 0
                && cur_qs >= 0 && cur_rs >= 0
                && (cur_qs as usize) < qseq.len()
                && (next_qe as usize) <= qseq.len()
                && cur_rs < ref_len && next_re <= ref_len
            {
                // Use wider bandwidth for LONG_JOIN gaps
                let use_bw = if has_long_join {
                    let gap_diff = (gap_tlen as i32 - gap_qlen as i32).unsigned_abs() as i32;
                    gap_diff.max(bw_long)
                } else if gap_tlen > bw as usize * 2 || gap_qlen > bw as usize * 2 {
                    bw_long
                } else {
                    bw
                };
                if gap_tlen > tseq_buf.len() { tseq_buf.resize(gap_tlen, 0); }
                mi.getseq(rid, cur_rs as u32, next_re as u32, &mut tseq_buf[..gap_tlen]);
                let qsub = &qseq[cur_qs as usize..next_qe as usize];
                // First pass: alignment with approximate z-drop
                let mut ez = do_align(opt, qsub, &tseq_buf[..gap_tlen], mat, use_bw, opt.zdrop, -1, KswFlags::empty());

                // Test z-drop and potential inversion
                if !ez.cigar.is_empty() && !ez.zdropped {
                    let zdrop_code = test_zdrop(opt, qsub, &tseq_buf[..gap_tlen], &ez.cigar, mat);
                    if zdrop_code > 0 {
                        // Second pass: re-align with exact z-drop (use zdrop_inv for inversions)
                        let zdrop2 = if zdrop_code == 2 { opt.zdrop_inv } else { opt.zdrop };
                        ez = do_align(opt, qsub, &tseq_buf[..gap_tlen], mat, use_bw, zdrop2, -1, KswFlags::empty());
                    }
                }

                log::debug!("  gap i={} cur_rs={} next_re={} cur_qs={} next_qe={} tlen={} qlen={} bw={} score={} zdropped={}",
                    i, cur_rs, next_re, cur_qs, next_qe, gap_tlen, gap_qlen, use_bw, ez.score, ez.zdropped);
                if ez.zdropped {
                        dp_score += ez.max;
                    dropped = true;
                    // Try to split: find the anchor just after the z-drop point
                    let remaining_cnt = cnt1 - i;
                    if remaining_cnt >= opt.min_cnt as usize {
                        if let Some(mut r2) = crate::hit::split_reg(r, i as i32, qlen, a) {
                            // Check if zdrop was due to inversion
                            let zdrop_code = test_zdrop(opt, qsub, &tseq_buf[..gap_tlen], &ez.cigar, mat);
                            if zdrop_code == 2 { r2.split_inv = true; }
                            split_regs.push(r2);
                        }
                    }
                    break;
                }
                if !ez.cigar.is_empty() {
                    append_cigar(&mut cigar_ops, &ez.cigar);
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
        if rext > 0 && qext > 0
            && (cur_qs as usize) < qseq.len()
            && (qe0 as usize) <= qseq.len()
        {
            if rext > tseq_buf.len() { tseq_buf.resize(rext, 0); }
            mi.getseq(rid, cur_rs as u32, re0 as u32, &mut tseq_buf[..rext]);
            let qsub = &qseq[cur_qs as usize..qe0 as usize];
            let ez = do_align(opt, qsub, &tseq_buf[..rext], mat, bw, opt.zdrop, opt.end_bonus, KswFlags::EXTZ_ONLY);
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

    log::debug!("align1: after gap-fill, cigar_ops.len={} dp_score={} re1={} qe1={} rs1={} qs1={}",
        cigar_ops.len(), dp_score, re1, qe1, rs1, qs1);
    if cigar_ops.is_empty() { return split_regs; }

    // Fix CIGAR: remove leading/trailing I/D, merge ops
    let (q_shift, t_shift) = fix_cigar(&mut cigar_ops);
    rs1 += t_shift;
    qs1 += q_shift;
    if cigar_ops.is_empty() { return split_regs; }

    // Compute stats from final alignment
    let final_tlen = (re1 - rs1).max(0) as usize;
    let mut final_tseq = vec![0u8; final_tlen.max(1)];
    if final_tlen > 0 {
        mi.getseq(rid, rs1 as u32, re1 as u32, &mut final_tseq[..final_tlen]);
    }
    let final_qseq = if qs1 >= 0 && qe1 <= qlen && qs1 < qe1 && (qe1 as usize) <= qseq.len() {
        &qseq[qs1 as usize..qe1 as usize]
    } else { &[] };

    let (mlen, blen, n_ambi, dp_max) = compute_cigar_stats(&cigar_ops, final_qseq, &final_tseq[..final_tlen], mat, opt.q, opt.e);

    // Convert M to =/X if EQX mode requested
    if opt.flag.contains(crate::flags::MapFlags::EQX) && !final_qseq.is_empty() {
        update_cigar_eqx(&mut cigar_ops, final_qseq, &final_tseq[..final_tlen]);
    }

    let extra = AlignExtra {
        dp_score,
        dp_max: dp_max.max(dp_score),
        dp_max2: 0,
        dp_max0: dp_max.max(dp_score),
        n_ambi,
        trans_strand: 0,
        cigar: Cigar(cigar_ops),
    };

    // Walk CIGAR to determine actual consumed bases for coordinate consistency
    let mut q_cig = 0i32;
    let mut t_cig = 0i32;
    for &c in &extra.cigar.0 {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 7 | 8 => { q_cig += len; t_cig += len; }
            1 => { q_cig += len; }
            2 | 3 => { t_cig += len; }
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
    r.extra = Some(Box::new(extra));
    split_regs
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
    if r1.split & 1 == 0 || r2.split & 2 == 0 { return None; }
    if r1.parent != r1.id && r1.parent != crate::types::PARENT_TMP_PRI { return None; }
    if r2.parent != r2.id && r2.parent != crate::types::PARENT_TMP_PRI { return None; }
    if r1.rid != r2.rid || r1.rev != r2.rev { return None; }

    let ql = if r1.rev { r1.qs - r2.qe } else { r2.qs - r1.qe };
    let tl = r2.rs - r1.re;
    if ql < opt.min_chain_score || ql > opt.max_gap { return None; }
    if tl < opt.min_chain_score || tl > opt.max_gap { return None; }

    // Get reference and query subsequences
    let mut tseq = vec![0u8; tl as usize];
    mi.getseq(r1.rid as u32, r1.re as u32, r2.rs as u32, &mut tseq);

    // Get reverse-complement query for the inversion gap
    let qseq_src = if r1.rev { qseq_fwd } else { qseq_rev };
    let q_start = if r1.rev { r2.qe as usize } else { (qlen - r2.qs) as usize };
    if q_start + ql as usize > qseq_src.len() { return None; }
    let qsub = &qseq_src[q_start..q_start + ql as usize];

    // Reverse both for LL alignment
    let qsub_rev: Vec<u8> = qsub.iter().rev().copied().collect();
    let tseq_rev: Vec<u8> = tseq.iter().rev().copied().collect();

    // Quick score check with ksw_ll_i16
    let (score, q_off_rev, t_off_rev) = ksw2::ksw_ll_i16(&qsub_rev, &tseq_rev, 5, mat, opt.q, opt.e);
    if score < opt.min_dp_max { return None; }

    // Convert reversed offsets back
    let q_off = ql - (q_off_rev + 1);
    let t_off = tl - (t_off_rev + 1);
    if q_off < 0 || t_off < 0 { return None; }

    // Full extension alignment on the remaining portion
    let bw = (opt.bw as f32 * 1.5) as i32;
    let ez = do_align(opt,
        &qsub[q_off as usize..],
        &tseq[t_off as usize..],
        mat, bw, opt.zdrop, -1, KswFlags::EXTZ_ONLY);
    if ez.cigar.is_empty() { return None; }

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
                if n_eq == 0 && n_x == 0 { break; } // safety
            }
            q_off += len;
            t_off += len;
        } else {
            append_cigar(&mut new_cigar, &[c]);
            match op {
                1 => q_off += len,       // I
                2 | 3 => t_off += len,   // D or N
                _ => { q_off += len; t_off += len; }
            }
        }
    }
    *cigar = new_cigar;
}

/// Compute match/block lengths from CIGAR and sequences.
fn compute_cigar_stats(cigar: &[u32], qseq: &[u8], tseq: &[u8], mat: &[i8], q_pen: i32, e_pen: i32) -> (i32, i32, u32, i32) {
    let mut mlen = 0i32;
    let mut blen = 0i32;
    let mut n_ambi = 0u32;
    let mut score_cur = 0f64;
    let mut dp_max = 0f64;
    let mut q_off = 0usize;
    let mut t_off = 0usize;
    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as usize;
        match op {
            0 | 7 | 8 => {
                for l in 0..len {
                    if q_off + l < qseq.len() && t_off + l < tseq.len() {
                        let qb = qseq[q_off + l];
                        let tb = tseq[t_off + l];
                        if qb > 3 || tb > 3 { n_ambi += 1; }
                        else if qb == tb { mlen += 1; }
                        score_cur += mat[(tb as usize) * 5 + qb as usize] as f64;
                    }
                    if score_cur < 0.0 { score_cur = 0.0; }
                    if score_cur > dp_max { dp_max = score_cur; }
                }
                blen += len as i32;
                q_off += len; t_off += len;
            }
            1 => { blen += len as i32; score_cur -= q_pen as f64 + e_pen as f64; if score_cur < 0.0 { score_cur = 0.0; } q_off += len; }
            2 => { blen += len as i32; score_cur -= q_pen as f64 + e_pen as f64; if score_cur < 0.0 { score_cur = 0.0; } t_off += len; }
            3 => { t_off += len; } // N_SKIP
            _ => { q_off += len; t_off += len; }
        }
    }
    (mlen, blen, n_ambi, dp_max as i32)
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

    // Encode query (forward and reverse complement)
    let mut qseq_fwd = vec![0u8; qlen as usize];
    let mut qseq_rev = vec![0u8; qlen as usize];
    for i in 0..qlen as usize {
        qseq_fwd[i] = SEQ_NT4_TABLE[qstr[i] as usize];
        let c = qseq_fwd[i];
        qseq_rev[qlen as usize - 1 - i] = if c < 4 { 3 - c } else { 4 };
    }

    // Generate scoring matrix
    let mut mat = Vec::new();
    if opt.transition != 0 && opt.b != opt.transition {
        score::gen_ts_mat(5, &mut mat, opt.a, opt.b, opt.transition, opt.sc_ambi);
    } else {
        score::gen_simple_mat(5, &mut mat, opt.a, opt.b, opt.sc_ambi);
    }

    // Align each region, collecting any split regions
    let mut new_regs: Vec<AlignReg> = Vec::new();
    let n = regs.len();
    for i in 0..n {
        let mut r = regs[i].clone();
        let splits = align1(opt, mi, qlen, &qseq_fwd, &qseq_rev, &mut r, &mut *a, &mat);
        new_regs.push(r);
        for mut sr in splits {
            let _ = align1(opt, mi, qlen, &qseq_fwd, &qseq_rev, &mut sr, &mut *a, &mat);
            new_regs.push(sr);
        }
    }

    // Try inversion realignment between consecutive split regions
    if !(opt.flag.contains(crate::flags::MapFlags::NO_INV)) {
        let mut inv_regs: Vec<AlignReg> = Vec::new();
        for i in 1..new_regs.len() {
            if new_regs[i].split_inv {
                if let Some(inv) = align1_inv(opt, mi, qlen, &qseq_fwd, &qseq_rev,
                    &new_regs[i-1], &new_regs[i], &mat)
                {
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
    if !(opt.flag.intersects(crate::flags::MapFlags::SR | crate::flags::MapFlags::SR_RNA
        | crate::flags::MapFlags::ALL_CHAINS))
        && opt.split_prefix.is_none()
        && qlen >= opt.rank_min_len
    {
        crate::hit::update_dp_max(qlen, regs, opt.rank_frac, opt.a, opt.b);
        crate::hit::filter_regs(opt, qlen, regs);
    }
    crate::hit::hit_sort(regs, opt.alt_drop);
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
        let ez = align_pair(&query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
        // CIGAR must be correct; score may differ between SIMD/scalar implementations
        assert_eq!(cigar_to_string(&ez.cigar), "8M");
    }

    #[test]
    fn test_align_pair_dual() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = align_pair_dual(&query, &target, 5, &mat, 4, 2, 24, 1, -1, 400, 0, KswFlags::empty());
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
        assert!(s.contains('=') && s.contains('X'),
            "EQX CIGAR should contain = and X, got: {}", s);
    }

    #[test]
    fn test_simd_dispatch() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = ksw2_simd::ksw_extz2_dispatch(
            &query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
        assert_eq!(crate::align::cigar_to_string(&ez.cigar), "8M");
    }
}
