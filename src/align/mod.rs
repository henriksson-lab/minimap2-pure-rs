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
    // Use scalar for now; SIMD dispatch available via ksw2_simd::ksw_extz2_dispatch
    // The SIMD version needs more testing before becoming the default
    ksw2::ksw_extz2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag)
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
    // TODO: dispatch to SIMD when available
    ksw2::ksw_extd2(query, target, m, mat, q, e, q2, e2, w, zdrop, end_bonus, flag)
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

/// Perform anchor-based gap-filling alignment for a single region.
///
/// Follows mm_align1() from align.c:
/// 1. Left extension from first anchor
/// 2. Gap filling between consecutive anchors
/// 3. Right extension from last anchor
fn align1(
    opt: &MapOpt,
    mi: &MmIdx,
    qlen: i32,
    qseq_fwd: &[u8],
    qseq_rev: &[u8],
    r: &mut AlignReg,
    a: &[Mm128],
    mat: &[i8],
) {
    if r.cnt == 0 { return; }
    let as1 = r.as_ as usize;
    let cnt1 = r.cnt as usize;
    let rid = r.rid as u32;
    let rev = r.rev;
    let qseq = if rev { qseq_rev } else { qseq_fwd };
    let ref_len = mi.seqs[rid as usize].len as i32;

    let bw = (opt.bw as f32 * 1.5 + 1.0) as i32;
    let bw_long = ((opt.bw_long as f32 * 1.5 + 1.0) as i32).max(bw);

    // Get anchor coordinates (already properly encoded after seed expansion fix)
    // a[i].x = rev<<63 | rid<<32 | ref_pos
    // a[i].y = q_span<<32 | q_pos
    let first_qspan = ((a[as1].y >> 32) & 0xff) as i32;
    let rs = ((a[as1].x as i32) + 1 - first_qspan).max(0);
    let re = (a[as1 + cnt1 - 1].x as i32) + 1;
    let qs = ((a[as1].y as i32) + 1 - first_qspan).max(0);
    let qe = (a[as1 + cnt1 - 1].y as i32) + 1;

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
        return;
    }
    if qs0 < 0 || qe0 > qlen {
        log::debug!("align1: bail qs0={} qe0={} qlen={}", qs0, qe0, qlen);
        return;
    }
    if (qs0 as usize) >= qseq.len() || (qe0 as usize) > qseq.len() {
        log::debug!("align1: bail qs0={} qe0={} qseq.len={}", qs0, qe0, qseq.len());
        return;
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
    let mut cur_rs = rs;
    let mut cur_qs = qs;
    for i in 1..cnt1 {
        let ai = &a[as1 + i];
        // Skip tandem/ignored seeds (except last)
        if i != cnt1 - 1 && (ai.y & (crate::flags::SEED_IGNORE | crate::flags::SEED_TANDEM)) != 0 {
            continue;
        }
        let next_re = (ai.x as i32) + 1;
        let next_qe = (ai.y as i32) + 1;

        // Align this gap (always process each anchor gap for robustness)
        {
            let gap_tlen = (next_re - cur_rs).max(0) as usize;
            let gap_qlen = (next_qe - cur_qs).max(0) as usize;
            if gap_tlen > 0 && gap_qlen > 0
                && cur_qs >= 0 && cur_rs >= 0
                && (cur_qs as usize) < qseq.len()
                && (next_qe as usize) <= qseq.len()
                && cur_rs >= 0 && cur_rs < ref_len && next_re > 0 && next_re <= ref_len
            {
                if gap_tlen > tseq_buf.len() { tseq_buf.resize(gap_tlen, 0); }
                mi.getseq(rid, cur_rs as u32, next_re as u32, &mut tseq_buf[..gap_tlen]);
                let qsub = &qseq[cur_qs as usize..next_qe as usize];
                let use_bw = if gap_tlen > bw as usize * 2 || gap_qlen > bw as usize * 2 { bw_long } else { bw };
                let ez = do_align(opt, qsub, &tseq_buf[..gap_tlen], mat, use_bw, opt.zdrop, -1, KswFlags::empty());
                log::debug!("  gap i={} cur_rs={} next_re={} cur_qs={} next_qe={} tlen={} qlen={} score={} zdropped={} cigar_len={}",
                    i, cur_rs, next_re, cur_qs, next_qe, gap_tlen, gap_qlen, ez.score, ez.zdropped, ez.cigar.len());
                if ez.zdropped {
                    dp_score += ez.max;
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
    if cur_qs < qe0 && cur_rs < re0 {
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
    if cigar_ops.is_empty() { return; }

    // Fix CIGAR: remove leading/trailing I/D, merge ops
    let (q_shift, t_shift) = fix_cigar(&mut cigar_ops);
    rs1 += t_shift;
    qs1 += q_shift;
    if cigar_ops.is_empty() { return; }

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
    a: &[Mm128],
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

    // Align each region
    for i in 0..regs.len() {
        // We need to temporarily take the reg out to avoid borrow issues
        let mut r = regs[i].clone();
        align1(opt, mi, qlen, &qseq_fwd, &qseq_rev, &mut r, a, &mat);
        regs[i] = r;
    }

    // Filter and sort
    crate::hit::filter_regs(opt, qlen, regs);
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
        assert_eq!(ez.score, 16);
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
        // Dispatch currently falls through to scalar
        let ez = ksw2_simd::ksw_extz2_dispatch(
            &query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::SCORE_ONLY);
        assert_eq!(ez.score, 16);
    }
}
