use crate::options::MapOpt;
use crate::sort::{radix_sort_mm128, radix_sort_u64};
use crate::types::{AlignReg, Mm128, PARENT_UNSET, PARENT_TMP_PRI};

/// Hash function for mixing chain hash values. Matches hash64() in hit.c.
fn hash64(mut key: u64) -> u64 {
    key = (!key).wrapping_add(key << 21);
    key ^= key >> 24;
    key = key.wrapping_add(key << 3).wrapping_add(key << 8);
    key ^= key >> 14;
    key = key.wrapping_add(key << 2).wrapping_add(key << 4);
    key ^= key >> 28;
    key = key.wrapping_add(key << 31);
    key
}

/// Compute fuzzy match/block lengths from anchors.
fn cal_fuzzy_len(r: &mut AlignReg, a: &[Mm128]) {
    r.mlen = 0;
    r.blen = 0;
    if r.cnt <= 0 {
        return;
    }
    let as_ = r.as_ as usize;
    let span = ((a[as_].y >> 32) & 0xff) as i32;
    r.mlen = span;
    r.blen = span;
    for i in 1..r.cnt as usize {
        let span = ((a[as_ + i].y >> 32) & 0xff) as i32;
        let tl = (a[as_ + i].x as i32).wrapping_sub(a[as_ + i - 1].x as i32);
        let ql = (a[as_ + i].y as i32).wrapping_sub(a[as_ + i - 1].y as i32);
        r.blen += tl.max(ql);
        r.mlen += if tl > span && ql > span { span } else { tl.min(ql) };
    }
}

/// Set coordinates from anchors. Matches mm_reg_set_coor().
fn reg_set_coor(r: &mut AlignReg, qlen: i32, a: &[Mm128], is_qstrand: bool) {
    let k = r.as_ as usize;
    let q_span = ((a[k].y >> 32) & 0xff) as i32;
    r.rev = (a[k].x >> 63) != 0;
    r.rid = ((a[k].x << 1) >> 33) as i32;
    r.rs = if (a[k].x as i32) + 1 > q_span { (a[k].x as i32) + 1 - q_span } else { 0 };
    r.re = (a[k + r.cnt as usize - 1].x as i32) + 1;
    if !r.rev || is_qstrand {
        r.qs = (a[k].y as i32) + 1 - q_span;
        r.qe = (a[k + r.cnt as usize - 1].y as i32) + 1;
    } else {
        r.qs = qlen - ((a[k + r.cnt as usize - 1].y as i32) + 1);
        r.qe = qlen - ((a[k].y as i32) + 1 - q_span);
    }
    cal_fuzzy_len(r, a);
}

/// Convert chains to alignment regions. Matches mm_gen_regs().
#[inline(never)]
pub fn gen_regs(
    hash: u32,
    qlen: i32,
    u: &[u64],
    a: &[Mm128],
    is_qstrand: bool,
) -> Vec<AlignReg> {
    let n_u = u.len();
    if n_u == 0 {
        return Vec::new();
    }
    // Build (mixed_score, index) and sort descending
    let mut z: Vec<Mm128> = Vec::with_capacity(n_u);
    let mut k = 0usize;
    for i in 0..n_u {
        let h = hash64(hash64(a[k].x).wrapping_add(hash64(a[k].y))) ^ hash as u64;
        let h = h as u32;
        z.push(Mm128::new(u[i] ^ h as u64, ((k as u64) << 32) | (u[i] as u32 as u64)));
        k += (u[i] as u32) as usize;
    }
    radix_sort_mm128(&mut z);
    z.reverse(); // largest score first

    let mut regs = Vec::with_capacity(n_u);
    for i in 0..n_u {
        let mut r = AlignReg::default();
        r.id = i as i32;
        r.parent = PARENT_UNSET;
        r.score = (z[i].x >> 32) as i32;
        r.score0 = r.score;
        r.hash = z[i].x as u32;
        r.cnt = z[i].y as u32 as i32;
        r.as_ = (z[i].y >> 32) as i32;
        r.div = -1.0;
        reg_set_coor(&mut r, qlen, a, is_qstrand);
        regs.push(r);
    }
    regs
}

/// Split a region at anchor position `n`. Matches mm_split_reg().
/// Returns the new second region (the tail), and modifies `r` (the head) in place.
pub fn split_reg(r: &mut AlignReg, n: i32, qlen: i32, a: &[Mm128]) -> Option<AlignReg> {
    if n <= 0 || n >= r.cnt { return None; }
    let mut r2 = r.clone();
    r2.id = -1;
    r2.sam_pri = false;
    r2.extra = None;
    r2.split_inv = false;
    r2.cnt = r.cnt - n;
    r2.score = (r.score as f32 * (r2.cnt as f32 / r.cnt as f32) + 0.499) as i32;
    r2.as_ = r.as_ + n;
    if r.parent == r.id { r2.parent = PARENT_TMP_PRI; }

    // Set coordinates for r2 from anchors
    if (r2.as_ as usize + r2.cnt as usize) <= a.len() {
        let k = r2.as_ as usize;
        let q_span = ((a[k].y >> 32) & 0xff) as i32;
        r2.rev = (a[k].x >> 63) != 0;
        r2.rid = ((a[k].x << 1) >> 33) as i32;
        r2.rs = ((a[k].x as i32) + 1 - q_span).max(0);
        r2.re = (a[k + r2.cnt as usize - 1].x as i32) + 1;
        if !r2.rev {
            r2.qs = ((a[k].y as i32) + 1 - q_span).max(0);
            r2.qe = (a[k + r2.cnt as usize - 1].y as i32) + 1;
        } else {
            r2.qs = qlen - ((a[k + r2.cnt as usize - 1].y as i32) + 1);
            r2.qe = qlen - ((a[k].y as i32) + 1 - q_span);
        }
    }

    // Update r (head)
    r.cnt -= r2.cnt;
    r.score -= r2.score;
    if r.cnt > 0 && (r.as_ as usize + r.cnt as usize) <= a.len() {
        let k = r.as_ as usize;
        let q_span = ((a[k].y >> 32) & 0xff) as i32;
        r.rs = ((a[k].x as i32) + 1 - q_span).max(0);
        r.re = (a[k + r.cnt as usize - 1].x as i32) + 1;
        if !r.rev {
            r.qs = ((a[k].y as i32) + 1 - q_span).max(0);
            r.qe = (a[k + r.cnt as usize - 1].y as i32) + 1;
        } else {
            r.qs = qlen - ((a[k + r.cnt as usize - 1].y as i32) + 1);
            r.qe = qlen - ((a[k].y as i32) + 1 - q_span);
        }
    }
    r.split |= 1;
    r2.split |= 2;
    Some(r2)
}

/// Mark ALT-contig alignments. Matches mm_mark_alt().
pub fn mark_alt(n_alt: i32, seq_is_alt: &[bool], regs: &mut [AlignReg]) {
    if n_alt == 0 {
        return;
    }
    for r in regs.iter_mut() {
        if r.rid >= 0 && (r.rid as usize) < seq_is_alt.len() && seq_is_alt[r.rid as usize] {
            r.is_alt = true;
        }
    }
}

fn alt_score(score: i32, alt_diff_frac: f32) -> i32 {
    if score < 0 { return score; }
    let s = (score as f32 * (1.0 - alt_diff_frac) + 0.499) as i32;
    if s > 0 { s } else { 1 }
}

/// Assign parent/secondary relationships. Matches mm_set_parent().
#[inline(never)]
pub fn set_parent(
    mask_level: f32,
    mask_len: i32,
    regs: &mut [AlignReg],
    sub_diff: i32,
    hard_mask_level: bool,
    alt_diff_frac: f32,
) {
    let n = regs.len();
    if n == 0 { return; }
    for i in 0..n { regs[i].id = i as i32; }

    let mut w: Vec<usize> = Vec::new(); // primary indices
    w.push(0);
    regs[0].parent = 0;
    let mut cov: Vec<u64> = Vec::new(); // reused across iterations

    for i in 1..n {
        let si = regs[i].qs;
        let ei = regs[i].qe;
        let mut uncov_len = 0i32;

        if !hard_mask_level {
            // Compute uncovered length
            cov.clear();
            for &j in &w {
                let sj = regs[j].qs;
                let ej = regs[j].qe;
                if ej <= si || sj >= ei { continue; }
                let sj = sj.max(si);
                let ej = ej.min(ei);
                cov.push(((sj as u64) << 32) | ej as u64);
            }
            if !cov.is_empty() {
                radix_sort_u64(&mut cov);
                let mut x = si;
                for &c in &cov {
                    let cs = (c >> 32) as i32;
                    let ce = c as u32 as i32;
                    if cs > x { uncov_len += cs - x; }
                    x = x.max(ce);
                }
                if ei > x { uncov_len += ei - x; }
            }
        }

        let mut found_parent = false;
        for &j in &w {
            let sj = regs[j].qs;
            let ej = regs[j].qe;
            if ej <= si || sj >= ei { continue; }
            let min = (ej - sj).min(ei - si);
            let max = (ej - sj).max(ei - si);
            let ol = {
                let (a, b, c, d) = (si, ei, sj, ej);
                if a < c { if b < c { 0 } else if b < d { b - c } else { d - c } }
                else { if d < a { 0 } else if d < b { d - a } else { b - a } }
            };
            if (ol as f32 / min as f32 - uncov_len as f32 / max as f32) > mask_level
                && uncov_len <= mask_len
            {
                regs[i].parent = regs[j].parent;
                let mut sci = regs[i].score;
                if !regs[j].is_alt && regs[i].is_alt {
                    sci = alt_score(sci, alt_diff_frac);
                }
                if regs[j].subsc < sci { regs[j].subsc = sci; }
                let mut cnt_sub = false;
                if regs[i].cnt >= regs[j].cnt { cnt_sub = true; }
                // Extract values from extras to avoid borrow conflicts
                let j_has_extra = regs[j].extra.is_some();
                let i_has_extra = regs[i].extra.is_some();
                if j_has_extra && i_has_extra {
                    let j_rid = regs[j].rid; let j_rs = regs[j].rs; let j_re = regs[j].re;
                    let i_rid = regs[i].rid; let i_rs = regs[i].rs; let i_re = regs[i].re;
                    let i_dp_max = regs[i].extra.as_ref().unwrap().dp_max;
                    let j_dp_max = regs[j].extra.as_ref().unwrap().dp_max;
                    if j_rid != i_rid || j_rs != i_rs || j_re != i_re || ol != min {
                        let mut sci2 = i_dp_max;
                        if !regs[j].is_alt && regs[i].is_alt {
                            sci2 = alt_score(sci2, alt_diff_frac);
                        }
                        let dp_max2 = &mut regs[j].extra.as_mut().unwrap().dp_max2;
                        if *dp_max2 < sci2 { *dp_max2 = sci2; }
                        if j_dp_max - i_dp_max <= sub_diff { cnt_sub = true; }
                    }
                }
                if cnt_sub { regs[j].n_sub += 1; }
                found_parent = true;
                break;
            }
        }
        if !found_parent {
            w.push(i);
            regs[i].parent = i as i32;
            regs[i].n_sub = 0;
        }
    }
}

/// Sort hits by DP score (or chain score). Matches mm_hit_sort().
pub fn hit_sort(regs: &mut Vec<AlignReg>, alt_diff_frac: f32) {
    let n = regs.len();
    if n <= 1 { return; }
    // Filter out cnt==0 entries (soft deleted) and sort
    let mut kept: Vec<(u64, usize)> = Vec::new();
    for (i, r) in regs.iter().enumerate() {
        if r.inv || r.cnt > 0 {
            let score = if let Some(ref p) = r.extra {
                p.dp_max
            } else {
                r.score
            };
            let score = if r.is_alt { alt_score(score, alt_diff_frac) } else { score };
            kept.push(((score as u64) << 32 | r.hash as u64, i));
        }
    }
    kept.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // descending
    let new_regs: Vec<AlignReg> = kept.iter().map(|&(_, i)| regs[i].clone()).collect();
    *regs = new_regs;
}

/// Set SAM primary flags. Matches mm_set_sam_pri().
pub fn set_sam_pri(regs: &mut [AlignReg]) -> i32 {
    let mut n_pri = 0;
    for r in regs.iter_mut() {
        if r.id == r.parent {
            n_pri += 1;
            r.sam_pri = n_pri == 1;
        } else {
            r.sam_pri = false;
        }
    }
    n_pri
}

/// Keep parent/id in sync after filtering. Matches mm_sync_regs().
pub fn sync_regs(regs: &mut [AlignReg]) {
    let n = regs.len();
    if n == 0 { return; }
    let max_id = regs.iter().map(|r| r.id).max().unwrap_or(-1);
    if max_id < 0 { return; }
    let mut tmp = vec![-1i32; (max_id + 1) as usize];
    for (i, r) in regs.iter().enumerate() {
        if r.id >= 0 {
            tmp[r.id as usize] = i as i32;
        }
    }
    for i in 0..n {
        let old_parent = regs[i].parent;
        regs[i].id = i as i32;
        if old_parent == PARENT_TMP_PRI {
            regs[i].parent = i as i32;
        } else if old_parent >= 0 && (old_parent as usize) < tmp.len() && tmp[old_parent as usize] >= 0 {
            regs[i].parent = tmp[old_parent as usize];
        } else {
            regs[i].parent = PARENT_UNSET;
        }
    }
    set_sam_pri(regs);
}

/// Recalibrate dp_max based on divergence for assembly ranking.
/// Matches mm_update_dp_max() from align.c.
pub fn update_dp_max(qlen: i32, regs: &mut [AlignReg], frac: f32, a: i32, b: i32) {
    if regs.len() < 2 { return; }
    let mut max = -1i32;
    let mut max2 = -1i32;
    let mut max_i: i32 = -1;
    for (i, r) in regs.iter().enumerate() {
        if let Some(ref p) = r.extra {
            if p.dp_max > max { max2 = max; max = p.dp_max; max_i = i as i32; }
            else if p.dp_max > max2 { max2 = p.dp_max; }
        }
    }
    if max_i < 0 || max < 0 || max2 < 0 { return; }
    let mi = max_i as usize;
    if (regs[mi].qe - regs[mi].qs) < (qlen as f64 * frac as f64) as i32 { return; }
    if (max2 as f64) < max as f64 * frac as f64 { return; }

    let identity = if regs[mi].blen > 0 { regs[mi].mlen as f64 / regs[mi].blen as f64 } else { 1.0 };
    let mut div = 1.0 - identity;
    if div < 0.02 { div = 0.02; }
    let mut b2 = 0.5 / div;
    if b2 * a as f64 > b as f64 { b2 = a as f64 / b as f64; }

    for r in regs.iter_mut() {
        if let Some(ref mut p) = r.extra {
            let mut n_gap = 0i32;
            let mut gap_cost = 0.0f64;
            for &c in &p.cigar.0 {
                let op = c & 0xf;
                let len = (c >> 4) as i32;
                if op == 1 || op == 2 { // I or D
                    gap_cost += b2 + crate::chain::mg_log2(1.0 + len as f32) as f64;
                    n_gap += len;
                }
            }
            let n_mis = r.blen + p.n_ambi as i32 - r.mlen - n_gap;
            let new_max = (a as f64 * (r.mlen as f64 - b2 * n_mis as f64 - gap_cost) + 0.499) as i32;
            p.dp_max = new_max.max(0);
        }
    }
}

/// Filter secondary hits. Matches mm_select_sub().
pub fn select_sub(
    pri_ratio: f32,
    min_diff: i32,
    best_n: i32,
    regs: &mut Vec<AlignReg>,
) {
    if pri_ratio <= 0.0 || regs.is_empty() { return; }
    let mut kept = Vec::new();
    let mut n_2nd = 0i32;
    for i in 0..regs.len() {
        let p = regs[i].parent as usize;
        if regs[i].parent == regs[i].id || regs[i].inv {
            kept.push(regs[i].clone());
        } else if p < regs.len()
            && (regs[i].score as f32 >= regs[p].score as f32 * pri_ratio
                || regs[i].score + min_diff >= regs[p].score)
            && n_2nd < best_n
        {
            // Not identical hits
            if !(regs[i].qs == regs[p].qs && regs[i].qe == regs[p].qe
                && regs[i].rid == regs[p].rid && regs[i].rs == regs[p].rs && regs[i].re == regs[p].re)
            {
                kept.push(regs[i].clone());
                n_2nd += 1;
            }
        }
    }
    let changed = kept.len() != regs.len();
    *regs = kept;
    if changed {
        sync_regs(regs);
    }
}

/// Filter regions by quality. Matches mm_filter_regs().
pub fn filter_regs(opt: &MapOpt, qlen: i32, regs: &mut Vec<AlignReg>) {
    regs.retain(|r| {
        if !r.inv && !r.seg_split && r.cnt < opt.min_cnt {
            return false;
        }
        if let Some(ref p) = r.extra {
            if r.mlen < opt.min_chain_score { return false; }
            if p.dp_max < opt.min_dp_max { return false; }
            if r.qs as f32 > qlen as f32 * opt.max_clip_ratio
                && (qlen - r.qe) as f32 > qlen as f32 * opt.max_clip_ratio
            {
                return false;
            }
        }
        true
    });
}

/// Compute MAPQ scores. Matches mm_set_mapq2().
pub fn set_mapq(
    regs: &mut [AlignReg],
    min_chain_sc: i32,
    match_sc: i32,
    rep_len: i32,
    is_sr: bool,
    is_splice: bool,
) {
    const Q_COEF: f32 = 40.0;
    if regs.is_empty() { return; }

    let mut sum_sc: i64 = 0;
    let mut n_2nd_splice = 0;
    for r in regs.iter() {
        if r.parent == r.id { sum_sc += r.score as i64; }
        else if r.is_spliced { n_2nd_splice += 1; }
    }
    let uniq_ratio = sum_sc as f32 / (sum_sc as f32 + rep_len as f32);

    for r in regs.iter_mut() {
        if r.inv {
            r.mapq = 0;
        } else if r.parent == r.id {
            let pen_s1 = (if r.score > 100 { 1.0 } else { 0.01 * r.score as f32 }) * uniq_ratio;
            let pen_cm = if r.cnt > 10 { 1.0f32 } else { 0.1 * r.cnt as f32 };
            let pen_cm = pen_s1.min(pen_cm);
            let subsc = if r.subsc > min_chain_sc { r.subsc } else { min_chain_sc };

            let mapq = if let Some(ref p) = r.extra {
                if p.dp_max2 > 0 && p.dp_max > 0 {
                    let identity = r.mlen as f32 / r.blen.max(1) as f32;
                    let x = if is_sr && is_splice {
                        p.dp_max2 as f32 / p.dp_max as f32
                    } else {
                        p.dp_max2 as f32 * subsc as f32 / p.dp_max as f32 / r.score0.max(1) as f32
                    };
                    let mut mq = (identity * pen_cm * Q_COEF * (1.0 - x * x)
                        * (p.dp_max as f32 / match_sc as f32).ln()) as i32;
                    if !is_sr {
                        let mq_alt = (6.02 * identity * identity
                            * (p.dp_max - p.dp_max2) as f32 / match_sc as f32 + 0.499) as i32;
                        mq = mq.min(mq_alt);
                    }
                    if is_splice && is_sr && r.is_spliced && n_2nd_splice == 0 {
                        mq += 10;
                    }
                    mq
                } else {
                    let x = subsc as f32 / r.score0.max(1) as f32;
                    if p.dp_max > 0 {
                        let identity = r.mlen as f32 / r.blen.max(1) as f32;
                        (identity * pen_cm * Q_COEF * (1.0 - x)
                            * (p.dp_max as f32 / match_sc as f32).ln()) as i32
                    } else {
                        (pen_cm * Q_COEF * (1.0 - x) * (r.score as f32).ln()) as i32
                    }
                }
            } else {
                let x = subsc as f32 / r.score0.max(1) as f32;
                (pen_cm * Q_COEF * (1.0 - x) * (r.score as f32).ln()) as i32
            };

            let mapq = mapq - (4.343 * ((r.n_sub + 1) as f32).ln() + 0.499) as i32;
            let mapq = mapq.max(0).min(60);
            r.mapq = mapq as u8;
            if let Some(ref p) = r.extra {
                if p.dp_max > p.dp_max2 && r.mapq == 0 { r.mapq = 1; }
            }
        } else {
            r.mapq = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_anchors_and_chains() -> (Vec<Mm128>, Vec<u64>) {
        // Simple chain: 5 anchors on rid=0, forward strand
        let mut a = Vec::new();
        for i in 0..5 {
            a.push(Mm128::new(
                0u64 << 32 | (i * 100 + 100) as u64,
                15u64 << 32 | (i * 100 + 100) as u64,
            ));
        }
        let u = vec![(75u64 << 32) | 5]; // score=75, 5 anchors
        (a, u)
    }

    #[test]
    fn test_gen_regs() {
        let (a, u) = make_anchors_and_chains();
        let regs = gen_regs(42, 600, &u, &a, false);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].cnt, 5);
        assert_eq!(regs[0].score, 75);
        assert!(regs[0].rs >= 0);
        assert!(regs[0].re > regs[0].rs);
    }

    #[test]
    fn test_set_parent_single() {
        let (a, u) = make_anchors_and_chains();
        let mut regs = gen_regs(42, 600, &u, &a, false);
        set_parent(0.5, i32::MAX, &mut regs, 0, false, 0.15);
        assert_eq!(regs[0].parent, 0); // primary
    }

    #[test]
    fn test_set_parent_two_overlapping() {
        // Create two overlapping chains
        let mut a = Vec::new();
        for i in 0..5 {
            a.push(Mm128::new(0u64 << 32 | (i * 50 + 50) as u64, 15u64 << 32 | (i * 50 + 50) as u64));
        }
        for i in 0..3 {
            a.push(Mm128::new(0u64 << 32 | (i * 50 + 75) as u64, 15u64 << 32 | (i * 50 + 75) as u64));
        }
        let u = vec![(80u64 << 32) | 5, (40u64 << 32) | 3];
        let mut regs = gen_regs(42, 600, &u, &a, false);
        set_parent(0.5, i32::MAX, &mut regs, 0, false, 0.15);
        // First should be primary
        assert_eq!(regs[0].parent, regs[0].id);
    }

    #[test]
    fn test_hit_sort() {
        let (a, u) = make_anchors_and_chains();
        let mut regs = gen_regs(42, 600, &u, &a, false);
        hit_sort(&mut regs, 0.15);
        assert_eq!(regs.len(), 1);
    }

    #[test]
    fn test_set_mapq() {
        let (a, u) = make_anchors_and_chains();
        let mut regs = gen_regs(42, 600, &u, &a, false);
        set_parent(0.5, i32::MAX, &mut regs, 0, false, 0.15);
        set_mapq(&mut regs, 40, 2, 0, false, false);
        assert!(regs[0].mapq > 0);
    }

    #[test]
    fn test_sync_regs() {
        let (a, u) = make_anchors_and_chains();
        let mut regs = gen_regs(42, 600, &u, &a, false);
        regs[0].parent = 0;
        sync_regs(&mut regs);
        assert_eq!(regs[0].id, 0);
        assert_eq!(regs[0].parent, 0);
    }
}
