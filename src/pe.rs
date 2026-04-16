//! Paired-end read handling.
//!
//! Implements mm_pair(), mm_set_pe_thru(), mm_select_sub_multi() from pe.c.

use crate::hit;
use crate::sort::radix_sort_u64;
use crate::types::AlignReg;

/// Paired-end pairing: find the best concordant pair and update MAPQ.
///
/// Matches mm_pair() from pe.c.
pub fn pair(
    max_gap_ref: i32,
    pe_bonus: i32,
    sub_diff: i32,
    match_sc: i32,
    qlens: &[i32],
    n_regs: &mut [usize],
    regs: &mut [Vec<AlignReg>],
) {
    if regs.len() < 2 {
        return;
    }

    // Build sorted array of (rid<<32 | rs<<1 | orientation, seg_idx, reg_idx)
    struct PairEntry {
        key: u64,
        seg: usize,
        idx: usize,
        rev: bool,
    }
    let mut entries: Vec<PairEntry> = Vec::new();
    let mut dp_thres = 0i32;
    let mut segs = 0u32;

    for s in 0..2 {
        let mut seg_max = 0i32;
        for i in 0..n_regs[s] {
            let r = &regs[s][i];
            if r.extra.is_none() {
                continue;
            }
            let dp = r.extra.as_ref().unwrap().dp_max;
            seg_max = seg_max.max(dp);
            let orientation = (s as u64 ^ r.rev as u64) & 1;
            let key = (r.rid as u64) << 32 | (r.rs as u64) << 1 | orientation;
            entries.push(PairEntry {
                key,
                seg: s,
                idx: i,
                rev: r.rev,
            });
            segs |= 1 << s;
        }
        dp_thres += seg_max;
    }
    if segs != 3 {
        return;
    } // only one end mapped
    dp_thres -= pe_bonus;
    if dp_thres < 0 {
        dp_thres = 0;
    }

    // Sort by key
    entries.sort_unstable_by_key(|e| e.key);

    // Find best concordant pair
    let n = entries.len();
    let mut max_score: i64 = -1;
    let mut max_idx = [usize::MAX; 2];
    let mut last = [-1i32; 2]; // last index with forward orientation per strand
    let mut pair_scores: Vec<u64> = Vec::new();

    for i in 0..n {
        if entries[i].key & 1 != 0 {
            // Reverse first or forward second: try to pair with previous forward entry
            let rev_idx = entries[i].rev as usize;
            if last[rev_idx] < 0 {
                continue;
            }
            let ri = &regs[entries[i].seg][entries[i].idx];
            let qi =
                &regs[entries[last[rev_idx] as usize].seg][entries[last[rev_idx] as usize].idx];
            if ri.rid != qi.rid || ri.rs - qi.re > max_gap_ref {
                continue;
            }

            // Search backwards for valid pairs
            let mut j = last[rev_idx] as usize;
            loop {
                let ej = &entries[j];
                if ej.rev != entries[i].rev || ej.seg == entries[i].seg {
                    if j == 0 {
                        break;
                    } else {
                        j -= 1;
                        continue;
                    }
                }
                let qj = &regs[ej.seg][ej.idx];
                if ri.rid != qj.rid || ri.rs - qj.re > max_gap_ref {
                    break;
                }

                if let (Some(pi), Some(pj)) = (&ri.extra, &qj.extra) {
                    if pi.dp_max + pj.dp_max >= dp_thres {
                        let hash = ri.hash.wrapping_add(qj.hash) as i64;
                        let score = ((pi.dp_max as i64 + pj.dp_max as i64) << 32) | hash;
                        if score > max_score {
                            max_score = score;
                            max_idx[ej.seg] = j;
                            max_idx[entries[i].seg] = i;
                        }
                        pair_scores.push(score as u64);
                    }
                }
                if j == 0 {
                    break;
                } else {
                    j -= 1;
                }
            }
        } else {
            // Forward first or reverse second: record position
            last[entries[i].rev as usize] = i as i32;
        }
    }

    if pair_scores.is_empty() || max_score < 0 {
        return;
    }
    radix_sort_u64(&mut pair_scores);

    // Update the best pair
    if max_idx[0] < n && max_idx[1] < n {
        let e0 = &entries[max_idx[0]];
        let e1 = &entries[max_idx[1]];
        regs[e0.seg][e0.idx].proper_frag = true;
        regs[e1.seg][e1.idx].proper_frag = true;

        // Lift to primary if needed
        for s in 0..2 {
            let ei = if s == 0 { e0 } else { e1 };
            let r_id = regs[ei.seg][ei.idx].id;
            let r_parent = regs[ei.seg][ei.idx].parent;
            if r_id != r_parent {
                // Lift child to primary
                let old_parent_id = r_parent;
                for k in 0..n_regs[ei.seg] {
                    if regs[ei.seg][k].parent == old_parent_id {
                        regs[ei.seg][k].parent = r_id;
                    }
                }
                // Zero out old primary's mapq
                for k in 0..n_regs[ei.seg] {
                    if regs[ei.seg][k].id == old_parent_id {
                        regs[ei.seg][k].mapq = 0;
                    }
                }
            }
            if !regs[ei.seg][ei.idx].sam_pri {
                for k in 0..n_regs[ei.seg] {
                    regs[ei.seg][k].sam_pri = false;
                }
                regs[ei.seg][ei.idx].sam_pri = true;
            }
        }

        // Compute PE MAPQ
        let r0 = &regs[e0.seg][e0.idx];
        let r1 = &regs[e1.seg][e1.idx];
        let mut mapq_pe = r0.mapq.max(r1.mapq) as i32;

        let mut n_sub = 0;
        for &sc in &pair_scores {
            if (sc >> 32) as i32 + sub_diff >= (max_score >> 32) as i32 {
                n_sub += 1;
            }
        }
        if pair_scores.len() > 1 {
            let second_best = pair_scores[pair_scores.len() - 2] >> 32;
            let best = (max_score >> 32) as u64;
            let mapq_alt = (6.02 * (best as f32 - second_best as f32) / match_sc as f32
                - 4.343 * (n_sub as f32).ln()) as i32;
            mapq_pe = mapq_pe.min(mapq_alt);
        }

        // Update MAPQ
        for s in 0..2 {
            let ei = if s == 0 { e0 } else { e1 };
            let cur_mapq = regs[ei.seg][ei.idx].mapq as i32;
            if cur_mapq < mapq_pe {
                regs[ei.seg][ei.idx].mapq =
                    (0.2 * cur_mapq as f32 + 0.8 * mapq_pe as f32 + 0.499) as u8;
            }
        }
        if pair_scores.len() == 1 {
            for s in 0..2 {
                let ei = if s == 0 { e0 } else { e1 };
                if regs[ei.seg][ei.idx].mapq < 2 {
                    regs[ei.seg][ei.idx].mapq = 2;
                }
            }
        } else if ((max_score >> 32) as u64) > (pair_scores[pair_scores.len() - 2] >> 32) {
            for s in 0..2 {
                let ei = if s == 0 { e0 } else { e1 };
                if regs[ei.seg][ei.idx].mapq < 1 {
                    regs[ei.seg][ei.idx].mapq = 1;
                }
            }
        }
    }

    // Check for through-alignment
    set_pe_thru(qlens, n_regs, regs);
}

/// Detect through-aligned pairs where one read's alignment spans into the mate.
/// Matches mm_set_pe_thru().
fn set_pe_thru(qlens: &[i32], n_regs: &[usize], regs: &mut [Vec<AlignReg>]) {
    if regs.len() < 2 {
        return;
    }
    let mut n_pri = [0i32; 2];
    let mut pri_idx = [-1i32; 2];
    for s in 0..2 {
        for i in 0..n_regs[s] {
            if regs[s][i].id == regs[s][i].parent {
                n_pri[s] += 1;
                pri_idx[s] = i as i32;
            }
        }
    }
    if n_pri[0] == 1 && n_pri[1] == 1 {
        let p = &regs[0][pri_idx[0] as usize];
        let q = &regs[1][pri_idx[1] as usize];
        if p.rid == q.rid
            && p.rev == q.rev
            && (p.rs - q.rs).abs() < 3
            && (p.re - q.re).abs() < 3
            && ((p.qs == 0 && qlens[1] - q.qe == 0) || (q.qs == 0 && qlens[0] - p.qe == 0))
        {
            regs[0][pri_idx[0] as usize].pe_thru = true;
            regs[1][pri_idx[1] as usize].pe_thru = true;
        }
    }
}

/// Select secondary hits for multi-segment mode.
/// Matches mm_select_sub_multi() from pe.c.
pub fn select_sub_multi(
    pri_ratio: f32,
    pri1: f32,
    pri2: f32,
    max_gap_ref: i32,
    min_diff: i32,
    best_n: i32,
    n_segs: i32,
    qlens: &[i32],
    n_regs: &mut usize,
    regs: &mut Vec<AlignReg>,
) {
    if pri_ratio <= 0.0 || regs.is_empty() {
        return;
    }
    let n = *n_regs;
    let max_dist = if n_segs == 2 {
        qlens[0] + qlens[1] + max_gap_ref
    } else {
        0
    };
    let mut n_2nd = 0i32;
    let mut k = 0usize;

    for i in 0..n {
        let mut to_keep = false;
        let parent_idx = regs[i].parent as usize;
        if regs[i].parent == i as i32 {
            to_keep = true; // primary
        } else if parent_idx < n && regs[i].score + min_diff >= regs[parent_idx].score {
            to_keep = true;
        } else if parent_idx < n {
            let p = &regs[parent_idx];
            let q = &regs[i];
            if p.rev == q.rev && p.rid == q.rid && q.re - p.rs < max_dist && p.re - q.rs < max_dist
            {
                if q.score as f32 >= p.score as f32 * pri1 {
                    to_keep = true;
                }
            } else {
                let is_par_both = n_segs == 2 && p.qs < qlens[0] && p.qe > qlens[0];
                let is_chi_both = n_segs == 2 && q.qs < qlens[0] && q.qe > qlens[0];
                if is_chi_both || is_chi_both == is_par_both {
                    if q.score as f32 >= p.score as f32 * pri_ratio {
                        to_keep = true;
                    }
                } else if q.score as f32 >= p.score as f32 * pri2 {
                    to_keep = true;
                }
            }
        }
        if to_keep && regs[i].parent != i as i32 {
            if n_2nd >= best_n {
                to_keep = false;
            } else {
                n_2nd += 1;
            }
        }
        if to_keep {
            if k != i {
                regs[k] = regs[i].clone();
            }
            k += 1;
        }
    }
    let changed = k != n;
    regs.truncate(k);
    *n_regs = regs.len();
    if changed {
        hit::sync_regs(regs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_no_crash_empty() {
        let mut n_regs = [0usize, 0usize];
        let mut regs: Vec<Vec<AlignReg>> = vec![Vec::new(), Vec::new()];
        pair(800, 33, 0, 2, &[150, 150], &mut n_regs, &mut regs);
    }
}
