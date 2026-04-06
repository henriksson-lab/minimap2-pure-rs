use std::cell::RefCell;
use crate::types::Mm128;
use super::backtrack::{chain_backtrack, compact_a};
use super::{comput_sc, ChainResult};

// Thread-local scratch buffers for chain DP to avoid per-call allocations.
thread_local! {
    static CHAIN_P: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    static CHAIN_F: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    static CHAIN_V: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    static CHAIN_T: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

/// DP-based chaining algorithm. Matches mg_lchain_dp() from lchain.c.
///
/// Input anchors must be sorted by (target_id, target_pos).
///
/// # Arguments
/// * `max_dist_x` - max reference gap
/// * `max_dist_y` - max query gap
/// * `bw` - bandwidth (max diagonal deviation)
/// * `max_skip` - max consecutive skips in DP
/// * `max_iter` - max predecessors to consider per anchor
/// * `min_cnt` - min anchors per chain
/// * `min_sc` - min chain score
/// * `chn_pen_gap` - gap penalty coefficient
/// * `chn_pen_skip` - skip penalty coefficient
/// * `is_cdna` - cDNA mode (allow large gaps)
/// * `n_seg` - number of query segments
/// * `a` - sorted anchor array (consumed)
#[inline(never)]
pub fn lchain_dp(
    mut max_dist_x: i32,
    mut max_dist_y: i32,
    bw: i32,
    max_skip: i32,
    max_iter: i32,
    min_cnt: i32,
    min_sc: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    is_cdna: bool,
    n_seg: i32,
    a: &[Mm128],
) -> Option<ChainResult> {
    let n = a.len() as i64;
    if n == 0 {
        return None;
    }

    if max_dist_x < bw {
        max_dist_x = bw;
    }
    if max_dist_y < bw && !is_cdna {
        max_dist_y = bw;
    }
    let max_drop = if is_cdna { i32::MAX } else { bw };

    let mut p = CHAIN_P.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut f = CHAIN_F.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut v_scores = CHAIN_V.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut t = CHAIN_T.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let nu = n as usize;
    p.clear(); p.resize(nu, -1i64);
    f.clear(); f.resize(nu, 0i32);
    v_scores.clear(); v_scores.resize(nu, 0i32);
    t.clear(); t.resize(nu, 0i32);

    // Fill score and backtrack arrays
    let mut st: i64 = 0;
    let mut max_ii: i64 = -1;

    // SAFETY: all indices i, j, st, max_ii are in 0..n; all arrays have length n
    let ap = a.as_ptr();
    let fp = f.as_mut_ptr();
    let pp = p.as_mut_ptr();
    let tp = t.as_mut_ptr();
    let vp = v_scores.as_mut_ptr();
    for i in 0..n {
        let iu = i as usize;
        let mut max_j: i64 = -1;
        let mut max_f = unsafe { (((*ap.add(iu)).y >> 32) & 0xff) as i32 }; // q_span
        let mut n_skip: i32 = 0;

        // Advance start pointer
        while st < i {
            let su = st as usize;
            unsafe {
                if (*ap.add(iu)).x >> 32 != (*ap.add(su)).x >> 32
                    || (*ap.add(iu)).x > (*ap.add(su)).x + max_dist_x as u64 {
                    st += 1;
                } else { break; }
            }
        }
        if i - st > max_iter as i64 {
            st = i - max_iter as i64;
        }

        let mut end_j = st;
        // DP: scan predecessors
        for j in (st..i).rev() {
            let ju = j as usize;
            let sc = unsafe { comput_sc(
                &*ap.add(iu), &*ap.add(ju), max_dist_x, max_dist_y, bw,
                chn_pen_gap, chn_pen_skip, is_cdna, n_seg,
            ) };
            if sc == i32::MIN {
                continue;
            }
            let sc = sc + unsafe { *fp.add(ju) };
            if sc > max_f {
                max_f = sc;
                max_j = j;
                if n_skip > 0 {
                    n_skip -= 1;
                }
            } else if unsafe { *tp.add(ju) } == i as i32 {
                n_skip += 1;
                if n_skip > max_skip {
                    break;
                }
            }
            unsafe {
                if *pp.add(ju) >= 0 {
                    *tp.add((*pp.add(ju)) as usize) = i as i32;
                }
            }
            end_j = j;
        }

        // Check the global maximum in range (for skip recovery)
        if max_ii < 0 || unsafe { (*ap.add(iu)).x.wrapping_sub((*ap.add(max_ii as usize)).x) } > max_dist_x as u64 {
            let mut max_val = i32::MIN;
            max_ii = -1;
            for j in (st..i).rev() {
                unsafe {
                    if max_val < *fp.add(j as usize) {
                        max_val = *fp.add(j as usize);
                        max_ii = j;
                    }
                }
            }
        }
        if max_ii >= 0 && max_ii < end_j {
            let tmp = unsafe { comput_sc(
                &*ap.add(iu), &*ap.add(max_ii as usize), max_dist_x, max_dist_y, bw,
                chn_pen_gap, chn_pen_skip, is_cdna, n_seg,
            ) };
            if tmp != i32::MIN && max_f < tmp + unsafe { *fp.add(max_ii as usize) } {
                max_f = tmp + unsafe { *fp.add(max_ii as usize) };
                max_j = max_ii;
            }
        }

        unsafe {
            *fp.add(iu) = max_f;
            *pp.add(iu) = max_j;
        }
        unsafe {
            *vp.add(iu) = if max_j >= 0 && *vp.add(max_j as usize) > max_f {
                *vp.add(max_j as usize)
            } else {
                max_f
            };
        }
        if max_ii < 0
            || unsafe { (*ap.add(iu)).x.wrapping_sub((*ap.add(max_ii as usize)).x) <= max_dist_x as u64
                && *fp.add(max_ii as usize) < *fp.add(iu) }
        {
            max_ii = i;
        }
    }

    // Backtrack
    let mut v_indices = Vec::new();
    let mut u = chain_backtrack(n, &f, &p, &mut v_indices, &mut t, min_cnt, min_sc, max_drop);

    // Return scratch buffers for reuse
    CHAIN_P.with(|c| *c.borrow_mut() = p);
    CHAIN_F.with(|c| *c.borrow_mut() = f);
    CHAIN_V.with(|c| *c.borrow_mut() = v_scores);
    CHAIN_T.with(|c| *c.borrow_mut() = t);

    if u.is_empty() {
        return None;
    }

    // Compact
    let anchors = compact_a(&mut u, &v_indices, a);
    Some(ChainResult {
        anchors,
        chains: u,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_anchors() -> Vec<Mm128> {
        // Create 10 anchors forming a nice diagonal chain
        // a[i].x = rid<<32 | ref_pos, a[i].y = q_span<<32 | q_pos
        let mut a = Vec::new();
        for i in 0..10 {
            let ref_pos = (i * 100 + 50) as u64;
            let q_pos = (i * 100 + 50) as u64;
            let q_span = 15u64;
            a.push(Mm128::new(
                0u64 << 32 | ref_pos,   // rid=0, ref_pos
                q_span << 32 | q_pos,    // q_span=15, q_pos
            ));
        }
        a
    }

    #[test]
    fn test_lchain_dp_basic() {
        let a = make_linear_anchors();
        let result = lchain_dp(
            5000, 5000, 500, 25, 5000,
            3,   // min_cnt
            40,  // min_sc
            0.8, // chn_pen_gap
            0.0, // chn_pen_skip
            false, 1,
            &a,
        );
        assert!(result.is_some(), "Should find at least one chain");
        let r = result.unwrap();
        assert!(!r.chains.is_empty());
        let total_anchors: u64 = r.chains.iter().map(|&u| u as u32 as u64).sum();
        assert_eq!(total_anchors as usize, r.anchors.len());
    }

    #[test]
    fn test_lchain_dp_empty() {
        let result = lchain_dp(
            5000, 5000, 500, 25, 5000,
            3, 40, 0.8, 0.0, false, 1,
            &[],
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_lchain_dp_two_chains() {
        // Two chains on different references
        let mut a = Vec::new();
        // Chain 1: rid=0
        for i in 0..5 {
            a.push(Mm128::new(
                0u64 << 32 | (i * 50 + 100) as u64,
                15u64 << 32 | (i * 50 + 100) as u64,
            ));
        }
        // Chain 2: rid=1
        for i in 0..5 {
            a.push(Mm128::new(
                1u64 << 32 | (i * 50 + 100) as u64,
                15u64 << 32 | (i * 50 + 400) as u64,
            ));
        }
        // Sort by (rid, ref_pos)
        a.sort_unstable();

        let result = lchain_dp(
            5000, 5000, 500, 25, 5000,
            2, 20, 0.8, 0.0, false, 1,
            &a,
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.chains.len() >= 2, "Should find at least 2 chains, found {}", r.chains.len());
    }
}
