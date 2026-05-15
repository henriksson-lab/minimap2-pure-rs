use super::backtrack::{chain_backtrack, compact_a};
use super::{comput_sc_simple, ChainResult};
use crate::types::Mm128;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct RmqElem {
    y: i32,
    i: i64,
    pri: f64,
}

/// Dynamic range-minimum structure for mg_lchain_rmq().
///
/// C minimap2 uses an augmented red-black tree keyed by `(y, i)` and stores the
/// minimum-priority element in each subtree. We know the full set of candidate
/// `(y, i)` keys up front, so a compressed-key segment tree provides the same
/// range-min query semantics without scanning all active anchors in the range.
struct RmqTree {
    keys: Vec<(i32, i64)>,
    seg: Vec<Option<RmqElem>>,
    tree: BTreeMap<(i32, i64), f64>,
}

impl RmqTree {
    fn new(keys: &[(i32, i64)]) -> Self {
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        let n = keys.len().next_power_of_two().max(1);
        Self {
            keys,
            seg: vec![None; n * 2],
            tree: BTreeMap::new(),
        }
    }

    fn better(a: Option<RmqElem>, b: Option<RmqElem>) -> Option<RmqElem> {
        match (a, b) {
            (None, x) | (x, None) => x,
            (Some(x), Some(y)) => {
                if y.pri < x.pri {
                    Some(y)
                } else {
                    Some(x)
                }
            }
        }
    }

    fn set_leaf(&mut self, key: (i32, i64), elem: Option<RmqElem>) {
        let Ok(pos) = self.keys.binary_search(&key) else {
            return;
        };
        let base = self.seg.len() / 2;
        let mut idx = base + pos;
        self.seg[idx] = elem;
        while idx > 1 {
            idx >>= 1;
            self.seg[idx] = Self::better(self.seg[idx << 1], self.seg[idx << 1 | 1]);
        }
    }

    fn insert(&mut self, y: i32, i: i64, pri: f64) {
        let key = (y, i);
        self.tree.insert(key, pri);
        self.set_leaf(key, Some(RmqElem { y, i, pri }));
    }

    fn remove(&mut self, y: i32, i: i64) {
        let key = (y, i);
        self.tree.remove(&key);
        self.set_leaf(key, None);
    }

    fn len(&self) -> usize {
        self.tree.len()
    }

    fn lower_bound(&self, key: (i32, i64)) -> usize {
        self.keys.partition_point(|&x| x < key)
    }

    fn upper_bound(&self, key: (i32, i64)) -> usize {
        self.keys.partition_point(|&x| x <= key)
    }

    /// Find the active element with minimum priority in C's closed key interval
    /// `[(lo_y, INT32_MAX), (hi_y, 0)]`.
    fn rmq(&self, lo_y: i32, hi_y: i32) -> Option<(i32, i64)> {
        let lo = self.lower_bound((lo_y, i32::MAX as i64));
        let hi = self.upper_bound((hi_y, 0));
        if lo >= hi {
            return None;
        }
        let base = self.seg.len() / 2;
        let mut l = base + lo;
        let mut r = base + hi;
        let mut best = None;
        while l < r {
            if l & 1 == 1 {
                best = Self::better(best, self.seg[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                best = Self::better(self.seg[r], best);
            }
            l >>= 1;
            r >>= 1;
        }
        best.map(|e| (e.y, e.i))
    }

    /// Iterate backwards from a given y position, yielding elements in decreasing
    /// `(y, i)` order. This mirrors the fallback inner-tree scan in C.
    fn iter_rev_from(&self, y: i32) -> impl Iterator<Item = (i32, i64, f64)> + '_ {
        let key = (y, i64::MAX);
        self.tree
            .range(..=key)
            .rev()
            .map(|(&(y, i), &pri)| (y, i, pri))
    }
}

/// RMQ-based chaining algorithm. Matches mg_lchain_rmq() from lchain.c.
///
/// Uses range-minimum queries for faster predecessor search compared to DP.
/// Better for very large anchor sets (>1000).
///
/// # Parameters
/// * `max_dist` - max reference gap for the outer RMQ window
/// * `max_dist_inner` - max reference gap for the inner refinement window (0 disables)
/// * `bw` - chaining bandwidth (max diagonal deviation `|dr - dq|`)
/// * `max_chn_skip` - max consecutive non-improving skips in inner-tree refinement
/// * `cap_rmq_size` - cap on active RMQ tree size; older anchors evicted past this
/// * `min_cnt` - minimum anchors required to keep a chain
/// * `min_sc` - minimum chain score to keep a chain
/// * `chn_pen_gap` - linear gap penalty coefficient
/// * `chn_pen_skip` - linear skip (per-minimizer) penalty coefficient
/// * `a` - sorted anchor array; `x = rev<<63 | tid<<32 | tpos`, `y = flags|seg_id<<48|q_span<<32|q_pos`
pub fn lchain_rmq(
    mut max_dist: i32,
    mut max_dist_inner: i32,
    bw: i32,
    max_chn_skip: i32,
    cap_rmq_size: i32,
    min_cnt: i32,
    min_sc: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    a: &[Mm128],
) -> Option<ChainResult> {
    let n = a.len() as i64;
    if n == 0 {
        return None;
    }

    if max_dist < bw {
        max_dist = bw;
    }
    if max_dist_inner < 0 {
        max_dist_inner = 0;
    }
    if max_dist_inner > max_dist {
        max_dist_inner = max_dist;
    }
    let max_drop = bw;

    let mut p = vec![-1i64; n as usize];
    let mut f = vec![0i32; n as usize];
    let mut t = vec![0i32; n as usize];
    let mut v = vec![0i32; n as usize];

    let keys: Vec<(i32, i64)> = a
        .iter()
        .enumerate()
        .map(|(i, x)| (x.y as i32, i as i64))
        .collect();
    let mut root = RmqTree::new(&keys);
    let mut root_inner = if max_dist_inner > 0 {
        Some(RmqTree::new(&keys))
    } else {
        None
    };

    let mut st: i64 = 0;
    let mut st_inner: i64 = 0;
    let mut i0: i64 = 0;

    for i in 0..n {
        let iu = i as usize;
        let mut max_j: i64 = -1;
        let q_span = ((a[iu].y >> 32) & 0xff) as i32;
        let mut max_f = q_span;

        // Add anchors at same x position that we haven't added yet
        if i0 < i && a[i0 as usize].x != a[iu].x {
            for j in i0..i {
                let ju = j as usize;
                let pri = -(f[ju] as f64
                    + 0.5 * chn_pen_gap as f64 * (a[ju].x as i32 as f64 + a[ju].y as i32 as f64));
                root.insert(a[ju].y as i32, j, pri);
                if let Some(ref mut ri) = root_inner {
                    ri.insert(a[ju].y as i32, j, pri);
                }
            }
            i0 = i;
        }

        // Remove out-of-range elements from outer tree
        while st < i
            && (a[iu].x >> 32 != a[st as usize].x >> 32
                || a[iu].x > a[st as usize].x + max_dist as u64
                || root.len() > cap_rmq_size as usize)
        {
            root.remove(a[st as usize].y as i32, st);
            st += 1;
        }

        // Remove out-of-range elements from inner tree
        if let Some(ref mut ri) = root_inner {
            while st_inner < i
                && (a[iu].x >> 32 != a[st_inner as usize].x >> 32
                    || a[iu].x > a[st_inner as usize].x + max_dist_inner as u64
                    || ri.len() > cap_rmq_size as usize)
            {
                ri.remove(a[st_inner as usize].y as i32, st_inner);
                st_inner += 1;
            }
        }

        // RMQ query on outer tree
        let lo_y = a[iu].y as i32 - max_dist;
        let hi_y = a[iu].y as i32;
        if let Some((_qy, qj)) = root.rmq(lo_y, hi_y) {
            let (sc, exact, width) =
                comput_sc_simple(&a[iu], &a[qj as usize], chn_pen_gap, chn_pen_skip);
            let sc = f[qj as usize] + sc;
            if width <= bw && sc > max_f {
                max_f = sc;
                max_j = qj;
            }

            // If not exact match, search inner tree for better candidates
            if !exact {
                if let Some(ref ri) = root_inner {
                    if (a[iu].y as i32) > 0 {
                        let mut n_skip = 0i32;
                        for (qy, qj2, _pri) in ri.iter_rev_from(a[iu].y as i32 - 1) {
                            if qy < a[iu].y as i32 - max_dist_inner {
                                break;
                            }
                            let (sc2, _, width2) = comput_sc_simple(
                                &a[iu],
                                &a[qj2 as usize],
                                chn_pen_gap,
                                chn_pen_skip,
                            );
                            let sc2 = f[qj2 as usize] + sc2;
                            if width2 <= bw {
                                if sc2 > max_f {
                                    max_f = sc2;
                                    max_j = qj2;
                                    if n_skip > 0 {
                                        n_skip -= 1;
                                    }
                                } else if t[qj2 as usize] == i as i32 {
                                    n_skip += 1;
                                    if n_skip > max_chn_skip {
                                        break;
                                    }
                                }
                                if p[qj2 as usize] >= 0 {
                                    t[p[qj2 as usize] as usize] = i as i32;
                                }
                            }
                        }
                    }
                }
            }
        }

        f[iu] = max_f;
        p[iu] = max_j;
        v[iu] = if max_j >= 0 && v[max_j as usize] > max_f {
            v[max_j as usize]
        } else {
            max_f
        };
    }

    // Backtrack
    let mut v_indices = Vec::new();
    let mut u = chain_backtrack(n, &f, &p, &mut v_indices, &mut t, min_cnt, min_sc, max_drop);
    if u.is_empty() {
        return None;
    }

    let anchors = compact_a(&mut u, &v_indices, a);
    Some(ChainResult { anchors, chains: u })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_anchors() -> Vec<Mm128> {
        let mut a = Vec::new();
        for i in 0..10 {
            let ref_pos = (i * 100 + 50) as u64;
            let q_pos = (i * 100 + 50) as u64;
            let q_span = 15u64;
            a.push(Mm128::new(0u64 << 32 | ref_pos, q_span << 32 | q_pos));
        }
        a
    }

    #[test]
    fn test_lchain_rmq_basic() {
        let a = make_linear_anchors();
        let result = lchain_rmq(
            5000,   // max_dist
            1000,   // max_dist_inner
            500,    // bw
            25,     // max_chn_skip
            100000, // cap_rmq_size
            3,      // min_cnt
            40,     // min_sc
            0.8,    // chn_pen_gap
            0.0,    // chn_pen_skip
            &a,
        );
        assert!(result.is_some(), "Should find at least one chain");
        let r = result.unwrap();
        assert!(!r.chains.is_empty());
    }

    #[test]
    fn test_lchain_rmq_empty() {
        let result = lchain_rmq(5000, 1000, 500, 25, 100000, 3, 40, 0.8, 0.0, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_rmq_tree() {
        let mut tree = RmqTree::new(&[(10, 0), (20, 1), (30, 2)]);
        tree.insert(10, 0, -100.0);
        tree.insert(20, 1, -200.0);
        tree.insert(30, 2, -150.0);

        // RMQ over full range should return element with lowest priority (most negative)
        let best = tree.rmq(0, 40);
        assert_eq!(best, Some((20, 1))); // pri=-200 is the minimum

        // RMQ over partial range
        let best = tree.rmq(25, 40);
        assert_eq!(best, Some((30, 2)));

        tree.remove(20, 1);
        let best = tree.rmq(0, 40);
        assert_eq!(best, Some((30, 2))); // pri=-150 is now the minimum
    }
}
