use crate::sort::radix_sort_mm128;
use crate::types::Mm128;

/// Find the end of a chain by backtracking with z-drop.
/// Matches mg_chain_bk_end() from lchain.c.
fn chain_bk_end(max_drop: i32, z: &[Mm128], f: &[i32], p: &[i64], t: &mut [i32], k: usize) -> i64 {
    let mut i = z[k].y as i64;
    let mut end_i: i64;
    let mut max_i = i;
    let mut max_s: i32 = 0;

    if i < 0 || t[i as usize] != 0 {
        return i;
    }

    loop {
        t[i as usize] = 2;
        i = p[i as usize];
        end_i = i;
        let s = if i < 0 {
            z[k].x as i32
        } else {
            z[k].x as i32 - f[i as usize]
        };
        if s > max_s {
            max_s = s;
            max_i = i;
        } else if max_s - s > max_drop {
            break;
        }
        if i < 0 || t[i as usize] != 0 {
            break;
        }
    }

    // Reset modified t[]
    i = z[k].y as i64;
    while i >= 0 && i != end_i {
        t[i as usize] = 0;
        i = p[i as usize];
    }

    max_i
}

/// Backtrack chains from DP arrays. Matches mg_chain_backtrack() from lchain.c.
///
/// Returns (u, v) where:
/// - `u[i]` = `score<<32 | n_anchors` for chain `i`
/// - v contains anchor indices in chain order
///
/// # Parameters
/// * `n` - number of anchors (length of `f` and `p`)
/// * `f` - per-anchor optimal DP score ending at that anchor
/// * `p` - predecessor index for each anchor (`-1` = chain head)
/// * `v_out` - output buffer for selected anchor indices in chain order
/// * `t` - scratch flag array (size `n`); used and reset internally
/// * `min_cnt` - minimum anchors required to keep a chain
/// * `min_sc` - minimum chain score to keep a chain
/// * `max_drop` - max score drop tolerated during chain end backtracking (z-drop)
pub fn chain_backtrack(
    n: i64,
    f: &[i32],
    p: &[i64],
    v_out: &mut Vec<i32>,
    t: &mut Vec<i32>,
    min_cnt: i32,
    min_sc: i32,
    max_drop: i32,
) -> Vec<u64> {
    // Count candidates meeting min_sc
    let mut n_z: usize = 0;
    for i in 0..n as usize {
        if f[i] >= min_sc {
            n_z += 1;
        }
    }
    if n_z == 0 {
        return Vec::new();
    }

    // Build and sort (score, index) array
    let mut z: Vec<Mm128> = Vec::with_capacity(n_z);
    for i in 0..n as usize {
        if f[i] >= min_sc {
            z.push(Mm128::new(f[i] as u64, i as u64));
        }
    }
    radix_sort_mm128(&mut z);

    // First pass: count chains and their total anchors
    t.iter_mut().for_each(|x| *x = 0);
    let mut n_v: usize = 0;
    let mut n_u: usize = 0;
    for k in (0..n_z).rev() {
        let idx = z[k].y as usize;
        if t[idx] == 0 {
            let n_v0 = n_v;
            let end_i = chain_bk_end(max_drop, &z, f, p, t, k);
            let mut i = z[k].y as i64;
            while i != end_i {
                n_v += 1;
                t[i as usize] = 1;
                i = p[i as usize];
            }
            let sc = if i < 0 {
                z[k].x as i32
            } else {
                z[k].x as i32 - f[i as usize]
            };
            if sc >= min_sc && n_v > n_v0 && (n_v - n_v0) as i32 >= min_cnt {
                n_u += 1;
            } else {
                n_v = n_v0;
            }
        }
    }

    // Second pass: populate u[] and v[]
    let mut u = Vec::with_capacity(n_u);
    v_out.clear();
    v_out.reserve(n_v);
    t.iter_mut().for_each(|x| *x = 0);
    for k in (0..n_z).rev() {
        let idx = z[k].y as usize;
        if t[idx] == 0 {
            let n_v0 = v_out.len();
            let end_i = chain_bk_end(max_drop, &z, f, p, t, k);
            let mut i = z[k].y as i64;
            while i != end_i {
                v_out.push(i as i32);
                t[i as usize] = 1;
                i = p[i as usize];
            }
            let sc = if i < 0 {
                z[k].x as i32
            } else {
                z[k].x as i32 - f[i as usize]
            };
            let cnt = v_out.len() - n_v0;
            if sc >= min_sc && cnt > 0 && cnt as i32 >= min_cnt {
                u.push(((sc as u64) << 32) | cnt as u64);
            } else {
                v_out.truncate(n_v0);
            }
        }
    }
    u
}

/// Compact the anchor array so chains are stored contiguously, sorted by target position.
/// Matches compact_a() from lchain.c.
///
/// # Parameters
/// * `u` - chain descriptors `score<<32 | n_anchors`; reordered in place by target position
/// * `v` - anchor indices in backtrack order (reverse-of-chain) for each chain
/// * `a` - source anchor array indexed by `v[]`
pub fn compact_a(u: &mut Vec<u64>, v: &[i32], a: &[Mm128]) -> Vec<Mm128> {
    let n_u = u.len();
    // Write chains to b[] (reversing within each chain since backtrack gives reverse order)
    let n_v: usize = v.len();
    let mut b = Vec::with_capacity(n_v);
    let mut k: usize = 0;
    for i in 0..n_u {
        let ni = (u[i] as u32) as usize;
        for j in 0..ni {
            b.push(a[v[k + (ni - j - 1)] as usize]);
        }
        k += ni;
    }

    // Sort chains by target position of first anchor
    let mut w: Vec<Mm128> = Vec::with_capacity(n_u);
    k = 0;
    for i in 0..n_u {
        w.push(Mm128::new(b[k].x, ((k as u64) << 32) | i as u64));
        k += (u[i] as u32) as usize;
    }
    radix_sort_mm128(&mut w);

    // Reorder u[] and build final anchor array
    let mut u2 = Vec::with_capacity(n_u);
    let mut result = Vec::with_capacity(n_v);
    for i in 0..n_u {
        let j = w[i].y as u32 as usize;
        let n = (u[j] as u32) as usize;
        u2.push(u[j]);
        let offset = (w[i].y >> 32) as usize;
        result.extend_from_slice(&b[offset..offset + n]);
    }
    *u = u2;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_backtrack_simple() {
        // Simple linear chain: 5 anchors with increasing scores
        let n = 5i64;
        let f = vec![10, 20, 30, 40, 50];
        let p: Vec<i64> = vec![-1, 0, 1, 2, 3]; // chain: 4->3->2->1->0
        let mut v = Vec::new();
        let mut t = vec![0i32; n as usize];

        let u = chain_backtrack(n, &f, &p, &mut v, &mut t, 2, 10, 100);
        assert!(!u.is_empty(), "Should find at least one chain");
        let chain_score = (u[0] >> 32) as i32;
        assert!(chain_score > 0);
    }

    #[test]
    fn test_compact_a() {
        let a = vec![
            Mm128::new(200, 10), // index 0
            Mm128::new(100, 20), // index 1
            Mm128::new(150, 30), // index 2
            Mm128::new(300, 40), // index 3
        ];
        // Two chains: [1,2] and [0,3]
        let mut u = vec![
            (20u64 << 32) | 2, // score=20, 2 anchors
            (30u64 << 32) | 2, // score=30, 2 anchors
        ];
        let v = vec![2i32, 1, 3, 0]; // chain1: [2,1], chain2: [3,0]

        let result = compact_a(&mut u, &v, &a);
        assert_eq!(result.len(), 4);
        // Should be sorted by target position of first anchor in each chain
        assert!(
            result[0].x <= result[2].x,
            "Chains should be sorted by target position"
        );
    }
}
