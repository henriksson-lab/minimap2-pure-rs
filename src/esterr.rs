use crate::index::MmIdx;
use crate::types::{AlignReg, Mm128};

/// Get forward-strand query position from an anchor.
fn get_for_qpos(qlen: i32, a: &Mm128) -> i32 {
    let x = a.y as i32;
    let q_span = ((a.y >> 32) & 0xff) as i32;
    if (a.x >> 63) != 0 {
        qlen - 1 - (x + 1 - q_span)
    } else {
        x
    }
}

/// Binary search for a minimizer position.
fn get_mini_idx(qlen: i32, a: &Mm128, mini_pos: &[u64]) -> i32 {
    let x = get_for_qpos(qlen, a);
    let mut l: i32 = 0;
    let mut r: i32 = mini_pos.len() as i32 - 1;
    while l <= r {
        let m = ((l as u64 + r as u64) >> 1) as i32;
        let y = mini_pos[m as usize] as u32 as i32;
        if y < x {
            l = m + 1;
        } else if y > x {
            r = m - 1;
        } else {
            return m;
        }
    }
    -1
}

/// Estimate sequence divergence for alignment regions.
/// Matches mm_est_err() from esterr.c.
pub fn est_err(mi: &MmIdx, qlen: i32, regs: &mut [AlignReg], a: &[Mm128], mini_pos: &[u64]) {
    let n = mini_pos.len();
    if n == 0 {
        return;
    }
    let mut sum_k: u64 = 0;
    for &mp in mini_pos {
        sum_k += (mp >> 32) & 0xff;
    }
    let avg_k = sum_k as f32 / n as f32;

    for r in regs.iter_mut() {
        r.div = -1.0;
        if r.cnt == 0 {
            continue;
        }
        let start_anchor = if r.rev {
            &a[r.as_ as usize + r.cnt as usize - 1]
        } else {
            &a[r.as_ as usize]
        };
        let st = get_mini_idx(qlen, start_anchor, mini_pos);
        if st < 0 {
            continue;
        }

        let l_ref = mi.seqs[r.rid as usize].len as i32;
        let mut k = 1i32;
        let mut en = st;
        let mut n_match = 1i32;
        let mut j = st + 1;
        while j < n as i32 && k < r.cnt {
            let anchor = if r.rev {
                &a[r.as_ as usize + r.cnt as usize - 1 - k as usize]
            } else {
                &a[r.as_ as usize + k as usize]
            };
            let x = get_for_qpos(qlen, anchor);
            if x == mini_pos[j as usize] as u32 as i32 {
                k += 1;
                en = j;
                n_match += 1;
            }
            j += 1;
        }

        let mut n_tot = en - st + 1;
        if r.qs as f32 > avg_k && r.rs as f32 > avg_k {
            n_tot += 1;
        }
        if (qlen - r.qe) as f32 > avg_k && (l_ref - r.re) as f32 > avg_k {
            n_tot += 1;
        }
        r.div = if n_match >= n_tot {
            0.0
        } else {
            (1.0 - (n_match as f64 / n_tot as f64).powf(1.0 / avg_k as f64)) as f32
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_for_qpos_forward() {
        let a = Mm128::new(0u64 << 63 | 100, 15u64 << 32 | 50);
        assert_eq!(get_for_qpos(100, &a), 50);
    }

    #[test]
    fn test_get_for_qpos_reverse() {
        let a = Mm128::new(1u64 << 63 | 100, 15u64 << 32 | 50);
        let qlen = 200;
        let x = get_for_qpos(qlen, &a);
        // reverse: qlen - 1 - (50 + 1 - 15) = 200 - 1 - 36 = 163
        assert_eq!(x, 163);
    }

    #[test]
    fn test_get_mini_idx() {
        let mini_pos = vec![
            15u64 << 32 | 10,
            15u64 << 32 | 20,
            15u64 << 32 | 30,
            15u64 << 32 | 40,
        ];
        let a = Mm128::new(0u64 << 63 | 100, 15u64 << 32 | 20);
        assert_eq!(get_mini_idx(100, &a, &mini_pos), 1);
    }
}
