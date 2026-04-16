pub mod backtrack;
pub mod dp;
pub mod rmq;

use crate::flags::{SEED_SEG_MASK, SEED_SEG_SHIFT};
use crate::types::Mm128;

/// Chain result: compacted anchors and chain scores.
pub struct ChainResult {
    /// Compacted anchor array (chains are contiguous).
    pub anchors: Vec<Mm128>,
    /// Chain descriptors: score << 32 | n_anchors.
    pub chains: Vec<u64>,
}

/// Fast approximate log2. Matches mg_log2() from mmpriv.h.
/// NB: doesn't work when x < 2.
#[inline]
pub fn mg_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut log_2 = (((bits >> 23) & 255) as f32) - 128.0;
    let mut z_bits = bits;
    z_bits &= !(255 << 23);
    z_bits += 127 << 23;
    let z_f = f32::from_bits(z_bits);
    log_2 += (-0.344_848_43 * z_f + 2.024_665_8) * z_f - 0.674_877_6;
    log_2
}

/// Compute chaining score between two anchors (full version for DP chaining).
///
/// Matches comput_sc() from lchain.c.
///
/// Input anchor encoding:
///   a[].x: rev<<63 | tid<<32 | tpos
///   a[].y: flags<<40 | q_span<<32 | q_pos
#[inline]
pub fn comput_sc(
    ai: &Mm128,
    aj: &Mm128,
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    is_cdna: bool,
    n_seg: i32,
) -> i32 {
    let dq = (ai.y as i32).wrapping_sub(aj.y as i32);
    if dq <= 0 || dq > max_dist_x {
        return i32::MIN;
    }
    let dr = (ai.x as i32).wrapping_sub(aj.x as i32);
    let sidi = ((ai.y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as i32;
    let sidj = ((aj.y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as i32;
    if sidi == sidj && (dr == 0 || dq > max_dist_y) {
        return i32::MIN;
    }
    let dd = if dr > dq { dr - dq } else { dq - dr };
    if sidi == sidj && dd > bw {
        return i32::MIN;
    }
    if n_seg > 1 && !is_cdna && sidi == sidj && dr > max_dist_y {
        return i32::MIN;
    }
    let dg = dr.min(dq);
    let q_span = ((aj.y >> 32) & 0xff) as i32;
    let mut sc = q_span.min(dg);
    if dd != 0 || dg > q_span {
        let lin_pen = chn_pen_gap * dd as f32 + chn_pen_skip * dg as f32;
        let log_pen = if dd >= 1 {
            mg_log2(dd as f32 + 1.0)
        } else {
            0.0
        };
        if is_cdna || sidi != sidj {
            if sidi != sidj && dr == 0 {
                sc += 1;
            } else if dr > dq || sidi != sidj {
                sc -= lin_pen.min(log_pen) as i32;
            } else {
                sc -= (lin_pen + 0.5 * log_pen) as i32;
            }
        } else {
            sc -= (lin_pen + 0.5 * log_pen) as i32;
        }
    }
    sc
}

/// Simplified scoring for RMQ chaining. Matches comput_sc_simple() from lchain.c.
#[inline]
pub fn comput_sc_simple(
    ai: &Mm128,
    aj: &Mm128,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
) -> (i32, bool, i32) {
    // returns (score, is_exact, width)
    let dq = (ai.y as i32).wrapping_sub(aj.y as i32);
    let dr = (ai.x as i32).wrapping_sub(aj.x as i32);
    let dd = if dr > dq { dr - dq } else { dq - dr };
    let dg = dr.min(dq);
    let q_span = ((aj.y >> 32) & 0xff) as i32;
    let mut sc = q_span.min(dg);
    let exact = dd == 0 && dg <= q_span;
    if dd != 0 || dq > q_span {
        let lin_pen = chn_pen_gap * dd as f32 + chn_pen_skip * dg as f32;
        let log_pen = if dd >= 1 {
            mg_log2(dd as f32 + 1.0)
        } else {
            0.0
        };
        sc -= (lin_pen + 0.5 * log_pen) as i32;
    }
    (sc, exact, dd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mg_log2() {
        // log2(4) = 2.0
        let v = mg_log2(4.0);
        assert!((v - 2.0).abs() < 0.05);
        // log2(8) = 3.0
        let v = mg_log2(8.0);
        assert!((v - 3.0).abs() < 0.05);
        // log2(1024) = 10.0
        let v = mg_log2(1024.0);
        assert!((v - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_comput_sc_basic() {
        // Two anchors on same strand, same ref, close together
        let aj = Mm128::new(
            0u64 << 63 | 0u64 << 32 | 100, // rev=0, rid=0, pos=100
            (15u64 << 32) | 50,            // qspan=15, qpos=50
        );
        let ai = Mm128::new(
            0u64 << 63 | 0u64 << 32 | 120, // pos=120
            (15u64 << 32) | 70,            // qspan=15, qpos=70
        );
        let sc = comput_sc(&ai, &aj, 5000, 5000, 500, 0.8, 0.0, false, 1);
        assert!(
            sc > 0,
            "Score should be positive for close colinear anchors: {}",
            sc
        );
    }

    #[test]
    fn test_comput_sc_too_far() {
        let aj = Mm128::new(0u64 << 32 | 100, (15u64 << 32) | 50);
        let ai = Mm128::new(0u64 << 32 | 10100, (15u64 << 32) | 10050);
        let sc = comput_sc(&ai, &aj, 5000, 5000, 500, 0.8, 0.0, false, 1);
        assert_eq!(sc, i32::MIN, "Should reject anchors beyond max_dist_x");
    }
}
