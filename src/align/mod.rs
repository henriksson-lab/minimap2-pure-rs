pub mod score;
pub mod ksw2;

use crate::flags::KswFlags;
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
    // TODO: dispatch to SIMD when available
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
}
