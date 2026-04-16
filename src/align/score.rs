/// Generate a simple scoring matrix for DNA (m=5: A,C,G,T,N).
///
/// mat[i*m+j] = match_score if i==j (and i<m-1), else mismatch_penalty (negated).
/// The last row/column (N) gets sc_ambi.
///
/// This is used by minimap2 for standard alignment.
pub fn gen_simple_mat(m: i32, mat: &mut Vec<i8>, a: i32, b: i32, sc_ambi: i32) {
    let m = m as usize;
    mat.clear();
    mat.resize(m * m, 0);
    for i in 0..m {
        for j in 0..m {
            if i == m - 1 || j == m - 1 {
                mat[i * m + j] = -sc_ambi as i8;
            } else if i == j {
                mat[i * m + j] = a as i8;
            } else {
                mat[i * m + j] = -(b as i8);
            }
        }
    }
}

/// Generate a scoring matrix with transition/transversion distinction.
///
/// Transitions (A↔G, C↔T) get a different penalty than transversions.
pub fn gen_ts_mat(m: i32, mat: &mut Vec<i8>, a: i32, b: i32, transition: i32, sc_ambi: i32) {
    let m = m as usize;
    mat.clear();
    mat.resize(m * m, 0);
    for i in 0..m {
        for j in 0..m {
            if i == m - 1 || j == m - 1 {
                mat[i * m + j] = -sc_ambi as i8;
            } else if i == j {
                mat[i * m + j] = a as i8;
            } else if (i ^ j) == 2 {
                // A(0)↔G(2) or C(1)↔T(3): transition
                mat[i * m + j] = -(b - transition) as i8;
            } else {
                mat[i * m + j] = -(b as i8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_simple_mat() {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        assert_eq!(mat.len(), 25);
        // Diagonal (match)
        assert_eq!(mat[0], 2); // A-A
        assert_eq!(mat[6], 2); // C-C
        assert_eq!(mat[12], 2); // G-G
        assert_eq!(mat[18], 2); // T-T
                                // Off-diagonal (mismatch)
        assert_eq!(mat[1], -4); // A-C
                                // N column/row
        assert_eq!(mat[4], -1); // A-N
        assert_eq!(mat[20], -1); // N-A
        assert_eq!(mat[24], -1); // N-N
    }

    #[test]
    fn test_gen_ts_mat() {
        let mut mat = Vec::new();
        gen_ts_mat(5, &mut mat, 2, 4, 1, 1);
        // A-G (transition): -(4-1) = -3
        assert_eq!(mat[2], -3);
        // A-C (transversion): -4
        assert_eq!(mat[1], -4);
        // A-A (match): 2
        assert_eq!(mat[0], 2);
    }
}
