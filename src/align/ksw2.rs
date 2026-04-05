use crate::flags::KswFlags;

pub const KSW_NEG_INF: i32 = -0x40000000;

/// KSW2 alignment result.
#[derive(Clone, Debug)]
pub struct KswResult {
    pub max: i32,
    pub zdropped: bool,
    pub max_q: i32,
    pub max_t: i32,
    pub mqe: i32,   // max score reaching end of query
    pub mqe_t: i32,  // target position when reaching end of query
    pub mte: i32,   // max score reaching end of target
    pub mte_q: i32,  // query position when reaching end of target
    pub score: i32,  // max score reaching both ends
    pub reach_end: bool,
    pub cigar: Vec<u32>,
}

impl Default for KswResult {
    fn default() -> Self {
        Self {
            max: 0,
            zdropped: false,
            max_q: -1,
            max_t: -1,
            mqe: KSW_NEG_INF,
            mqe_t: -1,
            mte: KSW_NEG_INF,
            mte_q: -1,
            score: KSW_NEG_INF,
            reach_end: false,
            cigar: Vec::new(),
        }
    }
}

impl KswResult {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Low-level 16-bit local alignment for inversion detection.
/// Scalar equivalent of ksw_ll_i16.
/// Returns (score, query_end, target_end).
pub fn ksw_ll_i16(
    query: &[u8], target: &[u8], m: i8, mat: &[i8], gapo: i32, gape: i32,
) -> (i32, i32, i32) {
    let qlen = query.len();
    let tlen = target.len();
    if qlen == 0 || tlen == 0 { return (0, -1, -1); }
    let m_u = m as usize;
    let gapoe = gapo + gape;
    let mut h = vec![0i32; qlen];
    let mut e = vec![0i32; qlen];
    let mut gmax = 0i32;
    let mut qe = -1i32;
    let mut te = -1i32;

    for i in 0..tlen {
        let ti = target[i] as usize;
        let mut f = 0i32;
        let mut h_prev = 0i32;
        let mut imax = 0i32;
        for j in 0..qlen {
            let sc = mat[ti * m_u + query[j] as usize] as i32;
            let h_new = (h_prev + sc).max(e[j]).max(f).max(0);
            h_prev = h[j];
            h[j] = h_new;
            let h_gap = (h_new - gapoe).max(0);
            e[j] = (e[j] - gape).max(h_gap);
            f = (f - gape).max(h_gap);
            if h_new > imax { imax = h_new; }
        }
        if imax >= gmax {
            gmax = imax; te = i as i32;
            for j in (0..qlen).rev() {
                if h[j] == gmax { qe = j as i32; break; }
            }
        }
    }
    (gmax, qe, te)
}

/// Push a CIGAR operation, merging with the last if same op.
pub(crate) fn push_cigar_fn(cigar: &mut Vec<u32>, op: u32, len: i32) {
    if let Some(last) = cigar.last_mut() {
        if (*last & 0xf) == op {
            *last += (len as u32) << 4;
            return;
        }
    }
    cigar.push((len as u32) << 4 | op);
}

/// Apply Z-drop check. Returns true if alignment should be terminated.
fn apply_zdrop(ez: &mut KswResult, h: i32, i: i32, j: i32, zdrop: i32, e: i8) -> bool {
    let r = i + j;
    let t = i;
    if h > ez.max {
        ez.max = h;
        ez.max_t = t;
        ez.max_q = r - t;
    } else if t >= ez.max_t && r - t >= ez.max_q {
        let tl = t - ez.max_t;
        let ql = (r - t) - ez.max_q;
        let l = (tl - ql).abs();
        if zdrop >= 0 && ez.max - h > zdrop + l * e as i32 {
            ez.zdropped = true;
            return true;
        }
    }
    false
}

/// Scalar banded extension alignment with single affine gap penalty.
///
/// This is a scalar equivalent of ksw_extz2_sse. It performs banded DP alignment
/// with optional CIGAR backtracking, Z-drop, and end bonus.
///
/// # Arguments
/// * `query` - query sequence (0-3 encoded)
/// * `target` - target sequence (0-3 encoded)
/// * `m` - alphabet size (typically 5 for DNA)
/// * `mat` - m*m scoring matrix
/// * `q` - gap open penalty
/// * `e` - gap extension penalty
/// * `w` - bandwidth (-1 for full)
/// * `zdrop` - Z-drop threshold (-1 to disable)
/// * `end_bonus` - bonus for reaching the end
/// * `flag` - KSW flags
pub fn ksw_extz2(
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
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();

    if qlen <= 0 || tlen <= 0 {
        return ez;
    }

    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let m_u = m as usize;

    // DP arrays
    let mut h_cur = vec![KSW_NEG_INF; tlen as usize + 1];
    let mut h_prv = vec![KSW_NEG_INF; tlen as usize + 1];
    let mut e_arr = vec![KSW_NEG_INF; tlen as usize + 1]; // gap in target (deletion)

    // Backtrack matrix (if needed)
    let mut bt: Vec<Vec<u8>> = if with_cigar {
        vec![vec![0u8; tlen as usize]; qlen as usize]
    } else {
        Vec::new()
    };

    let qe = q as i32 + e as i32;

    // Precompute scoring row for this query base
    let e_i32 = e as i32;
    let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
    let last_col = (tlen - 1) as usize;
    let last_row = qlen - 1;

    // Initialize h_prv: first row (i=0)
    // Only cell (0,0) gets score from scoring matrix; rest start at NEG_INF
    // We handle row 0 specially below

    for i in 0..qlen {
        let iu = i as usize;
        let qi = query[iu] as usize;
        let qi_row = qi * m_u; // precompute row offset

        let j_st = if i > w { (i - w) as usize } else { 0 };
        let j_en = ((i as i64 + w as i64 + 1) as usize).min(tlen as usize);

        let mut f = KSW_NEG_INF;
        let mut row_max = KSW_NEG_INF;
        let mut row_max_j = j_st;

        // SAFETY: all indices are bounded by array allocations above
        unsafe {
            let h_prv_ptr = h_prv.as_ptr();
            let h_cur_ptr = h_cur.as_mut_ptr();
            let e_ptr = e_arr.as_mut_ptr();
            let target_ptr = target.as_ptr();
            let mat_ptr = mat.as_ptr();

            for j in j_st..j_en {
                let tj = *target_ptr.add(j) as usize;
                let sc = *mat_ptr.add(qi_row + tj) as i32;

                // Diagonal
                let diag = if j > 0 && i > 0 {
                    let prev = *h_prv_ptr.add(j - 1);
                    prev + sc
                } else if i == 0 && j == 0 {
                    sc
                } else {
                    KSW_NEG_INF
                };

                // E state (deletion)
                let e_old = *e_ptr.add(j);
                let hp = *h_prv_ptr.add(j);
                let e_new = (e_old - e_i32).max(hp - qe);
                *e_ptr.add(j) = e_new;

                // F state (insertion)
                let f_new = (f - e_i32).max(
                    if j > 0 { *h_cur_ptr.add(j - 1) - qe } else { KSW_NEG_INF }
                );
                f = f_new;

                // H = max(diag, E, F)
                let mut h = diag.max(e_new).max(f_new);
                if is_extz && h < 0 { h = 0; }
                *h_cur_ptr.add(j) = h;

                if h > row_max { row_max = h; row_max_j = j; }

                // Backtrack: bits 0-2 = state (0=H/diag, 1=E/del, 2=F/ins)
                // bit 3 (0x08) = E continuation, bit 4 (0x10) = F continuation
                if with_cigar {
                    let e_ext = e_old - e_i32;
                    let _f_ext = f - e_i32; // f before update was f_new, f_ext is old f - e
                    let bt_val = if h == diag { 0u8 }
                        else if h == e_new {
                            1u8 | if e_new == e_ext { 0x08 } else { 0 }
                        } else {
                            // For F continuation: check if f_new came from extension
                            let _f_was_ext = f_new == (f_new + e_i32) - e_i32; // always true, use prev f
                            2u8 | if j > 0 && f_new != (*h_cur_ptr.add(j-1) - qe) { 0x10 } else { 0 }
                        };
                    bt[iu][j] = bt_val;
                }

                if j == last_col && h > ez.mte { ez.mte = h; ez.mte_q = i; }
            }
        }
        if i == last_row {
            for j in j_st..j_en {
                if h_cur[j] > ez.mqe { ez.mqe = h_cur[j]; ez.mqe_t = j as i32; }
            }
        }

        // Z-drop
        if zdrop >= 0 && row_max > KSW_NEG_INF / 2 {
            if apply_zdrop(&mut ez, row_max, i, row_max_j as i32, zdrop, e) {
                break;
            }
        }

        std::mem::swap(&mut h_cur, &mut h_prv);
        h_cur[j_st..j_en].fill(KSW_NEG_INF);
    }

    // Compute final score
    if qlen > 0 && tlen > 0 {
        {
            // Use mqe and mte to determine score
            let mut score = KSW_NEG_INF;
            if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
                score = ez.mqe;
            }
            if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 && ez.mte > score {
                score = ez.mte;
            }
            ez.score = score;
            if score > KSW_NEG_INF / 2 {
                ez.reach_end = true;
            }
        }
    }

    // Add end bonus
    if end_bonus > 0 {
        if ez.mqe != KSW_NEG_INF {
            ez.mqe += end_bonus;
        }
        if ez.mte != KSW_NEG_INF {
            ez.mte += end_bonus;
        }
        if ez.score != KSW_NEG_INF {
            ez.score += end_bonus;
        }
    }

    // Backtrack to generate CIGAR
    if with_cigar && !ez.zdropped {
        let (end_i, end_j) = if ez.score > KSW_NEG_INF / 2 {
            if ez.mte_q == qlen - 1 && ez.mte >= ez.mqe {
                (ez.mte_q, tlen - 1)
            } else {
                (qlen - 1, ez.mqe_t)
            }
        } else {
            // Use max position
            (ez.max_q.max(0), ez.max_t.max(0))
        };

        if end_i >= 0 && end_j >= 0 {
            let mut i = end_i;
            let mut j = end_j;
            let mut cigar = Vec::new();
            let mut state = 0u8;

            while i >= 0 && j >= 0 {
                let iu = i as usize;
                let ju = j as usize;
                if iu >= bt.len() || ju >= bt[0].len() {
                    break;
                }
                let tmp = bt[iu][ju];
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 {
                    state = tmp & 7;
                }

                match state {
                    0 => { push_cigar_fn(&mut cigar, 0, 1); i -= 1; j -= 1; }  // M
                    1 => { push_cigar_fn(&mut cigar, 2, 1); i -= 1; }          // D
                    _ => { push_cigar_fn(&mut cigar, 1, 1); j -= 1; }          // I
                }
            }
            if i >= 0 {
                push_cigar_fn(&mut cigar, 2, i + 1); // leading deletion
            }
            if j >= 0 {
                push_cigar_fn(&mut cigar, 1, j + 1); // leading insertion
            }
            // Reverse CIGAR unless REV_CIGAR flag
            if !flag.contains(KswFlags::REV_CIGAR) {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }

    ez
}

/// Scalar banded extension with dual affine gap penalty.
///
/// Supports two gap penalty sets: (q, e) and (q2, e2).
/// This is a scalar equivalent of ksw_extd2_sse.
pub fn ksw_extd2(
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
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();

    if qlen <= 0 || tlen <= 0 {
        return ez;
    }

    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let w = if w < 0 { qlen.max(tlen) } else { w };
    let m_u = m as usize;
    let qe1 = q as i32 + e as i32;
    let qe2 = q2 as i32 + e2 as i32;

    let mut h_cur = vec![KSW_NEG_INF; tlen as usize + 1];
    let mut h_prv = vec![KSW_NEG_INF; tlen as usize + 1];
    let mut e1_arr = vec![KSW_NEG_INF; tlen as usize + 1];
    let mut e2_arr = vec![KSW_NEG_INF; tlen as usize + 1];

    let mut bt: Vec<Vec<u8>> = if with_cigar {
        vec![vec![0u8; tlen as usize]; qlen as usize]
    } else {
        Vec::new()
    };

    for i in 0..qlen {
        let iu = i as usize;
        let qi = query[iu] as usize;
        let j_st = if i > w { (i - w) as usize } else { 0 };
        let j_en = ((i as i64 + w as i64 + 1) as usize).min(tlen as usize);
        let mut f1 = KSW_NEG_INF;
        let mut f2 = KSW_NEG_INF;

        for j in j_st..j_en {
            let tj = target[j] as usize;
            let diag = if i == 0 && j == 0 {
                mat[qi * m_u + tj] as i32
            } else if i > 0 && j > 0 && h_prv[j - 1] > KSW_NEG_INF / 2 {
                h_prv[j - 1] + mat[qi * m_u + tj] as i32
            } else {
                KSW_NEG_INF
            };

            // E states (deletion)
            let e1_ext = if e1_arr[j] > KSW_NEG_INF / 2 { e1_arr[j] - e as i32 } else { KSW_NEG_INF };
            let e1_open = if h_prv[j] > KSW_NEG_INF / 2 { h_prv[j] - qe1 } else { KSW_NEG_INF };
            e1_arr[j] = e1_ext.max(e1_open);

            let e2_ext = if e2_arr[j] > KSW_NEG_INF / 2 { e2_arr[j] - e2 as i32 } else { KSW_NEG_INF };
            let e2_open = if h_prv[j] > KSW_NEG_INF / 2 { h_prv[j] - qe2 } else { KSW_NEG_INF };
            e2_arr[j] = e2_ext.max(e2_open);

            // F states (insertion)
            let f1_ext = if f1 > KSW_NEG_INF / 2 { f1 - e as i32 } else { KSW_NEG_INF };
            let f1_open = if j > 0 && h_cur[j - 1] > KSW_NEG_INF / 2 { h_cur[j - 1] - qe1 } else { KSW_NEG_INF };
            f1 = f1_ext.max(f1_open);

            let f2_ext = if f2 > KSW_NEG_INF / 2 { f2 - e2 as i32 } else { KSW_NEG_INF };
            let f2_open = if j > 0 && h_cur[j - 1] > KSW_NEG_INF / 2 { h_cur[j - 1] - qe2 } else { KSW_NEG_INF };
            f2 = f2_ext.max(f2_open);

            let h = diag.max(e1_arr[j]).max(e2_arr[j]).max(f1).max(f2);
            h_cur[j] = h;

            if with_cigar {
                let bt_val = if h == diag {
                    0u8
                } else if h == e1_arr[j] {
                    1 | if e1_arr[j] == e1_ext { 0x08 } else { 0 }
                } else if h == f1 {
                    2 | if f1 == f1_ext { 0x10 } else { 0 }
                } else if h == e2_arr[j] {
                    3 | if e2_arr[j] == e2_ext { 0x20 } else { 0 }
                } else {
                    4 | if f2 == f2_ext { 0x40 } else { 0 }
                };
                bt[iu][j] = bt_val;
            }

            if j as i32 == tlen - 1 && h > ez.mte {
                ez.mte = h;
                ez.mte_q = i;
            }
            if i == qlen - 1 && h > ez.mqe {
                ez.mqe = h;
                ez.mqe_t = j as i32;
            }
        }

        if zdrop >= 0 {
            let best_j = (j_st..j_en).max_by_key(|&j| h_cur[j]).unwrap_or(j_st);
            if h_cur[best_j] > KSW_NEG_INF / 2 {
                if apply_zdrop(&mut ez, h_cur[best_j], i, best_j as i32, zdrop, e) {
                    break;
                }
            }
        }

        std::mem::swap(&mut h_cur, &mut h_prv);
        h_cur.iter_mut().for_each(|x| *x = KSW_NEG_INF);
    }

    // Score computation (same as extz2)
    let mut score = KSW_NEG_INF;
    if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 { score = ez.mqe; }
    if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 && ez.mte > score { score = ez.mte; }
    ez.score = score;
    if score > KSW_NEG_INF / 2 { ez.reach_end = true; }

    if end_bonus > 0 {
        if ez.mqe != KSW_NEG_INF { ez.mqe += end_bonus; }
        if ez.mte != KSW_NEG_INF { ez.mte += end_bonus; }
        if ez.score != KSW_NEG_INF { ez.score += end_bonus; }
    }

    // Backtrack (same logic as extz2 but with 5 states)
    if with_cigar && !ez.zdropped {
        let (end_i, end_j) = if ez.score > KSW_NEG_INF / 2 {
            if ez.mte_q == qlen - 1 && ez.mte >= ez.mqe {
                (ez.mte_q, tlen - 1)
            } else {
                (qlen - 1, ez.mqe_t)
            }
        } else {
            (ez.max_q.max(0), ez.max_t.max(0))
        };

        if end_i >= 0 && end_j >= 0 {
            let mut i = end_i;
            let mut j = end_j;
            let mut cigar = Vec::new();
            let mut state = 0u8;

            while i >= 0 && j >= 0 {
                let iu = i as usize;
                let ju = j as usize;
                if iu >= bt.len() || ju >= bt[0].len() { break; }
                let tmp = bt[iu][ju];
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 { state = tmp & 7; }
                match state {
                    0 => { push_cigar_fn(&mut cigar, 0, 1); i -= 1; j -= 1; }
                    1 | 3 => { push_cigar_fn(&mut cigar, 2, 1); i -= 1; }
                    _ => { push_cigar_fn(&mut cigar, 1, 1); j -= 1; }
                }
            }
            if i >= 0 { push_cigar_fn(&mut cigar, 2, i + 1); }
            if j >= 0 { push_cigar_fn(&mut cigar, 1, j + 1); }
            if !flag.contains(KswFlags::REV_CIGAR) { cigar.reverse(); }
            ez.cigar = cigar;
        }
    }

    ez
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::score::gen_simple_mat;

    fn make_mat() -> Vec<i8> {
        let mut mat = Vec::new();
        gen_simple_mat(5, &mut mat, 2, 4, 1);
        mat
    }

    #[test]
    fn test_extz2_identical() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = ksw_extz2(&query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
        assert_eq!(ez.score, 16); // 8 * 2 = 16
        assert!(ez.reach_end);
        assert!(!ez.cigar.is_empty());
        // CIGAR should be 8M
        assert_eq!(ez.cigar.len(), 1);
        assert_eq!(ez.cigar[0], (8 << 4) | 0); // 8M
    }

    #[test]
    fn test_extz2_with_mismatch() {
        let mat = make_mat();
        let query =  [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 0, 3, 0, 1, 2, 3]; // mismatch at pos 2
        let ez = ksw_extz2(&query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
        assert!(ez.score > 0);
        assert!(ez.score < 16); // less than perfect
        assert!(ez.reach_end);
    }

    #[test]
    fn test_extz2_with_gap() {
        let mat = make_mat();
        let query =  [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 2, 3, 0, 1, 2, 3]; // insertion in target
        let ez = ksw_extz2(&query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::empty());
        assert!(ez.score > 0);
    }

    #[test]
    fn test_extz2_score_only() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3];
        let target = [0u8, 1, 2, 3];
        let ez = ksw_extz2(&query, &target, 5, &mat, 4, 2, -1, 400, 0, KswFlags::SCORE_ONLY);
        assert_eq!(ez.score, 8);
        assert!(ez.cigar.is_empty());
    }

    #[test]
    fn test_extd2_identical() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = ksw_extd2(&query, &target, 5, &mat, 4, 2, 24, 1, -1, 400, 0, KswFlags::empty());
        assert_eq!(ez.score, 16);
        assert!(ez.reach_end);
    }

    #[test]
    fn test_extd2_long_gap() {
        let mat = make_mat();
        // Query is short, target has a long insertion
        let query = [0u8, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 0, 0, 0, 0, 0];
        let ez = ksw_extd2(&query, &target, 5, &mat, 4, 2, 24, 1, -1, 400, 0, KswFlags::empty());
        // Should still find the alignment
        assert!(ez.mqe > 0 || ez.mte > 0);
    }
}
