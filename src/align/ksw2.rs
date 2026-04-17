use crate::flags::KswFlags;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const KSW_NEG_INF: i32 = -0x40000000;

/// KSW2 alignment result.
#[derive(Clone, Debug)]
pub struct KswResult {
    pub max: i32,
    pub zdropped: bool,
    pub max_q: i32,
    pub max_t: i32,
    pub mqe: i32,   // max score reaching end of query
    pub mqe_t: i32, // target position when reaching end of query
    pub mte: i32,   // max score reaching end of target
    pub mte_q: i32, // query position when reaching end of target
    pub score: i32, // max score reaching both ends
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
/// Equivalent of minimap2's ksw_ll_i16.
/// Returns (score, query_end, target_end).
pub fn ksw_ll_i16(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    gapo: i32,
    gape: i32,
) -> (i32, i32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: the runtime check above guarantees SSE2 support.
            return unsafe { ksw_ll_i16_sse2(query, target, m, mat, gapo, gape) };
        }
    }
    ksw_ll_i16_scalar(query, target, m, mat, gapo, gape)
}

fn ksw_ll_i16_scalar(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    gapo: i32,
    gape: i32,
) -> (i32, i32, i32) {
    let qlen = query.len();
    let tlen = target.len();
    if qlen == 0 || tlen == 0 {
        return (0, -1, -1);
    }
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
            if h_new > imax {
                imax = h_new;
            }
        }
        if imax >= gmax {
            gmax = imax;
            te = i as i32;
            for j in (0..qlen).rev() {
                if h[j] == gmax {
                    qe = j as i32;
                    break;
                }
            }
        }
    }
    (gmax, qe, te)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn ksw_ll_i16_sse2(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    gapo: i32,
    gape: i32,
) -> (i32, i32, i32) {
    let qlen = query.len();
    let tlen = target.len();
    if qlen == 0 || tlen == 0 {
        return (0, -1, -1);
    }

    let m_u = m as usize;
    let slen = (qlen + 7) >> 3;
    let n_vec = slen * m_u;
    let mut qp = vec![_mm_setzero_si128(); n_vec];
    for a in 0..m_u {
        for i in 0..slen {
            let mut v = [0i16; 8];
            for lane in 0..8 {
                let k = i + lane * slen;
                if k < qlen {
                    v[lane] = *mat.get_unchecked(a * m_u + *query.get_unchecked(k) as usize) as i16;
                }
            }
            *qp.get_unchecked_mut(a * slen + i) =
                _mm_set_epi16(v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
        }
    }

    let mut h0 = vec![_mm_setzero_si128(); slen];
    let mut h1 = vec![_mm_setzero_si128(); slen];
    let mut e_arr = vec![_mm_setzero_si128(); slen];
    let mut hmax = vec![_mm_setzero_si128(); slen];

    let zero = _mm_setzero_si128();
    let gapoe = _mm_set1_epi16((gapo + gape) as i16);
    let gape_v = _mm_set1_epi16(gape as i16);
    let mut gmax = 0i32;
    let mut te = -1i32;

    for i in 0..tlen {
        let mut f = zero;
        let mut max_v = zero;
        let ti = *target.get_unchecked(i) as usize;
        let s_base = ti * slen;

        let mut h = _mm_loadu_si128(h0.as_ptr().add(slen - 1));
        h = _mm_slli_si128::<2>(h);
        for j in 0..slen {
            h = _mm_adds_epi16(h, _mm_loadu_si128(qp.as_ptr().add(s_base + j)));
            let e = _mm_loadu_si128(e_arr.as_ptr().add(j));
            h = _mm_max_epi16(h, e);
            h = _mm_max_epi16(h, f);
            max_v = _mm_max_epi16(max_v, h);
            _mm_storeu_si128(h1.as_mut_ptr().add(j), h);

            h = _mm_subs_epu16(h, gapoe);
            let e_new = _mm_max_epi16(_mm_subs_epu16(e, gape_v), h);
            _mm_storeu_si128(e_arr.as_mut_ptr().add(j), e_new);
            f = _mm_max_epi16(_mm_subs_epu16(f, gape_v), h);
            h = _mm_loadu_si128(h0.as_ptr().add(j));
        }

        for _ in 0..8 {
            f = _mm_slli_si128::<2>(f);
            let mut any_gt = false;
            for j in 0..slen {
                let h_old = _mm_loadu_si128(h1.as_ptr().add(j));
                let h_new = _mm_max_epi16(h_old, f);
                _mm_storeu_si128(h1.as_mut_ptr().add(j), h_new);
                let h_gap = _mm_subs_epu16(h_new, gapoe);
                f = _mm_subs_epu16(f, gape_v);
                let gt = _mm_cmpgt_epi16(f, h_gap);
                if _mm_movemask_epi8(gt) != 0 {
                    any_gt = true;
                }
            }
            if !any_gt {
                break;
            }
        }

        let imax = hmax_epi16(max_v) as i32;
        if imax >= gmax {
            gmax = imax;
            te = i as i32;
            hmax.copy_from_slice(&h1);
        }
        std::mem::swap(&mut h0, &mut h1);
    }

    let mut qe = -1i32;
    for i in 0..(slen * 8) {
        let vec_idx = i / 8;
        let lane = i & 7;
        let mut lanes = [0i16; 8];
        _mm_storeu_si128(
            lanes.as_mut_ptr() as *mut __m128i,
            _mm_loadu_si128(hmax.as_ptr().add(vec_idx)),
        );
        if lanes[lane] as i32 == gmax {
            let q = i / 8 + (i & 7) * slen;
            qe = q as i32;
        }
    }
    (gmax, qe, te)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hmax_epi16(mut x: __m128i) -> i16 {
    x = _mm_max_epi16(x, _mm_srli_si128::<8>(x));
    x = _mm_max_epi16(x, _mm_srli_si128::<4>(x));
    x = _mm_max_epi16(x, _mm_srli_si128::<2>(x));
    _mm_extract_epi16::<0>(x) as i16
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
                let f_new = (f - e_i32).max(if j > 0 {
                    *h_cur_ptr.add(j - 1) - qe
                } else {
                    KSW_NEG_INF
                });
                f = f_new;

                // H = max(diag, E, F)
                let mut h = diag.max(e_new).max(f_new);
                if is_extz && h < 0 {
                    h = 0;
                }
                *h_cur_ptr.add(j) = h;

                if h > row_max {
                    row_max = h;
                    row_max_j = j;
                }

                // Backtrack: bits 0-2 = state (0=H/diag, 1=E/del, 2=F/ins)
                // bit 3 (0x08) = E continuation, bit 4 (0x10) = F continuation
                if with_cigar {
                    let e_ext = e_old - e_i32;
                    let _f_ext = f - e_i32; // f before update was f_new, f_ext is old f - e
                    let bt_val = if h == diag {
                        0u8
                    } else if h == e_new {
                        1u8 | if e_new == e_ext { 0x08 } else { 0 }
                    } else {
                        // For F continuation: check if f_new came from extension
                        let _f_was_ext = f_new == (f_new + e_i32) - e_i32; // always true, use prev f
                        2u8 | if j > 0 && f_new != (*h_cur_ptr.add(j - 1) - qe) {
                            0x10
                        } else {
                            0
                        }
                    };
                    bt[iu][j] = bt_val;
                }

                if j == last_col && h > ez.mte {
                    ez.mte = h;
                    ez.mte_q = i;
                }
            }
        }
        if i == last_row {
            for j in j_st..j_en {
                if h_cur[j] > ez.mqe {
                    ez.mqe = h_cur[j];
                    ez.mqe_t = j as i32;
                }
            }
        }

        // Z-drop: skip for APPROX_MAX
        if zdrop >= 0 && !flag.contains(KswFlags::APPROX_MAX) && row_max > KSW_NEG_INF / 2 {
            if apply_zdrop(&mut ez, row_max, i, row_max_j as i32, zdrop, e) {
                break;
            }
        }

        std::mem::swap(&mut h_cur, &mut h_prv);
        h_cur[j_st..j_en].fill(KSW_NEG_INF);
    }

    // Compute final score for full alignment (non-extension)
    if !is_extz && qlen > 0 && tlen > 0 {
        if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
            ez.score = ez.mqe;
        } else if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 {
            ez.score = ez.mte;
        }
    }

    // Backtrace start — generate CIGAR even for zdropped
    if with_cigar {
        let (end_i, end_j) = if !ez.zdropped && !is_extz {
            ez.reach_end = ez.score > KSW_NEG_INF / 2;
            (qlen - 1, tlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max {
            ez.reach_end = true;
            (qlen - 1, ez.mqe_t)
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            (ez.max_t.max(0), ez.max_q.max(0))
        } else {
            (-1, -1)
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
                    // i=query, j=target: state 1 = E (query gap) = I, state 2 = F (target gap) = D
                    0 => {
                        push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 => {
                        push_cigar_fn(&mut cigar, 1, 1);
                        i -= 1;
                    }
                    _ => {
                        push_cigar_fn(&mut cigar, 2, 1);
                        j -= 1;
                    }
                }
            }
            if i >= 0 {
                push_cigar_fn(&mut cigar, 1, i + 1);
            }
            if j >= 0 {
                push_cigar_fn(&mut cigar, 2, j + 1);
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
            let e1_ext = if e1_arr[j] > KSW_NEG_INF / 2 {
                e1_arr[j] - e as i32
            } else {
                KSW_NEG_INF
            };
            let e1_open = if h_prv[j] > KSW_NEG_INF / 2 {
                h_prv[j] - qe1
            } else {
                KSW_NEG_INF
            };
            e1_arr[j] = e1_ext.max(e1_open);

            let e2_ext = if e2_arr[j] > KSW_NEG_INF / 2 {
                e2_arr[j] - e2 as i32
            } else {
                KSW_NEG_INF
            };
            let e2_open = if h_prv[j] > KSW_NEG_INF / 2 {
                h_prv[j] - qe2
            } else {
                KSW_NEG_INF
            };
            e2_arr[j] = e2_ext.max(e2_open);

            // F states (insertion)
            let f1_ext = if f1 > KSW_NEG_INF / 2 {
                f1 - e as i32
            } else {
                KSW_NEG_INF
            };
            let f1_open = if j > 0 && h_cur[j - 1] > KSW_NEG_INF / 2 {
                h_cur[j - 1] - qe1
            } else {
                KSW_NEG_INF
            };
            f1 = f1_ext.max(f1_open);

            let f2_ext = if f2 > KSW_NEG_INF / 2 {
                f2 - e2 as i32
            } else {
                KSW_NEG_INF
            };
            let f2_open = if j > 0 && h_cur[j - 1] > KSW_NEG_INF / 2 {
                h_cur[j - 1] - qe2
            } else {
                KSW_NEG_INF
            };
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

        // Z-drop check: skip for APPROX_MAX (matching SIMD behavior where the
        // approximate tracking path only checks z-drop with APPROX_DROP flag)
        if zdrop >= 0 && !flag.contains(KswFlags::APPROX_MAX) {
            let best_j = (j_st..j_en).max_by_key(|&j| h_cur[j]).unwrap_or(j_st);
            if h_cur[best_j] > KSW_NEG_INF / 2 {
                if apply_zdrop(&mut ez, h_cur[best_j], i, best_j as i32, zdrop, e2) {
                    break;
                }
            }
        }

        std::mem::swap(&mut h_cur, &mut h_prv);
        h_cur.iter_mut().for_each(|x| *x = KSW_NEG_INF);
    }

    // Score: for non-extension, score = H[qlen-1][tlen-1].
    // For extension, score is the max along the last query row (mqe) or last target col (mte).
    let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
    if !is_extz {
        // Full alignment: the score at (qlen-1, tlen-1) was tracked via mqe/mte
        if ez.mqe > KSW_NEG_INF / 2 && ez.mqe_t == tlen - 1 {
            ez.score = ez.mqe;
        } else if ez.mte > KSW_NEG_INF / 2 && ez.mte_q == qlen - 1 {
            ez.score = ez.mte;
        }
    }

    // Backtrace start — generate CIGAR even for zdropped (matching C SIMD behavior)
    if with_cigar {
        let (end_i, end_j) = if !ez.zdropped && !is_extz {
            ez.reach_end = ez.score > KSW_NEG_INF / 2;
            (qlen - 1, tlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max {
            ez.reach_end = true;
            (qlen - 1, ez.mqe_t)
        } else if ez.max_t >= 0 && ez.max_q >= 0 {
            // apply_zdrop sets max_t=query_pos, max_q=target_pos
            (ez.max_t.max(0), ez.max_q.max(0))
        } else {
            (-1, -1)
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
                // i=query, j=target: state 1/3 = E (query gap) = I, state 2/4 = F (target gap) = D
                match state {
                    0 => {
                        push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 | 3 => {
                        push_cigar_fn(&mut cigar, 1, 1);
                        i -= 1;
                    }
                    _ => {
                        push_cigar_fn(&mut cigar, 2, 1);
                        j -= 1;
                    }
                }
            }
            if i >= 0 {
                push_cigar_fn(&mut cigar, 1, i + 1);
            }
            if j >= 0 {
                push_cigar_fn(&mut cigar, 2, j + 1);
            }
            if !flag.contains(KswFlags::REV_CIGAR) {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }

    ez
}

/// Scalar splice-aware extension alignment.
///
/// This models the extra long-deletion splice state from minimap2's
/// `ksw_exts2`: ordinary affine insertion/deletion gaps use `q/e`, while long
/// reference skips use `q2` plus donor/acceptor splice signal penalties. It is
/// intentionally scalar and is only dispatched for splice mode.
pub fn ksw_exts2(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    noncan: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    let qlen = query.len();
    let tlen = target.len();
    let mut ez = KswResult::new();
    if qlen == 0 || tlen == 0 || q2 <= q + e {
        return ez;
    }

    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
    let band = if w < 0 { qlen.max(tlen) as i32 } else { w };
    let qe = q as i32 + e as i32;
    let splice_open = q2 as i32;
    let m_u = m as usize;
    let donor = splice_donor_scores(target, noncan as i32, flag);
    let acceptor = splice_acceptor_scores(target, noncan as i32, flag);

    let cols = tlen + 1;
    let idx = |i: usize, j: usize| i * cols + j;
    let mut h = vec![KSW_NEG_INF; (qlen + 1) * cols];
    let mut ins = vec![KSW_NEG_INF; (qlen + 1) * cols];
    let mut del = vec![KSW_NEG_INF; (qlen + 1) * cols];
    let mut bt = if with_cigar {
        vec![0u8; (qlen + 1) * cols]
    } else {
        Vec::new()
    };
    let mut skip_len = if with_cigar {
        vec![0usize; (qlen + 1) * cols]
    } else {
        Vec::new()
    };
    h[idx(0, 0)] = 0;

    for i in 0..=qlen {
        let j_st = if i as i32 > band {
            i - band as usize
        } else {
            0
        };
        let j_en = (i + band as usize).min(tlen);
        let mut best_splice = KSW_NEG_INF;
        let mut best_splice_j = 0usize;

        for j in j_st..=j_en {
            if i == 0 && j == 0 {
                continue;
            }
            let cur = idx(i, j);
            let mut best = KSW_NEG_INF;
            let mut state = 0u8;

            if i > 0 && j > 0 {
                let sc = mat[query[i - 1] as usize * m_u + target[j - 1] as usize] as i32;
                let diag = h[idx(i - 1, j - 1)] + sc;
                if diag > best {
                    best = diag;
                    state = 0;
                }
            }
            if i > 0 {
                let up = idx(i - 1, j);
                let ext = ins[up] - e as i32;
                let open = h[up] - qe;
                ins[cur] = ext.max(open);
                if ins[cur] > best {
                    best = ins[cur];
                    state = 1;
                }
            }
            if j > 0 {
                let left = idx(i, j - 1);
                let ext = del[left] - e as i32;
                let open = h[left] - qe;
                del[cur] = ext.max(open);
                if del[cur] > best {
                    best = del[cur];
                    state = 2;
                }
            }
            if j >= 6 && best_splice > KSW_NEG_INF / 2 {
                let sp = best_splice - splice_open + acceptor[j];
                if sp > best {
                    best = sp;
                    state = 3;
                }
            }
            if is_extz && best < 0 {
                best = 0;
                state = 0;
            }
            h[cur] = best;
            if with_cigar {
                bt[cur] = state;
                if state == 3 {
                    skip_len[cur] = j - best_splice_j;
                }
            }

            if best > ez.max {
                ez.max = best;
                ez.max_q = i as i32 - 1;
                ez.max_t = j as i32 - 1;
            }
            if j == tlen && best > ez.mte {
                ez.mte = best;
                ez.mte_q = i as i32 - 1;
            }
            if i == qlen && best > ez.mqe {
                ez.mqe = best;
                ez.mqe_t = j as i32 - 1;
            }
            if j <= tlen.saturating_sub(6) {
                let cand = best + donor[j];
                if cand > best_splice {
                    best_splice = cand;
                    best_splice_j = j;
                }
            }
        }

        if zdrop >= 0 && !flag.contains(KswFlags::APPROX_MAX) && ez.max_q >= 0 && ez.max_t >= 0 {
            let row_best_j = (j_st..=j_en).max_by_key(|&j| h[idx(i, j)]).unwrap_or(j_st);
            let row_best = h[idx(i, row_best_j)];
            if row_best > KSW_NEG_INF / 2
                && apply_zdrop(
                    &mut ez,
                    row_best,
                    i as i32 - 1,
                    row_best_j as i32 - 1,
                    zdrop,
                    e,
                )
            {
                break;
            }
        }
    }

    ez.score = h[idx(qlen, tlen)];
    ez.reach_end = ez.score > KSW_NEG_INF / 2;
    if with_cigar {
        let (mut i, mut j) = if !ez.zdropped && !is_extz {
            (qlen, tlen)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max {
            ez.reach_end = true;
            (qlen, (ez.mqe_t + 1).max(0) as usize)
        } else if ez.max_q >= 0 && ez.max_t >= 0 {
            ((ez.max_q + 1) as usize, (ez.max_t + 1) as usize)
        } else {
            (0, 0)
        };
        let mut cigar = Vec::new();
        while i > 0 || j > 0 {
            let state = bt[idx(i, j)] & 7;
            match state {
                0 if i > 0 && j > 0 => {
                    push_cigar_fn(&mut cigar, 0, 1);
                    i -= 1;
                    j -= 1;
                }
                1 if i > 0 => {
                    push_cigar_fn(&mut cigar, 1, 1);
                    i -= 1;
                }
                2 if j > 0 => {
                    push_cigar_fn(&mut cigar, 2, 1);
                    j -= 1;
                }
                3 if j > 0 => {
                    let mut len = skip_len[idx(i, j)];
                    if len == 0 || len > j {
                        len = 1;
                    }
                    push_cigar_fn(&mut cigar, 3, len as i32);
                    j -= len;
                }
                _ => {
                    if i > 0 {
                        push_cigar_fn(&mut cigar, 1, 1);
                        i -= 1;
                    } else if j > 0 {
                        push_cigar_fn(&mut cigar, 2, 1);
                        j -= 1;
                    }
                }
            }
        }
        if !flag.contains(KswFlags::REV_CIGAR) {
            cigar.reverse();
        }
        ez.cigar = cigar;
    }
    ez
}

fn splice_donor_scores(target: &[u8], noncan: i32, flag: KswFlags) -> Vec<i32> {
    let mut scores = vec![-noncan; target.len() + 1];
    if noncan <= 0 {
        scores.fill(0);
        return scores;
    }
    for pos in 0..target.len().saturating_sub(1) {
        scores[pos] = splice_signal_penalty_at(target, pos, true, flag, noncan);
    }
    scores
}

fn splice_acceptor_scores(target: &[u8], noncan: i32, flag: KswFlags) -> Vec<i32> {
    let mut scores = vec![-noncan; target.len() + 1];
    if noncan <= 0 {
        scores.fill(0);
        return scores;
    }
    for pos in 2..=target.len() {
        scores[pos] = splice_signal_penalty_at(target, pos, false, flag, noncan);
    }
    scores
}

fn splice_signal_penalty_at(
    target: &[u8],
    pos: usize,
    donor: bool,
    flag: KswFlags,
    noncan: i32,
) -> i32 {
    if flag.contains(KswFlags::SPLICE_CMPLX) {
        return splice_complex_signal_penalty_at(target, pos, donor, flag);
    }
    let half = if flag.contains(KswFlags::SPLICE_FLANK) {
        noncan / 2
    } else {
        0
    };
    if flag.contains(KswFlags::SPLICE_REV) {
        if donor {
            if pos + 1 < target.len() && target[pos] == 1 && target[pos + 1] == 3 {
                return 0;
            }
            if pos + 1 < target.len() && target[pos] == 2 && target[pos + 1] == 3 {
                return -noncan;
            }
        } else if pos >= 2 {
            if target[pos - 2] == 0 && target[pos - 1] == 1 {
                return 0;
            }
            if target[pos - 2] == 2 && target[pos - 1] == 1
                || target[pos - 2] == 0 && target[pos - 1] == 3
            {
                return -noncan;
            }
        }
    } else if donor {
        if pos + 1 < target.len() && target[pos] == 2 && target[pos + 1] == 3 {
            return 0;
        }
        if pos + 1 < target.len()
            && (target[pos] == 2 && target[pos + 1] == 1
                || target[pos] == 0 && target[pos + 1] == 3)
        {
            return -noncan;
        }
    } else if pos >= 2 {
        if target[pos - 2] == 0 && target[pos - 1] == 2 {
            return 0;
        }
        if target[pos - 2] == 0 && target[pos - 1] == 1 {
            return -noncan;
        }
    }
    -half.max(noncan)
}

fn splice_complex_signal_penalty_at(target: &[u8], pos: usize, donor: bool, flag: KswFlags) -> i32 {
    let sp = [3, 5, 7, 10];
    let rev_cigar = flag.contains(KswFlags::REV_CIGAR);
    let splice_rev = flag.contains(KswFlags::SPLICE_REV);
    let z = match (rev_cigar, splice_rev, donor) {
        (false, false, true) => {
            if pos + 2 < target.len() && target[pos] == 2 && target[pos + 1] == 3 {
                if target[pos + 2] == 0 || target[pos + 2] == 2 {
                    -1
                } else {
                    0
                }
            } else if pos + 1 < target.len() && target[pos] == 2 && target[pos + 1] == 1 {
                1
            } else if pos + 1 < target.len() && target[pos] == 0 && target[pos + 1] == 3 {
                2
            } else {
                3
            }
        }
        (false, false, false) => {
            if pos >= 3 && target[pos - 2] == 0 && target[pos - 1] == 2 {
                if target[pos - 3] == 1 || target[pos - 3] == 3 {
                    -1
                } else {
                    0
                }
            } else if pos >= 2 && target[pos - 2] == 0 && target[pos - 1] == 1 {
                2
            } else {
                3
            }
        }
        (false, true, true) => {
            if pos + 2 < target.len() && target[pos] == 1 && target[pos + 1] == 3 {
                if target[pos + 2] == 0 || target[pos + 2] == 2 {
                    -1
                } else {
                    0
                }
            } else if pos + 1 < target.len() && target[pos] == 2 && target[pos + 1] == 3 {
                2
            } else {
                3
            }
        }
        (false, true, false) => {
            if pos >= 3 && target[pos - 2] == 0 && target[pos - 1] == 1 {
                if target[pos - 3] == 1 || target[pos - 3] == 3 {
                    -1
                } else {
                    0
                }
            } else if pos >= 2 && target[pos - 2] == 2 && target[pos - 1] == 1 {
                1
            } else if pos >= 2 && target[pos - 2] == 0 && target[pos - 1] == 3 {
                2
            } else {
                3
            }
        }
        (true, false, true) => {
            if pos + 2 < target.len() && target[pos] == 2 && target[pos + 1] == 0 {
                if target[pos + 2] == 1 || target[pos + 2] == 3 {
                    -1
                } else {
                    0
                }
            } else if pos + 1 < target.len() && target[pos] == 1 && target[pos + 1] == 0 {
                2
            } else {
                3
            }
        }
        (true, false, false) => {
            if pos >= 3 && target[pos - 2] == 3 && target[pos - 1] == 2 {
                if target[pos - 3] == 0 || target[pos - 3] == 2 {
                    -1
                } else {
                    0
                }
            } else if pos >= 2 && target[pos - 2] == 1 && target[pos - 1] == 2 {
                1
            } else if pos >= 2 && target[pos - 2] == 3 && target[pos - 1] == 0 {
                2
            } else {
                3
            }
        }
        (true, true, true) => {
            if pos + 2 < target.len() && target[pos] == 1 && target[pos + 1] == 0 {
                if target[pos + 2] == 1 || target[pos + 2] == 3 {
                    -1
                } else {
                    0
                }
            } else if pos + 1 < target.len() && target[pos] == 1 && target[pos + 1] == 2 {
                1
            } else if pos + 1 < target.len() && target[pos] == 3 && target[pos + 1] == 0 {
                2
            } else {
                3
            }
        }
        (true, true, false) => {
            if pos >= 3 && target[pos - 2] == 3 && target[pos - 1] == 1 {
                if target[pos - 3] == 0 || target[pos - 3] == 2 {
                    -1
                } else {
                    0
                }
            } else if pos >= 2 && target[pos - 2] == 3 && target[pos - 1] == 2 {
                2
            } else {
                3
            }
        }
    };
    if z < 0 {
        0
    } else {
        -sp[z as usize]
    }
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
        let ez = ksw_extz2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert_eq!(ez.score, 16); // 8 * 2 = 16
        assert!(ez.reach_end);
        assert!(!ez.cigar.is_empty());
        // CIGAR should be 8M
        assert_eq!(ez.cigar.len(), 1);
        assert_eq!(ez.cigar[0], (8 << 4) | 0); // 8M
    }

    #[test]
    fn test_ksw_ll_i16_matches_scalar() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 4, 2, 3, 1, 1, 2, 0, 3, 2, 1, 0];
        let target = [3u8, 0, 1, 2, 3, 0, 1, 4, 2, 3, 1, 2, 0, 3, 2, 1, 0];
        assert_eq!(
            ksw_ll_i16(&query, &target, 5, &mat, 4, 2),
            ksw_ll_i16_scalar(&query, &target, 5, &mat, 4, 2)
        );
    }

    #[test]
    fn test_extz2_with_mismatch() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 0, 3, 0, 1, 2, 3]; // mismatch at pos 2
        let ez = ksw_extz2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert!(ez.score > 0);
        assert!(ez.score < 16); // less than perfect
        assert!(ez.reach_end);
    }

    #[test]
    fn test_extz2_with_gap() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 2, 3, 0, 1, 2, 3]; // insertion in target
        let ez = ksw_extz2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert!(ez.score > 0);
    }

    #[test]
    fn test_extz2_score_only() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3];
        let target = [0u8, 1, 2, 3];
        let ez = ksw_extz2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            -1,
            400,
            0,
            KswFlags::SCORE_ONLY,
        );
        assert_eq!(ez.score, 8);
        assert!(ez.cigar.is_empty());
    }

    #[test]
    fn test_extd2_identical() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let ez = ksw_extd2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            4,
            1,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        assert_eq!(ez.score, 16);
        assert!(ez.reach_end);
    }

    #[test]
    fn test_extd2_long_gap() {
        let mat = make_mat();
        // Query is short, target has a long insertion
        let query = [0u8, 1, 2, 3];
        let target = [0u8, 1, 2, 3, 0, 0, 0, 0, 0, 0];
        let ez = ksw_extd2(
            &query,
            &target,
            5,
            &mat,
            4,
            2,
            4,
            1,
            -1,
            400,
            0,
            KswFlags::empty(),
        );
        // Should still find the alignment
        assert!(ez.mqe > 0 || ez.mte > 0);
    }

    #[test]
    fn test_exts2_prefers_canonical_splice_skip() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [
            0u8, 1, 2, 3, // ACGT
            2, 3, 0, 0, 0, 0, 0, 2, // GT.....AG
            0, 1, 2, 3, // ACGT
        ];
        let ez = ksw_exts2(
            &query,
            &target,
            5,
            &mat,
            2,
            1,
            4,
            9,
            50,
            200,
            -1,
            KswFlags::SPLICE_FOR | KswFlags::SPLICE_FLANK,
        );
        assert!(ez.score > 0);
        assert!(ez.cigar.iter().any(|&c| (c & 0xf) == 3));
    }
}
