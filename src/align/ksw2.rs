use crate::flags::KswFlags;
use std::cell::RefCell;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
thread_local! {
    static KSW_LL_SCRATCH: RefCell<Vec<__m128i>> = const { RefCell::new(Vec::new()) };
}

#[derive(Default)]
struct Exts2Scratch {
    donor: Vec<i8>,
    acceptor: Vec<i8>,
    u: Vec<i8>,
    v: Vec<i8>,
    x: Vec<i8>,
    y: Vec<i8>,
    x2: Vec<i8>,
    s: Vec<i32>,
    h_arr: Vec<i32>,
    bt: Vec<u8>,
    bt_row_starts: Vec<usize>,
    offs: Vec<i32>,
    off_ends: Vec<i32>,
}

thread_local! {
    static KSW_EXTS2_SCRATCH: RefCell<Exts2Scratch> = RefCell::new(Exts2Scratch::default());
}

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
    let zero = _mm_setzero_si128();
    let total_vec = n_vec + 4 * slen;
    let mut scratch = KSW_LL_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if scratch.len() < total_vec {
        scratch.resize(total_vec, zero);
    }
    scratch[..total_vec].fill(zero);

    let result = {
        let scratch = &mut scratch[..total_vec];
        let (qp, rest) = scratch.split_at_mut(n_vec);
        let (mut h0, rest) = rest.split_at_mut(slen);
        let (mut h1, rest) = rest.split_at_mut(slen);
        let (e_arr, hmax) = rest.split_at_mut(slen);

        for a in 0..m_u {
            for i in 0..slen {
                let mut v = [0i16; 8];
                for lane in 0..8 {
                    let k = i + lane * slen;
                    if k < qlen {
                        v[lane] =
                            *mat.get_unchecked(a * m_u + *query.get_unchecked(k) as usize) as i16;
                    }
                }
                *qp.get_unchecked_mut(a * slen + i) =
                    _mm_set_epi16(v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
            }
        }

        let gapoe = _mm_set1_epi16((gapo + gape) as i16);
        let gape_v = _mm_set1_epi16(gape as i16);
        let mut gmax = 0i32;
        let mut te = -1i32;

        for i in 0..tlen {
            let mut f = zero;
            let mut max_v = zero;
            let ti = *target.get_unchecked(i) as usize;
            let s_base = ti * slen;

            let mut h = _mm_load_si128(h0.as_ptr().add(slen - 1));
            h = _mm_slli_si128::<2>(h);
            for j in 0..slen {
                h = _mm_adds_epi16(h, _mm_load_si128(qp.as_ptr().add(s_base + j)));
                let e = _mm_load_si128(e_arr.as_ptr().add(j));
                h = _mm_max_epi16(h, e);
                h = _mm_max_epi16(h, f);
                max_v = _mm_max_epi16(max_v, h);
                _mm_store_si128(h1.as_mut_ptr().add(j), h);

                h = _mm_subs_epu16(h, gapoe);
                let e_new = _mm_max_epi16(_mm_subs_epu16(e, gape_v), h);
                _mm_store_si128(e_arr.as_mut_ptr().add(j), e_new);
                f = _mm_max_epi16(_mm_subs_epu16(f, gape_v), h);
                h = _mm_load_si128(h0.as_ptr().add(j));
            }

            'lazy_f: for _ in 0..8 {
                f = _mm_slli_si128::<2>(f);
                for j in 0..slen {
                    let h_old = _mm_load_si128(h1.as_ptr().add(j));
                    let h_new = _mm_max_epi16(h_old, f);
                    _mm_store_si128(h1.as_mut_ptr().add(j), h_new);
                    let h_gap = _mm_subs_epu16(h_new, gapoe);
                    f = _mm_subs_epu16(f, gape_v);
                    if _mm_movemask_epi8(_mm_cmpgt_epi16(f, h_gap)) == 0 {
                        break 'lazy_f;
                    }
                }
            }

            let imax = hmax_epi16(max_v) as i32;
            if imax >= gmax {
                gmax = imax;
                te = i as i32;
                hmax.copy_from_slice(h1);
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
                _mm_load_si128(hmax.as_ptr().add(vec_idx)),
            );
            if lanes[lane] as i32 == gmax {
                let q = i / 8 + (i & 7) * slen;
                qe = q as i32;
            }
        }
        (gmax, qe, te)
    };

    KSW_LL_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    result
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
    // Minimum intron length — matches C's long_thres
    // (ksw2_extd2_sse.c:102-105). Below this, the splice path can't beat a
    // regular gap so don't even consider it. Without this guard, Rust would
    // call short stretches like 6-base regions as N-skips where C produces
    // simple D's (e.g. yeast SRR30335018.169095 had `7D` in C but `7N` in
    // Rust prior to this fix).
    let e_i = e as i32;
    let e2_i = 0i32; // ksw_exts2 effectively has e2 = 0 for the splice path
    let mut min_intron_len = if e_i != e2_i {
        (q2 as i32 - q as i32) / (e_i - e2_i) - 1
    } else {
        0
    };
    if (q2 as i32) + e2_i + min_intron_len * e2_i > (q as i32) + e_i + min_intron_len * e_i {
        min_intron_len += 1;
    }
    if min_intron_len < 6 {
        min_intron_len = 6;
    }

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
            // Order of state-update probing matches C's ksw_exts2_sse SIMD
            // tie-break (line 307-311 of ksw2_exts2_sse.c): a (DEL, "x array")
            // before b (INS, "y array") before a2 (splice). For equal scores,
            // earlier-probed states win — Rust used to probe ins before del,
            // producing different CIGAR shapes than C (e.g. SRR30335018.202275).
            //
            // Note: Rust's exts2 state encoding swaps 1↔2 vs C's, so this
            // probes DEL→state=2 first, then INS→state=1, mirroring C's
            // first-DEL-then-INS preference at the cost of in-Rust state
            // numbering inconsistency. The backtrack at lines 944-978 maps
            // state 1→INS and state 2→DEL, so DEL preference here yields the
            // same CIGAR-side preference as C.
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
            if j >= min_intron_len as usize
                && best_splice > KSW_NEG_INF / 2
                && (j - best_splice_j) >= min_intron_len as usize
            {
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
            if j <= tlen.saturating_sub(min_intron_len as usize) {
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

/// Rotated anti-diagonal DP version of ksw_exts2.
///
/// Faithfully follows minimap2/ksw2_exts2_sse.c — same u/v/x/y/x2 array layout,
/// same per-cell `d` byte encoding (state in bits 0-2, continuation flags in
/// bits 3-5), and same ksw_backtrack continuation logic. Scalar (one byte at a
/// time) instead of SSE. Replaces the standard (i, j) DP for splice mode so
/// CIGAR-shape parity with C SIMD ksw_exts2 is preserved.
pub fn ksw_exts2_rot(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    q2: i8,
    noncan: i8,
    _w: i32,
    zdrop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    junc: Option<&[u8]>,
    flag: KswFlags,
) -> KswResult {
    #[inline]
    fn i8_sem(v: i32) -> i32 {
        v as i8 as i32
    }

    let qlen = query.len() as i32;
    let tlen = target.len() as i32;
    let mut ez = KswResult::new();
    if m <= 1 || qlen <= 0 || tlen <= 0 || q2 <= q + e {
        return ez;
    }
    let with_cigar = !flag.contains(KswFlags::SCORE_ONLY);
    let approx_max = flag.contains(KswFlags::APPROX_MAX);
    let is_extz = flag.contains(KswFlags::EXTZ_ONLY);
    let is_right = flag.contains(KswFlags::RIGHT);
    let qe_i = (q as i32) + (e as i32);
    let q_i = q as i32;
    let q2_i = q2 as i32;
    let m_u = m as usize;
    let wildcard = m_u - 1;
    let sc_n = {
        let v = mat[m_u * m_u - 1] as i32;
        if v == 0 {
            -(e as i32)
        } else {
            v
        }
    };

    // long_thres mirrors C: long_thres = (q2-q)/e - 1; bumped by 1 if needed.
    let mut long_thres = (q2_i - q_i) / (e as i32) - 1;
    if q2_i > q_i + (e as i32) + long_thres * (e as i32) {
        long_thres += 1;
    }
    let long_diff = long_thres * (e as i32) - (q2_i - q_i);

    // Score sanity check matching C (line 88-92).
    let mut max_sc = mat[0] as i32;
    let mut min_sc = mat[1] as i32;
    for k in 1..(m_u * m_u) {
        let v = mat[k] as i32;
        if v > max_sc {
            max_sc = v;
        }
        if v < min_sc {
            min_sc = v;
        }
    }
    if -min_sc > 2 * (q_i + e as i32) {
        return ez;
    }
    let _ = max_sc;

    // Build donor/acceptor exactly like minimap2/ksw2_exts2_sse.c.
    let mut sp = [0i32; 4];
    if flag.contains(KswFlags::SPLICE_CMPLX) {
        let sp0 = [8, 15, 21, 30];
        for i in 0..4 {
            sp[i] = ((sp0[i] as f64) / 3.0 + 0.499) as i32;
        }
    } else {
        sp[0] = if flag.contains(KswFlags::SPLICE_FLANK) {
            (noncan as i32) / 2
        } else {
            0
        };
        sp[1] = noncan as i32;
        sp[2] = noncan as i32;
        sp[3] = noncan as i32;
    }
    let tlen_padded = ((tlen + 15) / 16) * 16;
    let n_anti = (qlen + tlen - 1) as usize;
    let max_bt = (qlen.min(tlen) as usize) * n_anti;

    let mut scratch = KSW_EXTS2_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    scratch.donor.resize(tlen_padded as usize, (-sp[3]) as i8);
    scratch
        .acceptor
        .resize(tlen_padded as usize, (-sp[3]) as i8);
    scratch.u.resize(tlen_padded as usize, 0);
    scratch.v.resize(tlen_padded as usize, 0);
    scratch.x.resize(tlen_padded as usize, 0);
    scratch.y.resize(tlen_padded as usize, 0);
    scratch.x2.resize(tlen_padded as usize, 0);
    scratch.s.resize(tlen_padded as usize, 0);
    scratch.h_arr.resize(tlen_padded as usize, KSW_NEG_INF);
    scratch.bt.clear();
    if with_cigar && scratch.bt.capacity() < max_bt {
        scratch.bt.reserve(max_bt - scratch.bt.capacity());
    }
    scratch.bt_row_starts.clear();
    scratch.bt_row_starts.reserve(n_anti);
    scratch.offs.clear();
    scratch.offs.reserve(n_anti);
    scratch.off_ends.clear();
    scratch.off_ends.reserve(n_anti);

    let donor = &mut scratch.donor;
    let acceptor = &mut scratch.acceptor;
    if !flag.contains(KswFlags::REV_CIGAR) {
        for t in 0..target.len().saturating_sub(4) {
            let mut z = 3i32;
            if flag.contains(KswFlags::SPLICE_FOR) {
                if target[t + 1] == 2 && target[t + 2] == 3 {
                    z = if target[t + 3] == 0 || target[t + 3] == 2 {
                        -1
                    } else {
                        0
                    };
                } else if target[t + 1] == 2 && target[t + 2] == 1 {
                    z = 1;
                } else if target[t + 1] == 0 && target[t + 2] == 3 {
                    z = 2;
                }
            } else if flag.contains(KswFlags::SPLICE_REV) {
                if target[t + 1] == 1 && target[t + 2] == 3 {
                    z = if target[t + 3] == 0 || target[t + 3] == 2 {
                        -1
                    } else {
                        0
                    };
                } else if target[t + 1] == 2 && target[t + 2] == 3 {
                    z = 2;
                }
            }
            donor[t] = if z < 0 { 0 } else { (-sp[z as usize]) as i8 };
        }
        for t in 2..target.len() {
            let mut z = 3i32;
            if flag.contains(KswFlags::SPLICE_FOR) {
                if target[t - 1] == 0 && target[t] == 2 {
                    z = if target[t - 2] == 1 || target[t - 2] == 3 {
                        -1
                    } else {
                        0
                    };
                } else if target[t - 1] == 0 && target[t] == 1 {
                    z = 2;
                }
            } else if flag.contains(KswFlags::SPLICE_REV) {
                if target[t - 1] == 0 && target[t] == 1 {
                    z = if target[t - 2] == 1 || target[t - 2] == 3 {
                        -1
                    } else {
                        0
                    };
                } else if target[t - 1] == 2 && target[t] == 1 {
                    z = 1;
                } else if target[t - 1] == 0 && target[t] == 3 {
                    z = 2;
                }
            }
            acceptor[t] = if z < 0 { 0 } else { (-sp[z as usize]) as i8 };
        }
    } else {
        for t in 0..target.len().saturating_sub(4) {
            let mut z = 3i32;
            if flag.contains(KswFlags::SPLICE_FOR) {
                if target[t + 1] == 2 && target[t + 2] == 0 {
                    z = if target[t + 3] == 1 || target[t + 3] == 3 {
                        -1
                    } else {
                        0
                    };
                } else if target[t + 1] == 1 && target[t + 2] == 0 {
                    z = 2;
                }
            } else if flag.contains(KswFlags::SPLICE_REV) {
                if target[t + 1] == 1 && target[t + 2] == 0 {
                    z = if target[t + 3] == 1 || target[t + 3] == 3 {
                        -1
                    } else {
                        0
                    };
                } else if target[t + 1] == 1 && target[t + 2] == 2 {
                    z = 1;
                } else if target[t + 1] == 3 && target[t + 2] == 0 {
                    z = 2;
                }
            }
            donor[t] = if z < 0 { 0 } else { (-sp[z as usize]) as i8 };
        }
        for t in 2..target.len() {
            let mut z = 3i32;
            if flag.contains(KswFlags::SPLICE_FOR) {
                if target[t - 1] == 3 && target[t] == 2 {
                    z = if target[t - 2] == 0 || target[t - 2] == 2 {
                        -1
                    } else {
                        0
                    };
                } else if target[t - 1] == 1 && target[t] == 2 {
                    z = 1;
                } else if target[t - 1] == 3 && target[t] == 0 {
                    z = 2;
                }
            } else if flag.contains(KswFlags::SPLICE_REV) {
                if target[t - 1] == 3 && target[t] == 1 {
                    z = if target[t - 2] == 0 || target[t - 2] == 2 {
                        -1
                    } else {
                        0
                    };
                } else if target[t - 1] == 3 && target[t] == 2 {
                    z = 2;
                }
            }
            acceptor[t] = if z < 0 { 0 } else { (-sp[z as usize]) as i8 };
        }
    }
    if let Some(junc) = junc {
        if flag.contains(KswFlags::SPLICE_SCORE) {
            let donor_val =
                if flag.contains(KswFlags::SPLICE_FOR) == !flag.contains(KswFlags::REV_CIGAR) {
                    0u8
                } else {
                    1u8
                };
            let limit = target
                .len()
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1));
            for t in 0..limit {
                donor[t] = (donor[t] as i32
                    + if junc[t + 1] == 0xff || (junc[t + 1] & 1) != donor_val {
                        -(junc_pen as i32)
                    } else {
                        ((junc[t + 1] >> 1) as i32) - 64
                    }) as i8;
                acceptor[t] = (acceptor[t] as i32
                    + if junc[t + 1] == 0xff || (junc[t + 1] & 1) == donor_val {
                        -(junc_pen as i32)
                    } else {
                        ((junc[t + 1] >> 1) as i32) - 64
                    }) as i8;
            }
        } else if !flag.contains(KswFlags::REV_CIGAR) {
            for t in 0..target
                .len()
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1))
            {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 8) != 0)
                {
                    donor[t] = (donor[t] as i32 + junc_bonus as i32) as i8;
                }
            }
            for t in 0..target.len().min(junc.len()) {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 4) != 0)
                {
                    acceptor[t] = (acceptor[t] as i32 + junc_bonus as i32) as i8;
                }
            }
        } else {
            for t in 0..target
                .len()
                .saturating_sub(1)
                .min(junc.len().saturating_sub(1))
            {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t + 1] & 2) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t + 1] & 4) != 0)
                {
                    donor[t] = (donor[t] as i32 + junc_bonus as i32) as i8;
                }
            }
            for t in 0..target.len().min(junc.len()) {
                if (flag.contains(KswFlags::SPLICE_FOR) && (junc[t] & 1) != 0)
                    || (flag.contains(KswFlags::SPLICE_REV) && (junc[t] & 8) != 0)
                {
                    acceptor[t] = (acceptor[t] as i32 + junc_bonus as i32) as i8;
                }
            }
        }
    }

    // Per-column DP state arrays use the same 16-byte padded width as C's
    // SIMD implementation. The padded tail participates in chunk-boundary
    // propagation, so clipping the scalar loop to the real target length is
    // not parity-correct.
    let neg_qe = i8_sem(-qe_i);
    let neg_q2 = i8_sem(-q2_i);
    let u = &mut scratch.u;
    let v = &mut scratch.v;
    let x = &mut scratch.x;
    let y = &mut scratch.y;
    let x2 = &mut scratch.x2;
    u.fill(neg_qe as i8);
    v.fill(neg_qe as i8);
    x.fill(neg_qe as i8);
    y.fill(neg_qe as i8);
    x2.fill(neg_q2 as i8);

    // Per-column score scratch. C uses aligned vectors; scalar uses reusable
    // scratch.
    let s = &mut scratch.s;
    s.fill(0);

    // Per-anti-diagonal H storage for exact max tracking.
    let h_arr = &mut scratch.h_arr;
    if !approx_max {
        h_arr.fill(KSW_NEG_INF);
    }
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    // CIGAR backtrack scratch: packed bytes in one contiguous buffer plus
    // per-row offsets, matching C's single allocation more closely.
    let bt = &mut scratch.bt;
    let bt_row_starts = &mut scratch.bt_row_starts;
    let offs = &mut scratch.offs;
    let off_ends = &mut scratch.off_ends;
    let query_ptr = query.as_ptr();
    let target_ptr = target.as_ptr();
    let mat_ptr = mat.as_ptr();
    let donor_ptr = donor.as_mut_ptr();
    let acceptor_ptr = acceptor.as_mut_ptr();
    let u_ptr = u.as_mut_ptr();
    let v_ptr = v.as_mut_ptr();
    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    let x2_ptr = x2.as_mut_ptr();
    let s_ptr = s.as_mut_ptr();
    let h_ptr = h_arr.as_mut_ptr();

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;

    for r in 0..(qlen + tlen - 1) {
        // Determine anti-diagonal column range [st0, en0]. C lines 219-228.
        let mut st = 0i32;
        let mut en = tlen - 1;
        if st < r - qlen + 1 {
            st = r - qlen + 1;
        }
        if en > r {
            en = r;
        }
        let st0 = st;
        let en0 = en;
        offs.push(st0);
        off_ends.push(en0);

        // Boundary state at column st0-1 (corresponds to SIMD x1/x21/v1).
        let (x1, x21, v1) = if st0 > 0 {
            if st0 - 1 >= last_st && st0 - 1 <= last_en {
                (
                    x[(st0 - 1) as usize],
                    x2[(st0 - 1) as usize],
                    v[(st0 - 1) as usize],
                )
            } else {
                (neg_qe as i8, neg_q2 as i8, neg_qe as i8)
            }
        } else {
            let v0 = if r == 0 {
                -qe_i
            } else if r < long_thres {
                -(e as i32)
            } else if r == long_thres {
                long_diff
            } else {
                0
            };
            (neg_qe as i8, neg_q2 as i8, i8_sem(v0) as i8)
        };
        // y[r] / u[r] boundary at en0 + 1 = r when en0 == r.
        if en0 >= r {
            y[r as usize] = neg_qe as i8;
            u[r as usize] = if r == 0 {
                neg_qe as i8
            } else if r < long_thres {
                i8_sem(-(e as i32)) as i8
            } else if r == long_thres {
                i8_sem(long_diff) as i8
            } else {
                0
            };
        }

        // Score lookup. Match C's non-GENERIC_SC SIMD path: if either base is
        // the wildcard symbol (m-1), use sc_N (mat[m*m-1], except force -e
        // when that matrix entry is zero). Only GENERIC_SC does the raw mat[]
        // lookup for all symbols.
        let chunk_st = (st0 / 16) * 16;
        let chunk_en_excl = ((en0 + 16) / 16) * 16; // exclusive upper bound
        let chunk_en = chunk_en_excl - 1;
        for t in chunk_st..=chunk_en {
            // qr[qlen - 1 - (r - t)] is equivalent to query[r - t]; read the
            // original query directly and avoid building a reversed copy per
            // alignment call.
            let q_pos = r - t;
            let qb = if q_pos >= 0 && q_pos < qlen {
                unsafe { *query_ptr.add(q_pos as usize) as usize }
            } else {
                0usize
            };
            let tb = if t >= 0 && t < tlen {
                unsafe { *target_ptr.add(t as usize) as usize }
            } else {
                0usize
            };
            unsafe {
                *s_ptr.add(t as usize) =
                    if !flag.contains(KswFlags::GENERIC_SC) && (qb == wildcard || tb == wildcard) {
                        sc_n
                    } else {
                        *mat_ptr.add(qb * m_u + tb) as i32
                    };
            }
        }

        // Per-row bt slot — one byte per t in [st0, en0].
        let row_len = (en0 - st0 + 1) as usize;
        let row_bt_start = bt.len();
        if with_cigar {
            bt_row_starts.push(row_bt_start);
            bt.resize(row_bt_start + row_len, 0);
        }

        // Iterate over CHUNK-ALIGNED bounds (multiples of 16), matching C
        // SIMD's lane-parallel processing. Cells outside [st0, en0] get
        // computed and stored with garbage values (since they're outside the
        // band), but C does this too — and subsequent rows may read these
        // garbage values as their boundary x1, propagating equivalent state.
        // Boundary at chunk_st - 1: C reads x[st-1] etc. unconditionally
        // when st > 0. Use prev row's value if chunk_st-1 was in prev row's
        // chunk range, else fall back to initial -qe boundary.
        let (mut prev_x, mut prev_v, mut prev_x2) = if chunk_st > 0 {
            if chunk_st - 1 >= last_st && chunk_st - 1 <= last_en {
                (
                    x[(chunk_st - 1) as usize],
                    v[(chunk_st - 1) as usize],
                    x2[(chunk_st - 1) as usize],
                )
            } else {
                (neg_qe as i8, neg_qe as i8, neg_q2 as i8)
            }
        } else {
            (x1, v1, x21)
        };
        // For chunk_st == 0, x1/v1/x21 are already correct (computed earlier
        // with the special r==0 / boundary logic).
        for t in chunk_st..=chunk_en {
            let tu = t as usize;
            let z_init = unsafe { *s_ptr.add(tu) };
            let xt1 = prev_x as i32;
            let vt1 = prev_v as i32;
            let x2t1 = prev_x2 as i32;
            let ut = unsafe { *u_ptr.add(tu) as i32 };
            // Snapshot the OLD t-th values for the next iteration before they
            // get overwritten by this iteration's stores.
            let next_prev_x = unsafe { *x_ptr.add(tu) };
            let next_prev_v = unsafe { *v_ptr.add(tu) };
            let next_prev_x2 = unsafe { *x2_ptr.add(tu) };
            let a = i8_sem(xt1 + vt1);
            let b = i8_sem(unsafe { *y_ptr.add(tu) as i32 } + ut);
            let a2 = i8_sem(x2t1 + vt1);
            // splice_*_scores() already stores values in C's DP column
            // coordinates: donor[t] scores target[t+1..t+2], acceptor[t]
            // scores target[t-1..t]. Read them directly by column index.
            let acceptor_t = unsafe { *acceptor_ptr.add(tu) as i32 };
            let a2a = i8_sem(a2 + acceptor_t);

            // State pick. left-alignment (also covers !is_right): probe in
            // order a (DEL→state 1), b (INS→state 2), a2a (splice→state 3).
            // Strict > so M wins ties over gaps; matches C ksw2_exts2_sse.c
            // line 307-311.
            let mut z = z_init;
            let mut state: u8 = 0;
            if !is_right {
                if a > z {
                    z = a;
                    state = 1;
                }
                if b > z {
                    z = b;
                    state = 2;
                }
                if a2a > z {
                    z = a2a;
                    state = 3;
                }
            } else {
                // RIGHT-alignment: ties prefer gap states. Matches C lines
                // 350-355: blendv with `z > X ? d : new_state`.
                if !(z > a) {
                    z = a;
                    state = 1;
                }
                if !(z > b) {
                    z = b;
                    state = 2;
                }
                if !(z > a2a) {
                    z = a2a;
                    state = 3;
                }
            }
            // Save the un-clamped match score before the post-block2 deltas
            // update u/v.
            let z_kept = z;
            // dp_code_block2: u[t]=z-vt1, v[t]=z-ut, then a/b/a2 deltas.
            unsafe {
                *u_ptr.add(tu) = i8_sem(z - vt1) as i8;
                *v_ptr.add(tu) = i8_sem(z - ut) as i8;
            }
            let a_post = i8_sem(a - i8_sem(z - q_i));
            let b_post = i8_sem(b - i8_sem(z - q_i));
            let a2_post = i8_sem(a2 - i8_sem(z - q2_i));
            // x/y/x2 update with continuation-flag tracking.
            let a_pos = a_post > 0;
            let b_pos = b_post > 0;
            // x[t] = max(a_post, 0) - qe (left); for right-alignment use
            // andnot semantics — but the resulting stored value is the same
            // since `max(a_post, 0)` doesn't depend on alignment direction
            // for the stored x[t].
            unsafe {
                *x_ptr.add(tu) = i8_sem((if a_pos { a_post } else { 0 }) - qe_i) as i8;
                *y_ptr.add(tu) = i8_sem((if b_pos { b_post } else { 0 }) - qe_i) as i8;
            }
            // x2[t] = max(a2_post, donor[t]) - q2  (left-alignment direction
            // matters: C uses max(a2, donor) for left, max(donor, a2) for
            // right with andnot pattern that flips bit 5's meaning).
            let donor_t = unsafe { *donor_ptr.add(tu) as i32 };
            // C ksw2_exts2_sse.c uses `_mm_cmpgt_epi8(a2, donor)` on the
            // left-aligned path, so donor wins exact ties. Using `>=` here
            // skews backtrack continuation bits and can flip a donor-tie
            // into a different insertion/deletion decomposition.
            let a2_wins_donor = if !is_right {
                a2_post > donor_t
            } else {
                a2_post >= donor_t
            };
            let chosen_x2 = if a2_wins_donor { a2_post } else { donor_t };
            unsafe {
                *x2_ptr.add(tu) = i8_sem(chosen_x2 - q2_i) as i8;
            }

            // Update ez.max for each cell (only meaningful with !approx_max,
            // where exact H[] is computed below). Here we just use z_kept
            // through the H update logic.
            let _ = z_kept;

            #[cfg(feature = "tracehash")]
            if with_cigar
                && ((r < 8)
                    || (qlen == 217 && tlen == 587 && r >= 120 && r <= 140)
                    || (qlen == 50 && tlen == 98 && r >= 12 && r <= 16)
                    || (qlen == 50 && tlen == 98 && r >= 84 && r <= 92))
                && t >= st0
                && t <= en0
            {
                let mut th = tracehash::th_call!("exts2_cell_dbg");
                th.input_i64(if is_right { 1 } else { 0 });
                th.input_i64(r as i64);
                th.input_i64(t as i64);
                th.input_i64(z_init as i64);
                th.input_i64(xt1 as i64);
                th.input_i64(vt1 as i64);
                th.input_i64(ut as i64);
                th.input_i64(x2t1 as i64);
                th.input_i64(acceptor_t as i64);
                th.input_i64(donor_t as i64);
                th.output_i64(a as i64);
                th.output_i64(b as i64);
                th.output_i64(a2 as i64);
                th.output_i64(a2a as i64);
                th.output_i64(z_kept as i64);
                th.output_i64(state as i64);
                th.output_i64(a_post as i64);
                th.output_i64(b_post as i64);
                th.output_i64(a2_post as i64);
                th.output_i64(if a_pos { 1 } else { 0 });
                th.output_i64(if b_pos { 1 } else { 0 });
                th.output_i64(if a2_wins_donor { 1 } else { 0 });
                th.finish();
            }

            if with_cigar && t >= st0 && t <= en0 {
                let mut d = state;
                // Continuation flags. Left-alignment encoding (C lines
                // 327/330/340): bit set iff respective gap path was active
                // (>0 for x/y, > donor[t] for x2).
                if !is_right {
                    if a_pos {
                        d |= 0x08;
                    }
                    if b_pos {
                        d |= 0x10;
                    }
                    if a2_wins_donor {
                        d |= 0x20;
                    }
                } else {
                    if a_post >= 0 {
                        d |= 0x08;
                    }
                    if b_post >= 0 {
                        d |= 0x10;
                    }
                    if a2_post >= donor_t {
                        d |= 0x20;
                    }
                }
                bt[row_bt_start + (t - st0) as usize] = d;
            }
            // Advance the prev-anti-diagonal snapshot.
            prev_x = next_prev_x;
            prev_v = next_prev_v;
            prev_x2 = next_prev_x2;
        }

        #[cfg(feature = "tracehash")]
        if with_cigar {
            let mut th = tracehash::th_call!("exts2_pr_row");
            th.input_i64(r as i64);
            th.input_i64(st0 as i64);
            th.input_i64(en0 as i64);
            th.input_i64(chunk_st as i64);
            th.input_i64(chunk_en as i64);
            for &d in &bt[row_bt_start..row_bt_start + row_len] {
                th.output_i64(d as i64);
            }
            th.finish();
        }

        // Exact H[] / approximate max tracking — controls ez.max/mqe/mte/score.
        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                // Special-case the last element: H[en0] = en0>0? H[en0-1] + u[en0] : H[en0] + v[en0]
                if en0 > 0 {
                    let new_h =
                        unsafe { *h_ptr.add((en0 - 1) as usize) + *u_ptr.add(en0 as usize) as i32 };
                    unsafe {
                        *h_ptr.add(en0 as usize) = new_h;
                    }
                } else {
                    unsafe {
                        *h_ptr.add(en0 as usize) += *v_ptr.add(en0 as usize) as i32;
                    }
                }
                max_h = unsafe { *h_ptr.add(en0 as usize) };
                max_t = en0;
                // Mirror C SIMD scan order from ksw2_exts2_sse.c:401-425 so
                // tie-breaking on max_t matches: 4-lane SIMD walks t in
                // column-major (lane k visits st0+k, st0+k+4, …) then a
                // scalar tail covers en1..en0. Strict-GT updates throughout
                // mean ties resolve to the earliest-visited t in this order.
                let en1 = st0 + (en0 - st0) / 4 * 4;
                // First update H[t] for all t in [st0, en0) (independent of order).
                for t in st0..en0 {
                    unsafe {
                        *h_ptr.add(t as usize) += *v_ptr.add(t as usize) as i32;
                    }
                }
                // Then scan in C SIMD order.
                for lane in 0..4 {
                    let mut t = st0 + lane;
                    while t < en1 {
                        let ht = unsafe { *h_ptr.add(t as usize) };
                        if ht > max_h {
                            max_h = ht;
                            max_t = t;
                        }
                        t += 4;
                    }
                }
                for t in en1..en0 {
                    let ht = unsafe { *h_ptr.add(t as usize) };
                    if ht > max_h {
                        max_h = ht;
                        max_t = t;
                    }
                }
            } else {
                unsafe {
                    *h_ptr.add(0) = *v_ptr.add(0) as i32 - qe_i;
                }
                max_h = unsafe { *h_ptr.add(0) };
                max_t = 0;
            }
            // mte / mqe / score updates per C lines 423-430.
            let h_en0 = unsafe { *h_ptr.add(en0 as usize) };
            if en0 == tlen - 1 && h_en0 > ez.mte {
                ez.mte = h_en0;
                ez.mte_q = r - en0;
            }
            let h_st0 = unsafe { *h_ptr.add(st0 as usize) };
            if r - st0 == qlen - 1 && h_st0 > ez.mqe {
                ez.mqe = h_st0;
                ez.mqe_t = st0;
            }
            // Track ez.max from the per-anti-diagonal exact max.
            if max_h > ez.max as i32 {
                ez.max = max_h;
                ez.max_t = max_t;
                ez.max_q = r - max_t;
            }
            // Per-anti-diagonal tracehash probe — captures (r, st0, en0,
            // max_h, max_t, h[st0], h[en0]) so a Rust/C deep diff can
            // pinpoint the first divergent anti-diagonal.
            #[cfg(feature = "tracehash")]
            {
                let mut th = tracehash::th_call!("exts2_anti_diag");
                th.input_i64(r as i64);
                th.input_i64(st0 as i64);
                th.input_i64(en0 as i64);
                th.output_i64(max_h as i64);
                th.output_i64(max_t as i64);
                th.output_i64(h_st0 as i64);
                th.output_i64(h_en0 as i64);
                th.finish();
            }
            // Z-drop check. C ksw_exts2_sse.c:428 passes e=0 for the splice
            // DP (long-gap path uses e2=0 effectively).
            if zdrop >= 0 && ksw_apply_zdrop_rot(&mut ez, max_h, r, max_t, zdrop, 0) {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = unsafe { *h_ptr.add((tlen - 1) as usize) };
            }
        } else {
            // Approximate-max mode: track diagonal H0 along last_h0_t.
            // C reads u8/v8 (the int8 byte view of u/v); the i32 we store can
            // exceed int8 range, so cast through i8 to match C's sign-extended
            // delta. (The exact-max path uses the same byte values implicitly
            // through the H array update; here we reach into u/v directly.)
            if r > 0 {
                if last_h0_t >= st0
                    && last_h0_t <= en0
                    && last_h0_t + 1 >= st0
                    && last_h0_t + 1 <= en0
                {
                    let d0 = v[last_h0_t as usize] as i8 as i32;
                    let d1 = u[(last_h0_t + 1) as usize] as i8 as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += v[last_h0_t as usize] as i8 as i32;
                } else {
                    last_h0_t += 1;
                    h0 += u[last_h0_t as usize] as i8 as i32;
                }
            } else {
                h0 = v[0] as i8 as i32 - qe_i;
                last_h0_t = 0;
            }
            #[cfg(feature = "tracehash")]
            {
                let mut th = tracehash::th_call!("exts2_approx_anti_diag");
                th.input_i64(r as i64);
                th.input_i64(st0 as i64);
                th.input_i64(en0 as i64);
                th.output_i64(h0 as i64);
                th.output_i64(last_h0_t as i64);
                th.output_i64(ez.max as i64);
                th.output_i64(ez.mqe as i64);
                th.output_i64(ez.mte as i64);
                th.finish();
            }
            // Match C ksw2_exts2_sse.c:461 — APPROX_MAX only updates ez.max
            // through ksw_apply_zdrop, and ksw_apply_zdrop is only called
            // when APPROX_DROP is set. So when APPROX_MAX is set without
            // APPROX_DROP, ez.max stays at its initial value (0). Earlier
            // Rust unconditionally tracked ez.max here, which over-counted
            // dp_score for splice long-intron gaps.
            if flag.contains(KswFlags::APPROX_DROP)
                && ksw_apply_zdrop_rot(&mut ez, h0, r, last_h0_t, zdrop, 0)
            {
                break;
            }
            if r == qlen + tlen - 2 && en0 == tlen - 1 {
                ez.score = h0;
            }
        }

        // Track CHUNK-ALIGNED bounds for the next row's boundary check, so
        // garbage-but-stored chunk-extension positions are visible to the
        // next iteration just as they are in C SIMD.
        last_st = chunk_st;
        last_en = chunk_en;
        let _ = (st, en);
    }

    #[cfg(feature = "tracehash")]
    {
        let mut th = tracehash::th_call!("exts2_call_end");
        th.input_i64(qlen as i64);
        th.input_i64(tlen as i64);
        th.input_i64(flag.bits() as i64);
        th.output_i64(ez.score as i64);
        th.output_i64(ez.max as i64);
        th.output_i64(ez.max_t as i64);
        th.output_i64(ez.max_q as i64);
        th.output_i64(ez.mqe as i64);
        th.output_i64(ez.mte as i64);
        th.output_i64(ez.mqe_t as i64);
        th.output_i64(ez.mte_q as i64);
        th.output_i64(if ez.zdropped { 1 } else { 0 });
        th.output_i64(if ez.reach_end { 1 } else { 0 });
        th.finish();
    }

    // Backtrack with C's continuation-flag semantics. C ksw_exts2_sse.c
    // lines 453-462: when zdropped, branch 3 fires (use max_t, max_q). Only
    // skip backtrack entirely when no max was recorded.
    if with_cigar {
        let (i0, j0) = if !ez.zdropped && !is_extz {
            (qlen - 1, tlen - 1)
        } else if !ez.zdropped && is_extz && ez.mqe + end_bonus > ez.max {
            ez.reach_end = true;
            (qlen - 1, ez.mqe_t)
        } else if ez.max_q >= 0 && ez.max_t >= 0 {
            (ez.max_q, ez.max_t)
        } else {
            (-1, -1)
        };
        if i0 >= 0 && j0 >= 0 {
            let (mut i, mut j) = (i0, j0);
            let mut state: u8 = 0;
            let mut cigar = Vec::new();
            while i >= 0 && j >= 0 {
                let r = i + j;
                if (r as usize) >= bt_row_starts.len() {
                    break;
                }
                let st = offs[r as usize];
                let en = off_ends[r as usize];
                let row_bt_start = bt_row_starts[r as usize];
                let mut force_state: i32 = -1;
                if j < st {
                    force_state = 2;
                }
                if j > en {
                    force_state = 1;
                }
                let tmp: u8 = if force_state < 0 {
                    bt[row_bt_start + (j - st) as usize]
                } else {
                    0
                };
                if state == 0 {
                    state = tmp & 7;
                } else if (tmp >> (state + 2)) & 1 == 0 {
                    state = 0;
                }
                if state == 0 {
                    state = tmp & 7;
                }
                if force_state >= 0 {
                    state = force_state as u8;
                }
                // C ksw_backtrack mapping (ksw2.h:151-154) with min_intron_len
                // = long_thres > 0:
                //   state 0 → MATCH, --i, --j
                //   state 1 → DEL, --j  (rotated: i is query, j is target;
                //                        DEL consumes target, decrement j)
                //   state 3 → N_SKIP if min_intron > 0, else DEL, --j
                //   state 2 → INS, --i (consume query)
                match state {
                    0 => {
                        push_cigar_fn(&mut cigar, 0, 1);
                        i -= 1;
                        j -= 1;
                    }
                    1 => {
                        push_cigar_fn(&mut cigar, 2, 1);
                        j -= 1;
                    }
                    3 if long_thres > 0 => {
                        push_cigar_fn(&mut cigar, 3, 1);
                        j -= 1;
                    }
                    3 => {
                        push_cigar_fn(&mut cigar, 2, 1);
                        j -= 1;
                    }
                    _ => {
                        push_cigar_fn(&mut cigar, 1, 1);
                        i -= 1;
                    }
                }
            }
            // Trailing leftovers (C ksw2.h:156-157). C checks `i >=
            // min_intron_len` where i is residual target; we name j our
            // target. Pushed length is j + 1 columns.
            if j >= 0 {
                let op = if long_thres > 0 && j >= long_thres {
                    3
                } else {
                    2
                };
                push_cigar_fn(&mut cigar, op, j + 1);
            }
            if i >= 0 {
                push_cigar_fn(&mut cigar, 1, i + 1);
            }
            if !flag.contains(KswFlags::REV_CIGAR) {
                cigar.reverse();
            }
            ez.cigar = cigar;
        }
    }

    KSW_EXTS2_SCRATCH.with(|c| *c.borrow_mut() = scratch);
    ez
}

fn ksw_apply_zdrop_rot(ez: &mut KswResult, h: i32, r: i32, t: i32, zdrop: i32, e: i8) -> bool {
    if h > ez.max {
        ez.max = h;
        ez.max_t = t;
        ez.max_q = r - t;
    } else if t >= ez.max_t && r - t >= ez.max_q {
        let tl = t - ez.max_t;
        let ql = (r - t) - ez.max_q;
        let l = if tl > ql { tl - ql } else { ql - tl };
        if zdrop >= 0 && (ez.max - h) > zdrop + l * (e as i32) {
            ez.zdropped = true;
            return true;
        }
    }
    false
}

fn splice_default_penalty(noncan: i32, flag: KswFlags) -> i32 {
    // Mirrors C ksw2_exts2_sse.c lines 122-130: in CMPLX mode the default
    // splice penalty is sp[3] (= round(30/3) = 10); in non-CMPLX it's
    // noncan. C uses this value to memset the donor/acceptor arrays so
    // unscored positions still get a reasonable per-cell penalty.
    if flag.contains(KswFlags::SPLICE_CMPLX) {
        let sp0 = [8, 15, 21, 30];
        ((sp0[3] as f64) / 3.0 + 0.499) as i32
    } else {
        noncan
    }
}

fn splice_donor_scores(target: &[u8], noncan: i32, flag: KswFlags) -> Vec<i32> {
    let default_pen = splice_default_penalty(noncan, flag);
    let mut scores = vec![-default_pen; target.len() + 1];
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
    let default_pen = splice_default_penalty(noncan, flag);
    let mut scores = vec![-default_pen; target.len() + 1];
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
    fn test_ksw_exts2_rot_vs_scalar() {
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let target = [
            0u8, 1, 2, 3, // ACGT
            2, 3, 0, 0, 0, 0, 0, 2, // GT.....AG
            0, 1, 2, 3, // ACGT
        ];
        let flag = KswFlags::SPLICE_FOR | KswFlags::SPLICE_FLANK;
        let scalar = ksw_exts2(&query, &target, 5, &mat, 2, 1, 4, 9, 50, 200, -1, flag);
        let rot = ksw_exts2_rot(
            &query, &target, 5, &mat, 2, 1, 4, 9, 50, 200, -1, 0, 0, None, flag,
        );
        eprintln!(
            "scalar: score={} max={} cigar={:?}",
            scalar.score, scalar.max, scalar.cigar
        );
        eprintln!(
            "rot:    score={} max={} cigar={:?}",
            rot.score, rot.max, rot.cigar
        );
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

    #[test]
    fn test_exts2_for_vs_rev_asymmetry() {
        // For a forward GT-AG intron, SPLICE_FOR should score higher than
        // SPLICE_REV (which expects CT-AC). If they tie, the splice signal
        // scoring is broken.
        let mat = make_mat();
        let query = [0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT (8 bp)
        let target = [
            0u8, 1, 2, 3, // ACGT
            2, 3, 0, 0, 0, 0, 0, 2, // GT.....AG (canonical FOR intron)
            0, 1, 2, 3, // ACGT
        ];
        let ez_for = ksw_exts2(
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
            KswFlags::SPLICE_FOR | KswFlags::SPLICE_CMPLX,
        );
        let ez_rev = ksw_exts2(
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
            KswFlags::SPLICE_REV | KswFlags::SPLICE_CMPLX,
        );
        eprintln!(
            "FOR score={} max={} cigar={:?}",
            ez_for.score, ez_for.max, ez_for.cigar
        );
        eprintln!(
            "REV score={} max={} cigar={:?}",
            ez_rev.score, ez_rev.max, ez_rev.cigar
        );
        assert!(
            ez_for.score > ez_rev.score,
            "FOR should outscore REV on canonical GT-AG intron, got FOR={} REV={}",
            ez_for.score,
            ez_rev.score
        );
    }
}
