//! Splice junction extension handling.
//!
//! Matches jump.c: mm_jump_split(), mm_jump_split_left(), mm_jump_split_right().
//!
use crate::index::MmIdx;
use crate::junc::JuncIntv;
use crate::options::MapOpt;
use crate::types::AlignReg;

const MIN_EXON_LEN: i32 = 20;

#[derive(Clone, Copy)]
struct JumpCandidate {
    st: i32,
    en: i32,
    off: i32,
    off2: i32,
    mm: i32,
}

/// Extend alignment across splice junctions.
pub fn jump_split(
    mi: &MmIdx,
    opt: &MapOpt,
    qlen: i32,
    qseq: &[u8],
    r: &mut AlignReg,
    _ts_strand: i32,
) {
    let aligned_qseq;
    let qseq_aln = if r.rev {
        aligned_qseq = qseq
            .iter()
            .rev()
            .map(|&c| if c < 4 { 3 - c } else { 4 })
            .collect::<Vec<_>>();
        aligned_qseq.as_slice()
    } else {
        qseq
    };
    jump_split_left(mi, opt, qlen, qseq_aln, r);
    jump_split_right(mi, opt, qlen, qseq_aln, r);
}

fn aligned_qs(qlen: i32, r: &AlignReg) -> i32 {
    if r.rev {
        qlen - r.qe
    } else {
        r.qs
    }
}

fn aligned_qe(qlen: i32, r: &AlignReg) -> i32 {
    if r.rev {
        qlen - r.qs
    } else {
        r.qe
    }
}

fn jump_split_left(mi: &MmIdx, opt: &MapOpt, qlen: i32, qseq: &[u8], r: &mut AlignReg) {
    let Some(db) = mi.junc_db.as_ref() else {
        return;
    };
    let Some(juncs) = db.juncs.get(r.rid as usize) else {
        return;
    };
    let clip = aligned_qs(qlen, r).max(0);
    let ext = 1 + (opt.b + opt.a - 1) / opt.a + 1;
    let extt = clip.min(ext);
    if !jump_check_left(mi, qlen, r, ext + MIN_EXON_LEN) {
        return;
    }
    let Some(extra_ref) = r.extra.as_ref() else {
        return;
    };
    if extra_ref.cigar.0.is_empty() || (extra_ref.cigar.0[0] & 0xf) != 0 {
        return;
    }
    let trans_strand = extra_ref.trans_strand;
    let mut candidates = Vec::new();
    for j in juncs {
        if !strand_compatible(trans_strand, j) {
            continue;
        }
        let off = j.en;
        let off2 = j.st;
        if off < r.rs - extt || off > r.rs + ext {
            continue;
        }
        if off - off2 < 6 || off2 < clip + ext {
            continue;
        }
        let tl1 = clip + (off - r.rs);
        if tl1 < 0 || tl1 > clip + ext {
            continue;
        }
        let Some(mm) = left_candidate_mismatches(mi, r, qseq, clip, ext, tl1, off, off2) else {
            continue;
        };
        candidates.push(JumpCandidate {
            st: j.st,
            en: j.en,
            off,
            off2,
            mm,
        });
    }
    if candidates.is_empty() {
        return;
    }
    let c = candidates[candidates.len() - 1];
    let l = c.off - r.rs;
    let Some(extra) = r.extra.as_mut() else {
        return;
    };
    if candidates.len() == 1 && clip + l >= opt.jump_min_match {
        let first_len = (extra.cigar.0[0] >> 4) as i32;
        let rem = first_len - l;
        if rem <= 0 || clip + l <= 0 {
            return;
        }
        let mut cigar = Vec::with_capacity(extra.cigar.0.len() + 2);
        cigar.push(((clip + l) as u32) << 4);
        cigar.push(((c.off - c.off2) as u32) << 4 | 3);
        cigar.push((rem as u32) << 4);
        cigar.extend_from_slice(&extra.cigar.0[1..]);
        extra.cigar.0 = cigar;
        r.rs = c.off2 - (clip + l);
        if r.rev {
            r.qe = qlen;
        } else {
            r.qs = 0;
        }
        r.mlen += clip - c.mm;
        r.blen += clip;
        if extra.trans_strand == 0 {
            extra.trans_strand = c.stand_in_strand(juncs);
        }
        let score_delta = (clip - c.mm) * opt.a - c.mm * opt.b;
        extra.dp_max += score_delta;
        extra.dp_max0 += score_delta;
        if !r.is_spliced {
            r.is_spliced = true;
            extra.dp_max += (opt.a + opt.b) + ((opt.a + opt.b) >> 1);
        }
    } else if candidates.len() > 1 && c.off > r.rs {
        let first_len = (extra.cigar.0[0] >> 4) as i32;
        if first_len > l {
            extra.cigar.0[0] = ((first_len - l) as u32) << 4;
            r.rs += l;
            if r.rev {
                r.qe -= l;
            } else {
                r.qs += l;
            }
        }
    }
}

fn jump_split_right(mi: &MmIdx, opt: &MapOpt, qlen: i32, qseq: &[u8], r: &mut AlignReg) {
    let Some(db) = mi.junc_db.as_ref() else {
        return;
    };
    let Some(juncs) = db.juncs.get(r.rid as usize) else {
        return;
    };
    let aqe = aligned_qe(qlen, r);
    let clip = (qlen - aqe).max(0);
    let ext = 1 + (opt.b + opt.a - 1) / opt.a + 1;
    let extt = clip.min(ext);
    if !jump_check_right(mi, qlen, r, ext + MIN_EXON_LEN) {
        return;
    }
    let Some(extra_ref) = r.extra.as_ref() else {
        return;
    };
    let Some(&last) = extra_ref.cigar.0.last() else {
        return;
    };
    if (last & 0xf) != 0 {
        return;
    }
    let trans_strand = extra_ref.trans_strand;
    let mut candidates = Vec::new();
    for j in juncs {
        if !strand_compatible(trans_strand, j) {
            continue;
        }
        let off = j.st;
        let off2 = j.en;
        if off < r.re - ext || off > r.re + extt {
            continue;
        }
        if off2 - off < 6 || off2 + clip + ext > mi.seqs[r.rid as usize].len as i32 {
            continue;
        }
        let tl1 = clip + (r.re - off);
        if tl1 < 0 || tl1 > clip + ext {
            continue;
        }
        let Some(mm) = right_candidate_mismatches(mi, r, qseq, aqe, clip, ext, tl1, off, off2)
        else {
            continue;
        };
        candidates.push(JumpCandidate {
            st: j.st,
            en: j.en,
            off,
            off2,
            mm,
        });
    }
    if candidates.is_empty() {
        return;
    }
    let c = candidates[0];
    let l = r.re - c.off;
    let Some(extra) = r.extra.as_mut() else {
        return;
    };
    if candidates.len() == 1 && clip + l >= opt.jump_min_match {
        let last_idx = extra.cigar.0.len() - 1;
        let last_len = (extra.cigar.0[last_idx] >> 4) as i32;
        let rem = last_len - l;
        if rem <= 0 || clip + l <= 0 {
            return;
        }
        extra.cigar.0[last_idx] = (rem as u32) << 4;
        extra.cigar.0.push(((c.off2 - c.off) as u32) << 4 | 3);
        extra.cigar.0.push(((clip + l) as u32) << 4);
        r.re = c.off2 + (clip + l);
        if r.rev {
            r.qs = 0;
        } else {
            r.qe = qlen;
        }
        r.mlen += clip - c.mm;
        r.blen += clip;
        if extra.trans_strand == 0 {
            extra.trans_strand = c.stand_in_strand(juncs);
        }
        let score_delta = (clip - c.mm) * opt.a - c.mm * opt.b;
        extra.dp_max += score_delta;
        extra.dp_max0 += score_delta;
        if !r.is_spliced {
            r.is_spliced = true;
            extra.dp_max += (opt.a + opt.b) + ((opt.a + opt.b) >> 1);
        }
    } else if candidates.len() > 1 && r.re > c.off {
        let last_idx = extra.cigar.0.len() - 1;
        let last_len = (extra.cigar.0[last_idx] >> 4) as i32;
        if last_len > l {
            extra.cigar.0[last_idx] = ((last_len - l) as u32) << 4;
            r.re -= l;
            if r.rev {
                r.qs += l;
            } else {
                r.qe -= l;
            }
        }
    }
}

impl JumpCandidate {
    fn stand_in_strand(self, juncs: &[JuncIntv]) -> u8 {
        juncs
            .iter()
            .find(|j| j.st == self.st && j.en == self.en)
            .map(|j| j.strand as u8)
            .unwrap_or(0)
    }
}

fn jump_check_left(mi: &MmIdx, qlen: i32, r: &AlignReg, ext: i32) -> bool {
    let Some(extra) = r.extra.as_ref() else {
        return false;
    };
    let Some(&cigar) = extra.cigar.0.first() else {
        return false;
    };
    let clip = if !r.rev { r.qs } else { qlen - r.qe };
    let clen = if (cigar & 0xf) == 0 {
        (cigar >> 4) as i32
    } else {
        0
    };
    clen > ext && clip < r.rs && (r.rid as usize) < mi.seqs.len()
}

fn jump_check_right(mi: &MmIdx, qlen: i32, r: &AlignReg, ext: i32) -> bool {
    let Some(extra) = r.extra.as_ref() else {
        return false;
    };
    let Some(&cigar) = extra.cigar.0.last() else {
        return false;
    };
    let clip = if !r.rev { qlen - r.qe } else { r.qs };
    let clen = if (cigar & 0xf) == 0 {
        (cigar >> 4) as i32
    } else {
        0
    };
    let Some(seq) = mi.seqs.get(r.rid as usize) else {
        return false;
    };
    clen > ext && clip < seq.len as i32 - r.re
}

fn strand_compatible(trans_strand: u8, j: &JuncIntv) -> bool {
    !(trans_strand == 1 && j.strand == 2 || trans_strand == 2 && j.strand == 1)
}

fn left_candidate_mismatches(
    mi: &MmIdx,
    r: &AlignReg,
    qseq: &[u8],
    clip: i32,
    ext: i32,
    tl1: i32,
    off: i32,
    off2: i32,
) -> Option<i32> {
    let total = clip + ext;
    if total <= 0 || total as usize > qseq.len() {
        return None;
    }
    let mut tseq = vec![4u8; total as usize];
    let left_st = off2 - tl1;
    if left_st < 0 || off > r.rs + ext || off < off2 {
        return None;
    }
    mi.getseq(r.rid as u32, left_st as u32, off2 as u32, &mut tseq[..tl1 as usize]);
    mi.getseq(
        r.rid as u32,
        off as u32,
        (r.rs + ext) as u32,
        &mut tseq[tl1 as usize..],
    );
    let mm1 = count_mismatch(&qseq[..tl1 as usize], &tseq[..tl1 as usize]);
    let mm2 = count_mismatch(&qseq[tl1 as usize..total as usize], &tseq[tl1 as usize..]);
    if mm1 == 0 && mm2 <= 1 {
        Some(mm1 + mm2)
    } else {
        None
    }
}

fn right_candidate_mismatches(
    mi: &MmIdx,
    r: &AlignReg,
    qseq: &[u8],
    aqe: i32,
    clip: i32,
    ext: i32,
    tl1: i32,
    off: i32,
    off2: i32,
) -> Option<i32> {
    let total = clip + ext;
    let q_start = aqe - ext;
    if total <= 0 || q_start < 0 || (q_start + total) as usize > qseq.len() {
        return None;
    }
    let mut tseq = vec![4u8; total as usize];
    let first_len = total - tl1;
    if first_len < 0 || r.re - ext < 0 {
        return None;
    }
    mi.getseq(
        r.rid as u32,
        (r.re - ext) as u32,
        off as u32,
        &mut tseq[..first_len as usize],
    );
    mi.getseq(
        r.rid as u32,
        off2 as u32,
        (off2 + tl1) as u32,
        &mut tseq[first_len as usize..],
    );
    let q = &qseq[q_start as usize..(q_start + total) as usize];
    let mm2 = count_mismatch(&q[..first_len as usize], &tseq[..first_len as usize]);
    let mm1 = count_mismatch(&q[first_len as usize..], &tseq[first_len as usize..]);
    if mm1 == 0 && mm2 <= 1 {
        Some(mm1 + mm2)
    } else {
        None
    }
}

fn count_mismatch(a: &[u8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b)
        .filter(|&(&x, &y)| x != y || x > 3 || y > 3)
        .count() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::junc::{JuncDb, JuncIntv};
    use crate::types::{AlignExtra, Cigar};

    #[test]
    fn test_jump_split_left_extends_annotated_junction() {
        let mut mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        mi.junc_db = Some(JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 30,
                en: 38,
                strand: 1,
            }]],
        });

        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 38;
        r.re = 68;
        r.qs = 4;
        r.qe = 34;
        r.mlen = 30;
        r.blen = 30;
        let mut extra = AlignExtra::default();
        extra.cigar = Cigar(vec![30u32 << 4]);
        r.extra = Some(Box::new(extra));

        let opt = MapOpt::default();
        let mut qseq = vec![0u8; 4];
        qseq.extend(std::iter::repeat(3u8).take(30));
        jump_split(&mi, &opt, 34, &qseq, &mut r, 0);

        assert_eq!(r.qs, 0);
        assert_eq!(r.rs, 26);
        assert!(r.is_spliced);
        assert_eq!(
            r.extra.as_ref().unwrap().cigar.0,
            vec![4u32 << 4, (8u32 << 4) | 3, 30u32 << 4]
        );
    }

    #[test]
    fn test_jump_split_left_extends_reverse_alignment() {
        let mut mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        mi.junc_db = Some(JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 30,
                en: 38,
                strand: 1,
            }]],
        });

        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 38;
        r.re = 68;
        r.qs = 0;
        r.qe = 30;
        r.rev = true;
        r.mlen = 30;
        r.blen = 30;
        let mut extra = AlignExtra::default();
        extra.cigar = Cigar(vec![30u32 << 4]);
        r.extra = Some(Box::new(extra));

        let opt = MapOpt::default();
        let mut qseq = vec![0u8; 30];
        qseq.extend(std::iter::repeat(3u8).take(4));
        jump_split(&mi, &opt, 34, &qseq, &mut r, 0);

        assert_eq!(r.qe, 34);
        assert_eq!(r.qs, 0);
        assert_eq!(r.rs, 26);
        assert!(r.is_spliced);
        assert_eq!(
            r.extra.as_ref().unwrap().cigar.0,
            vec![4u32 << 4, (8u32 << 4) | 3, 30u32 << 4]
        );
    }

    #[test]
    fn test_jump_split_right_extends_annotated_junction() {
        let mut mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        mi.junc_db = Some(JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 30,
                en: 38,
                strand: 1,
            }]],
        });

        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 0;
        r.re = 30;
        r.qs = 0;
        r.qe = 30;
        r.mlen = 30;
        r.blen = 30;
        let mut extra = AlignExtra::default();
        extra.cigar = Cigar(vec![30u32 << 4]);
        r.extra = Some(Box::new(extra));

        let opt = MapOpt::default();
        let mut qseq = vec![0u8; 30];
        qseq.extend(std::iter::repeat(3u8).take(4));
        jump_split(&mi, &opt, 34, &qseq, &mut r, 0);

        assert_eq!(r.qe, 34);
        assert_eq!(r.re, 42);
        assert!(r.is_spliced);
        assert_eq!(
            r.extra.as_ref().unwrap().cigar.0,
            vec![30u32 << 4, (8u32 << 4) | 3, 4u32 << 4]
        );
    }

    #[test]
    fn test_jump_split_left_trims_ambiguous_candidates() {
        let mut mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        mi.junc_db = Some(JuncDb {
            juncs: vec![vec![
                JuncIntv {
                    st: 30,
                    en: 38,
                    strand: 1,
                },
                JuncIntv {
                    st: 29,
                    en: 39,
                    strand: 1,
                },
            ]],
        });

        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 38;
        r.re = 68;
        r.qs = 5;
        r.qe = 35;
        r.mlen = 30;
        r.blen = 30;
        let mut extra = AlignExtra::default();
        extra.cigar = Cigar(vec![30u32 << 4]);
        r.extra = Some(Box::new(extra));

        let opt = MapOpt::default();
        let mut qseq = vec![0u8; 6];
        qseq.extend(std::iter::repeat(3u8).take(29));
        jump_split(&mi, &opt, 35, &qseq, &mut r, 0);

        assert_eq!(r.qs, 6);
        assert_eq!(r.rs, 39);
        assert!(!r.is_spliced);
        assert_eq!(r.extra.as_ref().unwrap().cigar.0, vec![29u32 << 4]);
    }
}
