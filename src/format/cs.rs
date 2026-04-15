use crate::index::MmIdx;
use crate::types::AlignReg;
use std::fmt::Write;

const NT_CHARS: &[u8] = b"acgtn";

/// Generate the cs tag for an alignment.
///
/// If `no_iden` is true, uses ':' instead of '=' for matches (short cs).
pub fn gen_cs(
    mi: &MmIdx,
    r: &AlignReg,
    qseq: &[u8], // 0-3 encoded query sequence (forward strand segment)
    no_iden: bool,
) -> Option<String> {
    let p = r.extra.as_ref()?;
    let mut s = String::with_capacity(256);
    let mut q_off = 0usize;
    let mut t_off = r.rs as usize;

    // Fetch reference region
    let rlen = (r.re - r.rs) as usize;
    let mut tseq = vec![0u8; rlen];
    mi.getseq2(r.rev, r.rid as u32, r.rs as u32, r.re as u32, &mut tseq);

    for &c in &p.cigar.0 {
        let op = (c & 0xf) as usize;
        let len = (c >> 4) as usize;
        match op {
            0 | 7 | 8 => {
                // M, =, X: match/mismatch
                let mut i = 0;
                while i < len {
                    let ti = t_off - r.rs as usize;
                    if ti >= tseq.len() || q_off >= qseq.len() {
                        break;
                    }
                    let tc = tseq[ti];
                    let qc = qseq[q_off];
                    if tc == qc {
                        // Count consecutive matches
                        let mut n = 0;
                        while i + n < len {
                            let ti2 = t_off + n - r.rs as usize;
                            if ti2 >= tseq.len() || q_off + n >= qseq.len() {
                                break;
                            }
                            if tseq[ti2] != qseq[q_off + n] {
                                break;
                            }
                            n += 1;
                        }
                        if no_iden {
                            write!(s, ":{}", n).unwrap();
                        } else {
                            s.push('=');
                            for k in 0..n {
                                let ti2 = t_off + k - r.rs as usize;
                                s.push(NT_CHARS[tseq[ti2].min(4) as usize] as char);
                            }
                        }
                        q_off += n;
                        t_off += n;
                        i += n;
                    } else {
                        // Mismatch
                        s.push('*');
                        s.push(NT_CHARS[tc.min(4) as usize] as char);
                        s.push(NT_CHARS[qc.min(4) as usize] as char);
                        q_off += 1;
                        t_off += 1;
                        i += 1;
                    }
                }
            }
            1 => {
                // I: insertion to ref
                s.push('+');
                for _ in 0..len {
                    if q_off < qseq.len() {
                        s.push(NT_CHARS[qseq[q_off].min(4) as usize] as char);
                    }
                    q_off += 1;
                }
            }
            2 => {
                // D: deletion from ref
                s.push('-');
                for _ in 0..len {
                    let ti = t_off - r.rs as usize;
                    if ti < tseq.len() {
                        s.push(NT_CHARS[tseq[ti].min(4) as usize] as char);
                    }
                    t_off += 1;
                }
            }
            3 => {
                // N: intron/skip
                write!(s, "~").unwrap();
                // Write splice site bases
                let ti = t_off - r.rs as usize;
                if ti + 1 < tseq.len() {
                    s.push(NT_CHARS[tseq[ti].min(4) as usize] as char);
                    s.push(NT_CHARS[tseq[ti + 1].min(4) as usize] as char);
                }
                write!(s, "{}", len).unwrap();
                let ti_end = t_off + len - r.rs as usize;
                if ti_end >= 2 && ti_end <= tseq.len() {
                    s.push(NT_CHARS[tseq[ti_end - 2].min(4) as usize] as char);
                    s.push(NT_CHARS[tseq[ti_end - 1].min(4) as usize] as char);
                }
                t_off += len;
            }
            _ => {
                t_off += len;
                q_off += len;
            }
        }
    }
    Some(s)
}

fn write_indel_ds(s: &mut String, len: usize, seq: &[u8], ll: usize, lr: usize) {
    if ll + lr >= len {
        s.push('[');
        for &c in &seq[..len.min(seq.len())] {
            s.push(NT_CHARS[c.min(4) as usize] as char);
        }
        s.push(']');
    } else {
        let mut k = 0usize;
        if ll > 0 {
            s.push('[');
            for &c in &seq[k..(k + ll).min(seq.len())] {
                s.push(NT_CHARS[c.min(4) as usize] as char);
            }
            s.push(']');
            k += ll;
        }
        let mid_len = len.saturating_sub(lr + ll);
        for &c in &seq[k..(k + mid_len).min(seq.len())] {
            s.push(NT_CHARS[c.min(4) as usize] as char);
        }
        k += mid_len;
        if lr > 0 {
            s.push('[');
            for &c in &seq[k..(k + lr).min(seq.len())] {
                s.push(NT_CHARS[c.min(4) as usize] as char);
            }
            s.push(']');
        }
    }
}

/// Generate the ds tag for an alignment.
///
/// `ds` is a cs-like tag that annotates indels with local repeat context.
pub fn gen_ds(mi: &MmIdx, r: &AlignReg, qseq: &[u8], no_iden: bool) -> Option<String> {
    let p = r.extra.as_ref()?;
    let mut s = String::with_capacity(256);
    let mut q_off = 0usize;
    let mut t_off = 0usize;

    let rlen = (r.re - r.rs) as usize;
    let mut tseq = vec![0u8; rlen];
    mi.getseq2(r.rev, r.rid as u32, r.rs as u32, r.re as u32, &mut tseq);

    for &c in &p.cigar.0 {
        let op = (c & 0xf) as usize;
        let len = (c >> 4) as usize;
        match op {
            0 | 7 | 8 => {
                let mut i = 0usize;
                while i < len {
                    if t_off + i >= tseq.len() || q_off + i >= qseq.len() {
                        break;
                    }
                    if tseq[t_off + i] != qseq[q_off + i] {
                        s.push('*');
                        s.push(NT_CHARS[tseq[t_off + i].min(4) as usize] as char);
                        s.push(NT_CHARS[qseq[q_off + i].min(4) as usize] as char);
                        i += 1;
                    } else {
                        let start = i;
                        while i < len
                            && t_off + i < tseq.len()
                            && q_off + i < qseq.len()
                            && tseq[t_off + i] == qseq[q_off + i]
                        {
                            i += 1;
                        }
                        if no_iden {
                            write!(s, ":{}", i - start).unwrap();
                        } else {
                            s.push('=');
                            for &base in &qseq[q_off + start..q_off + i] {
                                s.push((b"ACGTN"[base.min(4) as usize]) as char);
                            }
                        }
                    }
                }
                q_off += len;
                t_off += len;
            }
            1 => {
                let y = q_off;
                let mut z = 1usize;
                while z <= len {
                    if y < z || y + len < z || qseq.get(y + len - z) != qseq.get(y - z) {
                        break;
                    }
                    z += 1;
                }
                let lr = z - 1;
                z = 0;
                while z < len {
                    if y + len + z >= qseq.len()
                        || y + z >= qseq.len()
                        || qseq[y + len + z] != qseq[y + z]
                    {
                        break;
                    }
                    z += 1;
                }
                let ll = z;
                s.push('+');
                if y < qseq.len() {
                    write_indel_ds(&mut s, len, &qseq[y..], ll, lr);
                }
                q_off += len;
            }
            2 => {
                let x = t_off;
                let mut z = 1usize;
                while z <= len {
                    if x < z || x + len < z || tseq.get(x + len - z) != tseq.get(x - z) {
                        break;
                    }
                    z += 1;
                }
                let lr = z - 1;
                z = 0;
                while z < len {
                    if x + len + z >= tseq.len()
                        || x + z >= tseq.len()
                        || tseq[x + z] != tseq[x + len + z]
                    {
                        break;
                    }
                    z += 1;
                }
                let ll = z;
                s.push('-');
                if x < tseq.len() {
                    write_indel_ds(&mut s, len, &tseq[x..], ll, lr);
                }
                t_off += len;
            }
            3 => {
                let ti = t_off;
                if ti + len <= tseq.len() && len >= 2 {
                    s.push('~');
                    s.push(NT_CHARS[tseq[ti].min(4) as usize] as char);
                    s.push(NT_CHARS[tseq[ti + 1].min(4) as usize] as char);
                    write!(s, "{}", len).unwrap();
                    s.push(NT_CHARS[tseq[ti + len - 2].min(4) as usize] as char);
                    s.push(NT_CHARS[tseq[ti + len - 1].min(4) as usize] as char);
                }
                t_off += len;
            }
            _ => {
                t_off += len;
                q_off += len;
            }
        }
    }
    Some(s)
}

/// Generate the MD tag for an alignment.
pub fn gen_md(mi: &MmIdx, r: &AlignReg, qseq: &[u8]) -> Option<String> {
    let p = r.extra.as_ref()?;
    let mut s = String::with_capacity(128);
    let mut q_off = 0usize;
    let mut t_off = r.rs as usize;
    let mut n_match = 0u32;

    let rlen = (r.re - r.rs) as usize;
    let mut tseq = vec![0u8; rlen];
    mi.getseq2(r.rev, r.rid as u32, r.rs as u32, r.re as u32, &mut tseq);

    for &c in &p.cigar.0 {
        let op = (c & 0xf) as usize;
        let len = (c >> 4) as usize;
        match op {
            0 | 7 | 8 => {
                for _ in 0..len {
                    let ti = t_off - r.rs as usize;
                    if ti >= tseq.len() || q_off >= qseq.len() {
                        break;
                    }
                    if tseq[ti] == qseq[q_off] {
                        n_match += 1;
                    } else {
                        write!(s, "{}", n_match).unwrap();
                        n_match = 0;
                        s.push((b"ACGTN"[tseq[ti].min(4) as usize]) as char);
                    }
                    q_off += 1;
                    t_off += 1;
                }
            }
            1 => {
                q_off += len;
            }
            2 | 3 => {
                write!(s, "{}", n_match).unwrap();
                n_match = 0;
                s.push('^');
                for _ in 0..len {
                    let ti = t_off - r.rs as usize;
                    if ti < tseq.len() {
                        s.push((b"ACGTN"[tseq[ti].min(4) as usize]) as char);
                    }
                    t_off += 1;
                }
            }
            _ => {
                t_off += len;
                q_off += len;
            }
        }
    }
    write!(s, "{}", n_match).unwrap();
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::CigarOp;
    use crate::types::{AlignExtra, Cigar};

    fn make_index_and_reg() -> (MmIdx, AlignReg, Vec<u8>) {
        // ref: ACGTACGTACGT (encoded as 0,1,2,3,...)
        let mi = MmIdx::build_from_str(
            5,
            5,
            false,
            10,
            &[b"ACGTACGTACGT" as &[u8]],
            Some(&["ref1"]),
        )
        .unwrap();

        let mut cigar = Cigar::new();
        cigar.push(CigarOp::Match, 8); // 8M

        let extra = AlignExtra {
            dp_score: 16,
            dp_max: 16,
            dp_max2: 0,
            dp_max0: 16,
            n_ambi: 0,
            trans_strand: 0,
            cigar,
        };

        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 0;
        r.re = 8;
        r.qs = 0;
        r.qe = 8;
        r.mlen = 8;
        r.blen = 8;
        r.extra = Some(Box::new(extra));

        // query: ACGTACGT (matching ref)
        let qseq = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
        (mi, r, qseq)
    }

    #[test]
    fn test_gen_cs_perfect_match() {
        let (mi, r, qseq) = make_index_and_reg();
        let cs = gen_cs(&mi, &r, &qseq, true).unwrap();
        assert_eq!(cs, ":8"); // 8 matches in short form
    }

    #[test]
    fn test_gen_md_perfect_match() {
        let (mi, r, qseq) = make_index_and_reg();
        let md = gen_md(&mi, &r, &qseq).unwrap();
        assert_eq!(md, "8"); // 8 matches
    }

    #[test]
    fn test_gen_ds_perfect_match() {
        let (mi, r, qseq) = make_index_and_reg();
        let ds = gen_ds(&mi, &r, &qseq, true).unwrap();
        assert_eq!(ds, ":8");
    }
}
