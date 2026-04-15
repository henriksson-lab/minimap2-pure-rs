use crate::flags::MapFlags;
use crate::index::MmIdx;
use crate::types::AlignReg;
use std::fmt::Write;

/// Compute event identity from alignment: mlen / blen.
pub fn event_identity(r: &AlignReg) -> f64 {
    if r.blen == 0 {
        return 0.0;
    }
    let Some(extra) = r.extra.as_ref() else {
        return r.mlen as f64 / r.blen as f64;
    };
    let mut n_gap = 0i32;
    let mut n_gapo = 0i32;
    for &c in &extra.cigar.0 {
        let op = c & 0xf;
        if op == 1 || op == 2 {
            n_gapo += 1;
            n_gap += (c >> 4) as i32;
        }
    }
    r.mlen as f64 / (r.blen + extra.n_ambi as i32 - n_gap + n_gapo) as f64
}

/// Write PAF tags for an alignment. Matches write_tags() from format.c.
fn write_tags(s: &mut String, r: &AlignReg) {
    let type_char = if r.id == r.parent {
        if r.inv {
            'I'
        } else {
            'P'
        }
    } else {
        if r.inv {
            'i'
        } else {
            'S'
        }
    };
    if let Some(ref p) = r.extra {
        let nm = r.blen - r.mlen + p.n_ambi as i32;
        write!(
            s,
            "\tNM:i:{}\tms:i:{}\tAS:i:{}\tnn:i:{}",
            nm, p.dp_max, p.dp_score, p.n_ambi
        )
        .unwrap();
        if p.trans_strand == 1 || p.trans_strand == 2 {
            let ts = if p.trans_strand == 1 { '+' } else { '-' };
            write!(s, "\tts:A:{}", ts).unwrap();
        }
    }
    write!(s, "\ttp:A:{}\tcm:i:{}\ts1:i:{}", type_char, r.cnt, r.score).unwrap();
    if r.parent == r.id {
        write!(s, "\ts2:i:{}", r.subsc).unwrap();
    }
    if r.extra.is_some() {
        let div = 1.0 - event_identity(r);
        if div == 0.0 {
            write!(s, "\tde:f:0").unwrap();
        } else {
            write!(s, "\tde:f:{:.4}", div).unwrap();
        }
    } else if r.div >= 0.0 && r.div <= 1.0 {
        if r.div == 0.0 {
            write!(s, "\tdv:f:0").unwrap();
        } else {
            write!(s, "\tdv:f:{:.4}", r.div).unwrap();
        }
    }
    if r.split > 0 {
        write!(s, "\tzd:i:{}", r.split).unwrap();
    }
}

/// Format a single PAF record. Matches mm_write_paf4().
///
/// Returns the PAF line (without trailing newline).
pub fn write_paf(
    mi: &MmIdx,
    qname: &str,
    qlen: i32,
    r: Option<&AlignReg>,
    flag: MapFlags,
    rep_len: i32,
    n_seg: i32,
    seg_idx: i32,
    comment: Option<&str>,
) -> String {
    let mut s = String::with_capacity(256);

    s.push_str(qname);
    if flag.contains(MapFlags::FRAG_MODE) && n_seg >= 2 && seg_idx >= 0 {
        write!(s, "/{}", seg_idx + 1).unwrap();
    }

    let r = match r {
        Some(r) => r,
        None => {
            write!(s, "\t{}\t0\t0\t*\t*\t0\t0\t0\t0\t0\t0", qlen).unwrap();
            if rep_len >= 0 {
                write!(s, "\trl:i:{}", rep_len).unwrap();
            }
            if flag.contains(MapFlags::COPY_COMMENT) {
                if let Some(comment) = comment.filter(|c| !c.is_empty()) {
                    write!(s, "\t{}", comment).unwrap();
                }
            }
            return s;
        }
    };

    let strand = if r.rev { '-' } else { '+' };
    write!(s, "\t{}\t{}\t{}\t{}\t", qlen, r.qs, r.qe, strand).unwrap();

    // Reference name
    let rname = &mi.seqs[r.rid as usize].name;
    if !rname.is_empty() {
        s.push_str(rname);
    } else {
        write!(s, "{}", r.rid).unwrap();
    }

    let rlen = mi.seqs[r.rid as usize].len;
    if flag.contains(MapFlags::QSTRAND) && r.rev {
        write!(
            s,
            "\t{}\t{}\t{}",
            rlen,
            rlen as i32 - r.re,
            rlen as i32 - r.rs
        )
        .unwrap();
    } else {
        write!(s, "\t{}\t{}\t{}", rlen, r.rs, r.re).unwrap();
    }

    write!(s, "\t{}\t{}\t{}", r.mlen, r.blen, r.mapq).unwrap();
    write_tags(&mut s, r);

    if rep_len >= 0 {
        write!(s, "\trl:i:{}", rep_len).unwrap();
    }

    // CIGAR tag (cg:Z:)
    if r.extra.is_some() && flag.contains(MapFlags::OUT_CG) {
        let p = r.extra.as_ref().unwrap();
        s.push_str("\tcg:Z:");
        for &c in &p.cigar.0 {
            let op = c & 0xf;
            let len = c >> 4;
            const OPS: &[u8] = b"MIDNSHP=XB";
            write!(s, "{}{}", len, OPS[op as usize] as char).unwrap();
        }
    }

    if flag.contains(MapFlags::COPY_COMMENT) {
        if let Some(comment) = comment.filter(|c| !c.is_empty()) {
            write!(s, "\t{}", comment).unwrap();
        }
    }

    s
}

/// Format an unmapped PAF record.
pub fn write_paf_unmapped(qname: &str, qlen: i32) -> String {
    format!("{}\t{}\t0\t0\t*\t*\t0\t0\t0\t0\t0\t0", qname, qlen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_paf_unmapped() {
        let line = write_paf_unmapped("read1", 150);
        assert!(line.starts_with("read1\t150\t"));
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 12);
        assert_eq!(fields[4], "*");
    }

    #[test]
    fn test_write_paf_no_hit() {
        let mi = MmIdx::build_from_str(
            5,
            10,
            false,
            10,
            &[b"ACGTACGTACGT" as &[u8]],
            Some(&["ref1"]),
        )
        .unwrap();
        let line = write_paf(&mi, "read1", 100, None, MapFlags::empty(), -1, 0, 0, None);
        assert!(line.starts_with("read1\t100\t0\t0\t*"));
    }

    #[test]
    fn test_write_paf_no_hit_copy_comment() {
        let mi = MmIdx::build_from_str(
            5,
            10,
            false,
            10,
            &[b"ACGTACGTACGT" as &[u8]],
            Some(&["ref1"]),
        )
        .unwrap();
        let line = write_paf(
            &mi,
            "read1",
            100,
            None,
            MapFlags::COPY_COMMENT,
            -1,
            0,
            0,
            Some("comment text"),
        );
        assert!(line.ends_with("\tcomment text"));
    }

    #[test]
    fn test_write_paf_with_hit() {
        let mi = MmIdx::build_from_str(
            5,
            10,
            false,
            10,
            &[b"ACGTACGTACGTACGTACGTACGTACGTACGT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        let mut r = AlignReg::default();
        r.rid = 0;
        r.qs = 10;
        r.qe = 50;
        r.rs = 100;
        r.re = 140;
        r.rev = false;
        r.mlen = 35;
        r.blen = 40;
        r.mapq = 60;
        r.id = 0;
        r.parent = 0;
        r.score = 80;
        r.cnt = 5;

        let line = write_paf(
            &mi,
            "read1",
            100,
            Some(&r),
            MapFlags::empty(),
            -1,
            0,
            0,
            None,
        );
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields[0], "read1");
        assert_eq!(fields[1], "100"); // qlen
        assert_eq!(fields[4], "+"); // strand
        assert_eq!(fields[5], "chr1");
    }

    #[test]
    fn test_event_identity() {
        let mut r = AlignReg::default();
        r.mlen = 90;
        r.blen = 100;
        let id = event_identity(&r);
        assert!((id - 0.9).abs() < 0.001);
    }
}
