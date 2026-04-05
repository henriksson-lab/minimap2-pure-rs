use std::fmt::Write;
use crate::flags::{CigarOp, MapFlags};
use crate::index::MmIdx;
use crate::seq::SEQ_COMP_TABLE;
use crate::types::{AlignReg, MM_VERSION};

/// Write SAM header. Matches mm_write_sam_hdr().
pub fn write_sam_hdr(mi: &MmIdx, rg: Option<&str>, args: &[String]) -> String {
    let mut s = String::with_capacity(4096);
    writeln!(s, "@HD\tVN:1.6\tSO:unsorted\tGO:query").unwrap();
    for seq in &mi.seqs {
        writeln!(s, "@SQ\tSN:{}\tLN:{}", seq.name, seq.len).unwrap();
    }
    if let Some(rg) = rg {
        writeln!(s, "{}", rg).unwrap();
    }
    write!(s, "@PG\tID:minimap2\tPN:minimap2\tVN:{}", MM_VERSION).unwrap();
    if !args.is_empty() {
        write!(s, "\tCL:minimap2").unwrap();
        for arg in args {
            write!(s, " {}", arg).unwrap();
        }
    }
    s
}

/// Write a single SAM record.
///
/// Simplified version matching mm_write_sam3() core logic.
pub fn write_sam_record(
    mi: &MmIdx,
    qname: &str,
    qseq: &[u8],
    qual: &[u8],
    r: Option<&AlignReg>,
    _n_regs: usize,
    _regs: &[AlignReg],
    flag: MapFlags,
    rep_len: i32,
) -> String {
    let mut s = String::with_capacity(512);
    let qlen = qseq.len() as i32;

    // Unmapped
    let r = match r {
        Some(r) => r,
        None => {
            write!(s, "{}\t4\t*\t0\t0\t*\t*\t0\t0\t", qname).unwrap();
            // Sequence
            if flag.contains(MapFlags::NO_QUAL) || qseq.is_empty() {
                s.push('*');
            } else {
                for &b in qseq { s.push(b as char); }
            }
            s.push('\t');
            // Quality
            if qual.is_empty() || flag.contains(MapFlags::NO_QUAL) {
                s.push('*');
            } else {
                for &q in qual { s.push(q as char); }
            }
            return s;
        }
    };

    // SAM flag
    let mut sam_flag: u16 = 0;
    if r.split > 0 && !r.sam_pri {
        sam_flag |= 0x800; // supplementary
    } else if !r.sam_pri {
        sam_flag |= 0x100; // secondary
    }
    if r.rev { sam_flag |= 0x10; } // reverse

    // RNAME, POS
    let rname = &mi.seqs[r.rid as usize].name;
    let pos = r.rs + 1; // 1-based

    write!(s, "{}\t{}\t{}\t{}\t{}\t", qname, sam_flag, rname, pos, r.mapq).unwrap();

    // CIGAR
    if let Some(ref p) = r.extra {
        // Soft-clip at start
        if r.rev {
            let clip = qlen - r.qe;
            if clip > 0 {
                if flag.contains(MapFlags::SOFTCLIP) {
                    write!(s, "{}S", clip).unwrap();
                } else {
                    write!(s, "{}H", clip).unwrap();
                }
            }
        } else {
            if r.qs > 0 {
                if flag.contains(MapFlags::SOFTCLIP) {
                    write!(s, "{}S", r.qs).unwrap();
                } else {
                    write!(s, "{}H", r.qs).unwrap();
                }
            }
        }
        // Main CIGAR
        for &c in &p.cigar.0 {
            let op = c & 0xf;
            let len = c >> 4;
            let op_char = CigarOp::CHARS[op as usize] as char;
            write!(s, "{}{}", len, op_char).unwrap();
        }
        // Soft-clip at end
        if r.rev {
            if r.qs > 0 {
                if flag.contains(MapFlags::SOFTCLIP) {
                    write!(s, "{}S", r.qs).unwrap();
                } else {
                    write!(s, "{}H", r.qs).unwrap();
                }
            }
        } else {
            let clip = qlen - r.qe;
            if clip > 0 {
                if flag.contains(MapFlags::SOFTCLIP) {
                    write!(s, "{}S", clip).unwrap();
                } else {
                    write!(s, "{}H", clip).unwrap();
                }
            }
        }
    } else {
        s.push('*');
    }

    // RNEXT, PNEXT, TLEN
    write!(s, "\t*\t0\t0\t").unwrap();

    // Determine SEQ/QUAL range based on clipping mode
    let use_softclip = flag.contains(MapFlags::SOFTCLIP);
    let (seq_start, seq_end) = if use_softclip || r.extra.is_none() {
        (0usize, qseq.len()) // full sequence for soft clips
    } else {
        // Hard clip: output only aligned portion
        if r.rev {
            let qs_fwd = (qlen - r.qe).max(0) as usize;
            let qe_fwd = (qlen - r.qs).min(qlen) as usize;
            (qs_fwd.min(qseq.len()), qe_fwd.min(qseq.len()))
        } else {
            (r.qs.max(0) as usize, r.qe.min(qlen) as usize)
        }
    };

    // SEQ
    if flag.contains(MapFlags::NO_QUAL) || qseq.is_empty() {
        s.push('*');
    } else if r.rev {
        for i in (seq_start..seq_end).rev() {
            s.push(SEQ_COMP_TABLE[qseq[i] as usize] as char);
        }
    } else {
        for &b in &qseq[seq_start..seq_end] { s.push(b as char); }
    }

    s.push('\t');

    // QUAL
    if qual.is_empty() || flag.contains(MapFlags::NO_QUAL) {
        s.push('*');
    } else {
        let qual_start = seq_start.min(qual.len());
        let qual_end = seq_end.min(qual.len());
        if r.rev {
            for i in (qual_start..qual_end).rev() { s.push(qual[i] as char); }
        } else {
            for &q in &qual[qual_start..qual_end] { s.push(q as char); }
        }
    }

    // Tags
    if let Some(ref p) = r.extra {
        let nm = r.blen - r.mlen + p.n_ambi as i32;
        write!(s, "\tNM:i:{}\tms:i:{}\tAS:i:{}\tnn:i:{}", nm, p.dp_max, p.dp_score, p.n_ambi).unwrap();
    }
    let type_char = if r.id == r.parent {
        if r.inv { 'I' } else { 'P' }
    } else {
        if r.inv { 'i' } else { 'S' }
    };
    write!(s, "\ttp:A:{}\tcm:i:{}\ts1:i:{}", type_char, r.cnt, r.score).unwrap();
    if r.parent == r.id {
        write!(s, "\ts2:i:{}", r.subsc).unwrap();
    }
    if rep_len >= 0 {
        write!(s, "\trl:i:{}", rep_len).unwrap();
    }

    // SA tag for split/supplementary alignments
    if r.split > 0 && _regs.len() > 1 {
        let mut sa_parts: Vec<String> = Vec::new();
        for other in _regs.iter() {
            if other.id == r.id { continue; } // skip self
            if other.split == 0 { continue; } // only include split pieces
            if other.extra.is_none() { continue; }
            let other_rname = if (other.rid as usize) < mi.seqs.len() {
                &mi.seqs[other.rid as usize].name
            } else { continue; };
            let strand = if other.rev { '-' } else { '+' };
            let cigar_str = crate::align::cigar_to_string(&other.extra.as_ref().unwrap().cigar.0);
            let nm = other.blen - other.mlen + other.extra.as_ref().unwrap().n_ambi as i32;
            sa_parts.push(format!("{},{},{},{},{},{}", other_rname, other.rs + 1, strand, cigar_str, other.mapq, nm));
        }
        if !sa_parts.is_empty() {
            write!(s, "\tSA:Z:{}", sa_parts.join(";")).unwrap();
            s.push(';');
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_sam_hdr() {
        let mi = MmIdx::build_from_str(5, 10, false, 10, &[b"ACGTACGTACGT" as &[u8]], Some(&["chr1"])).unwrap();
        let hdr = write_sam_hdr(&mi, None, &[]);
        assert!(hdr.contains("@HD\tVN:1.6"));
        assert!(hdr.contains("@SQ\tSN:chr1\tLN:12"));
        assert!(hdr.contains("@PG\tID:minimap2"));
    }

    #[test]
    fn test_write_sam_unmapped() {
        let mi = MmIdx::build_from_str(5, 10, false, 10, &[b"ACGTACGT" as &[u8]], Some(&["chr1"])).unwrap();
        let rec = write_sam_record(&mi, "read1", b"ACGT", b"IIII", None, 0, &[], MapFlags::empty(), -1);
        let fields: Vec<&str> = rec.split('\t').collect();
        assert_eq!(fields[0], "read1");
        assert_eq!(fields[1], "4"); // unmapped flag
        assert_eq!(fields[2], "*");
    }

    #[test]
    fn test_write_sam_mapped() {
        let mi = MmIdx::build_from_str(5, 10, false, 10,
            &[b"ACGTACGTACGTACGTACGTACGTACGTACGT" as &[u8]], Some(&["chr1"])).unwrap();
        let mut r = AlignReg::default();
        r.rid = 0; r.qs = 0; r.qe = 8;
        r.rs = 10; r.re = 18;
        r.rev = false; r.mapq = 60;
        r.id = 0; r.parent = 0; r.sam_pri = true;
        r.score = 16; r.cnt = 3;
        r.mlen = 8; r.blen = 8;

        let rec = write_sam_record(&mi, "read1", b"ACGTACGT", b"", Some(&r), 1, &[r.clone()], MapFlags::empty(), -1);
        let fields: Vec<&str> = rec.split('\t').collect();
        assert_eq!(fields[2], "chr1");
        assert_eq!(fields[3], "11"); // 1-based pos
        assert_eq!(fields[4], "60"); // MAPQ
    }
}
