use crate::flags::{CigarOp, MapFlags};
use crate::format::paf::event_identity;
use crate::index::MmIdx;
use crate::seq::SEQ_COMP_TABLE;
use crate::types::{AlignExtra, AlignReg, MM_VERSION};
use std::fmt::Write;

const MAX_BAM_CIGAR_OP: usize = 65_535;

#[inline]
fn push_ascii_bytes(s: &mut String, bytes: &[u8]) {
    // FASTA/FASTQ sequence and quality fields are ASCII by construction.
    unsafe { s.push_str(std::str::from_utf8_unchecked(bytes)) };
}

/// Write SAM header. Matches mm_write_sam_hdr().
pub fn write_sam_hdr(mi: &MmIdx, rg: Option<&str>, args: &[String]) -> String {
    let mut s = String::with_capacity(4096);
    writeln!(s, "@HD\tVN:1.6\tSO:unsorted\tGO:query").unwrap();
    for seq in &mi.seqs {
        writeln!(s, "@SQ\tSN:{}\tLN:{}", seq.name, seq.len).unwrap();
    }
    if let Some(rg) = rg {
        writeln!(s, "{}", normalize_rg_line(rg)).unwrap();
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

pub fn normalize_rg_line(rg: &str) -> String {
    rg.replace("\\t", "\t")
}

pub fn read_group_id(rg: &str) -> Option<String> {
    normalize_rg_line(rg)
        .split('\t')
        .find_map(|field| field.strip_prefix("ID:").map(str::to_owned))
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
    write_sam_record_with_comment(
        mi, qname, qseq, qual, r, _n_regs, _regs, flag, rep_len, None,
    )
}

pub fn write_sam_record_with_comment(
    mi: &MmIdx,
    qname: &str,
    qseq: &[u8],
    qual: &[u8],
    r: Option<&AlignReg>,
    _n_regs: usize,
    _regs: &[AlignReg],
    flag: MapFlags,
    rep_len: i32,
    comment: Option<&str>,
) -> String {
    let mut s = String::with_capacity(512 + qseq.len() + qual.len());
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
                push_ascii_bytes(&mut s, qseq);
            }
            s.push('\t');
            // Quality
            if qual.is_empty() || flag.contains(MapFlags::NO_QUAL) {
                s.push('*');
            } else {
                push_ascii_bytes(&mut s, qual);
            }
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

    // SAM flag
    let mut sam_flag: u16 = 0;
    if r.id != r.parent {
        sam_flag |= 0x100; // secondary
    } else if !r.sam_pri {
        sam_flag |= 0x800; // supplementary
    }
    if r.rev {
        sam_flag |= 0x10;
    } // reverse

    // RNAME, POS
    let rname = &mi.seqs[r.rid as usize].name;
    let pos = r.rs + 1; // 1-based

    write!(
        s,
        "{}\t{}\t{}\t{}\t{}\t",
        qname, sam_flag, rname, pos, r.mapq
    )
    .unwrap();

    // CIGAR
    // Clip character: use hard-clip for supplementary/secondary (unless SOFTCLIP flag set),
    // soft-clip for primary. Matches C minimap2's write_sam_cigar() logic.
    let use_hard_clip = ((sam_flag & 0x800) != 0
        || ((sam_flag & 0x100) != 0 && flag.contains(MapFlags::SECONDARY_SEQ)))
        && !flag.contains(MapFlags::SOFTCLIP);
    let clip_char = if use_hard_clip { 'H' } else { 'S' };
    let cigar_in_tag = sam_long_cigar_in_tag(r, qlen, flag);

    if let Some(ref p) = r.extra {
        if cigar_in_tag {
            let slen = if (sam_flag & 0x900) == 0 || flag.contains(MapFlags::SOFTCLIP) {
                qlen
            } else if (sam_flag & 0x100) != 0 && !flag.contains(MapFlags::SECONDARY_SEQ) {
                0
            } else {
                r.qe - r.qs
            };
            write!(s, "{}S{}N", slen, r.re - r.rs).unwrap();
        } else {
            write_sam_cigar_text(&mut s, r, p, qlen, clip_char);
        }
    } else {
        s.push('*');
    }

    // RNEXT, PNEXT, TLEN
    write!(s, "\t*\t0\t0\t").unwrap();

    // Determine SEQ/QUAL range based on clipping mode
    let (seq_start, seq_end) = if !use_hard_clip || r.extra.is_none() {
        (0usize, qseq.len()) // full sequence for soft clips / primary
    } else {
        // Hard clip: output only aligned portion (supplementary/secondary)
        if r.rev {
            let qs_fwd = r.qs.max(0) as usize;
            let qe_fwd = r.qe.min(qlen) as usize;
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
        push_ascii_bytes(&mut s, &qseq[seq_start..seq_end]);
    }

    s.push('\t');

    // QUAL
    if qual.is_empty() || flag.contains(MapFlags::NO_QUAL) {
        s.push('*');
    } else {
        let qual_start = seq_start.min(qual.len());
        let qual_end = seq_end.min(qual.len());
        if r.rev {
            for i in (qual_start..qual_end).rev() {
                s.push(qual[i] as char);
            }
        } else {
            push_ascii_bytes(&mut s, &qual[qual_start..qual_end]);
        }
    }

    // Tags
    if let Some(ref p) = r.extra {
        let nm = r.blen - r.mlen + p.n_ambi as i32;
        write!(
            s,
            "\tNM:i:{}\tms:i:{}\tAS:i:{}\tnn:i:{}",
            nm, p.dp_max0, p.dp_score, p.n_ambi
        )
        .unwrap();
        if r.is_spliced && (p.trans_strand == 1 || p.trans_strand == 2) {
            let ts = if p.trans_strand == 1 { '+' } else { '-' };
            write!(s, "\tts:A:{}", ts).unwrap();
        }
    }
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
    }
    if r.split > 0 {
        write!(s, "\tzd:i:{}", r.split).unwrap();
    }
    // SA tag for split/supplementary alignments
    if r.parent == r.id && r.extra.is_some() && _regs.len() > 1 {
        let mut sa_parts: Vec<String> = Vec::new();
        for other in _regs.iter() {
            if other.id == r.id {
                continue;
            }
            if other.parent != other.id {
                continue;
            }
            if other.extra.is_none() {
                continue;
            }
            let other_rname = if (other.rid as usize) < mi.seqs.len() {
                &mi.seqs[other.rid as usize].name
            } else {
                continue;
            };
            let strand = if other.rev { '-' } else { '+' };
            let cigar_str = sam_sa_cigar(other, qlen);
            let nm = other.blen - other.mlen + other.extra.as_ref().unwrap().n_ambi as i32;
            sa_parts.push(format!(
                "{},{},{},{},{},{}",
                other_rname,
                other.rs + 1,
                strand,
                cigar_str,
                other.mapq,
                nm
            ));
        }
        if !sa_parts.is_empty() {
            write!(s, "\tSA:Z:{}", sa_parts.join(";")).unwrap();
            s.push(';');
        }
    }

    if cigar_in_tag {
        if let Some(ref p) = r.extra {
            write_sam_cigar_tag(&mut s, r, p, qlen, use_hard_clip);
        }
    }

    if rep_len >= 0 {
        write!(s, "\trl:i:{}", rep_len).unwrap();
    }

    if flag.contains(MapFlags::COPY_COMMENT) {
        if let Some(comment) = comment.filter(|c| !c.is_empty()) {
            write!(s, "\t{}", comment).unwrap();
        }
    }

    s
}

fn sam_sa_cigar(r: &AlignReg, qlen: i32) -> String {
    let q_span = r.qe - r.qs;
    let r_span = r.re - r.rs;
    let l_m = q_span.min(r_span).max(0);
    let l_i = (q_span - l_m).max(0);
    let l_d = (r_span - l_m).max(0);
    let clip5 = if r.rev { qlen - r.qe } else { r.qs }.max(0);
    let clip3 = if r.rev { r.qs } else { qlen - r.qe }.max(0);
    let mut cigar = String::new();
    if clip5 > 0 {
        write!(cigar, "{}S", clip5).unwrap();
    }
    if l_m > 0 {
        write!(cigar, "{}M", l_m).unwrap();
    }
    if l_i > 0 {
        write!(cigar, "{}I", l_i).unwrap();
    }
    if l_d > 0 {
        write!(cigar, "{}D", l_d).unwrap();
    }
    if clip3 > 0 {
        write!(cigar, "{}S", clip3).unwrap();
    }
    cigar
}

fn sam_long_cigar_in_tag(r: &AlignReg, qlen: i32, flag: MapFlags) -> bool {
    if !flag.contains(MapFlags::LONG_CIGAR) {
        return false;
    }
    let Some(ref p) = r.extra else {
        return false;
    };
    if p.cigar.0.len() <= MAX_BAM_CIGAR_OP - 2 {
        return false;
    }
    let mut n_cigar = p.cigar.0.len();
    if r.qs != 0 {
        n_cigar += 1;
    }
    if r.qe != qlen {
        n_cigar += 1;
    }
    n_cigar > MAX_BAM_CIGAR_OP
}

fn write_sam_cigar_text(s: &mut String, r: &AlignReg, p: &AlignExtra, qlen: i32, clip_char: char) {
    let clip_start = if r.rev { qlen - r.qe } else { r.qs };
    if clip_start > 0 {
        write!(s, "{}{}", clip_start, clip_char).unwrap();
    }
    for &c in &p.cigar.0 {
        let op = c & 0xf;
        let len = c >> 4;
        let op_char = CigarOp::CHARS[op as usize] as char;
        write!(s, "{}{}", len, op_char).unwrap();
    }
    let clip_end = if r.rev { r.qs } else { qlen - r.qe };
    if clip_end > 0 {
        write!(s, "{}{}", clip_end, clip_char).unwrap();
    }
}

fn write_sam_cigar_tag(
    s: &mut String,
    r: &AlignReg,
    p: &AlignExtra,
    qlen: i32,
    use_hard_clip: bool,
) {
    let clip_code = if use_hard_clip {
        CigarOp::HardClip as u32
    } else {
        CigarOp::SoftClip as u32
    };
    let clip_start = if r.rev { qlen - r.qe } else { r.qs };
    let clip_end = if r.rev { r.qs } else { qlen - r.qe };
    s.push_str("\tCG:B:I");
    if clip_start > 0 {
        write!(s, ",{}", ((clip_start as u32) << 4) | clip_code).unwrap();
    }
    for &c in &p.cigar.0 {
        write!(s, ",{}", c).unwrap();
    }
    if clip_end > 0 {
        write!(s, ",{}", ((clip_end as u32) << 4) | clip_code).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Cigar;

    #[test]
    fn test_write_sam_hdr() {
        let mi = MmIdx::build_from_str(
            5,
            10,
            false,
            10,
            &[b"ACGTACGTACGT" as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        let hdr = write_sam_hdr(&mi, None, &[]);
        assert!(hdr.contains("@HD\tVN:1.6"));
        assert!(hdr.contains("@SQ\tSN:chr1\tLN:12"));
        assert!(hdr.contains("@PG\tID:minimap2"));
    }

    #[test]
    fn test_write_sam_unmapped() {
        let mi = MmIdx::build_from_str(5, 10, false, 10, &[b"ACGTACGT" as &[u8]], Some(&["chr1"]))
            .unwrap();
        let rec = write_sam_record(
            &mi,
            "read1",
            b"ACGT",
            b"IIII",
            None,
            0,
            &[],
            MapFlags::empty(),
            -1,
        );
        let fields: Vec<&str> = rec.split('\t').collect();
        assert_eq!(fields[0], "read1");
        assert_eq!(fields[1], "4"); // unmapped flag
        assert_eq!(fields[2], "*");
    }

    #[test]
    fn test_write_sam_unmapped_copy_comment() {
        let mi = MmIdx::build_from_str(5, 10, false, 10, &[b"ACGTACGT" as &[u8]], Some(&["chr1"]))
            .unwrap();
        let rec = write_sam_record_with_comment(
            &mi,
            "read1",
            b"ACGT",
            b"IIII",
            None,
            0,
            &[],
            MapFlags::COPY_COMMENT,
            -1,
            Some("comment text"),
        );
        assert!(rec.ends_with("\tcomment text"));
    }

    #[test]
    fn test_write_sam_mapped() {
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
        r.qs = 0;
        r.qe = 8;
        r.rs = 10;
        r.re = 18;
        r.rev = false;
        r.mapq = 60;
        r.id = 0;
        r.parent = 0;
        r.sam_pri = true;
        r.score = 16;
        r.cnt = 3;
        r.mlen = 8;
        r.blen = 8;

        let rec = write_sam_record(
            &mi,
            "read1",
            b"ACGTACGT",
            b"",
            Some(&r),
            1,
            &[r.clone()],
            MapFlags::empty(),
            -1,
        );
        let fields: Vec<&str> = rec.split('\t').collect();
        assert_eq!(fields[2], "chr1");
        assert_eq!(fields[3], "11"); // 1-based pos
        assert_eq!(fields[4], "60"); // MAPQ
    }

    #[test]
    fn test_write_sam_mapped_copy_comment() {
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
        r.qs = 0;
        r.qe = 8;
        r.rs = 10;
        r.re = 18;
        r.rev = false;
        r.mapq = 60;
        r.id = 0;
        r.parent = 0;
        r.sam_pri = true;
        r.score = 16;
        r.cnt = 3;
        r.mlen = 8;
        r.blen = 8;

        let rec = write_sam_record_with_comment(
            &mi,
            "read1",
            b"ACGTACGT",
            b"",
            Some(&r),
            1,
            &[r.clone()],
            MapFlags::COPY_COMMENT,
            -1,
            Some("comment text"),
        );
        assert!(rec.ends_with("\tcomment text"));
    }

    #[test]
    fn test_write_sam_long_cigar_in_cg_tag() {
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
        r.qs = 0;
        r.qe = 8;
        r.rs = 10;
        r.re = 18;
        r.rev = false;
        r.mapq = 60;
        r.id = 0;
        r.parent = 0;
        r.sam_pri = true;
        r.score = 16;
        r.cnt = 3;
        r.mlen = 8;
        r.blen = 8;
        let mut extra = AlignExtra::default();
        extra.cigar = Cigar(vec![
            (1u32 << 4) | (CigarOp::Match as u32);
            MAX_BAM_CIGAR_OP + 1
        ]);
        r.extra = Some(Box::new(extra));

        let rec = write_sam_record(
            &mi,
            "read1",
            b"ACGTACGT",
            b"",
            Some(&r),
            1,
            &[r.clone()],
            MapFlags::LONG_CIGAR,
            -1,
        );
        let fields: Vec<&str> = rec.split('\t').collect();
        assert_eq!(fields[5], "8S8N");
        assert!(rec.contains("\tCG:B:I,16,16,16"));
    }
}
