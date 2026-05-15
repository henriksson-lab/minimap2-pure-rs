use super::MmIdx;
use crate::flags::MapFlags;
use crate::hit;
use crate::options::MapOpt;
use crate::types::{AlignExtra, AlignReg, Cigar, IdxSeq};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct SplitPartHeader {
    pub k: u32,
    pub seqs: Vec<IdxSeq>,
}

#[derive(Clone, Debug, Default)]
pub struct SplitQueryRecord {
    pub n_reg: i32,
    pub rep_len: i32,
    pub frag_gap: i32,
    pub regs: Vec<AlignReg>,
}

/// Build the per-part split temp-file path: `<prefix>.<part:04>.tmp`.
///
/// # Parameters
/// * `prefix` - shared filename prefix supplied by the user
/// * `part` - 0-based index part number
pub fn split_tmp_path(prefix: &str, part: usize) -> PathBuf {
    PathBuf::from(format!("{}.{:04}.tmp", prefix, part))
}

/// Write the split-part header (k + sequence metadata) to a writer.
///
/// # Parameters
/// * `writer` - destination writer; wrapped in a BufWriter
/// * `mi` - index part whose k-mer size and sequences are recorded
pub fn write_split_header<W: Write>(writer: &mut W, mi: &MmIdx) -> io::Result<()> {
    let mut writer = BufWriter::new(writer);
    writer.write_all(&(mi.k as u32).to_le_bytes())?;
    writer.write_all(&(mi.seqs.len() as u32).to_le_bytes())?;
    for seq in &mi.seqs {
        let name = seq.name.as_bytes();
        writer.write_all(&(name.len() as u32).to_le_bytes())?;
        writer.write_all(name)?;
        writer.write_all(&seq.len.to_le_bytes())?;
    }
    writer.flush()
}

/// Create a per-part split temp file and write its header.
///
/// # Parameters
/// * `prefix` - shared filename prefix
/// * `part` - 0-based index part number
/// * `mi` - index part to header-stamp
pub fn create_split_tmp(prefix: &str, part: usize, mi: &MmIdx) -> io::Result<File> {
    let path = split_tmp_path(prefix, part);
    let mut file = File::create(path)?;
    write_split_header(&mut file, mi)?;
    Ok(file)
}

/// Read a split-part header (k-mer size + sequence metadata).
///
/// # Parameters
/// * `reader` - source reader positioned at the start of a split-part file
pub fn read_split_header<R: Read>(reader: &mut R) -> io::Result<SplitPartHeader> {
    let mut reader = BufReader::new(reader);
    let k = read_u32(&mut reader)?;
    let n_seq = read_u32(&mut reader)?;
    let mut seqs = Vec::with_capacity(n_seq as usize);
    let mut offset = 0u64;
    for _ in 0..n_seq {
        let name_len = read_u32(&mut reader)? as usize;
        let mut name = vec![0u8; name_len];
        reader.read_exact(&mut name)?;
        let len = read_u32(&mut reader)?;
        seqs.push(IdxSeq {
            name: String::from_utf8_lossy(&name).into_owned(),
            offset,
            len,
            is_alt: false,
        });
        offset += len as u64;
    }
    Ok(SplitPartHeader { k, seqs })
}

/// Read all split-part headers for an existing run.
///
/// # Parameters
/// * `prefix` - shared filename prefix
/// * `n_parts` - number of parts to load (0..n_parts)
pub fn read_split_headers(prefix: &str, n_parts: usize) -> io::Result<Vec<SplitPartHeader>> {
    let mut headers = Vec::with_capacity(n_parts);
    for part in 0..n_parts {
        let mut file = File::open(split_tmp_path(prefix, part))?;
        headers.push(read_split_header(&mut file)?);
    }
    Ok(headers)
}

/// Remove all split-part temp files for a run (ignores missing files).
///
/// # Parameters
/// * `prefix` - shared filename prefix
/// * `n_parts` - number of parts to delete (0..n_parts)
pub fn remove_split_tmps(prefix: &str, n_parts: usize) -> io::Result<()> {
    for part in 0..n_parts {
        let path = split_tmp_path(prefix, part);
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Serialize a per-query split record (regions for one read against one part).
///
/// # Parameters
/// * `writer` - destination writer
/// * `record` - per-query result block to write
/// * `with_cigar` - include the optional AlignExtra (CIGAR + DP stats) per region
pub fn write_split_query_record<W: Write>(
    writer: &mut W,
    record: &SplitQueryRecord,
    with_cigar: bool,
) -> io::Result<()> {
    writer.write_all(&record.n_reg.to_le_bytes())?;
    writer.write_all(&record.rep_len.to_le_bytes())?;
    writer.write_all(&record.frag_gap.to_le_bytes())?;
    for r in &record.regs {
        write_align_reg(writer, r, with_cigar)?;
    }
    Ok(())
}

/// Deserialize a per-query split record written by `write_split_query_record`.
///
/// # Parameters
/// * `reader` - source reader positioned at the start of one record
/// * `with_cigar` - must match the value used when writing
pub fn read_split_query_record<R: Read>(
    reader: &mut R,
    with_cigar: bool,
) -> io::Result<SplitQueryRecord> {
    let n_reg = read_i32(reader)?;
    let rep_len = read_i32(reader)?;
    let frag_gap = read_i32(reader)?;
    let mut regs = Vec::with_capacity(n_reg.max(0) as usize);
    for _ in 0..n_reg.max(0) {
        regs.push(read_align_reg(reader, with_cigar)?);
    }
    Ok(SplitQueryRecord {
        n_reg,
        rep_len,
        frag_gap,
        regs,
    })
}

/// Merge per-part hits for one query into a unified region set; re-runs
/// hit filtering / MAPQ to mirror the C single-pass output.
///
/// # Parameters
/// * `records` - one record per index part, in part order
/// * `rid_shifts` - additive offsets applied to `rid` of part `i` (must align with `records`)
/// * `opt` - mapping options (preset thresholds, flags)
/// * `idx_k` - k-mer size of the merged index (any part — all use the same k)
/// * `qlen` - query length in bases
pub fn merge_split_query_records(
    records: &[SplitQueryRecord],
    rid_shifts: &[u32],
    opt: &MapOpt,
    idx_k: i32,
    qlen: i32,
) -> SplitQueryRecord {
    let mut merged = SplitQueryRecord::default();
    let rep_len_for_mapq = records.iter().map(|r| r.rep_len).max().unwrap_or(0);
    merged.rep_len = 0;
    merged.frag_gap = records.first().map(|r| r.frag_gap).unwrap_or(0);
    for (part, record) in records.iter().enumerate() {
        let rid_shift = rid_shifts.get(part).copied().unwrap_or(0) as i32;
        for r in &record.regs {
            let mut r = r.clone();
            r.rid += rid_shift;
            if let Some(ref mut p) = r.extra {
                p.dp_max2 = 0;
            }
            r.subsc = 0;
            r.n_sub = 0;
            merged.regs.push(r);
        }
    }

    if !(opt.flag.contains(MapFlags::SR)) && qlen >= opt.rank_min_len {
        hit::update_dp_max(qlen, &mut merged.regs, opt.rank_frac, opt.a, opt.b);
        for r in &mut merged.regs {
            if let Some(ref mut p) = r.extra {
                p.dp_max2 = 0;
            }
        }
    }
    hit::hit_sort(&mut merged.regs, opt.alt_drop);
    hit::set_parent(
        opt.mask_level,
        opt.mask_len,
        &mut merged.regs,
        opt.a * 2 + opt.b,
        opt.flag.contains(MapFlags::HARD_MLEVEL),
        opt.alt_drop,
    );
    if !opt.flag.contains(MapFlags::ALL_CHAINS) {
        hit::select_sub(opt.pri_ratio, idx_k * 2, opt.best_n, &mut merged.regs);
        hit::set_sam_pri(&mut merged.regs);
    }
    hit::set_mapq(
        &mut merged.regs,
        opt.min_chain_score,
        opt.a,
        rep_len_for_mapq,
        opt.flag.intersects(MapFlags::SR | MapFlags::SR_RNA),
        opt.flag.contains(MapFlags::SPLICE),
    );
    hit::sync_regs(&mut merged.regs);
    merged.n_reg = merged.regs.len() as i32;
    merged
}

fn write_align_reg<W: Write>(writer: &mut W, r: &AlignReg, with_cigar: bool) -> io::Result<()> {
    for v in [
        r.id, r.cnt, r.rid, r.score, r.qs, r.qe, r.rs, r.re, r.parent, r.subsc, r.as_, r.mlen,
        r.blen, r.n_sub, r.score0,
    ] {
        writer.write_all(&v.to_le_bytes())?;
    }
    writer.write_all(&[
        r.mapq,
        r.split,
        r.rev as u8,
        r.inv as u8,
        r.sam_pri as u8,
        r.proper_frag as u8,
        r.pe_thru as u8,
        r.seg_split as u8,
        r.seg_id,
        r.split_inv as u8,
        r.is_alt as u8,
        r.strand_retained as u8,
        r.is_spliced as u8,
    ])?;
    writer.write_all(&r.hash.to_le_bytes())?;
    writer.write_all(&r.div.to_le_bytes())?;
    if with_cigar {
        write_extra(writer, r.extra.as_deref())?;
    }
    Ok(())
}

fn read_align_reg<R: Read>(reader: &mut R, with_cigar: bool) -> io::Result<AlignReg> {
    let mut r = AlignReg::default();
    r.id = read_i32(reader)?;
    r.cnt = read_i32(reader)?;
    r.rid = read_i32(reader)?;
    r.score = read_i32(reader)?;
    r.qs = read_i32(reader)?;
    r.qe = read_i32(reader)?;
    r.rs = read_i32(reader)?;
    r.re = read_i32(reader)?;
    r.parent = read_i32(reader)?;
    r.subsc = read_i32(reader)?;
    r.as_ = read_i32(reader)?;
    r.mlen = read_i32(reader)?;
    r.blen = read_i32(reader)?;
    r.n_sub = read_i32(reader)?;
    r.score0 = read_i32(reader)?;
    let mut flags = [0u8; 13];
    reader.read_exact(&mut flags)?;
    r.mapq = flags[0];
    r.split = flags[1];
    r.rev = flags[2] != 0;
    r.inv = flags[3] != 0;
    r.sam_pri = flags[4] != 0;
    r.proper_frag = flags[5] != 0;
    r.pe_thru = flags[6] != 0;
    r.seg_split = flags[7] != 0;
    r.seg_id = flags[8];
    r.split_inv = flags[9] != 0;
    r.is_alt = flags[10] != 0;
    r.strand_retained = flags[11] != 0;
    r.is_spliced = flags[12] != 0;
    r.hash = read_u32(reader)?;
    r.div = read_f32(reader)?;
    if with_cigar {
        r.extra = read_extra(reader)?.map(Box::new);
    }
    Ok(r)
}

fn write_extra<W: Write>(writer: &mut W, extra: Option<&AlignExtra>) -> io::Result<()> {
    let Some(extra) = extra else {
        writer.write_all(&[0])?;
        return Ok(());
    };
    writer.write_all(&[1])?;
    for v in [extra.dp_score, extra.dp_max, extra.dp_max2, extra.dp_max0] {
        writer.write_all(&v.to_le_bytes())?;
    }
    writer.write_all(&extra.n_ambi.to_le_bytes())?;
    writer.write_all(&[extra.trans_strand])?;
    writer.write_all(&(extra.cigar.0.len() as u32).to_le_bytes())?;
    for &c in &extra.cigar.0 {
        writer.write_all(&c.to_le_bytes())?;
    }
    Ok(())
}

fn read_extra<R: Read>(reader: &mut R) -> io::Result<Option<AlignExtra>> {
    let mut present = [0u8; 1];
    reader.read_exact(&mut present)?;
    if present[0] == 0 {
        return Ok(None);
    }
    let dp_score = read_i32(reader)?;
    let dp_max = read_i32(reader)?;
    let dp_max2 = read_i32(reader)?;
    let dp_max0 = read_i32(reader)?;
    let n_ambi = read_u32(reader)?;
    let mut trans_strand = [0u8; 1];
    reader.read_exact(&mut trans_strand)?;
    let n_cigar = read_u32(reader)?;
    let mut cigar = Vec::with_capacity(n_cigar as usize);
    for _ in 0..n_cigar {
        cigar.push(read_u32(reader)?);
    }
    Ok(Some(AlignExtra {
        dp_score,
        dp_max,
        dp_max2,
        dp_max0,
        n_ambi,
        trans_strand: trans_strand[0],
        cigar: Cigar(cigar),
    }))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::IdxFlags;

    #[test]
    fn test_split_header_roundtrip() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"TGCATGCA"];
        let names = vec!["seq1", "seq2"];
        let mi = MmIdx::build_from_str(5, 4, false, 10, &seqs, Some(&names)).unwrap();

        let mut buf = Vec::new();
        write_split_header(&mut buf, &mi).unwrap();
        assert_eq!(&buf[..4], &(4u32.to_le_bytes()));
        assert_eq!(&buf[4..8], &(2u32.to_le_bytes()));

        let header = read_split_header(&mut &buf[..]).unwrap();
        assert_eq!(header.k, 4);
        assert_eq!(header.seqs.len(), 2);
        assert_eq!(header.seqs[0].name, "seq1");
        assert_eq!(header.seqs[0].len, 8);
        assert_eq!(header.seqs[1].name, "seq2");
        assert_eq!(header.seqs[1].offset, 8);
    }

    #[test]
    fn test_split_tmp_files_roundtrip_and_remove() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"TGCATGCATGCA"];
        let names = vec!["seq1", "seq2"];
        let mi = MmIdx::build_from_str(5, 4, false, 10, &seqs, Some(&names)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("split");
        let prefix = prefix.to_str().unwrap();

        let file = create_split_tmp(prefix, 0, &mi).unwrap();
        drop(file);
        let headers = read_split_headers(prefix, 1).unwrap();
        assert_eq!(headers[0].seqs.len(), 2);

        remove_split_tmps(prefix, 1).unwrap();
        assert!(!split_tmp_path(prefix, 0).exists());
    }

    #[test]
    fn test_split_header_from_build_parts() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"TGCATGCATGCA"];
        let names = vec!["seq1", "seq2"];
        let mi = MmIdx::build_from_str(5, 4, false, 10, &seqs, Some(&names)).unwrap();
        assert!(!mi.flag.contains(IdxFlags::NO_SEQ));
        let mut buf = Vec::new();
        write_split_header(&mut buf, &mi).unwrap();
        let header = read_split_header(&mut &buf[..]).unwrap();
        assert_eq!(header.k, mi.k as u32);
    }

    #[test]
    fn test_split_query_record_roundtrip() {
        let mut r = AlignReg::default();
        r.id = 3;
        r.rid = 2;
        r.qs = 5;
        r.qe = 42;
        r.rs = 100;
        r.re = 137;
        r.mapq = 60;
        r.rev = true;
        r.hash = 12345;
        r.div = 0.125;
        r.extra = Some(Box::new(AlignExtra {
            dp_score: 37,
            dp_max: 40,
            dp_max2: 3,
            dp_max0: 40,
            n_ambi: 1,
            trans_strand: 2,
            cigar: Cigar(vec![(10u32 << 4) | 0, (2u32 << 4) | 1, (25u32 << 4) | 0]),
        }));
        let record = SplitQueryRecord {
            n_reg: 1,
            rep_len: 7,
            frag_gap: 300,
            regs: vec![r],
        };

        let mut buf = Vec::new();
        write_split_query_record(&mut buf, &record, true).unwrap();
        let decoded = read_split_query_record(&mut &buf[..], true).unwrap();
        assert_eq!(decoded.n_reg, 1);
        assert_eq!(decoded.rep_len, 7);
        assert_eq!(decoded.frag_gap, 300);
        assert_eq!(decoded.regs[0].rid, 2);
        assert!(decoded.regs[0].rev);
        assert_eq!(decoded.regs[0].extra.as_ref().unwrap().trans_strand, 2);
        assert_eq!(decoded.regs[0].extra.as_ref().unwrap().cigar.0.len(), 3);
    }

    #[test]
    fn test_merge_split_query_records_shifts_rids_and_recomputes() {
        let mut r0 = AlignReg::default();
        r0.id = 0;
        r0.parent = 0;
        r0.rid = 0;
        r0.score = 90;
        r0.score0 = 90;
        r0.cnt = 10;
        r0.qs = 0;
        r0.qe = 90;
        r0.rs = 10;
        r0.re = 100;
        r0.mlen = 90;
        r0.blen = 90;
        r0.hash = 10;

        let mut r1 = r0.clone();
        r1.rid = 0;
        r1.score = 80;
        r1.score0 = 80;
        r1.rs = 20;
        r1.re = 100;
        r1.hash = 9;

        let records = vec![
            SplitQueryRecord {
                n_reg: 1,
                rep_len: 3,
                frag_gap: 100,
                regs: vec![r0],
            },
            SplitQueryRecord {
                n_reg: 1,
                rep_len: 7,
                frag_gap: 200,
                regs: vec![r1],
            },
        ];
        let opt = crate::options::MapOpt::default();
        let merged = merge_split_query_records(&records, &[0, 5], &opt, 15, 100);

        assert_eq!(merged.n_reg, 2);
        assert_eq!(merged.rep_len, 0);
        assert!(merged.regs.iter().any(|r| r.rid == 5));
        assert_eq!(merged.regs[0].id, 0);
        assert!(merged.regs[0].mapq <= 60);
    }
}
