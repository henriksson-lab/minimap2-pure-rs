use std::io::{self, Write, BufWriter};
use rayon::prelude::*;
use crate::bseq::{BseqFile, BseqRecord};
use crate::flags::MapFlags;
use crate::format::sam;
use crate::index::MmIdx;
use crate::map;
use crate::options::MapOpt;
use crate::pe;

/// Map a FASTA/FASTQ file against the index and write PAF output to stdout.
pub fn map_file_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Set up rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let batch = fp.read_batch(opt.mini_batch_size, false)?;
        if batch.is_empty() {
            break;
        }

        // Map in parallel
        let results: Vec<_> = pool.install(|| {
            batch.par_iter().map(|rec| {
                let result = map::map_query(mi, opt, &rec.name, &rec.seq);
                map::format_paf(mi, opt, &rec.name, &rec.seq, &result)
            }).collect()
        });

        // Write results sequentially
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Map a FASTA/FASTQ file and write SAM output to stdout.
pub fn map_file_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    // Write SAM header
    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let with_qual = true;
    loop {
        let batch = fp.read_batch(opt.mini_batch_size, with_qual)?;
        if batch.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            batch.par_iter().map(|rec| {
                let result = map::map_query(mi, opt, &rec.name, &rec.seq);
                let mut lines = Vec::new();
                if result.regs.is_empty() {
                    lines.push(sam::write_sam_record(
                        mi, &rec.name, &rec.seq, &rec.qual,
                        None, 0, &[], opt.flag, result.rep_len,
                    ));
                } else {
                    for (i, r) in result.regs.iter().enumerate() {
                        if i > 0 && opt.flag.contains(MapFlags::NO_PRINT_2ND) {
                            break;
                        }
                        lines.push(sam::write_sam_record(
                            mi, &rec.name, &rec.seq, &rec.qual,
                            Some(r), result.regs.len(), &result.regs, opt.flag, result.rep_len,
                        ));
                    }
                }
                lines
            }).collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Map interleaved paired-end reads from a single file.
///
/// Reads are consumed in pairs (R1, R2, R1, R2, ...).
pub fn map_file_interleaved_pe_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        // Read pairs from interleaved file
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp.read_record()?;
            let r2 = fp.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size { break; }
                }
                _ => break,
            }
        }
        if pairs.is_empty() { break; }

        let results: Vec<_> = pool.install(|| {
            pairs.par_iter().map(|(rec1, rec2)| {
                let mut res1 = map::map_query(mi, opt, &rec1.name, &rec1.seq);
                let mut res2 = map::map_query(mi, opt, &rec2.name, &rec2.seq);

                if !res1.regs.is_empty() && !res2.regs.is_empty()
                    && res1.regs[0].extra.is_some() && res2.regs[0].extra.is_some()
                {
                    let qlens = [rec1.l_seq as i32, rec2.l_seq as i32];
                    let mut n_regs = [res1.regs.len(), res2.regs.len()];
                    let mut regs_pair = [
                        std::mem::take(&mut res1.regs),
                        std::mem::take(&mut res2.regs),
                    ];
                    pe::pair(
                        opt.max_frag_len.max(opt.max_gap),
                        opt.pe_bonus, opt.a * 2 + opt.b, opt.a,
                        &qlens, &mut n_regs, &mut regs_pair,
                    );
                    res1.regs = regs_pair[0].clone();
                    res2.regs = regs_pair[1].clone();
                }

                let mut lines = Vec::new();
                // Use base name without /1 /2 suffix
                let base_name = strip_pe_suffix(&rec1.name);
                format_pe_sam_records(mi, opt, &BseqRecord {
                    name: base_name.to_string(), ..rec1.clone()
                }, &res1, &res2, true, &mut lines);
                format_pe_sam_records(mi, opt, &BseqRecord {
                    name: base_name.to_string(), ..rec2.clone()
                }, &res2, &res1, false, &mut lines);
                lines
            }).collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Strip /1 or /2 suffix from read name for PE output.
fn strip_pe_suffix(name: &str) -> &str {
    if name.len() >= 2 {
        let bytes = name.as_bytes();
        if bytes[bytes.len() - 2] == b'/' && (bytes[bytes.len() - 1] == b'1' || bytes[bytes.len() - 1] == b'2') {
            return &name[..name.len() - 2];
        }
    }
    name
}

/// Map paired-end FASTQ files against the index and write SAM output.
///
/// Reads pairs from two files simultaneously. Each pair is mapped independently,
/// then paired to set proper-pair flags and adjust MAPQ.
pub fn map_file_pe_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        // Read pairs
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp1.read_record()?;
            let r2 = fp2.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() { break; }

        // Map pairs in parallel
        let results: Vec<_> = pool.install(|| {
            pairs.par_iter().map(|(rec1, rec2)| {
                let mut res1 = map::map_query(mi, opt, &rec1.name, &rec1.seq);
                let mut res2 = map::map_query(mi, opt, &rec2.name, &rec2.seq);

                // Pair the alignments
                if !res1.regs.is_empty() && !res2.regs.is_empty()
                    && res1.regs[0].extra.is_some() && res2.regs[0].extra.is_some()
                {
                    let qlens = [rec1.l_seq as i32, rec2.l_seq as i32];
                    let mut n_regs = [res1.regs.len(), res2.regs.len()];
                    let mut regs_pair = [
                        std::mem::take(&mut res1.regs),
                        std::mem::take(&mut res2.regs),
                    ];
                    pe::pair(
                        opt.max_frag_len.max(opt.max_gap),
                        opt.pe_bonus,
                        opt.a * 2 + opt.b,
                        opt.a,
                        &qlens,
                        &mut n_regs,
                        &mut regs_pair,
                    );
                    res1.regs = regs_pair[0].clone();
                    res2.regs = regs_pair[1].clone();
                }

                // Format SAM records
                let mut lines = Vec::new();
                // Read 1
                format_pe_sam_records(mi, opt, rec1, &res1, &res2, true, &mut lines);
                // Read 2
                format_pe_sam_records(mi, opt, rec2, &res2, &res1, false, &mut lines);
                lines
            }).collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Format SAM records for one read of a pair.
fn format_pe_sam_records(
    mi: &MmIdx,
    opt: &MapOpt,
    rec: &BseqRecord,
    result: &map::MapResult,
    mate_result: &map::MapResult,
    is_read1: bool,
    lines: &mut Vec<String>,
) {
    

    if result.regs.is_empty() {
        let mut line = sam::write_sam_record(
            mi, &rec.name, &rec.seq, &rec.qual,
            None, 0, &[], opt.flag, result.rep_len,
        );
        // Add PE flags
        add_pe_flags(&mut line, is_read1, true, mate_result);
        lines.push(line);
    } else {
        for (i, r) in result.regs.iter().enumerate() {
            if i > 0 && opt.flag.contains(MapFlags::NO_PRINT_2ND) { break; }
            let mut line = sam::write_sam_record(
                mi, &rec.name, &rec.seq, &rec.qual,
                Some(r), result.regs.len(), &result.regs, opt.flag, result.rep_len,
            );
            add_pe_flags(&mut line, is_read1, false, mate_result);
            lines.push(line);
        }
    }
}

/// Modify SAM flag field to include PE flags.
fn add_pe_flags(line: &mut String, is_read1: bool, unmapped: bool, mate_result: &map::MapResult) {
    // Parse existing flag from SAM line and modify it
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 2 { return; }
    let mut flag: u16 = fields[1].parse().unwrap_or(0);

    flag |= 0x1; // paired
    if is_read1 { flag |= 0x40; } else { flag |= 0x80; }

    // Mate unmapped
    if mate_result.regs.is_empty() {
        flag |= 0x8;
    } else {
        // Mate reverse strand
        if mate_result.regs[0].rev { flag |= 0x20; }
        // Proper pair
        if !unmapped && mate_result.regs[0].proper_frag {
            flag |= 0x2;
        }
    }

    // Rebuild line with updated flag
    let mut new_line = String::with_capacity(line.len());
    new_line.push_str(fields[0]); // QNAME
    new_line.push('\t');
    new_line.push_str(&flag.to_string());
    for field in &fields[2..] {
        new_line.push('\t');
        new_line.push_str(field);
    }
    *line = new_line;
}

/// Map paired-end files, writing PAF output.
pub fn map_file_pe_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            match (fp1.read_record()?, fp2.read_record()?) {
                (Some(r1), Some(r2)) => {
                    total_len += r1.l_seq as i64 + r2.l_seq as i64;
                    pairs.push((r1, r2));
                    if total_len >= opt.mini_batch_size { break; }
                }
                _ => break,
            }
        }
        if pairs.is_empty() { break; }

        let results: Vec<_> = pool.install(|| {
            pairs.par_iter().map(|(rec1, rec2)| {
                let res1 = map::map_query(mi, opt, &rec1.name, &rec1.seq);
                let res2 = map::map_query(mi, opt, &rec2.name, &rec2.seq);
                let mut lines = map::format_paf(mi, opt, &rec1.name, &rec1.seq, &res1);
                lines.extend(map::format_paf(mi, opt, &rec2.name, &rec2.seq, &res2));
                lines
            }).collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    #[test]
    fn test_map_file_paf_basic() {
        // Create a temp reference and query
        let mut ref_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(ref_file, ">ref1").unwrap();
        writeln!(ref_file, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        ref_file.flush().unwrap();

        let mi = MmIdx::build_from_file(
            ref_file.path().to_str().unwrap(),
            10, 15, 14,
            crate::flags::IdxFlags::empty(),
            50_000_000, u64::MAX,
        ).unwrap().unwrap();

        let mut query_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(query_file, ">read1").unwrap();
        writeln!(query_file, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        query_file.flush().unwrap();

        let mut opt = MapOpt::default();
        crate::options::mapopt_update(&mut opt, &mi);
        // Test the core mapping
        let result = map::map_query(&mi, &opt, "read1",
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        assert!(!result.regs.is_empty());
    }
}
