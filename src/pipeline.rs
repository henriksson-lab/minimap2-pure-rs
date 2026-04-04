use std::io::{self, Write, BufWriter};
use rayon::prelude::*;
use crate::bseq::BseqFile;
use crate::flags::MapFlags;
use crate::format::sam;
use crate::index::MmIdx;
use crate::map;
use crate::options::MapOpt;

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
                let lines = map::format_paf(mi, opt, &rec.name, rec.l_seq as i32, &result);
                lines
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
        // Compute mid_occ from index
        opt.mid_occ = mi.cal_max_occ(opt.mid_occ_frac);
        if opt.mid_occ < opt.min_mid_occ { opt.mid_occ = opt.min_mid_occ; }
        // Test the core mapping
        let result = map::map_query(&mi, &opt, "read1",
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        assert!(!result.regs.is_empty());
    }
}
