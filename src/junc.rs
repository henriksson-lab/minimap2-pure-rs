//! BED/BED12 junction file parsing for splice-aware alignment.
//!
//! Reads junction annotations from BED/BED12 format files and stores them
//! in the index for use during splice-aware alignment.

use crate::index::MmIdx;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read};

/// A junction interval on a reference sequence.
#[derive(Clone, Debug)]
pub struct JuncIntv {
    pub st: i32,
    pub en: i32,
    pub strand: i8, // 1 = +, 2 = -, 0 = unknown
}

/// Junction database: stores junctions per reference sequence.
#[derive(Clone, Debug, Default)]
pub struct JuncDb {
    /// Junctions per reference, sorted by start position.
    pub juncs: Vec<Vec<JuncIntv>>,
}

impl JuncDb {
    /// Create a junction byte-array for a reference region.
    /// Returns a byte per reference position: 0 = no junction,
    /// 1 = donor (GT), 2 = acceptor (AG), based on strand.
    pub fn get_junc_array(&self, rid: u32, st: i32, en: i32, rev: bool) -> Vec<u8> {
        let len = (en - st) as usize;
        let mut junc = vec![0u8; len];
        if (rid as usize) >= self.juncs.len() {
            return junc;
        }
        for j in &self.juncs[rid as usize] {
            // Mark donor and acceptor sites
            let donor = j.st - st;
            let acceptor = j.en - st - 1;
            if donor >= 0 && (donor as usize) < len {
                junc[donor as usize] |= if rev { 2 } else { 1 };
            }
            if acceptor >= 0 && (acceptor as usize) < len {
                junc[acceptor as usize] |= if rev { 1 } else { 2 };
            }
        }
        junc
    }
}

fn open_maybe_gzip(path: &str) -> io::Result<Box<dyn BufRead>> {
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 2];
    let n = file.read(&mut prefix)?;
    let head = Cursor::new(prefix[..n].to_vec());
    let reader: Box<dyn Read> = if n == 2 && prefix == [0x1f, 0x8b] {
        Box::new(GzDecoder::new(head.chain(file)))
    } else {
        Box::new(head.chain(file))
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn parse_i32(s: Option<&&str>) -> Option<i32> {
    s?.parse().ok()
}

fn parse_strand(s: Option<&&str>) -> i8 {
    match s.copied() {
        Some("+") => 1,
        Some("-") => 2,
        _ => 0,
    }
}

fn parse_comma_i32s(s: &str) -> Vec<i32> {
    s.split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// Read splice junctions from a BED/BED12 file and annotate the index.
///
/// BED12 records contribute introns inferred from gaps between blocks. Records
/// with fewer than 12 fields are accepted as direct intervals, matching C
/// minimap2's fallback behavior in mm_idx_bed_read_core().
pub fn read_junc_bed(mi: &mut MmIdx, path: &str) -> io::Result<usize> {
    let reader = open_maybe_gzip(path)?;
    let mut db = JuncDb {
        juncs: vec![Vec::new(); mi.seqs.len()],
    };

    mi.index_names();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("track")
            || line.starts_with("browser")
        {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }

        let chrom = fields[0];
        let Some(chrom_start) = parse_i32(fields.get(1)) else {
            continue;
        };
        let Some(chrom_end) = parse_i32(fields.get(2)) else {
            continue;
        };
        if chrom_start < 0 || chrom_end <= chrom_start {
            continue;
        }

        let Some(rid) = mi.name2id(chrom) else {
            continue;
        };
        let strand = parse_strand(fields.get(5));

        if fields.len() >= 12 {
            if let Some(block_count) = fields.get(9).and_then(|s| s.parse::<usize>().ok()) {
                let block_sizes = parse_comma_i32s(fields[10]);
                let block_starts = parse_comma_i32s(fields[11]);
                if block_count >= 2
                    && block_sizes.len() >= block_count
                    && block_starts.len() >= block_count
                {
                    for i in 1..block_count {
                        let junc_start = chrom_start + block_starts[i - 1] + block_sizes[i - 1];
                        let junc_end = chrom_start + block_starts[i];
                        if junc_end > junc_start {
                            db.juncs[rid as usize].push(JuncIntv {
                                st: junc_start,
                                en: junc_end,
                                strand,
                            });
                        }
                    }
                }
                continue;
            }
        }

        db.juncs[rid as usize].push(JuncIntv {
            st: chrom_start,
            en: chrom_end,
            strand,
        });
    }

    for juncs in &mut db.juncs {
        juncs.sort_by_key(|j| (j.st, j.en));
        juncs.dedup_by(|a, b| a.st == b.st && a.en == b.en && a.strand == b.strand);
    }
    let n_junc = db.juncs.iter().map(Vec::len).sum();

    if n_junc > 0 {
        log::info!("read {} junctions from {}", n_junc, path);
    }
    mi.junc_db = Some(db);
    Ok(n_junc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn build_test_idx() -> MmIdx {
        let mut ref_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(ref_file, ">chr1").unwrap();
        writeln!(
            ref_file,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        ref_file.flush().unwrap();

        MmIdx::build_from_file(
            ref_file.path().to_str().unwrap(),
            10,
            15,
            14,
            crate::flags::IdxFlags::empty(),
            50_000_000,
            u64::MAX,
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn test_read_junc_bed12() {
        let mut mi = build_test_idx();

        let mut bed = tempfile::NamedTempFile::new().unwrap();
        writeln!(bed, "chr1\t0\t50\tjunc1\t100\t+\t0\t50\t0\t2\t10,10\t0,30").unwrap();
        bed.flush().unwrap();

        let n = read_junc_bed(&mut mi, bed.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 1);
        let j = &mi.junc_db.as_ref().unwrap().juncs[0][0];
        assert_eq!((j.st, j.en, j.strand), (10, 30, 1));
    }

    #[test]
    fn test_read_junc_bed6_interval() {
        let mut mi = build_test_idx();

        let mut bed = tempfile::NamedTempFile::new().unwrap();
        writeln!(bed, "chr1\t12\t34\tjunc1\t100\t-").unwrap();
        bed.flush().unwrap();

        let n = read_junc_bed(&mut mi, bed.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 1);
        let j = &mi.junc_db.as_ref().unwrap().juncs[0][0];
        assert_eq!((j.st, j.en, j.strand), (12, 34, 2));
    }

    #[test]
    fn test_read_junc_bed12_gzip() {
        let mut mi = build_test_idx();

        let gz = tempfile::NamedTempFile::new().unwrap();
        let mut encoder = GzEncoder::new(gz.reopen().unwrap(), Compression::default());
        writeln!(
            encoder,
            "chr1\t0\t50\tjunc1\t100\t+\t0\t50\t0\t2\t10,10\t0,30"
        )
        .unwrap();
        encoder.finish().unwrap();

        let n = read_junc_bed(&mut mi, gz.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 1);
        let j = &mi.junc_db.as_ref().unwrap().juncs[0][0];
        assert_eq!((j.st, j.en, j.strand), (10, 30, 1));
    }
}
