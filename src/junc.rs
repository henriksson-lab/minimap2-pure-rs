//! BED12 junction file parsing for splice-aware alignment.
//!
//! Reads junction annotations from BED12 format files and stores them
//! in the index for use during splice-aware alignment.

use crate::index::MmIdx;
use std::io::{self, BufRead};

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

/// Read splice junctions from a BED12 file and annotate the index.
///
/// BED12 format: chrom, start, end, name, score, strand, thickStart, thickEnd,
/// rgb, blockCount, blockSizes, blockStarts
///
/// Junctions are inferred from the gaps between blocks.
pub fn read_junc_bed(mi: &mut MmIdx, path: &str) -> io::Result<usize> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut n_junc = 0usize;
    let mut db = JuncDb {
        juncs: vec![Vec::new(); mi.seqs.len()],
    };

    // Build name index if not present
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
        if fields.len() < 12 {
            continue;
        }

        let chrom = fields[0];
        let chrom_start: i32 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let _strand = match fields[5] {
            "+" => 1i8,
            "-" => 2i8,
            _ => 0i8,
        };
        let block_count: usize = match fields[9].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if block_count < 2 {
            continue;
        }

        let block_sizes: Vec<i32> = fields[10]
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();
        let block_starts: Vec<i32> = fields[11]
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();

        if block_sizes.len() < block_count || block_starts.len() < block_count {
            continue;
        }

        let rid = match mi.name2id(chrom) {
            Some(id) => id,
            None => continue,
        };

        let strand_val = match fields[5] {
            "+" => 1i8,
            "-" => 2i8,
            _ => 0i8,
        };

        // Extract junctions from gaps between blocks
        for i in 1..block_count {
            let junc_start = chrom_start + block_starts[i - 1] + block_sizes[i - 1];
            let junc_end = chrom_start + block_starts[i];
            if junc_end > junc_start {
                db.juncs[rid as usize].push(JuncIntv {
                    st: junc_start,
                    en: junc_end,
                    strand: strand_val,
                });
                n_junc += 1;
            }
        }
    }

    // Sort junctions by start position
    for juncs in &mut db.juncs {
        juncs.sort_by_key(|j| j.st);
    }

    if n_junc > 0 {
        log::info!("read {} junctions from {}", n_junc, path);
    }
    mi.junc_db = Some(db);
    Ok(n_junc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_junc_bed() {
        let mut ref_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(ref_file, ">chr1").unwrap();
        writeln!(
            ref_file,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        ref_file.flush().unwrap();

        let mut mi = MmIdx::build_from_file(
            ref_file.path().to_str().unwrap(),
            10,
            15,
            14,
            crate::flags::IdxFlags::empty(),
            50_000_000,
            u64::MAX,
        )
        .unwrap()
        .unwrap();

        // Create BED12 file
        let mut bed = tempfile::NamedTempFile::new().unwrap();
        writeln!(bed, "chr1\t0\t50\tjunc1\t100\t+\t0\t50\t0\t2\t10,10\t0,30").unwrap();
        bed.flush().unwrap();

        let n = read_junc_bed(&mut mi, bed.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 1); // one junction gap between blocks
    }
}
