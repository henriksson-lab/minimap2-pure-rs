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

pub const JUNC_ANNO: u16 = 0x1;
pub const JUNC_MISC: u16 = 0x2;

/// Junction jump edge used by the post-MAPQ jump rescue path (`mi->J` in C).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JumpEdge {
    pub off: i32,
    pub off2: i32,
    pub cnt: i32,
    pub strand: i16, // +1 / -1 / 0
    pub flag: u16,
}

/// Separate jump database for `-j/--pass1` splice rescue. This is distinct
/// from `JuncDb`, which backs splice scoring for `--junc-bed`.
#[derive(Clone, Debug, Default)]
pub struct JumpDb {
    pub jumps: Vec<Vec<JumpEdge>>,
}

impl JuncDb {
    /// Build a per-position junction bitmask over a reference window.
    ///
    /// Each byte encodes donor/acceptor bits for the position: forward-strand
    /// donor = 1, acceptor = 2; reverse-strand donor = 8, acceptor = 4.
    /// Matches `mm_idx_bed_junc()`. Annotations whose endpoints fall outside
    /// `[st, en)` are skipped.
    ///
    /// # Parameters
    /// * `rid` - 0-based reference id; out-of-range returns an all-zero buffer
    /// * `st` - inclusive start of the reference window
    /// * `en` - exclusive end of the reference window
    /// * `_rev` - unused (junction bits stay in genomic-strand space; DP handles read orientation)
    pub fn get_junc_array(&self, rid: u32, st: i32, en: i32, _rev: bool) -> Vec<u8> {
        let len = (en - st) as usize;
        let mut junc = vec![0u8; len];
        if (rid as usize) >= self.juncs.len() {
            return junc;
        }
        for j in &self.juncs[rid as usize] {
            // Match minimap2/index.c:mm_idx_bed_junc():
            // BED annotations are stored in genomic-strand space and do not
            // depend on the current read/splice orientation. The downstream
            // DP interprets these fixed bits together with SPLICE_FOR/REV and
            // REV_CIGAR.
            if j.strand == 0 || j.st < st || j.en > en {
                continue;
            }
            let donor = j.st - st;
            let acceptor = j.en - st - 1;
            if donor >= 0 && (donor as usize) < len {
                junc[donor as usize] |= if j.strand == 2 { 8 } else { 1 };
            }
            if acceptor >= 0 && (acceptor as usize) < len {
                junc[acceptor as usize] |= if j.strand == 2 { 4 } else { 2 };
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

fn read_bed_intervals(
    mi: &mut MmIdx,
    path: &str,
    read_junc: bool,
    min_sc: i32,
) -> io::Result<Vec<Vec<JuncIntv>>> {
    let reader = open_maybe_gzip(path)?;
    let mut per_ref = vec![Vec::new(); mi.seqs.len()];

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
        if min_sc >= 0 {
            let score = fields
                .get(4)
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            if score < min_sc {
                continue;
            }
        }

        let Some(rid) = mi.name2id(chrom) else {
            continue;
        };
        let strand = parse_strand(fields.get(5));

        if read_junc && fields.len() >= 12 {
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
                            per_ref[rid as usize].push(JuncIntv {
                                st: junc_start,
                                en: junc_end,
                                strand,
                            });
                        }
                    }
                    continue;
                }
            }
        }

        per_ref[rid as usize].push(JuncIntv {
            st: chrom_start,
            en: chrom_end,
            strand,
        });
    }

    for juncs in &mut per_ref {
        juncs.sort_by_key(|j| (j.st, j.en, j.strand));
        juncs.dedup_by(|a, b| a.st == b.st && a.en == b.en && a.strand == b.strand);
    }
    Ok(per_ref)
}

/// Read splice junctions from a BED/BED12 file and attach them to the index.
///
/// BED12 records contribute introns inferred from gaps between blocks. Records
/// with fewer than 12 fields are accepted as direct intervals, matching C
/// minimap2's fallback in `mm_idx_bed_read_core()`. Returns the number of
/// junctions loaded.
///
/// # Parameters
/// * `mi` - index to annotate; `mi.junc_db` is replaced
/// * `path` - path to a plain or gzipped BED file
pub fn read_junc_bed(mi: &mut MmIdx, path: &str) -> io::Result<usize> {
    let db = JuncDb {
        juncs: read_bed_intervals(mi, path, true, -1)?,
    };
    let n_junc = db.juncs.iter().map(Vec::len).sum();

    if n_junc > 0 {
        log::info!("read {} junctions from {}", n_junc, path);
    }
    mi.junc_db = Some(db);
    Ok(n_junc)
}

fn merge_jump_edges(edges: &mut Vec<JumpEdge>) {
    edges.sort_by_key(|e| (e.off, e.off2));
    let mut merged: Vec<JumpEdge> = Vec::with_capacity(edges.len());
    for edge in edges.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.off == edge.off && last.off2 == edge.off2 {
                last.cnt += edge.cnt;
                last.flag |= edge.flag;
                if last.strand == 0 {
                    last.strand = edge.strand;
                }
                continue;
            }
        }
        merged.push(edge);
    }
    *edges = merged;
}

/// Read splice junctions and append bidirectional jump edges for `-j/--pass1` rescue.
///
/// For each BED12 intron `(st, en)` two `JumpEdge`s are pushed (`off→off2` and
/// reverse), then merged across calls. Returns the total edge count.
///
/// # Parameters
/// * `mi` - index to annotate; reuses `mi.jump_db` if present
/// * `path` - path to a plain or gzipped BED file
/// * `flag` - `JUNC_ANNO` or `JUNC_MISC`, merged via bitwise OR on duplicates
/// * `min_sc` - minimum BED score to accept; pass `-1` to disable the cutoff
pub fn read_jump_bed(mi: &mut MmIdx, path: &str, flag: u16, min_sc: i32) -> io::Result<usize> {
    let intervals = read_bed_intervals(mi, path, true, min_sc)?;
    let mut db = mi.jump_db.take().unwrap_or_else(|| JumpDb {
        jumps: vec![Vec::new(); mi.seqs.len()],
    });
    if db.jumps.len() < mi.seqs.len() {
        db.jumps.resize_with(mi.seqs.len(), Vec::new);
    }

    for (rid, juncs) in intervals.into_iter().enumerate() {
        for j in juncs {
            let strand = match j.strand {
                1 => 1,
                2 => -1,
                _ => 0,
            };
            db.jumps[rid].push(JumpEdge {
                off: j.st,
                off2: j.en,
                cnt: 1,
                strand,
                flag,
            });
            db.jumps[rid].push(JumpEdge {
                off: j.en,
                off2: j.st,
                cnt: 1,
                strand,
                flag,
            });
        }
        merge_jump_edges(&mut db.jumps[rid]);
    }

    let n_jump: usize = db.jumps.iter().map(Vec::len).sum();
    mi.jump_db = Some(db);
    Ok(n_jump)
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

    #[test]
    fn test_get_junc_array_preserves_minus_strand_bits() {
        let db = JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 12,
                en: 34,
                strand: 2,
            }]],
        };
        let junc = db.get_junc_array(0, 10, 40, false);
        assert_eq!(junc[2], 8);
        assert_eq!(junc[23], 4);
    }

    #[test]
    fn test_get_junc_array_skips_unknown_strand_bits() {
        let db = JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 12,
                en: 34,
                strand: 0,
            }]],
        };
        let junc = db.get_junc_array(0, 10, 40, false);
        assert!(junc.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_get_junc_array_skips_partial_window_overlap() {
        let db = JuncDb {
            juncs: vec![vec![JuncIntv {
                st: 12,
                en: 34,
                strand: 1,
            }]],
        };
        let left_cut = db.get_junc_array(0, 13, 40, false);
        let right_cut = db.get_junc_array(0, 10, 33, false);
        assert!(left_cut.iter().all(|&b| b == 0));
        assert!(right_cut.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_read_jump_bed_bed12_builds_bidirectional_edges() {
        let mut mi = build_test_idx();

        let mut bed = tempfile::NamedTempFile::new().unwrap();
        writeln!(bed, "chr1\t0\t50\tjunc1\t100\t+\t0\t50\t0\t2\t10,10\t0,30").unwrap();
        bed.flush().unwrap();

        let n = read_jump_bed(&mut mi, bed.path().to_str().unwrap(), JUNC_ANNO, -1).unwrap();
        assert_eq!(n, 2);
        let jumps = &mi.jump_db.as_ref().unwrap().jumps[0];
        assert_eq!(
            jumps,
            &vec![
                JumpEdge {
                    off: 10,
                    off2: 30,
                    cnt: 1,
                    strand: 1,
                    flag: JUNC_ANNO,
                },
                JumpEdge {
                    off: 30,
                    off2: 10,
                    cnt: 1,
                    strand: 1,
                    flag: JUNC_ANNO,
                },
            ]
        );
    }

    #[test]
    fn test_read_jump_bed_merges_duplicate_edges_and_flags() {
        let mut mi = build_test_idx();

        let mut anno = tempfile::NamedTempFile::new().unwrap();
        writeln!(anno, "chr1\t0\t50\tjunc1\t100\t+\t0\t50\t0\t2\t10,10\t0,30").unwrap();
        anno.flush().unwrap();
        read_jump_bed(&mut mi, anno.path().to_str().unwrap(), JUNC_ANNO, -1).unwrap();

        let mut pass1 = tempfile::NamedTempFile::new().unwrap();
        writeln!(pass1, "chr1\t0\t50\tjunc1\t10\t+\t0\t50\t0\t2\t10,10\t0,30").unwrap();
        pass1.flush().unwrap();
        read_jump_bed(&mut mi, pass1.path().to_str().unwrap(), JUNC_MISC, 5).unwrap();

        let jumps = &mi.jump_db.as_ref().unwrap().jumps[0];
        assert_eq!(jumps.len(), 2);
        assert_eq!(jumps[0].cnt, 2);
        assert_eq!(jumps[0].flag, JUNC_ANNO | JUNC_MISC);
        assert_eq!(jumps[1].cnt, 2);
        assert_eq!(jumps[1].flag, JUNC_ANNO | JUNC_MISC);
    }
}
