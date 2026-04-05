pub mod bucket;
pub mod io;

use hashbrown::HashMap;
use crate::bseq::BseqFile;
use crate::flags::IdxFlags;
use crate::seq::{SEQ_NT4_TABLE, packed};
use crate::sketch::mm_sketch;
use crate::sort::ksmall_u32;
use crate::types::{IdxSeq, Mm128};
use bucket::IdxBucket;

/// The minimap2 index. Mirrors mm_idx_t.
pub struct MmIdx {
    pub w: i32,
    pub k: i32,
    pub bucket_bits: i32,
    pub flag: IdxFlags,
    pub seqs: Vec<IdxSeq>,
    pub packed_seq: Vec<u32>,
    pub buckets: Vec<IdxBucket>,
    pub n_alt: i32,
    pub index_part: i32,
    name_map: Option<HashMap<String, u32>>,
}

impl MmIdx {
    /// Initialize an empty index. Matches mm_idx_init().
    pub fn new(w: i32, k: i32, bucket_bits: i32, flag: IdxFlags) -> Self {
        let b = if k * 2 < bucket_bits { k * 2 } else { bucket_bits };
        let w = if w < 1 { 1 } else { w };
        let n_buckets = 1usize << b;
        let mut buckets = Vec::with_capacity(n_buckets);
        for _ in 0..n_buckets {
            buckets.push(IdxBucket::new());
        }
        Self {
            w,
            k,
            bucket_bits: b,
            flag,
            seqs: Vec::new(),
            packed_seq: Vec::new(),
            buckets,
            n_alt: 0,
            index_part: 0,
            name_map: None,
        }
    }

    /// Look up positions for a minimizer. Matches mm_idx_get().
    ///
    /// `minier` is the full minimizer hash (before bucket selection).
    /// Returns (count, positions_slice).
    pub fn get(&self, minier: u64) -> (i32, &[u64]) {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket_idx = (minier & mask) as usize;
        let key = (minier >> self.bucket_bits) << 1;
        self.buckets[bucket_idx].get(key)
    }

    /// Get a subsequence from the packed reference. Matches mm_idx_getseq().
    /// Returns 2-bit encoded bases (0-3) in `buf`, or -1 on error.
    pub fn getseq(&self, rid: u32, st: u32, en: u32, buf: &mut [u8]) -> i32 {
        if rid as usize >= self.seqs.len() || st >= self.seqs[rid as usize].len {
            return -1;
        }
        let en = en.min(self.seqs[rid as usize].len);
        let offset = self.seqs[rid as usize].offset;
        let n = (en - st) as usize;
        if n > buf.len() { return -1; }
        for i in st..en {
            buf[(i - st) as usize] = packed::seq4_get(&self.packed_seq, (offset + i as u64) as usize);
        }
        n as i32
    }

    /// Get a reverse-complement subsequence. Matches mm_idx_getseq_rev().
    pub fn getseq_rev(&self, rid: u32, st: u32, en: u32, buf: &mut [u8]) -> i32 {
        if rid as usize >= self.seqs.len() || st >= self.seqs[rid as usize].len {
            return -1;
        }
        let s = &self.seqs[rid as usize];
        let en = en.min(s.len);
        let n = (en - st) as usize;
        if n > buf.len() { return -1; }
        let st1 = s.offset + (s.len - en) as u64;
        let en1 = s.offset + (s.len - st) as u64;
        for i in st1..en1 {
            let c = packed::seq4_get(&self.packed_seq, i as usize);
            buf[(en1 - i - 1) as usize] = if c < 4 { 3 - c } else { c };
        }
        n as i32
    }

    /// Get a subsequence, forward or reverse complement. Matches mm_idx_getseq2().
    pub fn getseq2(&self, is_rev: bool, rid: u32, st: u32, en: u32, buf: &mut [u8]) -> i32 {
        if is_rev {
            self.getseq_rev(rid, st, en, buf)
        } else {
            self.getseq(rid, st, en, buf)
        }
    }

    /// Build the name → id mapping. Matches mm_idx_index_name().
    pub fn index_names(&mut self) -> bool {
        if self.name_map.is_some() {
            return false;
        }
        let mut h = HashMap::with_capacity(self.seqs.len());
        let mut has_dup = false;
        for (i, seq) in self.seqs.iter().enumerate() {
            if h.insert(seq.name.clone(), i as u32).is_some() {
                has_dup = true;
            }
        }
        self.name_map = Some(h);
        has_dup
    }

    /// Look up a sequence by name. Matches mm_idx_name2id().
    pub fn name2id(&self, name: &str) -> Option<u32> {
        self.name_map.as_ref()?.get(name).copied()
    }

    /// Calculate max occurrence threshold from fractional cutoff.
    /// Matches mm_idx_cal_max_occ().
    pub fn cal_max_occ(&self, f: f32) -> i32 {
        if f <= 0.0 {
            return i32::MAX;
        }
        // Collect all occurrence counts
        let mut counts: Vec<u32> = Vec::new();
        for b in &self.buckets {
            for c in b.iter_counts() {
                counts.push(c);
            }
        }
        if counts.is_empty() {
            return i32::MAX;
        }
        let n = counts.len();
        let k = ((1.0 - f as f64) * n as f64) as usize;
        let k = k.min(n - 1);
        (ksmall_u32(&mut counts, k) + 1) as i32
    }

    /// Print index statistics. Matches mm_idx_stat().
    pub fn stat(&self) {
        let mut n_keys: i64 = 0;
        let mut n_singletons: i64 = 0;
        let mut sum_occ: u64 = 0;
        let mut total_len: u64 = 0;

        for seq in &self.seqs {
            total_len += seq.len as u64;
        }
        for b in &self.buckets {
            if let Some(h) = &b.h {
                n_keys += h.len() as i64;
                for (&key, &val) in h.iter() {
                    if key & 1 != 0 {
                        sum_occ += 1;
                        n_singletons += 1;
                    } else {
                        sum_occ += val as u32 as u64;
                    }
                }
            }
        }
        let is_hpc = if self.flag.contains(IdxFlags::HPC) { 1 } else { 0 };
        eprintln!(
            "[M::idx_stat] kmer size: {}; skip: {}; is_hpc: {}; #seq: {}",
            self.k, self.w, is_hpc, self.seqs.len()
        );
        if n_keys > 0 {
            eprintln!(
                "[M::idx_stat] distinct minimizers: {} ({:.2}% are singletons); average occurrences: {:.3}; average spacing: {:.3}; total length: {}",
                n_keys,
                100.0 * n_singletons as f64 / n_keys as f64,
                sum_occ as f64 / n_keys as f64,
                total_len as f64 / sum_occ as f64,
                total_len
            );
        }
    }

    /// Dispatch a minimizer to the appropriate bucket during building.
    fn add_minimizers(&mut self, minimizers: &[Mm128]) {
        let mask = (1i64 << self.bucket_bits) - 1;
        for m in minimizers {
            let bucket_idx = ((m.x >> 8) & mask as u64) as usize;
            self.buckets[bucket_idx].add(*m);
        }
    }

    /// Post-process all buckets (sort + build hash tables).
    fn post_process(&mut self) {
        let bb = self.bucket_bits;
        for b in &mut self.buckets {
            b.post_process(bb);
        }
    }

    /// Build an index from a FASTA/FASTQ file. Matches mm_idx_gen().
    pub fn build_from_file(
        path: &str,
        w: i32,
        k: i32,
        bucket_bits: i16,
        flag: IdxFlags,
        mini_batch_size: i64,
        batch_size: u64,
    ) -> std::io::Result<Option<Self>> {
        let mut fp = BseqFile::open(path)?;
        if fp.is_eof() {
            return Ok(None);
        }

        let mut mi = MmIdx::new(w, k, bucket_bits as i32, flag);
        let is_hpc = flag.contains(IdxFlags::HPC);
        let store_seq = !flag.contains(IdxFlags::NO_SEQ);
        let store_name = !flag.contains(IdxFlags::NO_NAME);
        let mut sum_len: u64 = 0;

        let effective_batch = if (mini_batch_size as u64) < batch_size {
            mini_batch_size
        } else {
            batch_size as i64
        };

        loop {
            if sum_len > batch_size {
                break;
            }
            let records = fp.read_batch(effective_batch, false)?;
            if records.is_empty() {
                break;
            }

            // Step 0: store sequence metadata and packed sequence
            for rec in &records {
                let seq_entry = IdxSeq {
                    name: if store_name { rec.name.clone() } else { String::new() },
                    offset: sum_len,
                    len: rec.l_seq as u32,
                    is_alt: false,
                };

                if store_seq {
                    // Ensure packed_seq has enough room
                    let needed = ((sum_len + rec.l_seq as u64 + 7) / 8) as usize;
                    if mi.packed_seq.len() < needed {
                        mi.packed_seq.resize(needed, 0);
                    }
                    for (j, &b) in rec.seq.iter().enumerate() {
                        let c = SEQ_NT4_TABLE[b as usize];
                        packed::seq4_set(&mut mi.packed_seq, (sum_len + j as u64) as usize, c);
                    }
                }

                sum_len += rec.l_seq as u64;
                mi.seqs.push(seq_entry);
            }

            // Step 1: compute sketches
            let mut all_minimizers = Vec::new();
            let n_prev_seqs = mi.seqs.len() - records.len();
            for (i, rec) in records.iter().enumerate() {
                if rec.l_seq > 0 {
                    let rid = (n_prev_seqs + i) as u32;
                    mm_sketch(&rec.seq, w as usize, k as usize, rid, is_hpc, &mut all_minimizers);
                }
            }

            // Step 2: dispatch to buckets
            mi.add_minimizers(&all_minimizers);
        }

        // Post-process: sort and build hash tables
        mi.post_process();

        Ok(Some(mi))
    }

    /// Build an index from in-memory sequences. Matches mm_idx_str().
    pub fn build_from_str(
        w: i32,
        k: i32,
        is_hpc: bool,
        bucket_bits: i32,
        seqs: &[&[u8]],
        names: Option<&[&str]>,
    ) -> Option<Self> {
        if seqs.is_empty() {
            return None;
        }
        let mut flag = IdxFlags::empty();
        if is_hpc {
            flag |= IdxFlags::HPC;
        }
        if names.is_none() {
            flag |= IdxFlags::NO_NAME;
        }
        let bb = if bucket_bits < 0 { 14 } else { bucket_bits };

        let sum_len: u64 = seqs.iter().map(|s| s.len() as u64).sum();
        let mut mi = MmIdx::new(w, k, bb, flag);
        mi.packed_seq = vec![0u32; ((sum_len + 7) / 8) as usize];

        let mut offset: u64 = 0;
        for (i, &seq) in seqs.iter().enumerate() {
            let name = names.map_or(String::new(), |n| n[i].to_string());
            mi.seqs.push(IdxSeq {
                name,
                offset,
                len: seq.len() as u32,
                is_alt: false,
            });
            // Pack sequence
            for (j, &b) in seq.iter().enumerate() {
                let c = SEQ_NT4_TABLE[b as usize];
                packed::seq4_set(&mut mi.packed_seq, (offset + j as u64) as usize, c);
            }
            // Sketch
            if !seq.is_empty() {
                let mut minimizers = Vec::new();
                mm_sketch(seq, w as usize, k as usize, i as u32, is_hpc, &mut minimizers);
                mi.add_minimizers(&minimizers);
            }
            offset += seq.len() as u64;
        }

        // Build name index if names provided
        if names.is_some() {
            mi.index_names();
        }

        mi.post_process();
        Some(mi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_from_str() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGT",
            b"TGCATGCATGCATGCATGCATGCATGCATGCA",
        ];
        let names = vec!["seq1", "seq2"];
        let mi = MmIdx::build_from_str(10, 15, false, 14, &seqs, Some(&names)).unwrap();

        assert_eq!(mi.seqs.len(), 2);
        assert_eq!(mi.seqs[0].name, "seq1");
        assert_eq!(mi.seqs[0].len, 32);
        assert_eq!(mi.seqs[1].name, "seq2");
        assert_eq!(mi.seqs[1].len, 32);

        // Verify getseq retrieves correct bases
        let mut buf = vec![0u8; 4];
        mi.getseq(0, 0, 4, &mut buf);
        assert_eq!(buf, vec![0, 1, 2, 3]); // ACGT

        // Verify name lookup
        assert_eq!(mi.name2id("seq1"), Some(0));
        assert_eq!(mi.name2id("seq2"), Some(1));
        assert_eq!(mi.name2id("nope"), None);
    }

    #[test]
    fn test_index_get() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGT",
        ];
        let mi = MmIdx::build_from_str(5, 10, false, 10, &seqs, None).unwrap();

        // The index should have some minimizers
        let mut total_keys = 0usize;
        for b in &mi.buckets {
            total_keys += b.n_keys();
        }
        assert!(total_keys > 0, "Index should have minimizers");
    }

    #[test]
    fn test_getseq_rev() {
        let seqs: Vec<&[u8]> = vec![b"ACGTAAAA"];
        let mi = MmIdx::build_from_str(5, 5, false, 10, &seqs, None).unwrap();

        let mut buf = vec![0u8; 4];
        mi.getseq(0, 0, 4, &mut buf);
        assert_eq!(buf, vec![0, 1, 2, 3]); // ACGT

        mi.getseq_rev(0, 0, 4, &mut buf);
        // reverse complement of last 4 bases "AAAA" = "TTTT" = [3,3,3,3]
        assert_eq!(buf, vec![3, 3, 3, 3]);
    }

    #[test]
    fn test_cal_max_occ() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        ];
        let mi = MmIdx::build_from_str(5, 10, false, 10, &seqs, None).unwrap();
        let occ = mi.cal_max_occ(0.02);
        assert!(occ > 0);
    }

    #[test]
    fn test_build_from_file() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, ">seq1").unwrap();
        writeln!(f, "ACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        writeln!(f, ">seq2").unwrap();
        writeln!(f, "TGCATGCATGCATGCATGCATGCATGCATGCA").unwrap();
        f.flush().unwrap();

        let mi = MmIdx::build_from_file(
            f.path().to_str().unwrap(),
            10, 15, 14,
            IdxFlags::empty(),
            1_000_000,
            u64::MAX,
        )
        .unwrap()
        .unwrap();

        assert_eq!(mi.seqs.len(), 2);
        assert_eq!(mi.seqs[0].name, "seq1");
        assert_eq!(mi.seqs[1].name, "seq2");
    }
}
