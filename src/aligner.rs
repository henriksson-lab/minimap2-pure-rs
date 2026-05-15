//! High-level aligner API for easy library usage.
//!
//! ```no_run
//! use minimap2::aligner::Aligner;
//!
//! let aligner = Aligner::builder()
//!     .preset("map-ont")
//!     .index("ref.fa")
//!     .build()
//!     .unwrap();
//!
//! let hits = aligner.map(b"ACGTACGTACGTACGT...");
//! for hit in &hits {
//!     println!("{} {}:{}-{}", hit.rid, hit.rs, hit.re, hit.mapq);
//! }
//! ```

use crate::flags::MapFlags;
use crate::index::MmIdx;
use crate::map;
use crate::options::{self, IdxOpt, MapOpt};
use crate::types::AlignReg;

/// High-level aligner wrapping index + options.
pub struct Aligner {
    pub idx: MmIdx,
    pub map_opt: MapOpt,
}

/// Builder for constructing an Aligner.
pub struct AlignerBuilder {
    idx_opt: IdxOpt,
    map_opt: MapOpt,
    index_path: Option<String>,
    cigar: bool,
}

impl Aligner {
    /// Create a new builder with default index and mapping options.
    pub fn builder() -> AlignerBuilder {
        AlignerBuilder {
            idx_opt: IdxOpt::default(),
            map_opt: MapOpt::default(),
            index_path: None,
            cigar: false,
        }
    }

    /// Map a single anonymous query sequence and return its alignment regions.
    ///
    /// # Parameters
    /// * `seq` - query sequence as ASCII bases (`A`/`C`/`G`/`T`/`N`); case-insensitive
    pub fn map(&self, seq: &[u8]) -> Vec<AlignReg> {
        let result = map::map_query(&self.idx, &self.map_opt, "", seq);
        result.regs
    }

    /// Map a named query sequence. The full `MapResult` (regs + auxiliary state) is returned.
    ///
    /// # Parameters
    /// * `name` - query name (used in PAF/SAM output and for hashing)
    /// * `seq` - query sequence as ASCII bases
    pub fn map_named(&self, name: &str, seq: &[u8]) -> map::MapResult {
        map::map_query(&self.idx, &self.map_opt, name, seq)
    }

    /// Format the result of `map_named` as one PAF line per mapping region.
    ///
    /// # Parameters
    /// * `name` - query name (PAF column 1)
    /// * `seq` - query sequence (used to derive query length and, with `OUT_CG`, CIGAR context)
    /// * `result` - the `MapResult` returned by `map_named`
    pub fn format_paf(&self, name: &str, seq: &[u8], result: &map::MapResult) -> Vec<String> {
        map::format_paf(&self.idx, &self.map_opt, name, seq, result)
    }

    /// Return the number of reference sequences in the index.
    pub fn n_seq(&self) -> usize {
        self.idx.seqs.len()
    }

    /// Look up a reference sequence name by its target id.
    ///
    /// # Parameters
    /// * `rid` - reference id (0..n_seq); matches `AlignReg::rid`
    pub fn seq_name(&self, rid: usize) -> &str {
        &self.idx.seqs[rid].name
    }

    /// Look up a reference sequence length (in bases) by its target id.
    ///
    /// # Parameters
    /// * `rid` - reference id (0..n_seq)
    pub fn seq_len(&self, rid: usize) -> u32 {
        self.idx.seqs[rid].len
    }
}

impl AlignerBuilder {
    /// Apply a minimap2 preset. Silently ignores unknown names.
    ///
    /// # Parameters
    /// * `name` - one of `map-ont`, `map-pb`, `map-hifi`, `map-ccs`, `sr`, `asm5`, `asm10`,
    ///   `asm20`, `ava-ont`, `ava-pb`, `splice`, `splice:hq`, `splice:sr`, `cdna`,
    ///   `lr:hq`, `lr:hqae`, `map-iclr`, `map-iclr-prerender`
    pub fn preset(mut self, name: &str) -> Self {
        options::set_opt(Some(name), &mut self.idx_opt, &mut self.map_opt).ok();
        self
    }

    /// Set the path used at `build()` time. May be a FASTA/FASTQ reference or a prebuilt `.mmi` index.
    ///
    /// # Parameters
    /// * `path` - filesystem path to the reference FASTA or `.mmi` index file
    pub fn index(mut self, path: &str) -> Self {
        self.index_path = Some(path.to_string());
        self
    }

    /// Enable CIGAR generation (`MapFlags::CIGAR | MapFlags::OUT_CG`).
    pub fn with_cigar(mut self) -> Self {
        self.cigar = true;
        self
    }

    /// Set the minimizer k-mer size used when building the index.
    ///
    /// # Parameters
    /// * `k` - k-mer length; minimap2 defaults range from 15 (long-read) to 21 (sr)
    pub fn k(mut self, k: i16) -> Self {
        self.idx_opt.k = k;
        self
    }

    /// Set the minimizer window size used when building the index.
    ///
    /// # Parameters
    /// * `w` - window size in k-mers (one minimizer is picked per window)
    pub fn w(mut self, w: i16) -> Self {
        self.idx_opt.w = w;
        self
    }

    /// Set the number of secondary alignments to report.
    ///
    /// # Parameters
    /// * `n` - maximum secondary alignments per primary (`-N` in the CLI)
    pub fn best_n(mut self, n: i32) -> Self {
        self.map_opt.best_n = n;
        self
    }

    /// Finalize the builder. Loads the prebuilt index or builds one from the FASTA path,
    /// then calls `mapopt_update` so options like `mid_occ` are derived from the index.
    pub fn build(mut self) -> Result<Aligner, String> {
        let path = self
            .index_path
            .as_deref()
            .ok_or_else(|| "No index path provided".to_string())?;

        if self.cigar {
            self.map_opt.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
        }

        let is_idx = crate::index::io::is_idx_file(path)
            .map_err(|e| format!("Failed to check index: {}", e))?;

        let idx = if is_idx {
            let mut f =
                std::fs::File::open(path).map_err(|e| format!("Failed to open index: {}", e))?;
            crate::index::io::idx_load(&mut f)
                .map_err(|e| format!("Failed to load index: {}", e))?
                .ok_or_else(|| "Empty index file".to_string())?
        } else {
            MmIdx::build_from_file(
                path,
                self.idx_opt.w as i32,
                self.idx_opt.k as i32,
                self.idx_opt.bucket_bits,
                self.idx_opt.flag,
                self.idx_opt.mini_batch_size,
                self.idx_opt.batch_size,
            )
            .map_err(|e| format!("Failed to build index: {}", e))?
            .ok_or_else(|| "Empty reference".to_string())?
        };

        options::mapopt_update(&mut self.map_opt, &idx);

        Ok(Aligner {
            idx,
            map_opt: self.map_opt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_aligner_basic() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, ">ref1").unwrap();
        writeln!(
            f,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        f.flush().unwrap();

        let aligner = Aligner::builder()
            .index(f.path().to_str().unwrap())
            .build()
            .unwrap();

        assert_eq!(aligner.n_seq(), 1);
        assert_eq!(aligner.seq_name(0), "ref1");

        let hits = aligner.map(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_aligner_with_preset() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, ">ref1").unwrap();
        writeln!(
            f,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        f.flush().unwrap();

        let aligner = Aligner::builder()
            .preset("map-ont")
            .index(f.path().to_str().unwrap())
            .build()
            .unwrap();

        let hits = aligner.map(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_aligner_with_cigar() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, ">ref1").unwrap();
        writeln!(
            f,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        f.flush().unwrap();

        let aligner = Aligner::builder()
            .index(f.path().to_str().unwrap())
            .with_cigar()
            .build()
            .unwrap();

        let hits = aligner.map(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
        assert!(!hits.is_empty());
        assert!(hits[0].extra.is_some(), "Should have CIGAR");
    }

    #[test]
    fn test_aligner_no_hit() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, ">ref1").unwrap();
        writeln!(
            f,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        f.flush().unwrap();

        let aligner = Aligner::builder()
            .index(f.path().to_str().unwrap())
            .build()
            .unwrap();

        let hits = aligner.map(b"NNNNNNNNNNNNNNNNNNNN");
        assert!(hits.is_empty());
    }
}
