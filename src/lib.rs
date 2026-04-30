#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::identity_op)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::let_and_return)]
//! Pure Rust reimplementation of minimap2.
//!
//! # Quick Start
//!
//! ```no_run
//! use minimap2::prelude::*;
//!
//! // Build index from FASTA
//! let mi = MmIdx::build_from_file(
//!     "ref.fa", 10, 15, 14,
//!     IdxFlags::empty(), 50_000_000, u64::MAX,
//! ).unwrap().unwrap();
//!
//! // Set up mapping options with preset
//! let (mut io, mut mo) = preset("map-ont").unwrap();
//! mapopt_update(&mut mo, &mi);
//!
//! // Map a query
//! let result = map_query(&mi, &mo, "read1", b"ACGTACGT...");
//! for reg in &result.regs {
//!     println!("{}:{}-{} mapq={}", reg.rid, reg.rs, reg.re, reg.mapq);
//! }
//! ```
//!
//! # Library API
//!
//! The main entry points are:
//! - [`index::MmIdx`] — build or load an index
//! - [`map::map_query`] — map a single query sequence
//! - [`options::set_opt`] — configure presets
//! - [`pipeline`] — multi-threaded file mapping

pub mod align;
pub mod aligner;
pub mod bseq;
pub mod chain;
pub mod cli;
pub mod esterr;
pub mod flags;
pub mod format;
pub mod hit;
pub mod index;
pub mod jump;
pub mod junc;
pub mod map;
pub mod options;
pub mod pe;
pub mod pipeline;
pub mod sdust;
pub mod seed;
pub mod seq;
pub mod sketch;
pub mod sort;
pub mod types;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::flags::{IdxFlags, MapFlags};
    pub use crate::index::MmIdx;
    pub use crate::map::{map_query, MapResult};
    pub use crate::options::{mapopt_update, IdxOpt, MapOpt};
    pub use crate::types::AlignReg;

    /// Set up options with a preset. Returns (IdxOpt, MapOpt).
    pub fn preset(name: &str) -> Result<(IdxOpt, MapOpt), String> {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        crate::options::set_opt(Some(name), &mut io, &mut mo)?;
        Ok((io, mo))
    }
}
