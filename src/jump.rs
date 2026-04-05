//! Splice junction extension handling.
//!
//! Matches jump.c: mm_jump_split(), mm_jump_split_left(), mm_jump_split_right().
//!
//! TODO: implement junction jump support.

use crate::index::MmIdx;
use crate::options::MapOpt;
use crate::types::AlignReg;

/// Extend alignment across splice junctions.
///
/// Stub — not yet implemented.
pub fn jump_split(
    _mi: &MmIdx,
    _opt: &MapOpt,
    _qlen: i32,
    _qseq: &[u8],
    _r: &mut AlignReg,
    _ts_strand: i32,
) {
    // TODO: implement junction jump
}
