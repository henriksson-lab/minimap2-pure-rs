//! SSE2-accelerated KSW2 extension alignment.
//!
//! The rotated-band DP from ksw2_extz2_sse.c uses anti-diagonal processing
//! with 16-way SIMD parallelism. The full implementation requires careful
//! management of offset-encoded byte values and the h0 accumulator.
//!
//! Current status: scaffolded with runtime dispatch; SIMD scoring core
//! needs further refinement. Falls back to scalar for all operations.


use crate::flags::KswFlags;
use super::ksw2::KswResult;

/// Check if SSE2 is available at runtime.
pub fn has_sse2() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("sse2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Runtime-dispatched alignment.
///
/// Currently uses the scalar implementation. The SIMD kernel is
/// scaffolded but needs the rotated-band h0-accumulator logic
/// to be fully correct before activation.
pub fn ksw_extz2_dispatch(
    query: &[u8],
    target: &[u8],
    m: i8,
    mat: &[i8],
    q: i8,
    e: i8,
    w: i32,
    zdrop: i32,
    end_bonus: i32,
    flag: KswFlags,
) -> KswResult {
    // Use scalar for correctness; SIMD activation pending
    super::ksw2::ksw_extz2(query, target, m, mat, q, e, w, zdrop, end_bonus, flag)
}
