use crate::index::MmIdx;
use crate::sort::radix_sort_mm128;
use crate::types::Mm128;

/// Seed match from index lookup.
#[derive(Clone, Debug)]
pub struct Seed {
    pub q_pos: u32,
    pub q_span: u32,
    pub n: i32,
    pub flt: bool,
    pub seg_id: u32,
    pub is_tandem: bool,
    /// Positions slice from the index (offset and length into index bucket data).
    /// For singletons, `positions` has exactly 1 element.
    pub positions: SeedPositions,
}

/// Seed positions — either a single value or a reference into the index.
#[derive(Clone, Debug)]
pub enum SeedPositions {
    /// Single position (singleton minimizer in index).
    Single(u64),
    /// Multiple positions: (bucket_index, offset, count) to look up later.
    Multi {
        offset: usize,
        count: usize,
        bucket_idx: usize,
    },
}

/// Result of seed collection.
pub struct SeedResult {
    pub seeds: Vec<Seed>,
    pub n_a: i64,
    pub rep_len: i32,
    pub mini_pos: Vec<u64>,
}

/// Filter query minimizers by occurrence frequency.
///
/// Removes minimizers that appear more than `q_occ_max` times AND more than
/// `q_occ_frac` fraction of total minimizers.
///
/// Matches mm_seed_mz_flt() from seed.c.
pub fn seed_mz_flt(mv: &mut Vec<Mm128>, q_occ_max: i32, q_occ_frac: f32) {
    if mv.len() as i32 <= q_occ_max || q_occ_frac <= 0.0 || q_occ_max <= 0 {
        return;
    }
    // Build (hash, original_index) pairs and sort by hash
    let n = mv.len();
    let mut a: Vec<Mm128> = mv
        .iter()
        .enumerate()
        .map(|(i, m)| Mm128::new(m.x, i as u64))
        .collect();
    radix_sort_mm128(&mut a);

    // Find runs of identical hashes and mark high-frequency ones
    let mut st = 0;
    for i in 1..=n {
        if i == n || a[i].x != a[st].x {
            let cnt = (i - st) as i32;
            if cnt > q_occ_max && cnt as f32 > n as f32 * q_occ_frac {
                for j in st..i {
                    mv[a[j].y as usize].x = 0;
                }
            }
            st = i;
        }
    }
    // Remove filtered minimizers (x == 0)
    mv.retain(|m| m.x != 0);
}

/// Collect all seed matches from the index for query minimizers.
///
/// Matches mm_seed_collect_all() from seed.c.
fn seed_collect_all(mi: &MmIdx, mv: &[Mm128]) -> Vec<Seed> {
    let mut seeds = Vec::with_capacity(mv.len());
    let mask = (1u64 << mi.bucket_bits) - 1;
    for (i, p) in mv.iter().enumerate() {
        let q_pos = p.y as u32;
        let q_span = (p.x & 0xff) as u32;
        let minier = p.x >> 8;

        // Single index lookup — get count and positions together
        let bucket_idx = (minier & mask) as usize;
        let key = (minier >> mi.bucket_bits) << 1;
        let (t, pos_slice) = mi.buckets[bucket_idx].get(key);
        if t == 0 {
            continue;
        }
        let seg_id = (p.y >> 32) as u32;

        // Detect tandem repeats (same minimizer hash as neighbors)
        let is_tandem = (i > 0 && p.x >> 8 == mv[i - 1].x >> 8)
            || (i + 1 < mv.len() && p.x >> 8 == mv[i + 1].x >> 8);

        // Store position info from the single lookup
        let positions = if t == 1 {
            SeedPositions::Single(pos_slice[0])
        } else {
            // Multi-hit — compute offset from slice pointer
            let bucket = &mi.buckets[bucket_idx];
            let base = bucket.p.as_ptr();
            let offset = unsafe { pos_slice.as_ptr().offset_from(base) as usize };
            SeedPositions::Multi {
                offset,
                count: t as usize,
                bucket_idx,
            }
        };

        seeds.push(Seed {
            q_pos,
            q_span,
            n: t,
            flt: false,
            seg_id,
            is_tandem,
            positions,
        });
    }
    seeds
}

const MAX_MAX_HIGH_OCC: usize = 128;

/// For high-occurrence minimizers, keep only the top `max_high_occ` lowest-frequency
/// ones in each high-occurrence streak.
///
/// Matches mm_seed_select() from seed.c.
fn seed_select(seeds: &mut [Seed], qlen: i32, max_occ: i32, max_max_occ: i32, dist: i32) {
    let n = seeds.len() as i32;
    if n <= 1 {
        return;
    }
    // Check if there are any high-frequency seeds
    let has_high = seeds.iter().any(|s| s.n > max_occ);
    if !has_high {
        return;
    }

    let mut last0: i32 = -1;
    for i in 0..=n {
        if i == n || seeds[i as usize].n <= max_occ {
            if i - last0 > 1 {
                let ps = if last0 < 0 {
                    0
                } else {
                    (seeds[last0 as usize].q_pos >> 1) as i32
                };
                let pe = if i == n {
                    qlen
                } else {
                    (seeds[i as usize].q_pos >> 1) as i32
                };
                let st = (last0 + 1) as usize;
                let en = i as usize;
                let max_high_occ = ((pe - ps) as f64 / dist as f64 + 0.499) as usize;
                if max_high_occ > 0 {
                    let max_high_occ = max_high_occ.min(MAX_MAX_HIGH_OCC);
                    // Use a max-heap of (occurrence_count, index) to select lowest-occ seeds
                    let mut heap: Vec<u64> = Vec::with_capacity(max_high_occ);
                    let mut k = 0;
                    for j in st..en {
                        if k < max_high_occ {
                            heap.push((seeds[j].n as u64) << 32 | j as u64);
                            k += 1;
                            if k == max_high_occ {
                                // Build max-heap
                                heap_make(&mut heap);
                            }
                        } else if seeds[j].n < (heap[0] >> 32) as i32 {
                            heap[0] = (seeds[j].n as u64) << 32 | j as u64;
                            heap_down(&mut heap, 0);
                        }
                    }
                    // Mark selected seeds (the ones in the heap)
                    for j in 0..k {
                        seeds[(heap[j] as u32) as usize].flt = true;
                    }
                }
                // Flip: flt=1 means KEEP (selected), so flip to flt=0 means keep
                for j in st..en {
                    seeds[j].flt = !seeds[j].flt;
                }
                // Also filter seeds above max_max_occ
                for j in st..en {
                    if seeds[j].n > max_max_occ {
                        seeds[j].flt = true;
                    }
                }
            }
            last0 = i;
        }
    }
}

/// Build a max-heap from an array.
fn heap_make(a: &mut [u64]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let mut i = n / 2;
    loop {
        if i == 0 {
            break;
        }
        i -= 1;
        heap_sift_down(a, i, n);
    }
    heap_sift_down(a, 0, n);
}

fn heap_down(a: &mut [u64], i: usize) {
    heap_sift_down(a, i, a.len());
}

fn heap_sift_down(a: &mut [u64], mut i: usize, n: usize) {
    loop {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < n && a[left] > a[largest] {
            largest = left;
        }
        if right < n && a[right] > a[largest] {
            largest = right;
        }
        if largest == i {
            break;
        }
        a.swap(i, largest);
        i = largest;
    }
}

/// Collect seed matches from the index, with occurrence filtering.
///
/// Matches mm_collect_matches() from seed.c.
///
/// # Arguments
/// * `mi` - the minimap2 index
/// * `mv` - query minimizers
/// * `qlen` - query length
/// * `max_occ` - max occurrence for a minimizer (higher are filtered or selected)
/// * `max_max_occ` - absolute maximum occurrence (always filtered above this)
/// * `dist` - distance parameter for adaptive selection
#[inline(never)]
pub fn collect_matches(
    mi: &MmIdx,
    mv: &[Mm128],
    qlen: i32,
    max_occ: i32,
    max_max_occ: i32,
    dist: i32,
) -> SeedResult {
    let mut seeds = seed_collect_all(mi, mv);

    // Apply occurrence filtering
    if dist > 0 && max_max_occ > max_occ {
        seed_select(&mut seeds, qlen, max_occ, max_max_occ, dist);
    } else {
        for s in &mut seeds {
            if s.n > max_occ {
                s.flt = true;
            }
        }
    }

    // Compute repeat length and collect kept seeds
    let mut rep_len: i32 = 0;
    let mut rep_st: i32 = 0;
    let mut rep_en: i32 = 0;
    let mut n_a: i64 = 0;

    for q in &seeds {
        if q.flt {
            let en = (q.q_pos >> 1) as i32 + 1;
            let st = en - q.q_span as i32;
            if st > rep_en {
                rep_len += rep_en - rep_st;
                rep_st = st;
                rep_en = en;
            } else {
                rep_en = en;
            }
        } else {
            n_a += q.n as i64;
        }
    }
    rep_len += rep_en - rep_st;

    // Matches C's mm_collect_matches in seed.c: push to mini_pos only for
    // seeds that both had index matches (i.e. survived seed_collect_all) and
    // passed the occurrence filter. Minimizers from `mv` with no matches are
    // not in `seeds` and must not be emitted — otherwise mini_pos is too
    // long and mm_est_err's avg_k / n_match / n_tot diverge from C.
    let mut mini_pos = Vec::with_capacity(seeds.len());
    let mut seed_i = 0usize;
    for p in mv {
        let q_pos = p.y as u32;
        let q_span = (p.x & 0xff) as u32;
        if seed_i < seeds.len() && seeds[seed_i].q_pos == q_pos && seeds[seed_i].q_span == q_span {
            if !seeds[seed_i].flt {
                mini_pos.push((q_span as u64) << 32 | (q_pos >> 1) as u64);
            }
            seed_i += 1;
        }
    }

    // Filter in-place instead of cloning into a new Vec
    seeds.retain(|s| !s.flt);

    // Cross-language parity probe. Must emit bytes identically to the
    // TH_CALL probe at the tail of mm_collect_matches in minimap2/seed.c.
    // Canonicalization (both sides):
    //   inputs  = qlen, max_occ, max_max_occ, dist, mv.len,
    //             for each Mm128 in mv: (x, y)
    //   outputs = n_a, rep_len, seeds.len(), mini_pos.len(),
    //             for each u64 in mini_pos: value
    // Per-element u64 encoding (rather than bulk bytes) keeps C and Rust
    // trivially in sync — the public API exposes u64 writers on both sides.
    #[cfg(feature = "tracehash")]
    {
        let mut th = tracehash::th_call!("mm_collect_matches");
        th.input_i64(qlen as i64);
        th.input_i64(max_occ as i64);
        th.input_i64(max_max_occ as i64);
        th.input_i64(dist as i64);
        th.input_u64(mv.len() as u64);
        for m in mv {
            th.input_u64(m.x);
            th.input_u64(m.y);
        }
        th.output_i64(n_a);
        th.output_i64(rep_len as i64);
        th.output_u64(seeds.len() as u64);
        th.output_u64(mini_pos.len() as u64);
        for &mp in &mini_pos {
            th.output_u64(mp);
        }
        th.finish();
    }

    SeedResult {
        seeds,
        n_a,
        rep_len,
        mini_pos,
    }
}

/// Expand seeds into anchor array (Mm128 format) for chaining.
///
/// Matches collect_seed_hits() from map.c.
///
/// Output encoding:
/// - `a[i].x = rev << 63 | rid << 32 | ref_pos` (ref_pos has strand bit stripped)
/// - `a[i].y = flags | seg_id << 48 | q_span << 32 | q_pos` (q_pos has strand bit stripped; reverse strand gets coordinate flip)
///
/// `qlen` is needed to flip reverse-strand query coordinates.
#[inline]
fn skip_seed(
    mi: &MmIdx,
    flag: crate::flags::MapFlags,
    qname: Option<&str>,
    qlen: i32,
    r: u64,
    q_pos_raw: u32,
) -> (bool, bool) {
    let mut is_self = false;
    if let Some(qname) = qname {
        if flag.intersects(crate::flags::MapFlags::NO_DIAG | crate::flags::MapFlags::NO_DUAL) {
            let rid = (r >> 32) as usize;
            if let Some(seq) = mi.seqs.get(rid) {
                match qname.cmp(seq.name.as_str()) {
                    std::cmp::Ordering::Equal => {
                        if flag.contains(crate::flags::MapFlags::NO_DIAG) && seq.len as i32 == qlen
                        {
                            if ((r as u32) >> 1) == (q_pos_raw >> 1) {
                                return (true, false);
                            }
                            if ((r as u32) & 1) == (q_pos_raw & 1) {
                                is_self = true;
                            }
                        }
                    }
                    std::cmp::Ordering::Greater => {
                        if flag.contains(crate::flags::MapFlags::NO_DUAL) {
                            return (true, false);
                        }
                    }
                    std::cmp::Ordering::Less => {}
                }
            }
        }
    }
    (false, is_self)
}

#[inline(never)]
pub fn expand_seeds_to_anchors(
    mi: &MmIdx,
    seeds: &[Seed],
    qname: Option<&str>,
    qlen: i32,
    flag: crate::flags::MapFlags,
) -> Vec<Mm128> {
    // Pre-compute total anchor count for capacity hint
    let n_a: usize = seeds
        .iter()
        .filter(|s| !s.flt)
        .map(|s| match &s.positions {
            SeedPositions::Single(_) => 1,
            SeedPositions::Multi { count, .. } => *count,
        })
        .sum();
    let mut anchors = Vec::with_capacity(n_a);
    for seed in seeds {
        if seed.flt {
            continue;
        }
        let q_pos_raw = seed.q_pos; // pos<<1 | strand from minimizer
        let q_strand = q_pos_raw & 1;
        let q_pos = q_pos_raw >> 1; // actual query position
        let q_span = seed.q_span;
        let seg_id = seed.seg_id;

        // Iterate over positions without allocating
        let single_buf: [u64; 1];
        let positions: &[u64] = match &seed.positions {
            SeedPositions::Single(pos) => {
                single_buf = [*pos];
                &single_buf
            }
            SeedPositions::Multi {
                offset,
                count,
                bucket_idx,
            } => {
                let bucket = &mi.buckets[*bucket_idx];
                &bucket.p[*offset..*offset + *count]
            }
        };

        for &r in positions {
            let (skip, is_self) = skip_seed(mi, flag, qname, qlen, r, q_pos_raw);
            if skip {
                continue;
            }

            let r_strand = (r as u32) & 1;
            let is_forward = r_strand == q_strand;

            // Strand filtering
            if is_forward && flag.contains(crate::flags::MapFlags::REV_ONLY) {
                continue;
            }
            if !is_forward && flag.contains(crate::flags::MapFlags::FOR_ONLY) {
                continue;
            }

            let rpos = (r as u32) >> 1; // strip strand bit from ref position
            let rid_bits = r & 0xffffffff00000000u64;

            let (x, qp);
            if is_forward {
                // Forward strand
                x = rid_bits | rpos as u64;
                qp = q_pos;
            } else {
                // Reverse strand
                x = (1u64 << 63) | rid_bits | rpos as u64;
                // Flip query coordinate for reverse strand
                qp = (qlen as u32)
                    .wrapping_sub(q_pos + 1 - q_span)
                    .wrapping_sub(1);
            }

            let mut y = (q_span as u64) << 32 | qp as u64;
            y |= (seg_id as u64) << crate::flags::SEED_SEG_SHIFT;
            if seed.is_tandem {
                y |= crate::flags::SEED_TANDEM;
            }
            if is_self {
                y |= crate::flags::SEED_SELF;
            }
            anchors.push(Mm128::new(x, y));
        }
    }
    anchors
}

/// Expand seeds using C minimap2's collect_seed_hits_heap() ordering.
///
/// This is used by the short-read preset via MM_F_HEAP_SORT. It merges target
/// positions across query minimizers with a heap, writes forward-strand anchors
/// from the front and reverse-strand anchors from the back, then moves reverse
/// anchors after forward anchors.
#[inline(never)]
pub fn expand_seeds_to_anchors_heap(
    mi: &MmIdx,
    seeds: &[Seed],
    qname: Option<&str>,
    qlen: i32,
    flag: crate::flags::MapFlags,
) -> Vec<Mm128> {
    let n_a: usize = seeds
        .iter()
        .map(|s| match &s.positions {
            SeedPositions::Single(_) => 1,
            SeedPositions::Multi { count, .. } => *count,
        })
        .sum();
    let mut positions: Vec<&[u64]> = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let pos: &[u64] = match &seed.positions {
            SeedPositions::Single(pos) => std::slice::from_ref(pos),
            SeedPositions::Multi {
                offset,
                count,
                bucket_idx,
            } => &mi.buckets[*bucket_idx].p[*offset..*offset + *count],
        };
        positions.push(pos);
    }

    let mut heap: Vec<Mm128> = Vec::with_capacity(positions.len());
    for (i, pos) in positions.iter().enumerate() {
        if !pos.is_empty() {
            heap.push(Mm128::new(pos[0], (i as u64) << 32));
        }
    }
    heap_make_by_x(&mut heap);

    let mut a = vec![Mm128::default(); n_a];
    let mut n_for = 0usize;
    let mut n_rev = 0usize;

    while !heap.is_empty() {
        let r = heap[0].x;
        let seed_idx = (heap[0].y >> 32) as usize;
        let pos_idx = heap[0].y as u32 as usize;
        let q = &seeds[seed_idx];
        let q_pos_raw = q.q_pos;
        let q_strand = q_pos_raw & 1;
        let r_strand = (r as u32) & 1;
        let is_forward = r_strand == q_strand;
        let (skip, is_self) = skip_seed(mi, flag, qname, qlen, r, q_pos_raw);

        if !skip
            && !(is_forward && flag.contains(crate::flags::MapFlags::REV_ONLY))
            && !(!is_forward && flag.contains(crate::flags::MapFlags::FOR_ONLY))
        {
            let rpos = (r as u32) >> 1;
            let rid_bits = r & 0xffffffff00000000u64;
            let q_span = q.q_span;
            let mut anchor = if is_forward {
                Mm128::new(
                    rid_bits | rpos as u64,
                    (q_span as u64) << 32 | ((q_pos_raw >> 1) as u64),
                )
            } else {
                Mm128::new(
                    (1u64 << 63) | rid_bits | rpos as u64,
                    (q_span as u64) << 32
                        | (qlen as u32)
                            .wrapping_sub((q_pos_raw >> 1) + 1 - q_span)
                            .wrapping_sub(1) as u64,
                )
            };
            anchor.y |= (q.seg_id as u64) << crate::flags::SEED_SEG_SHIFT;
            if q.is_tandem {
                anchor.y |= crate::flags::SEED_TANDEM;
            }
            if is_self {
                anchor.y |= crate::flags::SEED_SELF;
            }
            if is_forward {
                a[n_for] = anchor;
                n_for += 1;
            } else {
                n_rev += 1;
                a[n_a - n_rev] = anchor;
            }
        }

        let next = pos_idx + 1;
        if next < positions[seed_idx].len() {
            heap[0].y += 1;
            heap[0].x = positions[seed_idx][next];
        } else {
            let last = heap.pop().unwrap();
            if !heap.is_empty() {
                heap[0] = last;
            }
        }
        if !heap.is_empty() {
            heap_down_by_x(&mut heap, 0);
        }
    }

    let rev_start = n_a - n_rev;
    a[rev_start..].reverse();
    if rev_start > n_for && n_rev > 0 {
        a.copy_within(rev_start..rev_start + n_rev, n_for);
    }
    a.truncate(n_for + n_rev);
    a
}

fn heap_less_by_x(a: &Mm128, b: &Mm128) -> bool {
    a.x > b.x
}

fn heap_down_by_x(heap: &mut [Mm128], mut i: usize) {
    let n = heap.len();
    let mut k = i;
    let tmp = heap[i];
    while {
        k = (k << 1) + 1;
        k < n
    } {
        if k != n - 1 && heap_less_by_x(&heap[k], &heap[k + 1]) {
            k += 1;
        }
        if heap_less_by_x(&heap[k], &tmp) {
            break;
        }
        heap[i] = heap[k];
        i = k;
    }
    heap[i] = tmp;
}

fn heap_make_by_x(heap: &mut [Mm128]) {
    if heap.len() < 2 {
        return;
    }
    for i in (0..=(heap.len() / 2 - 1)).rev() {
        heap_down_by_x(heap, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::mm_sketch;

    fn make_test_index() -> MmIdx {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"];
        MmIdx::build_from_str(5, 10, false, 10, &seqs, None).unwrap()
    }

    #[test]
    fn test_seed_mz_flt_no_filter() {
        let mut mv = vec![
            Mm128::new(100 << 8 | 10, 0),
            Mm128::new(200 << 8 | 10, 1),
            Mm128::new(300 << 8 | 10, 2),
        ];
        let orig_len = mv.len();
        seed_mz_flt(&mut mv, 100, 0.01);
        assert_eq!(mv.len(), orig_len); // nothing filtered
    }

    #[test]
    fn test_seed_mz_flt_filters_high_occ() {
        let mut mv = Vec::new();
        // Add 50 minimizers with the same hash and 50 with distinct hashes
        for i in 0..50 {
            mv.push(Mm128::new(0xABC00 | 10, i));
        }
        for i in 0..50 {
            mv.push(Mm128::new(((i + 1) * 0x1000) << 8 | 10, 50 + i));
        }
        // q_occ_max=10, q_occ_frac=0.01 -> the hash with 50 copies should be filtered
        seed_mz_flt(&mut mv, 10, 0.01);
        assert!(mv.len() < 100); // some were filtered
    }

    #[test]
    fn test_collect_matches() {
        let mi = make_test_index();
        let mut mv = Vec::new();
        mm_sketch(
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            5,
            10,
            0,
            false,
            &mut mv,
        );
        assert!(!mv.is_empty());

        let result = collect_matches(&mi, &mv, 54, 1000, 5000, 500);
        // Should have some seeds
        assert!(!result.seeds.is_empty());
        assert!(result.n_a > 0);
    }

    /// Regression test for mm_est_err parity with C: mini_pos must contain
    /// only entries for query minimizers that (a) found matches in the index
    /// during seed_collect_all, and (b) were not filtered out by occurrence
    /// cutoffs. Matches mm_collect_matches() in seed.c:125.
    ///
    /// Prior bug: the Rust loop had an `else` branch that pushed to mini_pos
    /// for every minimizer with no index match, inflating len and corrupting
    /// avg_k / n_match / n_tot in est_err → wrong `dv` tag values.
    #[test]
    fn test_collect_matches_mini_pos_excludes_unmatched_minimizers() {
        // Small index built from a specific sequence — query minimizers NOT
        // present in this sequence will have no index matches.
        let mi = make_test_index();
        let ref_seq: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 10usize;
        let w = 5usize;

        // Sketch a query that shares only a prefix with the reference; the
        // unique suffix will produce minimizers with no matches.
        let mut ref_mv = Vec::new();
        mm_sketch(ref_seq, w, k, 0, false, &mut ref_mv);
        let mut mv = Vec::new();
        let mixed_query: &[u8] =
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGG";
        mm_sketch(mixed_query, w, k, 1, false, &mut mv);
        assert!(!mv.is_empty());

        let result = collect_matches(&mi, &mv, mixed_query.len() as i32, 1000, 5000, 500);
        // mini_pos must never exceed the number of seeds that actually had
        // matches in the index; the unmatched minimizers from the unique
        // suffix must NOT leak into mini_pos.
        assert!(
            result.mini_pos.len() <= result.seeds.len(),
            "mini_pos ({}) must be ≤ seeds ({}); extra entries come from \
             unmatched minimizers and break mm_est_err parity",
            result.mini_pos.len(),
            result.seeds.len()
        );
        // And there must be fewer mini_pos entries than total query
        // minimizers, proving unmatched ones were dropped.
        assert!(
            result.mini_pos.len() < mv.len(),
            "mv has {} minimizers but all {} ended up in mini_pos — the \
             no-match branch is leaking again",
            mv.len(),
            result.mini_pos.len()
        );
    }

    /// Invariant check: mini_pos entries must encode the exact
    /// `(q_span << 32) | (q_pos >> 1)` layout C emits at seed.c:125.
    #[test]
    fn test_collect_matches_mini_pos_encoding() {
        let mi = make_test_index();
        let mut mv = Vec::new();
        mm_sketch(
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            5,
            10,
            0,
            false,
            &mut mv,
        );
        let result = collect_matches(&mi, &mv, 54, 1000, 5000, 500);
        for (i, &mp) in result.mini_pos.iter().enumerate() {
            let span = (mp >> 32) & 0xff;
            assert!(span > 0, "mini_pos[{i}] has zero q_span");
            // Every mini_pos entry must correspond to one of the retained seeds.
            let pos = mp as u32;
            let found = result
                .seeds
                .iter()
                .any(|s| !s.flt && s.q_span as u64 == span && s.q_pos >> 1 == pos);
            assert!(
                found,
                "mini_pos[{i}] (span={span}, pos={pos}) has no matching \
                 non-filtered seed"
            );
        }
    }

    #[test]
    fn test_expand_seeds() {
        let mi = make_test_index();
        let mut mv = Vec::new();
        mm_sketch(
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            5,
            10,
            0,
            false,
            &mut mv,
        );
        let result = collect_matches(&mi, &mv, 54, 1000, 5000, 500);
        let anchors = expand_seeds_to_anchors(
            &mi,
            &result.seeds,
            None,
            54,
            crate::flags::MapFlags::empty(),
        );
        assert_eq!(anchors.len() as i64, result.n_a);
    }

    #[test]
    fn test_heap() {
        let mut a = vec![
            5u64 << 32 | 0,
            3u64 << 32 | 1,
            8u64 << 32 | 2,
            1u64 << 32 | 3,
        ];
        heap_make(&mut a);
        // Max-heap: root should be the largest
        assert_eq!(a[0] >> 32, 8);
    }
}
