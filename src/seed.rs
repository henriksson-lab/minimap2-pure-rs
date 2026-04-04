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
    Multi { offset: usize, count: usize, bucket_idx: usize },
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
    for (i, p) in mv.iter().enumerate() {
        let q_pos = p.y as u32;
        let q_span = (p.x & 0xff) as u32;
        let minier = p.x >> 8;
        let (t, _positions) = mi.get(minier);
        if t == 0 {
            continue;
        }
        let seg_id = (p.y >> 32) as u32;

        // Detect tandem repeats (same minimizer hash as neighbors)
        let is_tandem = (i > 0 && p.x >> 8 == mv[i - 1].x >> 8)
            || (i + 1 < mv.len() && p.x >> 8 == mv[i + 1].x >> 8);

        // Store position info — we need to re-fetch from index when expanding
        let mask = (1u64 << mi.bucket_bits) - 1;
        let bucket_idx = (minier & mask) as usize;
        let key = (minier >> mi.bucket_bits) << 1;
        let positions = if t == 1 {
            // Singleton — get the value directly
            let (_, pos_slice) = mi.buckets[bucket_idx].get(key);
            SeedPositions::Single(pos_slice[0])
        } else {
            // Multi-hit — store reference info
            let bucket = &mi.buckets[bucket_idx];
            if let Some(h) = &bucket.h {
                if let Some(&val) = h.get(&key) {
                    let offset = (val >> 32) as usize;
                    let count = (val as u32) as usize;
                    SeedPositions::Multi { offset, count, bucket_idx }
                } else {
                    // Try singleton key
                    SeedPositions::Single(0)
                }
            } else {
                SeedPositions::Single(0)
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
                let ps = if last0 < 0 { 0 } else { (seeds[last0 as usize].q_pos >> 1) as i32 };
                let pe = if i == n { qlen } else { (seeds[i as usize].q_pos >> 1) as i32 };
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
pub fn collect_matches(
    mi: &MmIdx,
    mv: &[Mm128],
    qlen: i32,
    max_occ: i32,
    max_max_occ: i32,
    dist: i32,
) -> SeedResult {
    let mut seeds = seed_collect_all(mi, mv);
    let n_m0 = seeds.len();

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
    let mut mini_pos = Vec::with_capacity(n_m0);
    let mut kept = Vec::with_capacity(n_m0);

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
            mini_pos.push((q.q_span as u64) << 32 | (q.q_pos >> 1) as u64);
            kept.push(q.clone());
        }
    }
    rep_len += rep_en - rep_st;

    SeedResult {
        seeds: kept,
        n_a,
        rep_len,
        mini_pos,
    }
}

/// Expand seeds into anchor array (Mm128 format) for chaining.
///
/// Each seed with `n` positions generates `n` anchors:
/// - `a[i].x = rev << 63 | rid << 32 | ref_pos`
/// - `a[i].y = flags | seg_id << 48 | q_span << 32 | q_pos`
///
/// This is the core of what happens in mm_map_frag() after collect_matches.
pub fn expand_seeds_to_anchors(mi: &MmIdx, seeds: &[Seed]) -> Vec<Mm128> {
    let mut anchors = Vec::new();
    for seed in seeds {
        if seed.flt {
            continue;
        }
        let q_pos = seed.q_pos;
        let q_span = seed.q_span;
        let seg_id = seed.seg_id;

        match &seed.positions {
            SeedPositions::Single(pos) => {
                let mut y = q_pos as u64 | (q_span as u64) << 32;
                y |= (seg_id as u64) << 48;
                if seed.is_tandem {
                    y |= crate::flags::SEED_TANDEM;
                }
                anchors.push(Mm128::new(*pos, y));
            }
            SeedPositions::Multi { offset, count, bucket_idx } => {
                let bucket = &mi.buckets[*bucket_idx];
                for k in 0..*count {
                    let pos = bucket.p[*offset + k];
                    let mut y = q_pos as u64 | (q_span as u64) << 32;
                    y |= (seg_id as u64) << 48;
                    if seed.is_tandem {
                        y |= crate::flags::SEED_TANDEM;
                    }
                    anchors.push(Mm128::new(pos, y));
                }
            }
        }
    }
    anchors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::mm_sketch;

    fn make_test_index() -> MmIdx {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        ];
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
            5, 10, 0, false, &mut mv,
        );
        assert!(!mv.is_empty());

        let result = collect_matches(&mi, &mv, 54, 1000, 5000, 500);
        // Should have some seeds
        assert!(!result.seeds.is_empty());
        assert!(result.n_a > 0);
    }

    #[test]
    fn test_expand_seeds() {
        let mi = make_test_index();
        let mut mv = Vec::new();
        mm_sketch(
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            5, 10, 0, false, &mut mv,
        );
        let result = collect_matches(&mi, &mv, 54, 1000, 5000, 500);
        let anchors = expand_seeds_to_anchors(&mi, &result.seeds);
        assert_eq!(anchors.len() as i64, result.n_a);
    }

    #[test]
    fn test_heap() {
        let mut a = vec![5u64 << 32 | 0, 3u64 << 32 | 1, 8u64 << 32 | 2, 1u64 << 32 | 3];
        heap_make(&mut a);
        // Max-heap: root should be the largest
        assert_eq!(a[0] >> 32, 8);
    }
}
