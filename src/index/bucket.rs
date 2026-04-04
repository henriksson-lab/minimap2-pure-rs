use hashbrown::HashMap;
use crate::types::Mm128;
use crate::sort::{radix_sort_mm128, radix_sort_u64};

/// Index bucket. Each bucket holds minimizers whose hash falls into this bucket.
///
/// After construction (post-processing), the bucket contains:
/// - `h`: hash table mapping `minimizer >> bucket_bits << 1` to either:
///   - a direct position value (if singleton, key has LSB set), or
///   - `(offset << 32 | count)` into the `p` array
/// - `p`: position array for multi-occurrence minimizers
///
/// This matches mm_idx_bucket_s from index.c.
pub struct IdxBucket {
    /// Temporary storage during index building (cleared after post-processing).
    pub build_buf: Vec<Mm128>,
    /// Position array for multi-hit minimizers.
    pub p: Vec<u64>,
    /// Hash table: key = minimizer >> bucket_bits << 1 (LSB=1 if singleton).
    /// Value = position (singleton) or offset<<32|count (multi).
    pub h: Option<HashMap<u64, u64>>,
}

impl IdxBucket {
    pub fn new() -> Self {
        Self {
            build_buf: Vec::new(),
            p: Vec::new(),
            h: None,
        }
    }

    /// Add a minimizer during index construction.
    #[inline]
    pub fn add(&mut self, m: Mm128) {
        self.build_buf.push(m);
    }

    /// Post-process: sort minimizers, count occurrences, build hash table.
    /// `bucket_bits` is the number of bits used for bucket selection.
    ///
    /// Matches worker_post() from index.c.
    pub fn post_process(&mut self, bucket_bits: i32) {
        if self.build_buf.is_empty() {
            return;
        }

        // Sort by minimizer hash (x field)
        radix_sort_mm128(&mut self.build_buf);

        // Count distinct keys and total multi-hit positions
        let n = self.build_buf.len();
        let mut n_keys = 0u32;
        let mut n_multi = 0usize;
        let mut run_len = 1usize;
        for j in 1..=n {
            if j == n || self.build_buf[j].x >> 8 != self.build_buf[j - 1].x >> 8 {
                n_keys += 1;
                if run_len > 1 {
                    n_multi += run_len;
                }
                run_len = 1;
            } else {
                run_len += 1;
            }
        }

        // Build hash table and position array
        let mut h: HashMap<u64, u64> = HashMap::with_capacity(n_keys as usize);
        self.p = vec![0u64; n_multi];

        let mut start_a = 0usize;
        let mut start_p = 0usize;
        run_len = 1;
        for j in 1..=n {
            if j == n || self.build_buf[j].x >> 8 != self.build_buf[j - 1].x >> 8 {
                let last = &self.build_buf[j - 1];
                let key = (last.x >> 8 >> bucket_bits as u64) << 1;
                if run_len == 1 {
                    // Singleton: set LSB of key, value is the position
                    h.insert(key | 1, last.y);
                } else {
                    // Multi-hit: copy positions to p array
                    for k in 0..run_len {
                        self.p[start_p + k] = self.build_buf[start_a + k].y;
                    }
                    // Sort positions within this group
                    radix_sort_u64(&mut self.p[start_p..start_p + run_len]);
                    h.insert(key, ((start_p as u64) << 32) | run_len as u64);
                    start_p += run_len;
                }
                start_a = j;
                run_len = 1;
            } else {
                run_len += 1;
            }
        }
        debug_assert_eq!(n_multi, start_p);

        self.h = Some(h);
        // Clear build buffer
        self.build_buf = Vec::new();
    }

    /// Look up a minimizer. Returns (count, slice of positions).
    ///
    /// Matches mm_idx_get() from index.c.
    pub fn get(&self, minier_key: u64) -> (i32, &[u64]) {
        let h = match &self.h {
            Some(h) => h,
            None => return (0, &[]),
        };
        // Try singleton first (key with LSB set)
        if let Some(val) = h.get(&(minier_key | 1)) {
            // Singleton: need to return a reference to the value
            // This is tricky since we need a &[u64]. Use slice::from_ref for the value in the map.
            // But we can't get a stable reference to the value inside HashMap easily.
            // Instead, check both forms.
            return (1, std::slice::from_ref(val));
        }
        // Try multi-hit (key without LSB)
        if let Some(&val) = h.get(&minier_key) {
            let offset = (val >> 32) as usize;
            let count = (val as u32) as usize;
            return (count as i32, &self.p[offset..offset + count]);
        }
        (0, &[])
    }

    /// Count total distinct minimizers in this bucket.
    pub fn n_keys(&self) -> usize {
        self.h.as_ref().map_or(0, |h| h.len())
    }

    /// Iterate over all entries, yielding (count) for each minimizer.
    pub fn iter_counts(&self) -> impl Iterator<Item = u32> + '_ {
        self.h.iter().flat_map(|h| {
            h.iter().map(|(&key, &val)| {
                if key & 1 != 0 {
                    1u32 // singleton
                } else {
                    val as u32 // count
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_singleton() {
        let mut b = IdxBucket::new();
        // Add one minimizer: x = (hash << 8 | span), y = position
        b.add(Mm128::new(0xABCD00 | 15, 42));
        b.post_process(14);
        assert!(b.h.is_some());
        assert_eq!(b.n_keys(), 1);
    }

    #[test]
    fn test_bucket_multi() {
        let mut b = IdxBucket::new();
        let hash = 0xABCD00u64 | 15; // same hash
        b.add(Mm128::new(hash, 10));
        b.add(Mm128::new(hash, 20));
        b.add(Mm128::new(hash, 30));
        b.post_process(14);
        assert_eq!(b.n_keys(), 1);
        // Lookup
        let key = (hash >> 8 >> 14) << 1;
        let (n, positions) = b.get(key);
        assert_eq!(n, 3);
        assert_eq!(positions.len(), 3);
        // Positions should be sorted
        assert!(positions[0] <= positions[1] && positions[1] <= positions[2]);
    }

    #[test]
    fn test_bucket_empty() {
        let mut b = IdxBucket::new();
        b.post_process(14);
        assert!(b.h.is_none());
        let (n, _) = b.get(0);
        assert_eq!(n, 0);
    }
}
