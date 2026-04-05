use crate::seq::SEQ_NT4_TABLE;
use crate::types::Mm128;

/// Invertible hash function for k-mers. Matches hash64() from sketch.c.
#[inline]
fn hash64(mut key: u64, mask: u64) -> u64 {
    key = (!key).wrapping_add(key << 21) & mask;
    key ^= key >> 24;
    key = key.wrapping_add(key << 3).wrapping_add(key << 8) & mask; // * 265
    key ^= key >> 14;
    key = key.wrapping_add(key << 2).wrapping_add(key << 4) & mask; // * 21
    key ^= key >> 28;
    key = key.wrapping_add(key << 31) & mask;
    key
}

/// Simplified circular deque for HPC k-mer span tracking.
struct TinyQueue {
    front: usize,
    count: usize,
    a: [i32; 32],
}

impl TinyQueue {
    fn new() -> Self {
        Self {
            front: 0,
            count: 0,
            a: [0; 32],
        }
    }

    #[inline]
    fn push(&mut self, x: i32) {
        self.a[(self.count + self.front) & 0x1f] = x;
        self.count += 1;
    }

    #[inline]
    fn shift(&mut self) -> i32 {
        if self.count == 0 {
            return -1;
        }
        let x = self.a[self.front];
        self.front = (self.front + 1) & 0x1f;
        self.count -= 1;
        x
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.front = 0;
    }
}

#[allow(clippy::needless_range_loop)]
/// Find symmetric (w,k)-minimizers on a DNA sequence.
///
/// Faithful port of mm_sketch() from sketch.c.
///
/// # Output encoding
/// - `p[i].x = hash64(kmer) << 8 | kmer_span`
/// - `p[i].y = rid << 32 | last_pos << 1 | strand`
///
/// Results are appended to `p`.
pub fn mm_sketch(seq: &[u8], w: usize, k: usize, rid: u32, is_hpc: bool, p: &mut Vec<Mm128>) {
    assert!(!seq.is_empty() && w > 0 && w < 256 && k > 0 && k <= 28);

    let shift1 = 2 * (k - 1);
    let mask: u64 = (1u64 << (2 * k)) - 1;
    let mut kmer = [0u64; 2];
    let mut l: usize = 0;
    let mut buf_pos: usize = 0;
    let mut min_pos: usize = 0;
    let mut kmer_span: usize = 0;
    let mut buf = vec![Mm128 { x: u64::MAX, y: u64::MAX }; w];
    let mut min = Mm128 { x: u64::MAX, y: u64::MAX };
    let mut tq = TinyQueue::new();

    p.reserve(seq.len() / w);

    let len = seq.len();
    let mut i: usize = 0;
    while i < len {
        let c = SEQ_NT4_TABLE[seq[i] as usize];
        let mut info = Mm128 { x: u64::MAX, y: u64::MAX };

        if c < 4 {
            if is_hpc {
                let mut skip_len: usize = 1;
                if i + 1 < len && SEQ_NT4_TABLE[seq[i + 1] as usize] == c {
                    skip_len = 2;
                    while i + skip_len < len {
                        if SEQ_NT4_TABLE[seq[i + skip_len] as usize] != c {
                            break;
                        }
                        skip_len += 1;
                    }
                    i += skip_len - 1;
                }
                tq.push(skip_len as i32);
                kmer_span += skip_len;
                if tq.count > k {
                    kmer_span -= tq.shift() as usize;
                }
            } else {
                kmer_span = if l + 1 < k { l + 1 } else { k };
            }
            kmer[0] = (kmer[0] << 2 | c as u64) & mask;
            kmer[1] = (kmer[1] >> 2) | ((3u64 ^ c as u64) << shift1);
            if kmer[0] == kmer[1] {
                i += 1;
                continue; // skip symmetric k-mers
            }
            let z = if kmer[0] < kmer[1] { 0usize } else { 1usize };
            l += 1;
            if l >= k && kmer_span < 256 {
                info.x = hash64(kmer[z], mask) << 8 | kmer_span as u64;
                info.y = (rid as u64) << 32 | (i as u64) << 1 | z as u64;
            }
        } else {
            l = 0;
            tq.reset();
            kmer_span = 0;
        }

        buf[buf_pos] = info;

        if l == w + k - 1 && min.x != u64::MAX {
            // special case for the first window
            for j in (buf_pos + 1)..w {
                if min.x == buf[j].x && buf[j].y != min.y {
                    p.push(buf[j]);
                }
            }
            for j in 0..buf_pos {
                if min.x == buf[j].x && buf[j].y != min.y {
                    p.push(buf[j]);
                }
            }
        }

        if info.x <= min.x {
            if l >= w + k && min.x != u64::MAX {
                p.push(min);
            }
            min = info;
            min_pos = buf_pos;
        } else if buf_pos == min_pos {
            if l >= w + k - 1 && min.x != u64::MAX {
                p.push(min);
            }
            // find new minimum in buffer
            min.x = u64::MAX;
            for j in (buf_pos + 1)..w {
                if min.x >= buf[j].x {
                    min = buf[j];
                    min_pos = j;
                }
            }
            for j in 0..=buf_pos {
                if min.x >= buf[j].x {
                    min = buf[j];
                    min_pos = j;
                }
            }
            if l >= w + k - 1 && min.x != u64::MAX {
                // write identical k-mers
                for j in (buf_pos + 1)..w {
                    if min.x == buf[j].x && min.y != buf[j].y {
                        p.push(buf[j]);
                    }
                }
                for j in 0..=buf_pos {
                    if min.x == buf[j].x && min.y != buf[j].y {
                        p.push(buf[j]);
                    }
                }
            }
        }

        buf_pos += 1;
        if buf_pos == w {
            buf_pos = 0;
        }
        i += 1;
    }
    if min.x != u64::MAX {
        p.push(min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash64() {
        let mask = (1u64 << 30) - 1; // k=15
        let h = hash64(0x12345, mask);
        assert_ne!(h, 0x12345);
        // hash should be invertible-ish (different inputs → different outputs)
        let h2 = hash64(0x12346, mask);
        assert_ne!(h, h2);
    }

    #[test]
    fn test_sketch_simple() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT"; // 32 bases
        let mut minimizers = Vec::new();
        mm_sketch(seq, 10, 15, 0, false, &mut minimizers);
        assert!(!minimizers.is_empty());

        // verify encoding
        for m in &minimizers {
            let span = m.x & 0xff;
            assert!(span > 0 && span <= 15);
            let strand = m.y & 1;
            assert!(strand <= 1);
            let pos = ((m.y >> 1) & 0x7fffffff) as usize;
            assert!(pos < seq.len());
            let rid = (m.y >> 32) as u32;
            assert_eq!(rid, 0);
        }
    }

    #[test]
    fn test_sketch_hpc() {
        let seq = b"AAACCCGGGTTTTACGTACGTACGTACGTACGT";
        let mut minimizers = Vec::new();
        mm_sketch(seq, 10, 15, 0, true, &mut minimizers);
        assert!(!minimizers.is_empty());
    }

    #[test]
    fn test_sketch_with_n() {
        let seq = b"ACGTACGTACNACGTACGTACGTACGTACGTACGT";
        let mut minimizers = Vec::new();
        mm_sketch(seq, 5, 10, 0, false, &mut minimizers);
        // N should break the k-mer chain but we should still get minimizers
        assert!(!minimizers.is_empty());
    }

    #[test]
    fn test_sketch_rid() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let mut minimizers = Vec::new();
        mm_sketch(seq, 10, 15, 42, false, &mut minimizers);
        for m in &minimizers {
            assert_eq!((m.y >> 32) as u32, 42);
        }
    }

    #[test]
    fn test_sketch_append() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let mut minimizers = Vec::new();
        mm_sketch(seq, 10, 15, 0, false, &mut minimizers);
        let n1 = minimizers.len();
        mm_sketch(seq, 10, 15, 1, false, &mut minimizers);
        assert!(minimizers.len() > n1); // appended
    }
}
