use crate::types::Mm128;

/// Radix sort for Mm128, sorting by the x field.
/// This matches radix_sort_128x from misc.c.
///
/// # Parameters
/// * `data` - slice sorted in-place by ascending `x` (minimizer hash<<8 | span)
pub fn radix_sort_mm128(data: &mut [Mm128]) {
    if data.len() <= RS_MIN_SIZE {
        insertion_sort_mm128_x(data);
        return;
    }
    radix_sort_mm128_x(data, 56);
}

const RS_MIN_SIZE: usize = 64;

fn insertion_sort_mm128_x(data: &mut [Mm128]) {
    for i in 1..data.len() {
        if data[i].x < data[i - 1].x {
            let tmp = data[i];
            let mut j = i;
            while j > 0 && tmp.x < data[j - 1].x {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = tmp;
        }
    }
}

fn radix_sort_mm128_x(data: &mut [Mm128], shift: u32) {
    const SIZE: usize = 256;
    let mut bucket_b = [0usize; SIZE];
    let mut bucket_e = [0usize; SIZE];

    for item in data.iter() {
        bucket_e[((item.x >> shift) & 0xff) as usize] += 1;
    }
    for k in 1..SIZE {
        bucket_e[k] += bucket_e[k - 1];
        bucket_b[k] = bucket_e[k - 1];
    }

    let mut k = 0usize;
    while k < SIZE {
        if bucket_b[k] != bucket_e[k] {
            let mut l = ((data[bucket_b[k]].x >> shift) & 0xff) as usize;
            if l != k {
                let mut tmp = data[bucket_b[k]];
                loop {
                    let swap = tmp;
                    tmp = data[bucket_b[l]];
                    data[bucket_b[l]] = swap;
                    bucket_b[l] += 1;
                    l = ((tmp.x >> shift) & 0xff) as usize;
                    if l == k {
                        break;
                    }
                }
                data[bucket_b[k]] = tmp;
                bucket_b[k] += 1;
            } else {
                bucket_b[k] += 1;
            }
        } else {
            k += 1;
        }
    }

    bucket_b[0] = 0;
    for k in 1..SIZE {
        bucket_b[k] = bucket_e[k - 1];
    }
    if shift > 0 {
        let next_shift = shift.saturating_sub(8);
        for k in 0..SIZE {
            let start = bucket_b[k];
            let end = bucket_e[k];
            let len = end - start;
            if len > RS_MIN_SIZE {
                radix_sort_mm128_x(&mut data[start..end], next_shift);
            } else if len > 1 {
                insertion_sort_mm128_x(&mut data[start..end]);
            }
        }
    }
}

/// Radix sort for u64 (8 bytes, LSB first).
/// This matches radix_sort_64 from misc.c.
///
/// # Parameters
/// * `data` - slice sorted in-place in ascending order
pub fn radix_sort_u64(data: &mut [u64]) {
    if data.len() < 256 {
        data.sort_unstable();
        return;
    }
    let mut buf = vec![0u64; data.len()];
    for shift in (0..64).step_by(8) {
        let mut counts = [0u32; 256];
        for &val in data.iter() {
            let byte = ((val >> shift) & 0xff) as usize;
            counts[byte] += 1;
        }
        let mut offsets = [0u32; 256];
        for i in 1..256 {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        for &val in data.iter() {
            let byte = ((val >> shift) & 0xff) as usize;
            buf[offsets[byte] as usize] = val;
            offsets[byte] += 1;
        }
        data.copy_from_slice(&buf);
    }
}

/// Selection algorithm: find the k-th smallest element in an unsorted array.
/// Matches ks_ksmall_uint32_t. Modifies array in-place.
///
/// # Parameters
/// * `arr` - non-empty slice; partially reordered around the k-th element
/// * `k` - 0-based rank to select; must be `< arr.len()`
pub fn ksmall_u32(arr: &mut [u32], k: usize) -> u32 {
    assert!(!arr.is_empty() && k < arr.len());
    let (_, &mut val, _) = arr.select_nth_unstable(k);
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_mm128() {
        let mut data: Vec<Mm128> = (0..1000)
            .map(|i| {
                let x = (i * 7 + 13) % 1000;
                Mm128::new(x, i)
            })
            .collect();

        radix_sort_mm128(&mut data);

        for i in 1..data.len() {
            assert!(data[i - 1].x <= data[i].x);
        }
    }

    #[test]
    fn test_radix_sort_u64() {
        let mut data: Vec<u64> = (0..1000).map(|i| (i * 7 + 13) % 1000).collect();
        radix_sort_u64(&mut data);
        for i in 1..data.len() {
            assert!(data[i - 1] <= data[i]);
        }
    }

    #[test]
    fn test_radix_sort_small() {
        let mut data = vec![Mm128::new(3, 0), Mm128::new(1, 0), Mm128::new(2, 0)];
        radix_sort_mm128(&mut data);
        assert_eq!(data[0].x, 1);
        assert_eq!(data[1].x, 2);
        assert_eq!(data[2].x, 3);
    }

    #[test]
    fn test_ksmall() {
        let mut arr = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        assert_eq!(ksmall_u32(&mut arr, 0), 0);
        let mut arr2 = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        assert_eq!(ksmall_u32(&mut arr2, 4), 4);
    }
}
