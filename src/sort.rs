use crate::types::Mm128;

/// Radix sort for Mm128, sorting by the x field (8 bytes, LSB first).
/// This matches radix_sort_128x from misc.c.
pub fn radix_sort_mm128(data: &mut [Mm128]) {
    if data.len() < 256 {
        data.sort_unstable();
        return;
    }
    let mut buf = vec![Mm128::default(); data.len()];
    // 8 passes of 8-bit radix sort on the x field
    for shift in (0..64).step_by(8) {
        let mut counts = [0u32; 256];
        for item in data.iter() {
            let byte = ((item.x >> shift) & 0xff) as usize;
            counts[byte] += 1;
        }
        let mut offsets = [0u32; 256];
        for i in 1..256 {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        for item in data.iter() {
            let byte = ((item.x >> shift) & 0xff) as usize;
            buf[offsets[byte] as usize] = *item;
            offsets[byte] += 1;
        }
        data.copy_from_slice(&buf);
    }
}

/// Radix sort for u64 (8 bytes, LSB first).
/// This matches radix_sort_64 from misc.c.
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
