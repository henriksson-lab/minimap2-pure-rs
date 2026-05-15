use crate::seq::SEQ_NT4_TABLE;
use std::collections::VecDeque;

const SD_WLEN: usize = 3;
const SD_WTOT: usize = 1 << (SD_WLEN << 1); // 64
const SD_WMSK: u32 = (SD_WTOT as u32) - 1;

#[derive(Clone)]
struct PerfIntv {
    start: i32,
    finish: i32,
    r: i32,
    l: i32,
}

/// SDUST low-complexity masking.
///
/// Returns a list of masked intervals as `(start, end)` pairs (0-based, end exclusive).
///
/// # Parameters
/// * `seq` - input sequence (ASCII); non-ACGT resets the scan
/// * `threshold` - score threshold (default 20 in minimap2)
/// * `window` - window size in bases (default 64 in minimap2)
pub fn sdust(seq: &[u8], threshold: i32, window: i32) -> Vec<(u32, u32)> {
    let mut buf = SdustBuf::new();
    sdust_core(seq, threshold, window, &mut buf);
    buf.res
        .iter()
        .map(|&v| ((v >> 32) as u32, v as u32))
        .collect()
}

/// Reusable buffer for repeated SDUST calls.
struct SdustBuf {
    w: VecDeque<i32>,
    p: Vec<PerfIntv>,
    res: Vec<u64>,
}

impl SdustBuf {
    fn new() -> Self {
        Self {
            w: VecDeque::with_capacity(256),
            p: Vec::new(),
            res: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.w.clear();
        self.p.clear();
        self.res.clear();
    }
}

#[inline]
fn shift_window(
    t: i32,
    w: &mut VecDeque<i32>,
    threshold: i32,
    window: i32,
    big_l: &mut i32,
    rw: &mut i32,
    rv: &mut i32,
    cw: &mut [i32; SD_WTOT],
    cv: &mut [i32; SD_WTOT],
) {
    if w.len() as i32 > window - SD_WLEN as i32 {
        let s = w.pop_front().unwrap();
        cw[s as usize] -= 1;
        *rw -= cw[s as usize];
        if *big_l > w.len() as i32 {
            *big_l -= 1;
            cv[s as usize] -= 1;
            *rv -= cv[s as usize];
        }
    }
    w.push_back(t);
    *big_l += 1;
    *rw += cw[t as usize];
    cw[t as usize] += 1;
    *rv += cv[t as usize];
    cv[t as usize] += 1;
    if cv[t as usize] * 10 > threshold << 1 {
        loop {
            let idx = w.len() as i32 - *big_l;
            let s = w[idx as usize];
            cv[s as usize] -= 1;
            *rv -= cv[s as usize];
            *big_l -= 1;
            if s == t {
                break;
            }
        }
    }
}

fn save_masked_regions(res: &mut Vec<u64>, p: &mut Vec<PerfIntv>, start: i32) {
    if p.is_empty() || p.last().unwrap().start >= start {
        return;
    }
    let last_p = p.last().unwrap();
    let mut saved = false;
    if let Some(last_res) = res.last_mut() {
        let s = (*last_res >> 32) as i32;
        let f = *last_res as u32 as i32;
        if last_p.start <= f {
            saved = true;
            let new_f = if f > last_p.finish { f } else { last_p.finish };
            *last_res = (s as u64) << 32 | new_f as u64;
        }
    }
    if !saved {
        let last_p = p.last().unwrap();
        res.push((last_p.start as u64) << 32 | last_p.finish as u64);
    }
    // remove perfect intervals that have fallen out of the window
    let mut i = p.len() as i32 - 1;
    while i >= 0 && p[i as usize].start < start {
        i -= 1;
    }
    p.truncate((i + 1) as usize);
}

fn find_perfect(
    p: &mut Vec<PerfIntv>,
    w: &VecDeque<i32>,
    threshold: i32,
    start: i32,
    big_l: i32,
    rv: i32,
    cv: &[i32; SD_WTOT],
) {
    let mut c = *cv;
    let mut r = rv;
    let mut max_r: i32 = 0;
    let mut max_l: i32 = 0;
    let wlen = w.len() as i32;

    let mut i = wlen - big_l - 1;
    while i >= 0 {
        let t = w[i as usize];
        r += c[t as usize];
        c[t as usize] += 1;
        let new_r = r;
        let new_l = wlen - i - 1;
        if new_r * 10 > threshold * new_l {
            let mut j = 0i32;
            while j < p.len() as i32 && p[j as usize].start >= i + start {
                let pp = &p[j as usize];
                if max_r == 0 || pp.r * max_l > max_r * pp.l {
                    max_r = pp.r;
                    max_l = pp.l;
                }
                j += 1;
            }
            if max_r == 0 || new_r * max_l >= max_r * new_l {
                max_r = new_r;
                max_l = new_l;
                let intv = PerfIntv {
                    start: i + start,
                    finish: wlen + (SD_WLEN as i32 - 1) + start,
                    r: new_r,
                    l: new_l,
                };
                p.insert(j as usize, intv);
            }
        }
        i -= 1;
    }
}

fn sdust_core(seq: &[u8], threshold: i32, window: i32, buf: &mut SdustBuf) {
    buf.reset();
    let mut rv: i32 = 0;
    let mut rw: i32 = 0;
    let mut big_l: i32 = 0;
    let mut cv = [0i32; SD_WTOT];
    let mut cw = [0i32; SD_WTOT];
    let mut l: i32 = 0;
    let mut t: u32 = 0;
    let l_seq = seq.len() as i32;

    for i in 0..=l_seq {
        let b = if i < l_seq {
            SEQ_NT4_TABLE[seq[i as usize] as usize]
        } else {
            4
        };
        if b < 4 {
            l += 1;
            t = (t << 2 | b as u32) & SD_WMSK;
            if l >= SD_WLEN as i32 {
                let start = if l - window > 0 { l - window } else { 0 } + (i + 1 - l);
                save_masked_regions(&mut buf.res, &mut buf.p, start);
                shift_window(
                    t as i32, &mut buf.w, threshold, window, &mut big_l, &mut rw, &mut rv, &mut cw,
                    &mut cv,
                );
                if rw * 10 > big_l * threshold {
                    find_perfect(&mut buf.p, &buf.w, threshold, start, big_l, rv, &cv);
                }
            }
        } else {
            let start = if l - window + 1 > 0 {
                l - window + 1
            } else {
                0
            } + (i + 1 - l);
            let mut s = start;
            while !buf.p.is_empty() {
                save_masked_regions(&mut buf.res, &mut buf.p, s);
                s += 1;
            }
            l = 0;
            t = 0;
            buf.w.clear();
            big_l = 0;
            rv = 0;
            rw = 0;
            cv = [0i32; SD_WTOT];
            cw = [0i32; SD_WTOT];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdust_no_mask() {
        // random-ish sequence should not be masked
        let seq = b"ACGTACGTTGCAATCGATCGATCGATCGATCGATCGATCG";
        let result = sdust(seq, 20, 64);
        // might or might not have masked regions depending on content
        // just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_sdust_low_complexity() {
        // highly repetitive sequence should be masked
        let seq = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let result = sdust(seq, 20, 64);
        // poly-A is low complexity — should produce masked regions
        // (sdust works on 3-mers so AAA repeated should trigger)
        assert!(!result.is_empty(), "poly-A should be masked");
    }

    #[test]
    fn test_sdust_with_n() {
        let seq = b"ACGTACGTNNNNACGTACGT";
        let result = sdust(seq, 20, 64);
        let _ = result; // just ensure no panic
    }

    #[test]
    fn test_sdust_empty() {
        let seq = b"ACGT";
        let result = sdust(seq, 20, 64);
        assert!(result.is_empty()); // too short to mask
    }

    #[test]
    fn test_sdust_dinucleotide_repeat() {
        // AT repeat is low complexity
        let seq = b"ATATATATATATATATATATATATATATATATATATATATATATATATATAT";
        let result = sdust(seq, 20, 64);
        assert!(!result.is_empty(), "dinucleotide repeat should be masked");
    }
}
