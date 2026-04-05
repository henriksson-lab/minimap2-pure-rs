use std::io::{self, Read, Write, BufReader, BufWriter};
use crate::flags::IdxFlags;
use crate::types::{IdxSeq, MM_IDX_MAGIC};
use super::MmIdx;


/// Write an index to a binary .mmi file. Matches mm_idx_dump().
pub fn idx_dump<W: Write>(w: &mut W, mi: &MmIdx) -> io::Result<()> {
    let mut w = BufWriter::new(w);
    // Magic
    w.write_all(MM_IDX_MAGIC)?;
    // Header: w, k, b, n_seq, flag
    let header: [u32; 5] = [
        mi.w as u32,
        mi.k as u32,
        mi.bucket_bits as u32,
        mi.seqs.len() as u32,
        mi.flag.bits() as u32,
    ];
    for &val in &header {
        w.write_all(&val.to_le_bytes())?;
    }
    // Sequence metadata
    let mut sum_len: u64 = 0;
    for seq in &mi.seqs {
        let name_bytes = seq.name.as_bytes();
        let l = name_bytes.len() as u8;
        w.write_all(&[l])?;
        if l > 0 {
            w.write_all(name_bytes)?;
        }
        w.write_all(&seq.len.to_le_bytes())?;
        sum_len += seq.len as u64;
    }
    // Buckets
    for b in &mi.buckets {
        let n = b.p.len() as i32;
        w.write_all(&n.to_le_bytes())?;
        for &val in &b.p {
            w.write_all(&val.to_le_bytes())?;
        }
        let size: u32 = b.h.as_ref().map_or(0, |h| h.len() as u32);
        w.write_all(&size.to_le_bytes())?;
        if let Some(h) = &b.h {
            for (&key, &val) in h.iter() {
                w.write_all(&key.to_le_bytes())?;
                w.write_all(&val.to_le_bytes())?;
            }
        }
    }
    // Packed sequence
    if !mi.flag.contains(IdxFlags::NO_SEQ) {
        let n_words = (sum_len.div_ceil(8)) as usize;
        for i in 0..n_words {
            let val = if i < mi.packed_seq.len() { mi.packed_seq[i] } else { 0 };
            w.write_all(&val.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Load an index from a binary .mmi file. Matches mm_idx_load().
pub fn idx_load<R: Read>(r: &mut R) -> io::Result<Option<MmIdx>> {
    let mut r = BufReader::new(r);
    // Magic
    let mut magic = [0u8; 4];
    if r.read_exact(&mut magic).is_err() {
        return Ok(None);
    }
    if &magic != MM_IDX_MAGIC {
        return Ok(None);
    }
    // Header
    let mut header = [0u32; 5];
    for h in &mut header {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        *h = u32::from_le_bytes(buf);
    }
    let [w, k, b, n_seq, flag] = header;
    let idx_flags = IdxFlags::from_bits_truncate(flag as i32);

    let mut mi = MmIdx::new(w as i32, k as i32, b as i32, idx_flags);

    // Sequence metadata
    let mut sum_len: u64 = 0;
    for _ in 0..n_seq {
        let mut l_buf = [0u8; 1];
        r.read_exact(&mut l_buf)?;
        let l = l_buf[0] as usize;
        let name = if l > 0 {
            let mut name_buf = vec![0u8; l];
            r.read_exact(&mut name_buf)?;
            String::from_utf8_lossy(&name_buf).to_string()
        } else {
            String::new()
        };
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf);
        mi.seqs.push(IdxSeq {
            name,
            offset: sum_len,
            len,
            is_alt: false,
        });
        sum_len += len as u64;
    }

    // Buckets
    for bucket in &mut mi.buckets {
        let mut n_buf = [0u8; 4];
        r.read_exact(&mut n_buf)?;
        let n = i32::from_le_bytes(n_buf) as usize;
        bucket.p = Vec::with_capacity(n);
        for _ in 0..n {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            bucket.p.push(u64::from_le_bytes(buf));
        }
        let mut size_buf = [0u8; 4];
        r.read_exact(&mut size_buf)?;
        let size = u32::from_le_bytes(size_buf) as usize;
        if size > 0 {
            let mut h = hashbrown::HashMap::with_capacity(size);
            for _ in 0..size {
                let mut kv = [0u8; 16];
                r.read_exact(&mut kv)?;
                let key = u64::from_le_bytes(kv[0..8].try_into().unwrap());
                let val = u64::from_le_bytes(kv[8..16].try_into().unwrap());
                h.insert(key, val);
            }
            bucket.h = Some(h);
        }
    }

    // Packed sequence
    if !idx_flags.contains(IdxFlags::NO_SEQ) {
        let n_words = (sum_len.div_ceil(8)) as usize;
        mi.packed_seq = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            mi.packed_seq.push(u32::from_le_bytes(buf));
        }
    }

    Ok(Some(mi))
}

/// Check if a file is a minimap2 index by reading the magic bytes.
pub fn is_idx_file(path: &str) -> io::Result<bool> {
    if path == "-" {
        return Ok(false);
    }
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    match f.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == MM_IDX_MAGIC),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_load_roundtrip() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGT",
            b"TGCATGCATGCATGCATGCATGCATGCATGCA",
        ];
        let names = vec!["seq1", "seq2"];
        let mi = MmIdx::build_from_str(10, 15, false, 14, &seqs, Some(&names)).unwrap();

        // Dump to buffer
        let mut buf = Vec::new();
        idx_dump(&mut buf, &mi).unwrap();

        // Load from buffer
        let mut cursor = io::Cursor::new(buf);
        let mi2 = idx_load(&mut cursor).unwrap().unwrap();

        assert_eq!(mi2.w, mi.w);
        assert_eq!(mi2.k, mi.k);
        assert_eq!(mi2.bucket_bits, mi.bucket_bits);
        assert_eq!(mi2.seqs.len(), mi.seqs.len());
        assert_eq!(mi2.seqs[0].name, "seq1");
        assert_eq!(mi2.seqs[1].name, "seq2");
        assert_eq!(mi2.seqs[0].len, 32);
        assert_eq!(mi2.seqs[1].len, 32);

        // Verify sequence data survived
        let mut buf1 = vec![0u8; 4];
        let mut buf2 = vec![0u8; 4];
        mi.getseq(0, 0, 4, &mut buf1);
        mi2.getseq(0, 0, 4, &mut buf2);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_is_idx_file() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(MM_IDX_MAGIC).unwrap();
        f.flush().unwrap();
        assert!(is_idx_file(f.path().to_str().unwrap()).unwrap());

        let mut f2 = tempfile::NamedTempFile::new().unwrap();
        f2.write_all(b"NOPE").unwrap();
        f2.flush().unwrap();
        assert!(!is_idx_file(f2.path().to_str().unwrap()).unwrap());
    }
}
