use super::MmIdx;
use crate::flags::IdxFlags;
#[cfg(test)]
use crate::junc::JUNC_ANNO;
use crate::junc::{JumpDb, JumpEdge};
use crate::types::{IdxSeq, MM_IDX_MAGIC};
use std::io::{self, BufReader, BufWriter, Read, Write};

const MM2RS_ALT_MAGIC: &[u8; 4] = b"ALT\0";
const MM2RS_JUMP_MAGIC: &[u8; 4] = b"JJP\0";

/// Write an index to a binary .mmi file. Matches mm_idx_dump().
pub fn idx_dump<W: Write>(w: &mut W, mi: &MmIdx) -> io::Result<()> {
    // Little-endian-only: we cast &[u32]/&[u64] directly to &[u8] to match
    // C's fwrite(ptr, elem_size, count, fp). Rewrite per-element on BE if needed.
    const _: () = assert!(cfg!(target_endian = "little"));

    // 64 KiB matches stdio's buffering granularity. Larger buffers batch
    // kernel writes into giant syscalls, which on NFS backlogs ACKs until
    // close() — causing multi-second close stalls.
    let mut w = BufWriter::with_capacity(1 << 16, w);

    w.write_all(MM_IDX_MAGIC)?;
    let header: [u32; 5] = [
        mi.w as u32,
        mi.k as u32,
        mi.bucket_bits as u32,
        mi.seqs.len() as u32,
        mi.flag.bits() as u32,
    ];
    w.write_all(u32_slice_as_bytes(&header))?;

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

    // Bucket output matches C's on-disk layout:
    //   [n : i32]               # number of u64 positions in p (multi-hit only)
    //   [p[0..n] : u64]         # multi-hit positions (singletons excluded)
    //   [size : u32]            # hash entry count
    //   (key, val) pairs : u64  # key LSB = 1 marks singleton (val = position)
    //                           # key LSB = 0 marks multi  (val = (offset<<32)|count)
    //
    // Our in-memory bucket always stores every position in p with
    // val = (offset<<32)|count, so we compact singletons into the kv stream
    // and rebuild a contiguous multi-only p[] at write time. idx_load()
    // expands both forms back into the unified all-in-p runtime layout.
    let mut compact_p: Vec<u64> = Vec::new();
    let mut kv_pairs: Vec<u64> = Vec::new();
    for b in &mi.buckets {
        compact_p.clear();
        kv_pairs.clear();
        if let Some(h) = &b.h {
            kv_pairs.reserve(h.len() * 2);
            for (&key, &val) in h.iter() {
                let offset = (val >> 32) as usize;
                let count = (val as u32) as usize;
                if count == 1 {
                    kv_pairs.push(key | 1);
                    kv_pairs.push(b.p[offset]);
                } else {
                    let new_offset = compact_p.len() as u64;
                    compact_p.extend_from_slice(&b.p[offset..offset + count]);
                    kv_pairs.push(key);
                    kv_pairs.push((new_offset << 32) | count as u64);
                }
            }
        }
        let n = compact_p.len() as i32;
        w.write_all(&n.to_le_bytes())?;
        w.write_all(u64_slice_as_bytes(&compact_p))?;
        let size: u32 = b.h.as_ref().map_or(0, |h| h.len() as u32);
        w.write_all(&size.to_le_bytes())?;
        w.write_all(u64_slice_as_bytes(&kv_pairs))?;
    }

    if !mi.flag.contains(IdxFlags::NO_SEQ) {
        let n_words = (sum_len.div_ceil(8)) as usize;
        let avail = mi.packed_seq.len().min(n_words);
        // Chunked write: C's stdio pushes a steady stream of ~4KB writes to
        // the kernel, letting NFS ACK them in parallel with computation. A
        // single giant write() syscall backlogs all ACKs to close() time and
        // stalls for ~10s on NFS. Chunking matches NFS wsize (512KB) and keeps
        // the pipeline full.
        let bytes = u32_slice_as_bytes(&mi.packed_seq[..avail]);
        const WRITE_CHUNK: usize = 1 << 19; // 512 KiB
        for chunk in bytes.chunks(WRITE_CHUNK) {
            w.write_all(chunk)?;
        }
        if n_words > avail {
            let pad_bytes = (n_words - avail) * 4;
            let zeros = [0u8; 4096];
            let mut remaining = pad_bytes;
            while remaining > 0 {
                let chunk = remaining.min(zeros.len());
                w.write_all(&zeros[..chunk])?;
                remaining -= chunk;
            }
        }
    }
    if mi.n_alt > 0 || mi.seqs.iter().any(|seq| seq.is_alt) {
        w.write_all(MM2RS_ALT_MAGIC)?;
        w.write_all(&(mi.seqs.len() as u32).to_le_bytes())?;
        let alt_flags: Vec<u8> = mi.seqs.iter().map(|s| s.is_alt as u8).collect();
        w.write_all(&alt_flags)?;
    }
    if let Some(jump_db) = &mi.jump_db {
        w.write_all(MM2RS_JUMP_MAGIC)?;
        w.write_all(&(jump_db.jumps.len() as u32).to_le_bytes())?;
        for jumps in &jump_db.jumps {
            w.write_all(&(jumps.len() as u32).to_le_bytes())?;
            for jump in jumps {
                w.write_all(&jump.off.to_le_bytes())?;
                w.write_all(&jump.off2.to_le_bytes())?;
                w.write_all(&jump.cnt.to_le_bytes())?;
                w.write_all(&jump.strand.to_le_bytes())?;
                w.write_all(&jump.flag.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

#[inline]
fn u32_slice_as_bytes(s: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding; len * 4 bytes always fit into the Vec's
    // allocation. target_endian assertion in idx_dump guarantees LE layout.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[inline]
fn u64_slice_as_bytes(s: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
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
                let mut key = u64::from_le_bytes(kv[0..8].try_into().unwrap());
                let val = u64::from_le_bytes(kv[8..16].try_into().unwrap());
                if key & 1 != 0 {
                    key &= !1;
                    let offset = bucket.p.len() as u64;
                    bucket.p.push(val);
                    h.insert(key, (offset << 32) | 1);
                    continue;
                }
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
    read_optional_extensions(&mut r, &mut mi)?;

    Ok(Some(mi))
}

fn read_optional_extensions<R: Read>(r: &mut R, mi: &mut MmIdx) -> io::Result<()> {
    loop {
        let mut magic = [0u8; 4];
        match r.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
        if &magic == MM2RS_ALT_MAGIC {
            read_alt_extension(r, mi)?;
        } else if &magic == MM2RS_JUMP_MAGIC {
            read_jump_extension(r, mi)?;
        } else {
            return Ok(());
        }
    }
}

fn read_alt_extension<R: Read>(r: &mut R, mi: &mut MmIdx) -> io::Result<()> {
    let mut n_buf = [0u8; 4];
    r.read_exact(&mut n_buf)?;
    let n_seq = u32::from_le_bytes(n_buf) as usize;
    let mut n_alt = 0i32;
    for i in 0..n_seq {
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        if let Some(seq) = mi.seqs.get_mut(i) {
            seq.is_alt = flag[0] != 0;
            if seq.is_alt {
                n_alt += 1;
            }
        }
    }
    mi.n_alt = n_alt;
    Ok(())
}

fn read_jump_extension<R: Read>(r: &mut R, mi: &mut MmIdx) -> io::Result<()> {
    let mut n_buf = [0u8; 4];
    r.read_exact(&mut n_buf)?;
    let n_seq = u32::from_le_bytes(n_buf) as usize;
    let mut jumps = Vec::with_capacity(n_seq);
    for _ in 0..n_seq {
        r.read_exact(&mut n_buf)?;
        let n_jump = u32::from_le_bytes(n_buf) as usize;
        let mut seq_jumps = Vec::with_capacity(n_jump);
        for _ in 0..n_jump {
            let mut off = [0u8; 4];
            let mut off2 = [0u8; 4];
            let mut cnt = [0u8; 4];
            let mut strand = [0u8; 2];
            let mut flag = [0u8; 2];
            r.read_exact(&mut off)?;
            r.read_exact(&mut off2)?;
            r.read_exact(&mut cnt)?;
            r.read_exact(&mut strand)?;
            r.read_exact(&mut flag)?;
            seq_jumps.push(JumpEdge {
                off: i32::from_le_bytes(off),
                off2: i32::from_le_bytes(off2),
                cnt: i32::from_le_bytes(cnt),
                strand: i16::from_le_bytes(strand),
                flag: u16::from_le_bytes(flag),
            });
        }
        jumps.push(seq_jumps);
    }
    mi.jump_db = Some(JumpDb { jumps });
    Ok(())
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
    fn test_dump_load_roundtrip_preserves_alt() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGTACGTACGTACGTACGT",
            b"TGCATGCATGCATGCATGCATGCATGCATGCA",
        ];
        let names = vec!["seq1", "seq2_alt"];
        let mut mi = MmIdx::build_from_str(10, 15, false, 14, &seqs, Some(&names)).unwrap();
        mi.seqs[1].is_alt = true;
        mi.n_alt = 1;

        let mut buf = Vec::new();
        idx_dump(&mut buf, &mi).unwrap();

        let mut cursor = io::Cursor::new(buf);
        let mi2 = idx_load(&mut cursor).unwrap().unwrap();

        assert_eq!(mi2.n_alt, 1);
        assert!(!mi2.seqs[0].is_alt);
        assert!(mi2.seqs[1].is_alt);
    }

    #[test]
    fn test_dump_load_roundtrip_preserves_jump_db() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let names = vec!["seq1"];
        let mut mi = MmIdx::build_from_str(10, 15, false, 14, &seqs, Some(&names)).unwrap();
        mi.jump_db = Some(JumpDb {
            jumps: vec![vec![
                JumpEdge {
                    off: 10,
                    off2: 30,
                    cnt: 2,
                    strand: 1,
                    flag: JUNC_ANNO,
                },
                JumpEdge {
                    off: 30,
                    off2: 10,
                    cnt: 2,
                    strand: 1,
                    flag: JUNC_ANNO,
                },
            ]],
        });

        let mut buf = Vec::new();
        idx_dump(&mut buf, &mi).unwrap();

        let mut cursor = io::Cursor::new(buf);
        let mi2 = idx_load(&mut cursor).unwrap().unwrap();
        let jump_db = mi2.jump_db.as_ref().unwrap();
        assert_eq!(jump_db.jumps.len(), 1);
        assert_eq!(jump_db.jumps[0].len(), 2);
        assert_eq!(jump_db.jumps[0][0].off, 10);
        assert_eq!(jump_db.jumps[0][0].off2, 30);
        assert_eq!(jump_db.jumps[0][0].cnt, 2);
        assert_eq!(jump_db.jumps[0][0].strand, 1);
        assert_eq!(jump_db.jumps[0][0].flag, JUNC_ANNO);
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
