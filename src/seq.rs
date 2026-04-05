/// DNA base to 2-bit encoding: A=0, C=1, G=2, T/U=3, other=4.
/// Matches seq_nt4_table from sketch.c.
pub const SEQ_NT4_TABLE: [u8; 256] = [
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, // @ABCDEFGHIJKLMNO
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, // PQRSTUVWXYZ
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, // `abcdefghijklmno
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, // pqrstuvwxyz
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
];

/// Complement table for ASCII bases. Matches seq_comp_table from bseq.c.
pub const SEQ_COMP_TABLE: [u8; 256] = [
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
    // @    A    B    C    D    E    F    G    H    I    J    K    L    M    N    O
     64, b'T',b'V',b'G',b'H',b'E',b'F',b'C',b'D',b'I',b'J',b'M',b'L',b'K',b'N',b'O',
    // P    Q    R    S    T    U    V    W    X    Y    Z
    b'P',b'Q',b'Y',b'S',b'A',b'A',b'B',b'W',b'X',b'R',b'Z', 91,  92,  93,  94,  95,
    // `    a    b    c    d    e    f    g    h    i    j    k    l    m    n    o
     96, b't',b'v',b'g',b'h',b'e',b'f',b'c',b'd',b'i',b'j',b'm',b'l',b'k',b'n',b'o',
    // p    q    r    s    t    u    v    w    x    y    z
    b'p',b'q',b'y',b's',b'a',b'a',b'b',b'w',b'x',b'r',b'z',123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
];

/// "ACGTN" for converting 2-bit encoding back to ASCII.
pub const NT4_TO_CHAR: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];

/// Encode an ASCII base to 2-bit (0-3) or 4 for N/ambiguous.
#[inline]
pub fn encode_base(b: u8) -> u8 {
    SEQ_NT4_TABLE[b as usize]
}

/// Complement of an ASCII base.
#[inline]
pub fn complement(b: u8) -> u8 {
    SEQ_COMP_TABLE[b as usize]
}

/// Complement of a 2-bit encoded base (0-3). 0↔3, 1↔2.
#[inline]
pub fn complement_nt4(c: u8) -> u8 {
    3 - c
}

/// Reverse complement a sequence of ASCII bases in-place.
pub fn revcomp_ascii(seq: &mut [u8]) {
    let n = seq.len();
    for i in 0..n / 2 {
        let j = n - 1 - i;
        let a = complement(seq[i]);
        let b = complement(seq[j]);
        seq[i] = b;
        seq[j] = a;
    }
    if n % 2 == 1 {
        seq[n / 2] = complement(seq[n / 2]);
    }
}

/// Reverse complement a sequence of 2-bit encoded bases (0-3).
pub fn revcomp_nt4(seq: &mut [u8]) {
    let n = seq.len();
    for i in 0..n / 2 {
        let j = n - 1 - i;
        let a = complement_nt4(seq[i]);
        let b = complement_nt4(seq[j]);
        seq[i] = b;
        seq[j] = a;
    }
    if n % 2 == 1 {
        seq[n / 2] = complement_nt4(seq[n / 2]);
    }
}

/// Reverse a quality string in-place.
pub fn reverse_qual(qual: &mut [u8]) {
    qual.reverse();
}

/// 4-bit packed sequence operations.
/// 8 bases per u32, 4 bits each. Matches mm_seq4_set/mm_seq4_get from mmpriv.h.
pub mod packed {
    /// Set base `c` (0-15) at position `i` in packed sequence.
    #[inline]
    pub fn seq4_set(s: &mut [u32], i: usize, c: u8) {
        s[i >> 3] |= (c as u32) << (((i & 7) << 2) as u32);
    }

    /// Get base at position `i` from packed sequence.
    #[inline]
    pub fn seq4_get(s: &[u32], i: usize) -> u8 {
        ((s[i >> 3] >> ((i & 7) << 2)) & 0xf) as u8
    }

    /// Pack an ASCII sequence into 4-bit packed u32 array.
    /// Returns the packed array. Each u32 stores 8 bases.
    pub fn pack_seq(seq: &[u8]) -> Vec<u32> {
        let n_words = seq.len().div_ceil(8);
        let mut packed = vec![0u32; n_words];
        for (i, &b) in seq.iter().enumerate() {
            let c = super::encode_base(b);
            seq4_set(&mut packed, i, c);
        }
        packed
    }

    /// Unpack a 4-bit packed sequence to 2-bit encoded bases.
    pub fn unpack_seq(packed: &[u32], len: usize) -> Vec<u8> {
        let mut seq = Vec::with_capacity(len);
        for i in 0..len {
            seq.push(seq4_get(packed, i));
        }
        seq
    }
}

/// Compare read names, ignoring /1 or /2 suffixes.
pub fn qname_len(s: &[u8]) -> usize {
    let l = s.len();
    if l >= 3 && s[l - 1] >= b'0' && s[l - 1] <= b'9' && s[l - 2] == b'/' {
        l - 2
    } else {
        l
    }
}

pub fn qname_same(s1: &[u8], s2: &[u8]) -> bool {
    let l1 = qname_len(s1);
    let l2 = qname_len(s2);
    l1 == l2 && s1[..l1] == s2[..l2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_base() {
        assert_eq!(encode_base(b'A'), 0);
        assert_eq!(encode_base(b'C'), 1);
        assert_eq!(encode_base(b'G'), 2);
        assert_eq!(encode_base(b'T'), 3);
        assert_eq!(encode_base(b'a'), 0);
        assert_eq!(encode_base(b'N'), 4);
        assert_eq!(encode_base(b'U'), 3);
    }

    #[test]
    fn test_complement() {
        assert_eq!(complement(b'A'), b'T');
        assert_eq!(complement(b'T'), b'A');
        assert_eq!(complement(b'C'), b'G');
        assert_eq!(complement(b'G'), b'C');
        assert_eq!(complement(b'a'), b't');
        assert_eq!(complement(b'N'), b'N');
    }

    #[test]
    fn test_revcomp_ascii() {
        let mut seq = b"ACGT".to_vec();
        revcomp_ascii(&mut seq);
        assert_eq!(&seq, b"ACGT"); // palindrome

        let mut seq2 = b"AACG".to_vec();
        revcomp_ascii(&mut seq2);
        assert_eq!(&seq2, b"CGTT");

        let mut seq3 = b"ACG".to_vec();
        revcomp_ascii(&mut seq3);
        assert_eq!(&seq3, b"CGT");
    }

    #[test]
    fn test_packed_seq() {
        let seq = b"ACGTACGT";
        let packed = packed::pack_seq(seq);
        assert_eq!(packed.len(), 1); // 8 bases in 1 u32

        for (i, &b) in seq.iter().enumerate() {
            assert_eq!(packed::seq4_get(&packed, i), encode_base(b));
        }

        let unpacked = packed::unpack_seq(&packed, 8);
        for (i, &b) in seq.iter().enumerate() {
            assert_eq!(unpacked[i], encode_base(b));
        }
    }

    #[test]
    fn test_qname_same() {
        assert!(qname_same(b"read1/1", b"read1/2"));
        assert!(qname_same(b"read1", b"read1"));
        assert!(!qname_same(b"read1", b"read2"));
        assert!(!qname_same(b"read1/1", b"read2/1"));
    }
}
