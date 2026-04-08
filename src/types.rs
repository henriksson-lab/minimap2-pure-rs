use crate::flags::CigarOp;

/// 128-bit minimizer/seed pair. Replaces mm128_t.
///
/// For minimizers (index building / sketch):
///   x = hash64(kmer) << 8 | kmer_span
///   y = rid << 32 | pos << 1 | strand
///
/// For seed hits (after collect_matches):
///   x = is_rev << 63 | rid << 32 | ref_pos
///   y = flags | seg_id << 48 | q_span << 32 | query_pos
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Mm128 {
    pub x: u64,
    pub y: u64,
}

impl Mm128 {
    #[inline]
    pub fn new(x: u64, y: u64) -> Self {
        Self { x, y }
    }
}

impl PartialOrd for Mm128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Mm128 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.x.cmp(&other.x).then(self.y.cmp(&other.y))
    }
}

/// Index sequence metadata. Replaces mm_idx_seq_t.
#[derive(Clone, Debug)]
pub struct IdxSeq {
    pub name: String,
    pub offset: u64,
    pub len: u32,
    pub is_alt: bool,
}

/// CIGAR string stored as BAM-style u32 values (len << 4 | op).
#[derive(Clone, Debug, Default)]
pub struct Cigar(pub Vec<u32>);

impl Cigar {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, op: CigarOp, len: u32) {
        self.0.push((len << 4) | (op as u32));
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Decode a single CIGAR element into (operation, length).
    #[inline]
    pub fn decode(cigar_val: u32) -> (CigarOp, u32) {
        let op = CigarOp::from_u8((cigar_val & 0xf) as u8).unwrap();
        let len = cigar_val >> 4;
        (op, len)
    }

    /// Format CIGAR as a human-readable string (e.g., "10M2I5M").
    pub fn format(&self) -> String {
        let mut s = String::new();
        for &c in &self.0 {
            let (op, len) = Self::decode(c);
            s.push_str(&len.to_string());
            s.push(op.to_char() as char);
        }
        s
    }
}

/// Extra alignment info attached to AlignReg. Replaces mm_extra_t.
#[derive(Clone, Debug)]
pub struct AlignExtra {
    pub dp_score: i32,
    pub dp_max: i32,
    pub dp_max2: i32,
    pub dp_max0: i32,
    pub n_ambi: u32,
    pub trans_strand: u8, // 0=unknown, 1=+, 2=-
    pub cigar: Cigar,
}

impl Default for AlignExtra {
    fn default() -> Self {
        Self {
            dp_score: 0,
            dp_max: 0,
            dp_max2: 0,
            dp_max0: 0,
            n_ambi: 0,
            trans_strand: 0,
            cigar: Cigar::new(),
        }
    }
}

/// Alignment region. Replaces mm_reg1_t.
#[derive(Clone, Debug)]
pub struct AlignReg {
    pub id: i32,
    pub cnt: i32,
    pub rid: i32,
    pub score: i32,
    pub qs: i32,
    pub qe: i32,
    pub rs: i32,
    pub re: i32,
    pub parent: i32,
    pub subsc: i32,
    pub as_: i32,
    pub mlen: i32,
    pub blen: i32,
    pub n_sub: i32,
    pub score0: i32,
    pub mapq: u8,
    pub split: u8,
    pub rev: bool,
    pub inv: bool,
    pub sam_pri: bool,
    pub proper_frag: bool,
    pub pe_thru: bool,
    pub seg_split: bool,
    pub seg_id: u8,
    pub split_inv: bool,
    pub is_alt: bool,
    pub strand_retained: bool,
    pub is_spliced: bool,
    pub hash: u32,
    pub div: f32,
    pub extra: Option<Box<AlignExtra>>,
}

impl Default for AlignReg {
    fn default() -> Self {
        Self {
            id: 0,
            cnt: 0,
            rid: 0,
            score: 0,
            qs: 0,
            qe: 0,
            rs: 0,
            re: 0,
            parent: -1, // MM_PARENT_UNSET
            subsc: 0,
            as_: 0,
            mlen: 0,
            blen: 0,
            n_sub: 0,
            score0: 0,
            mapq: 0,
            split: 0,
            rev: false,
            inv: false,
            sam_pri: false,
            proper_frag: false,
            pe_thru: false,
            seg_split: false,
            seg_id: 0,
            split_inv: false,
            is_alt: false,
            strand_retained: false,
            is_spliced: false,
            hash: 0,
            div: 0.0,
            extra: None,
        }
    }
}

/// Seed match from index lookup. Replaces mm_seed_t.
#[derive(Clone, Debug)]
pub struct Seed {
    pub n: u32,
    pub q_pos: u32,
    pub q_span: u32,
    pub flt: bool,
    pub seg_id: u32,
    pub is_tandem: bool,
    pub cr_offset: u32, // offset into index positions array
}

/// Segment info for multi-segment (paired-end) alignment. Replaces mm_seg_t.
#[derive(Clone, Debug)]
pub struct Seg {
    pub n_u: usize,
    pub n_a: usize,
    pub u: Vec<u64>,
    pub a: Vec<Mm128>,
}

/// Query sequence record for I/O. Replaces mm_bseq1_t.
#[derive(Clone, Debug)]
pub struct BseqRecord {
    pub name: String,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
    pub comment: String,
    pub rid: i32,
}

/// KSW2 alignment result. Replaces ksw_extz_t.
#[derive(Clone, Debug, Default)]
pub struct KswResult {
    pub max: i32,
    pub zdropped: bool,
    pub max_q: i32,
    pub max_t: i32,
    pub mqe: i32,
    pub mqe_t: i32,
    pub mte: i32,
    pub mte_q: i32,
    pub score: i32,
    pub cigar: Vec<u32>,
}

/// Constants
pub const MM_VERSION: &str = "2.30-rs";
pub const MM_IDX_MAGIC: &[u8; 4] = b"MMI\x02";
pub const MM_MAX_SEG: usize = 255;
pub const PARENT_UNSET: i32 = -1;
pub const PARENT_TMP_PRI: i32 = -2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm128_ordering() {
        let a = Mm128::new(1, 2);
        let b = Mm128::new(1, 3);
        let c = Mm128::new(2, 0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_cigar() {
        let mut c = Cigar::new();
        c.push(CigarOp::Match, 10);
        c.push(CigarOp::Ins, 2);
        c.push(CigarOp::Match, 5);
        assert_eq!(c.len(), 3);
        assert_eq!(c.format(), "10M2I5M");

        let (op, len) = Cigar::decode(c.0[0]);
        assert_eq!(op, CigarOp::Match);
        assert_eq!(len, 10);
    }

    #[test]
    fn test_align_reg_default() {
        let r = AlignReg::default();
        assert_eq!(r.parent, PARENT_UNSET);
        assert!(!r.rev);
        assert!(r.extra.is_none());
    }
}
