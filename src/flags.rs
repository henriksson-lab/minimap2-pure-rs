use bitflags::bitflags;

bitflags! {
    /// Mapping flags (mm_mapopt_t::flag). Mirrors MM_F_* from minimap.h.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MapFlags: u64 {
        const NO_DIAG        = 0x001;
        const NO_DUAL        = 0x002;
        const CIGAR          = 0x004;
        const OUT_SAM        = 0x008;
        const NO_QUAL        = 0x010;
        const OUT_CG         = 0x020;
        const OUT_CS         = 0x040;
        const SPLICE         = 0x080;
        const SPLICE_FOR     = 0x100;
        const SPLICE_REV     = 0x200;
        const NO_LJOIN       = 0x400;
        const OUT_CS_LONG    = 0x800;
        const SR             = 0x1000;
        const FRAG_MODE      = 0x2000;
        const NO_PRINT_2ND   = 0x4000;
        const TWO_IO_THREADS = 0x8000;
        const LONG_CIGAR     = 0x10000;
        const INDEPEND_SEG   = 0x20000;
        const SPLICE_FLANK   = 0x40000;
        const SOFTCLIP       = 0x80000;
        const FOR_ONLY       = 0x100000;
        const REV_ONLY       = 0x200000;
        const HEAP_SORT      = 0x400000;
        const ALL_CHAINS     = 0x800000;
        const OUT_MD         = 0x1000000;
        const COPY_COMMENT   = 0x2000000;
        const EQX            = 0x4000000;
        const PAF_NO_HIT     = 0x8000000;
        const NO_END_FLT     = 0x10000000;
        const HARD_MLEVEL    = 0x20000000;
        const SAM_HIT_ONLY   = 0x40000000;
        const RMQ            = 0x80000000;
        const QSTRAND        = 0x100000000;
        const NO_INV         = 0x200000000;
        const NO_HASH_NAME   = 0x400000000;
        const SPLICE_OLD     = 0x800000000;
        const SECONDARY_SEQ  = 0x1000000000;
        const OUT_DS         = 0x2000000000;
        const WEAK_PAIRING   = 0x4000000000;
        const SR_RNA         = 0x8000000000;
        const OUT_JUNC       = 0x10000000000;
    }

    /// Index flags (mm_idxopt_t::flag). Mirrors MM_I_* from minimap.h.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct IdxFlags: i32 {
        const HPC     = 0x1;
        const NO_SEQ  = 0x2;
        const NO_NAME = 0x4;
    }

    /// KSW2 extension flags. Mirrors KSW_EZ_* from ksw2.h.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct KswFlags: i32 {
        const SCORE_ONLY  = 0x01;
        const RIGHT       = 0x02;
        const GENERIC_SC  = 0x04;
        const APPROX_MAX  = 0x08;
        const APPROX_DROP = 0x10;
        const EXTZ_ONLY   = 0x20;
        const REV_CIGAR   = 0x40;
        const SPLICE_FOR  = 0x80;
        const SPLICE_REV  = 0x100;
        const SPLICE_FLANK= 0x200;
        const SPLICE_CMPLX= 0x400;
        const SPLICE_SCORE= 0x800;
    }
}

/// CIGAR operation codes, matching BAM specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CigarOp {
    Match = 0,     // M
    Ins = 1,       // I
    Del = 2,       // D
    NSkip = 3,     // N
    SoftClip = 4,  // S
    HardClip = 5,  // H
    Padding = 6,   // P
    EqMatch = 7,   // =
    XMismatch = 8, // X
}

impl CigarOp {
    pub const CHARS: &[u8] = b"MIDNSHP=XB";

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Match),
            1 => Some(Self::Ins),
            2 => Some(Self::Del),
            3 => Some(Self::NSkip),
            4 => Some(Self::SoftClip),
            5 => Some(Self::HardClip),
            6 => Some(Self::Padding),
            7 => Some(Self::EqMatch),
            8 => Some(Self::XMismatch),
            _ => None,
        }
    }

    pub fn to_char(self) -> u8 {
        Self::CHARS[self as usize]
    }
}

bitflags! {
    /// Debug flags. Mirrors MM_DBG_* from mmpriv.h.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DbgFlags: i32 {
        const NO_KALLOC      = 0x1;
        const PRINT_QNAME    = 0x2;
        const PRINT_SEED     = 0x4;
        const PRINT_ALN_SEQ  = 0x8;
        const PRINT_CHAIN    = 0x10;
        const SEED_FREQ      = 0x20;
    }
}

/// Seed flags packed into the y field of Mm128. Mirrors MM_SEED_* from mmpriv.h.
pub const SEED_LONG_JOIN: u64 = 1 << 40;
pub const SEED_IGNORE: u64 = 1 << 41;
pub const SEED_TANDEM: u64 = 1 << 42;
pub const SEED_SELF: u64 = 1 << 43;
pub const SEED_SEG_SHIFT: u32 = 48;
pub const SEED_SEG_MASK: u64 = 0xff << SEED_SEG_SHIFT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_flags_values() {
        assert_eq!(MapFlags::NO_DIAG.bits(), 0x001);
        assert_eq!(MapFlags::SPLICE.bits(), 0x080);
        assert_eq!(MapFlags::SR.bits(), 0x1000);
        assert_eq!(MapFlags::RMQ.bits(), 0x80000000);
        assert_eq!(MapFlags::QSTRAND.bits(), 0x100000000);
        assert_eq!(MapFlags::OUT_JUNC.bits(), 0x10000000000);
    }

    #[test]
    fn test_cigar_op_chars() {
        assert_eq!(CigarOp::Match.to_char(), b'M');
        assert_eq!(CigarOp::Ins.to_char(), b'I');
        assert_eq!(CigarOp::Del.to_char(), b'D');
        assert_eq!(CigarOp::EqMatch.to_char(), b'=');
        assert_eq!(CigarOp::XMismatch.to_char(), b'X');
    }

    #[test]
    fn test_idx_flags() {
        assert_eq!(IdxFlags::HPC.bits(), 0x1);
        assert_eq!(IdxFlags::NO_SEQ.bits(), 0x2);
    }

    #[test]
    fn test_seed_constants() {
        assert_eq!(SEED_LONG_JOIN, 1u64 << 40);
        assert_eq!(SEED_SEG_SHIFT, 48);
    }
}
