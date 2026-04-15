use crate::flags::{IdxFlags, MapFlags};

/// Index options. Mirrors mm_idxopt_t.
#[derive(Clone, Debug)]
pub struct IdxOpt {
    pub k: i16,
    pub w: i16,
    pub flag: IdxFlags,
    pub bucket_bits: i16,
    pub mini_batch_size: i64,
    pub batch_size: u64,
}

impl Default for IdxOpt {
    /// Matches mm_idxopt_init() from options.c.
    fn default() -> Self {
        Self {
            k: 15,
            w: 10,
            flag: IdxFlags::empty(),
            bucket_bits: 14,
            mini_batch_size: 50_000_000,
            batch_size: 8_000_000_000,
        }
    }
}

/// Mapping options. Mirrors mm_mapopt_t.
#[derive(Clone, Debug)]
pub struct MapOpt {
    pub flag: MapFlags,
    pub seed: i32,
    pub sdust_thres: i32,
    pub max_qlen: i32,
    pub bw: i32,
    pub bw_long: i32,
    pub max_gap: i32,
    pub max_gap_ref: i32,
    pub max_frag_len: i32,
    pub max_chain_skip: i32,
    pub max_chain_iter: i32,
    pub min_cnt: i32,
    pub min_chain_score: i32,
    pub chain_gap_scale: f32,
    pub chain_skip_scale: f32,
    pub rmq_size_cap: i32,
    pub rmq_inner_dist: i32,
    pub rmq_rescue_size: i32,
    pub rmq_rescue_ratio: f32,
    pub mask_level: f32,
    pub mask_len: i32,
    pub pri_ratio: f32,
    pub best_n: i32,
    pub alt_drop: f32,
    pub a: i32,
    pub b: i32,
    pub q: i32,
    pub e: i32,
    pub q2: i32,
    pub e2: i32,
    pub transition: i32,
    pub sc_ambi: i32,
    pub noncan: i32,
    pub junc_bonus: i32,
    pub junc_pen: i32,
    pub zdrop: i32,
    pub zdrop_inv: i32,
    pub end_bonus: i32,
    pub min_dp_max: i32,
    pub min_ksw_len: i32,
    pub anchor_ext_len: i32,
    pub anchor_ext_shift: i32,
    pub max_clip_ratio: f32,
    pub rank_min_len: i32,
    pub rank_frac: f32,
    pub pe_ori: i32,
    pub pe_bonus: i32,
    pub jump_min_match: i32,
    pub mid_occ_frac: f32,
    pub q_occ_frac: f32,
    pub min_mid_occ: i32,
    pub max_mid_occ: i32,
    pub mid_occ: i32,
    pub max_occ: i32,
    pub max_max_occ: i32,
    pub occ_dist: i32,
    pub mini_batch_size: i64,
    pub max_sw_mat: i64,
    pub cap_kalloc: i64,
    pub split_prefix: Option<String>,
}

impl Default for MapOpt {
    /// Matches mm_mapopt_init() from options.c.
    fn default() -> Self {
        Self {
            flag: MapFlags::empty(),
            seed: 11,
            sdust_thres: 0,
            max_qlen: 0,
            bw: 500,
            bw_long: 20_000,
            max_gap: 5000,
            max_gap_ref: -1,
            max_frag_len: 0,
            max_chain_skip: 25,
            max_chain_iter: 5000,
            min_cnt: 3,
            min_chain_score: 40,
            chain_gap_scale: 0.8,
            chain_skip_scale: 0.0,
            rmq_size_cap: 100_000,
            rmq_inner_dist: 1000,
            rmq_rescue_size: 1000,
            rmq_rescue_ratio: 0.1,
            mask_level: 0.5,
            mask_len: i32::MAX,
            pri_ratio: 0.8,
            best_n: 5,
            alt_drop: 0.15,
            a: 2,
            b: 4,
            q: 4,
            e: 2,
            q2: 24,
            e2: 1,
            transition: 0,
            sc_ambi: 1,
            noncan: 0,
            junc_bonus: 0,
            junc_pen: 0,
            zdrop: 400,
            zdrop_inv: 200,
            end_bonus: -1,
            min_dp_max: 40 * 2, // min_chain_score * a
            min_ksw_len: 200,
            anchor_ext_len: 20,
            anchor_ext_shift: 6,
            max_clip_ratio: 1.0,
            rank_min_len: 500,
            rank_frac: 0.9,
            pe_ori: 0,
            pe_bonus: 33,
            jump_min_match: 3,
            mid_occ_frac: 2e-4,
            q_occ_frac: 0.01,
            min_mid_occ: 10,
            max_mid_occ: 1_000_000,
            mid_occ: 0,
            max_occ: 0,
            max_max_occ: 4095,
            occ_dist: 500,
            mini_batch_size: 500_000_000,
            max_sw_mat: 100_000_000,
            cap_kalloc: 500_000_000,
            split_prefix: None,
        }
    }
}

/// Apply a preset to index and mapping options.
/// Pass `None` to reset to defaults.
/// Returns `Ok(())` on success, `Err(preset_name)` if unknown.
pub fn set_opt(preset: Option<&str>, io: &mut IdxOpt, mo: &mut MapOpt) -> Result<(), String> {
    match preset {
        None => {
            *io = IdxOpt::default();
            *mo = MapOpt::default();
        }
        Some("lr" | "map-ont") => {
            // same as default — no changes
        }
        Some("ava-ont") => {
            io.flag = IdxFlags::empty();
            io.k = 15;
            io.w = 5;
            mo.flag |=
                MapFlags::ALL_CHAINS | MapFlags::NO_DIAG | MapFlags::NO_DUAL | MapFlags::NO_LJOIN;
            mo.min_chain_score = 100;
            mo.pri_ratio = 0.0;
            mo.max_chain_skip = 25;
            mo.bw = 2000;
            mo.bw_long = 2000;
            mo.occ_dist = 0;
        }
        Some("map10k" | "map-pb") => {
            io.flag |= IdxFlags::HPC;
            io.k = 19;
        }
        Some("ava-pb") => {
            io.flag |= IdxFlags::HPC;
            io.k = 19;
            io.w = 5;
            mo.flag |=
                MapFlags::ALL_CHAINS | MapFlags::NO_DIAG | MapFlags::NO_DUAL | MapFlags::NO_LJOIN;
            mo.min_chain_score = 100;
            mo.pri_ratio = 0.0;
            mo.max_chain_skip = 25;
            mo.bw_long = mo.bw;
            mo.occ_dist = 0;
        }
        Some(p @ ("lr:hq" | "map-hifi" | "map-ccs")) => {
            io.flag = IdxFlags::empty();
            io.k = 19;
            io.w = 19;
            mo.max_gap = 10000;
            mo.min_mid_occ = 50;
            mo.max_mid_occ = 500;
            if p == "map-hifi" || p == "map-ccs" {
                mo.a = 1;
                mo.b = 4;
                mo.q = 6;
                mo.q2 = 26;
                mo.e = 2;
                mo.e2 = 1;
                mo.min_dp_max = 200;
            }
        }
        Some("lr:hqae") => {
            io.flag = IdxFlags::empty();
            io.k = 25;
            io.w = 51;
            mo.flag |= MapFlags::RMQ;
            mo.min_mid_occ = 50;
            mo.max_mid_occ = 500;
            mo.rmq_inner_dist = 5000;
            mo.occ_dist = 200;
            mo.best_n = 100;
            mo.chain_gap_scale = 5.0;
        }
        Some("map-iclr-prerender") => {
            io.flag = IdxFlags::empty();
            io.k = 15;
            mo.b = 6;
            mo.transition = 1;
            mo.q = 10;
            mo.q2 = 50;
        }
        Some("map-iclr") => {
            io.flag = IdxFlags::empty();
            io.k = 19;
            mo.b = 6;
            mo.transition = 4;
            mo.q = 10;
            mo.q2 = 50;
        }
        Some(p) if p.starts_with("asm") => {
            io.flag = IdxFlags::empty();
            io.k = 19;
            io.w = 19;
            mo.bw = 1000;
            mo.bw_long = 100_000;
            mo.max_gap = 10000;
            mo.flag |= MapFlags::RMQ;
            mo.min_mid_occ = 50;
            mo.max_mid_occ = 500;
            mo.min_dp_max = 200;
            mo.best_n = 50;
            match p {
                "asm5" => {
                    mo.a = 1;
                    mo.b = 19;
                    mo.q = 39;
                    mo.q2 = 81;
                    mo.e = 3;
                    mo.e2 = 1;
                    mo.zdrop = 200;
                    mo.zdrop_inv = 200;
                }
                "asm10" => {
                    mo.a = 1;
                    mo.b = 9;
                    mo.q = 16;
                    mo.q2 = 41;
                    mo.e = 2;
                    mo.e2 = 1;
                    mo.zdrop = 200;
                    mo.zdrop_inv = 200;
                }
                "asm20" => {
                    mo.a = 1;
                    mo.b = 4;
                    mo.q = 6;
                    mo.q2 = 26;
                    mo.e = 2;
                    mo.e2 = 1;
                    mo.zdrop = 200;
                    mo.zdrop_inv = 200;
                    io.w = 10;
                }
                _ => return Err(p.to_string()),
            }
        }
        Some("short" | "sr") => {
            io.flag = IdxFlags::empty();
            io.k = 21;
            io.w = 11;
            mo.flag |= MapFlags::SR
                | MapFlags::FRAG_MODE
                | MapFlags::NO_PRINT_2ND
                | MapFlags::TWO_IO_THREADS
                | MapFlags::HEAP_SORT;
            mo.pe_ori = 1; // FR orientation
            mo.a = 2;
            mo.b = 8;
            mo.q = 12;
            mo.e = 2;
            mo.q2 = 24;
            mo.e2 = 1;
            mo.zdrop = 100;
            mo.zdrop_inv = 100;
            mo.end_bonus = 10;
            mo.max_frag_len = 800;
            mo.max_gap = 100;
            mo.bw = 100;
            mo.bw_long = 100;
            mo.pri_ratio = 0.5;
            mo.min_cnt = 2;
            mo.min_chain_score = 25;
            mo.min_dp_max = 40;
            mo.best_n = 20;
            mo.mid_occ = 1000;
            mo.max_occ = 5000;
            mo.mini_batch_size = 50_000_000;
        }
        Some(p @ ("splice" | "splice:hq" | "splice:sr" | "cdna")) => {
            io.flag = IdxFlags::empty();
            io.k = 15;
            io.w = 5;
            mo.flag |= MapFlags::SPLICE
                | MapFlags::SPLICE_FOR
                | MapFlags::SPLICE_REV
                | MapFlags::SPLICE_FLANK;
            mo.max_sw_mat = 0;
            mo.max_gap = 2000;
            mo.max_gap_ref = 200_000;
            mo.bw = 200_000;
            mo.bw_long = 200_000;
            mo.a = 1;
            mo.b = 2;
            mo.q = 2;
            mo.e = 1;
            mo.q2 = 32;
            mo.e2 = 0;
            mo.noncan = 9;
            mo.junc_bonus = 9;
            mo.junc_pen = 5;
            mo.zdrop = 200;
            mo.zdrop_inv = 100;
            if p == "splice:hq" {
                mo.noncan = 5;
                mo.b = 4;
                mo.q = 6;
                mo.q2 = 24;
            } else if p == "splice:sr" {
                mo.flag |= MapFlags::NO_PRINT_2ND
                    | MapFlags::TWO_IO_THREADS
                    | MapFlags::HEAP_SORT
                    | MapFlags::FRAG_MODE
                    | MapFlags::WEAK_PAIRING
                    | MapFlags::SR_RNA;
                mo.noncan = 5;
                mo.b = 4;
                mo.q = 6;
                mo.q2 = 24;
                mo.min_chain_score = 25;
                mo.min_dp_max = 40;
                mo.min_ksw_len = 20;
                mo.pe_ori = 1; // FR orientation
                mo.best_n = 10;
                mo.mini_batch_size = 100_000_000;
            }
        }
        Some(unknown) => return Err(unknown.to_string()),
    }
    Ok(())
}

/// Update mapping options based on the index. Mirrors mm_mapopt_update().
///
/// Computes `mid_occ` from the index if not set, and ensures `bw_long >= bw`.
pub fn mapopt_update(opt: &mut MapOpt, mi: &crate::index::MmIdx) {
    if opt.flag.contains(MapFlags::SPLICE_FOR) || opt.flag.contains(MapFlags::SPLICE_REV) {
        opt.flag |= MapFlags::SPLICE;
    }
    if opt.mid_occ <= 0 {
        opt.mid_occ = mi.cal_max_occ(opt.mid_occ_frac);
        if opt.mid_occ < opt.min_mid_occ {
            opt.mid_occ = opt.min_mid_occ;
        }
        if opt.max_mid_occ > opt.min_mid_occ && opt.mid_occ > opt.max_mid_occ {
            opt.mid_occ = opt.max_mid_occ;
        }
    }
    if opt.bw_long < opt.bw {
        opt.bw_long = opt.bw;
    }
}

/// Compute max splice score bonus. Mirrors mm_max_spsc_bonus().
pub fn max_spsc_bonus(mo: &MapOpt) -> i32 {
    let mut max_sc = (mo.q2 + 1) / 2 - 1;
    let alt = mo.q2 - mo.q;
    if alt > max_sc {
        max_sc = alt;
    }
    max_sc
}

/// Validate option combinations. Mirrors mm_check_opt().
/// Returns Ok(()) if valid, Err(msg) if not.
pub fn check_opt(io: &IdxOpt, mo: &MapOpt) -> Result<(), String> {
    if mo.bw > mo.bw_long {
        return Err(format!(
            "with '-rNUM1,NUM2', NUM1 ({}) can't be larger than NUM2 ({})",
            mo.bw, mo.bw_long
        ));
    }
    if mo.flag.contains(MapFlags::RMQ) && mo.flag.intersects(MapFlags::SR | MapFlags::SPLICE) {
        return Err("--rmq doesn't work with --sr or --splice".into());
    }
    if mo.split_prefix.is_some() && mo.flag.intersects(MapFlags::OUT_CS | MapFlags::OUT_MD) {
        return Err("--cs or --MD doesn't work with --split-prefix".into());
    }
    if io.k <= 0 || io.w <= 0 {
        return Err("-k and -w must be positive".into());
    }
    if mo.flag.contains(MapFlags::CIGAR) && io.flag.contains(IdxFlags::NO_SEQ) {
        return Err("CIGAR/SAM/cs/MD output requires target sequences in the index".into());
    }
    if mo.flag.contains(MapFlags::QSTRAND) && mo.flag.contains(MapFlags::CIGAR) {
        return Err("--qstrand is currently supported for PAF without CIGAR/SAM only".into());
    }
    if mo.best_n < 0 {
        return Err("-N must be no less than 0".into());
    }
    if mo.pri_ratio < 0.0 || mo.pri_ratio > 1.0 {
        return Err("-p must be within 0 and 1 (including 0 and 1)".into());
    }
    if mo.flag.contains(MapFlags::FOR_ONLY) && mo.flag.contains(MapFlags::REV_ONLY) {
        return Err("--for-only and --rev-only can't be applied at the same time".into());
    }
    if mo.e <= 0 || mo.q <= 0 {
        return Err("-O and -E must be positive".into());
    }
    if (mo.q != mo.q2 || mo.e != mo.e2) && !(mo.e > mo.e2 && mo.q + mo.e < mo.q2 + mo.e2) {
        return Err("dual gap penalties violating E1>E2 and O1+E1<O2+E2".into());
    }
    if (mo.q + mo.e) + (mo.q2 + mo.e2) > 127 {
        return Err("scoring system violating ({-O}+{-E})+({-O2}+{-E2}) <= 127".into());
    }
    if mo.sc_ambi < 0 || mo.sc_ambi >= mo.b {
        return Err("--score-N should be within [0,{-B})".into());
    }
    if mo.zdrop < mo.zdrop_inv {
        return Err("Z-drop should not be less than inversion-Z-drop".into());
    }
    if mo.flag.contains(MapFlags::NO_PRINT_2ND) && mo.flag.contains(MapFlags::ALL_CHAINS) {
        return Err("-X/-P and --secondary=no can't be applied at the same time".into());
    }
    if mo.flag.contains(MapFlags::QSTRAND)
        && (mo
            .flag
            .intersects(MapFlags::OUT_SAM | MapFlags::SPLICE | MapFlags::FRAG_MODE)
            || io.flag.contains(IdxFlags::HPC))
    {
        return Err("--qstrand doesn't work with -a, -H, --frag or --splice".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_idx_opt() {
        let opt = IdxOpt::default();
        assert_eq!(opt.k, 15);
        assert_eq!(opt.w, 10);
        assert_eq!(opt.bucket_bits, 14);
        assert_eq!(opt.mini_batch_size, 50_000_000);
        assert_eq!(opt.batch_size, 8_000_000_000);
    }

    #[test]
    fn test_default_map_opt() {
        let opt = MapOpt::default();
        assert_eq!(opt.seed, 11);
        assert_eq!(opt.a, 2);
        assert_eq!(opt.b, 4);
        assert_eq!(opt.q, 4);
        assert_eq!(opt.e, 2);
        assert_eq!(opt.q2, 24);
        assert_eq!(opt.e2, 1);
        assert_eq!(opt.min_chain_score, 40);
        assert_eq!(opt.min_dp_max, 80); // 40 * 2
        assert_eq!(opt.zdrop, 400);
        assert_eq!(opt.zdrop_inv, 200);
        assert_eq!(opt.best_n, 5);
    }

    #[test]
    fn test_preset_sr() {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        set_opt(Some("sr"), &mut io, &mut mo).unwrap();
        assert_eq!(io.k, 21);
        assert_eq!(io.w, 11);
        assert!(mo.flag.contains(MapFlags::SR));
        assert!(mo.flag.contains(MapFlags::FRAG_MODE));
        assert_eq!(mo.a, 2);
        assert_eq!(mo.b, 8);
        assert_eq!(mo.max_gap, 100);
        assert_eq!(mo.mid_occ, 1000);
    }

    #[test]
    fn test_preset_map_hifi() {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        set_opt(Some("map-hifi"), &mut io, &mut mo).unwrap();
        assert_eq!(io.k, 19);
        assert_eq!(io.w, 19);
        assert_eq!(mo.a, 1);
        assert_eq!(mo.b, 4);
        assert_eq!(mo.min_dp_max, 200);
    }

    #[test]
    fn test_preset_asm5() {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        set_opt(Some("asm5"), &mut io, &mut mo).unwrap();
        assert_eq!(io.k, 19);
        assert_eq!(io.w, 19);
        assert!(mo.flag.contains(MapFlags::RMQ));
        assert_eq!(mo.a, 1);
        assert_eq!(mo.b, 19);
        assert_eq!(mo.q, 39);
    }

    #[test]
    fn test_preset_splice() {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        set_opt(Some("splice"), &mut io, &mut mo).unwrap();
        assert_eq!(io.k, 15);
        assert_eq!(io.w, 5);
        assert!(mo.flag.contains(MapFlags::SPLICE));
        assert_eq!(mo.noncan, 9);
    }

    #[test]
    fn test_preset_unknown() {
        let mut io = IdxOpt::default();
        let mut mo = MapOpt::default();
        assert!(set_opt(Some("bogus"), &mut io, &mut mo).is_err());
    }

    #[test]
    fn test_check_opt_valid() {
        let io = IdxOpt::default();
        let mo = MapOpt::default();
        assert!(check_opt(&io, &mo).is_ok());
    }

    #[test]
    fn test_check_opt_bw_invalid() {
        let io = IdxOpt::default();
        let mut mo = MapOpt::default();
        mo.bw = 1000;
        mo.bw_long = 500;
        assert!(check_opt(&io, &mo).is_err());
    }
}
