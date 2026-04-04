use crate::chain;
use crate::flags::{IdxFlags, MapFlags};
use crate::format::paf;
use crate::hit;
use crate::index::MmIdx;
use crate::options::MapOpt;
use crate::seed;
use crate::sketch::mm_sketch;
use crate::sort::radix_sort_mm128;
use crate::types::{AlignReg, Mm128};

/// Result of mapping a single query.
pub struct MapResult {
    pub regs: Vec<AlignReg>,
    pub rep_len: i32,
}

/// Map a single query sequence against the index.
///
/// This is the core mapping function, equivalent to mm_map_frag_core() for a single segment.
pub fn map_query(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qseq: &[u8],
) -> MapResult {
    let qlen = qseq.len() as i32;
    if qlen == 0 {
        return MapResult { regs: Vec::new(), rep_len: 0 };
    }
    if opt.max_qlen > 0 && qlen > opt.max_qlen {
        return MapResult { regs: Vec::new(), rep_len: 0 };
    }

    let is_splice = opt.flag.contains(MapFlags::SPLICE);
    let is_sr = opt.flag.contains(MapFlags::SR);

    // Hash the query name for deterministic tie-breaking
    let hash = if !opt.flag.contains(MapFlags::NO_HASH_NAME) {
        hash_string(qname.as_bytes())
    } else {
        0u32
    };
    let hash = wang_hash(wang_hash(qlen as u32).wrapping_add(wang_hash(opt.seed as u32))) ^ hash;

    // Step 1: Collect minimizers
    let is_hpc = mi.flag.contains(IdxFlags::HPC);
    let mut mv: Vec<Mm128> = Vec::new();
    mm_sketch(qseq, mi.w as usize, mi.k as usize, 0, is_hpc, &mut mv);

    // Step 2: Filter by query occurrence
    if opt.q_occ_frac > 0.0 {
        seed::seed_mz_flt(&mut mv, opt.mid_occ, opt.q_occ_frac);
    }

    // Step 3: Collect seed hits
    let seed_result = seed::collect_matches(mi, &mv, qlen, opt.mid_occ, opt.max_max_occ, opt.occ_dist);
    let rep_len = seed_result.rep_len;

    // Step 4: Expand seeds to anchors and sort
    let mut anchors = seed::expand_seeds_to_anchors(mi, &seed_result.seeds);
    if anchors.is_empty() {
        return MapResult { regs: Vec::new(), rep_len };
    }
    radix_sort_mm128(&mut anchors);

    // Step 5: Chain
    let chn_pen_gap = opt.chain_gap_scale * 0.01 * mi.k as f32;
    let chn_pen_skip = opt.chain_skip_scale * 0.01 * mi.k as f32;

    let max_chain_gap_qry = if is_sr {
        qlen.max(opt.max_gap)
    } else {
        opt.max_gap
    };
    let max_chain_gap_ref = if opt.max_gap_ref > 0 {
        opt.max_gap_ref
    } else if opt.max_frag_len > 0 {
        (opt.max_frag_len - qlen).max(opt.max_gap)
    } else {
        opt.max_gap
    };

    let chain_result = if opt.flag.contains(MapFlags::RMQ) {
        chain::rmq::lchain_rmq(
            opt.max_gap, opt.rmq_inner_dist, opt.bw,
            opt.max_chain_skip, opt.rmq_size_cap,
            opt.min_cnt, opt.min_chain_score,
            chn_pen_gap, chn_pen_skip,
            &anchors,
        )
    } else {
        chain::dp::lchain_dp(
            max_chain_gap_ref, max_chain_gap_qry, opt.bw,
            opt.max_chain_skip, opt.max_chain_iter,
            opt.min_cnt, opt.min_chain_score,
            chn_pen_gap, chn_pen_skip,
            is_splice, 1, // n_segs=1
            &anchors,
        )
    };

    let (chain_anchors, chains) = match chain_result {
        Some(r) => (r.anchors, r.chains),
        None => return MapResult { regs: Vec::new(), rep_len },
    };

    // Step 6: Generate alignment regions
    let mut regs = hit::gen_regs(hash, qlen, &chains, &chain_anchors, false);

    // Step 7: Mark ALT
    if mi.n_alt > 0 {
        let is_alt: Vec<bool> = mi.seqs.iter().map(|s| s.is_alt).collect();
        hit::mark_alt(mi.n_alt, &is_alt, &mut regs);
    }

    // Step 8: Set parent/secondary, select sub
    if !opt.flag.contains(MapFlags::ALL_CHAINS) {
        hit::set_parent(
            opt.mask_level, opt.mask_len,
            &mut regs, opt.a * 2 + opt.b,
            opt.flag.contains(MapFlags::HARD_MLEVEL),
            opt.alt_drop,
        );
        hit::select_sub(opt.pri_ratio, mi.k as i32 * 2, opt.best_n, &mut regs);
    }

    // Step 9: Filter
    hit::filter_regs(opt, qlen, &mut regs);

    // Step 10: Set MAPQ
    hit::set_mapq(&mut regs, opt.min_chain_score, opt.a, rep_len, is_sr, is_splice);

    // Step 11: Sort by score
    hit::hit_sort(&mut regs, opt.alt_drop);
    hit::sync_regs(&mut regs);

    MapResult { regs, rep_len }
}

/// Format mapping results as PAF lines.
pub fn format_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qlen: i32,
    result: &MapResult,
) -> Vec<String> {
    let mut lines = Vec::new();
    if result.regs.is_empty() {
        if opt.flag.contains(MapFlags::PAF_NO_HIT) {
            lines.push(paf::write_paf(mi, qname, qlen, None, opt.flag, result.rep_len, 0, 0));
        }
    } else {
        for (i, r) in result.regs.iter().enumerate() {
            if i > 0 && opt.flag.contains(MapFlags::NO_PRINT_2ND) {
                break;
            }
            lines.push(paf::write_paf(mi, qname, qlen, Some(r), opt.flag, result.rep_len, 0, 0));
        }
    }
    lines
}

// Simple hash functions matching minimap2
fn hash_string(s: &[u8]) -> u32 {
    let mut h = 0u32;
    for &b in s {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    h
}

fn wang_hash(mut key: u32) -> u32 {
    key = !key.wrapping_add(key << 15);
    key ^= key >> 12;
    key = key.wrapping_add(key << 2);
    key ^= key >> 4;
    key = key.wrapping_mul(2057);
    key ^= key >> 16;
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (MmIdx, MapOpt) {
        let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let mi = MmIdx::build_from_str(
            10, 15, false, 14,
            &[ref_seq as &[u8]],
            Some(&["ref1"]),
        ).unwrap();
        let mut io = crate::options::IdxOpt::default();
        let mut mo = MapOpt::default();
        crate::options::set_opt(None, &mut io, &mut mo).unwrap();
        // Compute mid_occ from index (equivalent to mm_mapopt_update)
        if mo.mid_occ <= 0 {
            mo.mid_occ = mi.cal_max_occ(mo.mid_occ_frac);
            if mo.mid_occ < mo.min_mid_occ { mo.mid_occ = mo.min_mid_occ; }
        }
        (mi, mo)
    }

    #[test]
    fn test_map_query_self() {
        let (mi, opt) = make_test_data();
        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let result = map_query(&mi, &opt, "query1", query);
        // Self-mapping should produce at least one hit
        assert!(!result.regs.is_empty(), "Self-mapping should produce hits");
        assert_eq!(result.regs[0].rid, 0);
    }

    #[test]
    fn test_map_query_no_hit() {
        let (mi, opt) = make_test_data();
        let query = b"NNNNNNNNNNNNNNNN";
        let result = map_query(&mi, &opt, "query1", query);
        assert!(result.regs.is_empty());
    }

    #[test]
    fn test_map_query_empty() {
        let (mi, opt) = make_test_data();
        let result = map_query(&mi, &opt, "query1", b"");
        assert!(result.regs.is_empty());
    }

    #[test]
    fn test_format_paf() {
        let (mi, opt) = make_test_data();
        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let result = map_query(&mi, &opt, "query1", query);
        let lines = format_paf(&mi, &opt, "query1", query.len() as i32, &result);
        assert!(!lines.is_empty());
        // Check PAF format: should have at least 12 tab-separated fields
        let fields: Vec<&str> = lines[0].split('\t').collect();
        assert!(fields.len() >= 12, "PAF should have >=12 fields, got {}", fields.len());
        assert_eq!(fields[0], "query1");
        assert_eq!(fields[5], "ref1");
    }

    #[test]
    fn test_map_with_preset() {
        let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let mut io = crate::options::IdxOpt::default();
        let mut mo = MapOpt::default();
        crate::options::set_opt(Some("map-ont"), &mut io, &mut mo).unwrap();

        let mi = MmIdx::build_from_str(
            io.w as i32, io.k as i32, false, io.bucket_bits as i32,
            &[ref_seq as &[u8]],
            Some(&["chr1"]),
        ).unwrap();
        if mo.mid_occ <= 0 {
            mo.mid_occ = mi.cal_max_occ(mo.mid_occ_frac);
            if mo.mid_occ < mo.min_mid_occ { mo.mid_occ = mo.min_mid_occ; }
        }

        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let result = map_query(&mi, &mo, "read1", query);
        assert!(!result.regs.is_empty());
    }
}
