use crate::align;
use crate::chain;
use crate::esterr;
use crate::flags::{IdxFlags, MapFlags, SEED_SEG_MASK, SEED_SEG_SHIFT};
use crate::format::paf;
use crate::hit;
use crate::index::MmIdx;
use crate::options::MapOpt;
use crate::sdust;
use crate::seed;
use crate::sketch::mm_sketch;
use crate::sort::radix_sort_mm128;
use crate::types::{AlignReg, Mm128};

/// Result of mapping a single query.
pub struct MapResult {
    pub regs: Vec<AlignReg>,
    pub rep_len: i32,
}

struct FragChainResult {
    anchors: Vec<Mm128>,
    chains: Vec<u64>,
    seed_result: seed::SeedResult,
    rep_len: i32,
    max_chain_gap_ref: i32,
}

fn dust_minimizers(minimizers: &mut Vec<Mm128>, qseq: &[u8], sdust_thres: i32) {
    if sdust_thres <= 0 || minimizers.is_empty() {
        return;
    }
    let intervals = sdust::sdust(qseq, sdust_thres, 64);
    if intervals.is_empty() {
        return;
    }

    let mut u = 0usize;
    minimizers.retain(|m| {
        let qpos = (m.y as u32 >> 1) as i32;
        let span = (m.x & 0xff) as i32;
        let start = qpos - (span - 1);
        let end = start + span;
        while u < intervals.len() && intervals[u].1 as i32 <= start {
            u += 1;
        }
        if u >= intervals.len() || intervals[u].0 as i32 >= end {
            return true;
        }

        let mut overlap = 0i32;
        let mut v = u;
        while v < intervals.len() && (intervals[v].0 as i32) < end {
            let ss = start.max(intervals[v].0 as i32);
            let ee = end.min(intervals[v].1 as i32);
            overlap += ee - ss;
            v += 1;
        }
        overlap <= span >> 1
    });
}

/// Map a single query sequence against the index.
///
/// This is the core mapping function, equivalent to mm_map_frag_core() for a single segment.
pub fn map_query(mi: &MmIdx, opt: &MapOpt, qname: &str, qseq: &[u8]) -> MapResult {
    let qlen = qseq.len() as i32;
    if qlen == 0 {
        return MapResult {
            regs: Vec::new(),
            rep_len: 0,
        };
    }
    if opt.max_qlen > 0 && qlen > opt.max_qlen {
        return MapResult {
            regs: Vec::new(),
            rep_len: 0,
        };
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
    let mut mv: Vec<Mm128> = Vec::with_capacity(qseq.len() / mi.w as usize + 16);
    mm_sketch(qseq, mi.w as usize, mi.k as usize, 0, is_hpc, &mut mv);
    dust_minimizers(&mut mv, qseq, opt.sdust_thres);

    // Step 2: Filter by query occurrence
    if opt.q_occ_frac > 0.0 {
        seed::seed_mz_flt(&mut mv, opt.mid_occ, opt.q_occ_frac);
    }

    // Step 3: Collect seed hits
    let mut seed_result =
        seed::collect_matches(mi, &mv, qlen, opt.mid_occ, opt.max_max_occ, opt.occ_dist);
    let mut rep_len = seed_result.rep_len;

    // Step 4: Expand seeds to anchors and sort
    let mut anchors = seed::expand_seeds_to_anchors(mi, &seed_result.seeds, qlen, opt.flag);
    if anchors.is_empty() {
        return MapResult {
            regs: Vec::new(),
            rep_len,
        };
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

    let mut chain_result = if opt.flag.contains(MapFlags::RMQ) {
        chain::rmq::lchain_rmq(
            opt.max_gap,
            opt.rmq_inner_dist,
            opt.bw,
            opt.max_chain_skip,
            opt.rmq_size_cap,
            opt.min_cnt,
            opt.min_chain_score,
            chn_pen_gap,
            chn_pen_skip,
            &anchors,
        )
    } else {
        chain::dp::lchain_dp(
            max_chain_gap_ref,
            max_chain_gap_qry,
            opt.bw,
            opt.max_chain_skip,
            opt.max_chain_iter,
            opt.min_cnt,
            opt.min_chain_score,
            chn_pen_gap,
            chn_pen_skip,
            is_splice,
            1, // n_segs=1
            &anchors,
        )
    };

    // C minimap2 re-runs chaining with max_occ for repetitive short-read cases.
    // For single-segment mapping this matters when the first pass finds no chain.
    if chain_result.is_none()
        && opt.max_occ > opt.mid_occ
        && rep_len > 0
        && !opt.flag.contains(MapFlags::RMQ)
    {
        seed_result =
            seed::collect_matches(mi, &mv, qlen, opt.max_occ, opt.max_max_occ, opt.occ_dist);
        rep_len = seed_result.rep_len;
        anchors = seed::expand_seeds_to_anchors(mi, &seed_result.seeds, qlen, opt.flag);
        if !anchors.is_empty() {
            radix_sort_mm128(&mut anchors);
            chain_result = chain::dp::lchain_dp(
                max_chain_gap_ref,
                max_chain_gap_qry,
                opt.bw,
                opt.max_chain_skip,
                opt.max_chain_iter,
                opt.min_cnt,
                opt.min_chain_score,
                chn_pen_gap,
                chn_pen_skip,
                is_splice,
                1,
                &anchors,
            );
        }
    }

    let (mut chain_anchors, chains) = match chain_result {
        Some(r) => (r.anchors, r.chains),
        None => {
            return MapResult {
                regs: Vec::new(),
                rep_len,
            }
        }
    };

    // Step 6: Generate alignment regions
    let mut regs = hit::gen_regs(
        hash,
        qlen,
        &chains,
        &chain_anchors,
        opt.flag.contains(MapFlags::QSTRAND),
    );

    // Step 6b: Estimate divergence
    if !seed_result.mini_pos.is_empty() {
        esterr::est_err(mi, qlen, &mut regs, &chain_anchors, &seed_result.mini_pos);
    }

    // Step 7: Mark ALT
    if mi.n_alt > 0 {
        let is_alt: Vec<bool> = mi.seqs.iter().map(|s| s.is_alt).collect();
        hit::mark_alt(mi.n_alt, &is_alt, &mut regs);
    }

    // Step 7b: Chain post-processing (set parent/secondary before alignment)
    if !opt.flag.contains(MapFlags::ALL_CHAINS) {
        hit::set_parent(
            opt.mask_level,
            opt.mask_len,
            &mut regs,
            opt.a * 2 + opt.b,
            opt.flag.contains(MapFlags::HARD_MLEVEL),
            opt.alt_drop,
        );
        hit::select_sub(opt.pri_ratio, mi.k * 2, opt.best_n, &mut regs);
    }

    // Step 8: DP alignment for CIGAR (if requested)
    if opt.flag.contains(MapFlags::CIGAR) {
        align::align_skeleton(opt, mi, qlen, qseq, &mut regs, &mut chain_anchors);
        // Re-do parent/secondary after alignment refinement
        if !opt.flag.contains(MapFlags::ALL_CHAINS) {
            hit::set_parent(
                opt.mask_level,
                opt.mask_len,
                &mut regs,
                opt.a * 2 + opt.b,
                opt.flag.contains(MapFlags::HARD_MLEVEL),
                opt.alt_drop,
            );
            hit::select_sub(opt.pri_ratio, mi.k * 2, opt.best_n, &mut regs);
            hit::set_sam_pri(&mut regs);
        }
    }

    // Step 9: Filter
    hit::filter_regs(opt, qlen, &mut regs);

    // Step 10: Set MAPQ
    hit::set_mapq(
        &mut regs,
        opt.min_chain_score,
        opt.a,
        rep_len,
        is_sr,
        is_splice,
    );

    // Step 11: Sort by score
    hit::hit_sort(&mut regs, opt.alt_drop);
    hit::sync_regs(&mut regs);

    MapResult { regs, rep_len }
}

/// Map a multi-segment fragment, such as paired-end reads.
///
/// This mirrors the multi-segment branch of C minimap2's mm_map_frag_core():
/// segment minimizers are chained together first, then split back into
/// per-segment chains for alignment and pairing.
pub fn map_frag_queries(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qseqs: &[&[u8]],
) -> Vec<MapResult> {
    let n_segs = qseqs.len();
    if n_segs == 0 {
        return Vec::new();
    }
    if n_segs == 1 {
        return vec![map_query(mi, opt, qname, qseqs[0])];
    }
    let qlens: Vec<i32> = qseqs.iter().map(|s| s.len() as i32).collect();
    let qlen_sum: i32 = qlens.iter().sum();
    if qlen_sum == 0 || opt.max_qlen > 0 && qlen_sum > opt.max_qlen {
        return qseqs.iter().map(|_| MapResult { regs: Vec::new(), rep_len: 0 }).collect();
    }

    let Some(frag) = chain_fragment(mi, opt, qname, qseqs, &qlens) else {
        return qseqs.iter().map(|_| MapResult { regs: Vec::new(), rep_len: 0 }).collect();
    };

    let hash = query_hash(qname, qlen_sum, opt);
    let mut regs0 = hit::gen_regs(
        hash,
        qlen_sum,
        &frag.chains,
        &frag.anchors,
        opt.flag.contains(MapFlags::QSTRAND),
    );

    if mi.n_alt > 0 {
        let is_alt: Vec<bool> = mi.seqs.iter().map(|s| s.is_alt).collect();
        hit::mark_alt(mi.n_alt, &is_alt, &mut regs0);
    }

    if !opt.flag.contains(MapFlags::ALL_CHAINS) {
        hit::set_parent(
            opt.mask_level,
            opt.mask_len,
            &mut regs0,
            opt.a * 2 + opt.b,
            opt.flag.contains(MapFlags::HARD_MLEVEL),
            opt.alt_drop,
        );
        let mut n_regs0 = regs0.len();
        crate::pe::select_sub_multi(
            opt.pri_ratio,
            0.2,
            0.7,
            frag.max_chain_gap_ref,
            mi.k * 2,
            opt.best_n,
            n_segs as i32,
            &qlens,
            &mut n_regs0,
            &mut regs0,
        );
    }

    if !opt.flag.contains(MapFlags::SR) && !opt.flag.contains(MapFlags::QSTRAND) && !frag.seed_result.mini_pos.is_empty() {
        esterr::est_err(mi, qlen_sum, &mut regs0, &frag.anchors, &frag.seed_result.mini_pos);
    }

    let mut segs = split_fragment_regs(hash, &qlens, &regs0, &frag.anchors);
    let is_splice = opt.flag.contains(MapFlags::SPLICE);
    let is_sr = opt.flag.contains(MapFlags::SR) || opt.flag.contains(MapFlags::SR_RNA);

    for s in 0..n_segs {
        if !opt.flag.contains(MapFlags::ALL_CHAINS) {
            hit::set_parent(
                opt.mask_level,
                opt.mask_len,
                &mut segs[s].regs,
                opt.a * 2 + opt.b,
                opt.flag.contains(MapFlags::HARD_MLEVEL),
                opt.alt_drop,
            );
        }
        if opt.flag.contains(MapFlags::CIGAR) {
            let seg = &mut segs[s];
            align::align_skeleton(opt, mi, qlens[s], qseqs[s], &mut seg.regs, &mut seg.anchors);
            if !opt.flag.contains(MapFlags::ALL_CHAINS) {
                hit::set_parent(
                    opt.mask_level,
                    opt.mask_len,
                    &mut segs[s].regs,
                    opt.a * 2 + opt.b,
                    opt.flag.contains(MapFlags::HARD_MLEVEL),
                    opt.alt_drop,
                );
                hit::select_sub(opt.pri_ratio, mi.k * 2, opt.best_n, &mut segs[s].regs);
                hit::set_sam_pri(&mut segs[s].regs);
            }
        } else {
            hit::filter_regs(opt, qlens[s], &mut segs[s].regs);
            hit::hit_sort(&mut segs[s].regs, opt.alt_drop);
        }
        hit::set_mapq(&mut segs[s].regs, opt.min_chain_score, opt.a, frag.rep_len, is_sr, is_splice);
        hit::sync_regs(&mut segs[s].regs);
    }

    segs.into_iter()
        .map(|seg| MapResult { regs: seg.regs, rep_len: frag.rep_len })
        .collect()
}

fn query_hash(qname: &str, qlen: i32, opt: &MapOpt) -> u32 {
    let hash = if !opt.flag.contains(MapFlags::NO_HASH_NAME) {
        hash_string(qname.as_bytes())
    } else {
        0u32
    };
    wang_hash(wang_hash(qlen as u32).wrapping_add(wang_hash(opt.seed as u32))) ^ hash
}

fn chain_fragment(
    mi: &MmIdx,
    opt: &MapOpt,
    _qname: &str,
    qseqs: &[&[u8]],
    qlens: &[i32],
) -> Option<FragChainResult> {
    let n_segs = qseqs.len() as i32;
    let qlen_sum: i32 = qlens.iter().sum();
    let is_splice = opt.flag.contains(MapFlags::SPLICE);
    let is_sr = opt.flag.contains(MapFlags::SR);

    let mut mv = collect_fragment_minimizers(mi, opt, qseqs, qlens);
    if opt.q_occ_frac > 0.0 {
        seed::seed_mz_flt(&mut mv, opt.mid_occ, opt.q_occ_frac);
    }

    let max_chain_gap_qry = if is_sr { qlen_sum.max(opt.max_gap) } else { opt.max_gap };
    let max_chain_gap_ref = if opt.max_gap_ref > 0 {
        opt.max_gap_ref
    } else if opt.max_frag_len > 0 {
        (opt.max_frag_len - qlen_sum).max(opt.max_gap)
    } else {
        opt.max_gap
    };
    let chn_pen_gap = opt.chain_gap_scale * 0.01 * mi.k as f32;
    let chn_pen_skip = opt.chain_skip_scale * 0.01 * mi.k as f32;

    let mut seed_result = seed::collect_matches(mi, &mv, qlen_sum, opt.mid_occ, opt.max_max_occ, opt.occ_dist);
    let mut rep_len = seed_result.rep_len;
    let mut anchors = seed::expand_seeds_to_anchors(mi, &seed_result.seeds, qlen_sum, opt.flag);
    if anchors.is_empty() {
        return None;
    }
    radix_sort_mm128(&mut anchors);
    let mut chain_result = chain::dp::lchain_dp(
        max_chain_gap_ref,
        max_chain_gap_qry,
        opt.bw,
        opt.max_chain_skip,
        opt.max_chain_iter,
        opt.min_cnt,
        opt.min_chain_score,
        chn_pen_gap,
        chn_pen_skip,
        is_splice,
        n_segs,
        &anchors,
    );

    let mut needs_rechain = chain_result.is_none();
    if !needs_rechain && opt.max_occ > opt.mid_occ && rep_len > 0 {
        let ch = chain_result.as_ref().unwrap();
        needs_rechain = best_chain_segment_count(&ch.chains, &ch.anchors) < n_segs as usize;
    }

    if needs_rechain
        && opt.max_occ > opt.mid_occ
        && rep_len > 0
        && !opt.flag.contains(MapFlags::RMQ)
    {
        seed_result = seed::collect_matches(mi, &mv, qlen_sum, opt.max_occ, opt.max_max_occ, opt.occ_dist);
        rep_len = seed_result.rep_len;
        anchors = seed::expand_seeds_to_anchors(mi, &seed_result.seeds, qlen_sum, opt.flag);
        if anchors.is_empty() {
            return None;
        }
        radix_sort_mm128(&mut anchors);
        chain_result = chain::dp::lchain_dp(
            max_chain_gap_ref,
            max_chain_gap_qry,
            opt.bw,
            opt.max_chain_skip,
            opt.max_chain_iter,
            opt.min_cnt,
            opt.min_chain_score,
            chn_pen_gap,
            chn_pen_skip,
            is_splice,
            n_segs,
            &anchors,
        );
    }

    let chain_result = chain_result?;
    Some(FragChainResult {
        anchors: chain_result.anchors,
        chains: chain_result.chains,
        seed_result,
        rep_len,
        max_chain_gap_ref,
    })
}

fn collect_fragment_minimizers(mi: &MmIdx, opt: &MapOpt, qseqs: &[&[u8]], qlens: &[i32]) -> Vec<Mm128> {
    let is_hpc = mi.flag.contains(IdxFlags::HPC);
    let cap = qseqs.iter().map(|s| s.len() / mi.w as usize + 16).sum();
    let mut mv = Vec::with_capacity(cap);
    let mut sum = 0i32;
    for (seg_id, seq) in qseqs.iter().enumerate() {
        let start = mv.len();
        mm_sketch(seq, mi.w as usize, mi.k as usize, seg_id as u32, is_hpc, &mut mv);
        if opt.sdust_thres > 0 {
            let mut segment = mv.split_off(start);
            dust_minimizers(&mut segment, seq, opt.sdust_thres);
            mv.extend(segment);
        }
        for m in &mut mv[start..] {
            m.y = m.y.wrapping_add((sum as u64) << 1);
        }
        sum += qlens[seg_id];
    }
    mv
}

fn best_chain_segment_count(chains: &[u64], anchors: &[Mm128]) -> usize {
    let Some((best_idx, _)) = chains.iter().enumerate().max_by_key(|(_, u)| *u >> 32) else {
        return 0;
    };
    let off: usize = chains[..best_idx].iter().map(|u| *u as u32 as usize).sum();
    let len = chains[best_idx] as u32 as usize;
    let mut seen = [false; crate::types::MM_MAX_SEG];
    let mut n = 0usize;
    for a in &anchors[off..off + len] {
        let sid = ((a.y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as usize;
        if sid < seen.len() && !seen[sid] {
            seen[sid] = true;
            n += 1;
        }
    }
    n
}

struct SegmentMap {
    regs: Vec<AlignReg>,
    anchors: Vec<Mm128>,
}

fn split_fragment_regs(hash: u32, qlens: &[i32], regs0: &[AlignReg], anchors: &[Mm128]) -> Vec<SegmentMap> {
    let n_segs = qlens.len();
    let mut acc = vec![0i32; n_segs + 1];
    for s in 1..=n_segs {
        acc[s] = acc[s - 1] + qlens[s - 1];
    }
    let qlen_sum = acc[n_segs];
    let mut seg_chains: Vec<Vec<u64>> = (0..n_segs).map(|_| Vec::with_capacity(regs0.len())).collect();
    let mut seg_anchors: Vec<Vec<Mm128>> = (0..n_segs).map(|_| Vec::new()).collect();

    for r in regs0 {
        let mut counts = vec![0u32; n_segs];
        for j in 0..r.cnt as usize {
            let sid = ((anchors[r.as_ as usize + j].y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as usize;
            if sid < n_segs {
                counts[sid] += 1;
            }
        }
        for s in 0..n_segs {
            if counts[s] > 0 {
                seg_chains[s].push(((r.score as u64) << 32) | counts[s] as u64);
            }
        }
        for j in 0..r.cnt as usize {
            let mut a = anchors[r.as_ as usize + j];
            let sid = ((a.y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as usize;
            if sid >= n_segs {
                continue;
            }
            let shift = if (a.x >> 63) != 0 {
                qlen_sum - (qlens[sid] + acc[sid])
            } else {
                acc[sid]
            };
            a.y = a.y.wrapping_sub(shift as u64);
            seg_anchors[sid].push(a);
        }
    }

    (0..n_segs)
        .map(|s| {
            let mut regs = hit::gen_regs(hash, qlens[s], &seg_chains[s], &seg_anchors[s], false);
            for r in &mut regs {
                r.seg_split = true;
                r.seg_id = s as u8;
            }
            SegmentMap { regs, anchors: std::mem::take(&mut seg_anchors[s]) }
        })
        .collect()
}

/// Format mapping results as PAF lines.
pub fn format_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qseq: &[u8],
    result: &MapResult,
) -> Vec<String> {
    format_paf_with_comment(mi, opt, qname, qseq, None, result)
}

pub fn format_paf_with_comment(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qseq: &[u8],
    comment: Option<&str>,
    result: &MapResult,
) -> Vec<String> {
    format_paf_segment_with_comment(mi, opt, qname, qseq, comment, result, 0, 0)
}

pub fn format_paf_segment_with_comment(
    mi: &MmIdx,
    opt: &MapOpt,
    qname: &str,
    qseq: &[u8],
    comment: Option<&str>,
    result: &MapResult,
    n_seg: i32,
    seg_idx: i32,
) -> Vec<String> {
    let qlen = qseq.len() as i32;
    let mut lines = Vec::new();
    if opt.flag.contains(MapFlags::OUT_JUNC) {
        for r in &result.regs {
            if r.id == r.parent && r.mapq >= 10 {
                lines.extend(format_junc(mi, qname, r));
            }
        }
        return lines;
    }
    if result.regs.is_empty() {
        if opt.flag.contains(MapFlags::PAF_NO_HIT) {
            lines.push(paf::write_paf(
                mi,
                qname,
                qlen,
                None,
                opt.flag,
                result.rep_len,
                n_seg,
                seg_idx,
                comment,
            ));
        }
    } else {
        // Encode query for cs/MD tag generation
        let qseq_enc: Vec<u8> = qseq
            .iter()
            .map(|&b| crate::seq::SEQ_NT4_TABLE[b as usize])
            .collect();
        let qseq_rc: Vec<u8> = qseq_enc
            .iter()
            .rev()
            .map(|&c| if c < 4 { 3 - c } else { 4 })
            .collect();

        for (i, r) in result.regs.iter().enumerate() {
            if i > 0 && opt.flag.contains(MapFlags::NO_PRINT_2ND) {
                break;
            }
            let mut line = paf::write_paf(
                mi,
                qname,
                qlen,
                Some(r),
                opt.flag,
                result.rep_len,
                n_seg,
                seg_idx,
                None,
            );

            // Append cs/MD tags if requested and CIGAR is available
            if r.extra.is_some() {
                let aligned_qseq = if r.rev { &qseq_rc } else { &qseq_enc };
                let qs = if r.rev { qlen - r.qe } else { r.qs };
                let qe = if r.rev { qlen - r.qs } else { r.qe };
                if qs >= 0 && qe > qs && (qe as usize) <= aligned_qseq.len() {
                    let qsub = &aligned_qseq[qs as usize..qe as usize];
                    if opt.flag.contains(MapFlags::OUT_CS) {
                        if let Some(cs_str) = crate::format::cs::gen_cs(
                            mi,
                            r,
                            qsub,
                            !opt.flag.contains(MapFlags::OUT_CS_LONG),
                        ) {
                            use std::fmt::Write;
                            write!(line, "\tcs:Z:{}", cs_str).unwrap();
                        }
                    }
                    if opt.flag.contains(MapFlags::OUT_DS) {
                        if let Some(ds_str) = crate::format::cs::gen_ds(
                            mi,
                            r,
                            qsub,
                            !opt.flag.contains(MapFlags::OUT_CS_LONG),
                        ) {
                            use std::fmt::Write;
                            write!(line, "\tds:Z:{}", ds_str).unwrap();
                        }
                    }
                    if opt.flag.contains(MapFlags::OUT_MD) {
                        if let Some(md_str) = crate::format::cs::gen_md(mi, r, qsub) {
                            use std::fmt::Write;
                            write!(line, "\tMD:Z:{}", md_str).unwrap();
                        }
                    }
                }
            }
            if opt.flag.contains(MapFlags::COPY_COMMENT) {
                if let Some(comment) = comment.filter(|c| !c.is_empty()) {
                    use std::fmt::Write;
                    write!(line, "\t{}", comment).unwrap();
                }
            }
            lines.push(line);
        }
    }
    lines
}

fn format_junc(mi: &MmIdx, qname: &str, r: &AlignReg) -> Vec<String> {
    let Some(ref p) = r.extra else {
        return Vec::new();
    };
    if !r.is_spliced || (p.trans_strand != 1 && p.trans_strand != 2) {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut t_off = r.rs;
    for &c in &p.cigar.0 {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 2 | 7 | 8 => t_off += len,
            3 => {
                if len >= 2 {
                    let rev = (p.trans_strand == 2) ^ r.rev;
                    let mut donor = [4u8; 2];
                    let mut acceptor = [4u8; 2];
                    if !rev {
                        mi.getseq(r.rid as u32, t_off as u32, (t_off + 2) as u32, &mut donor);
                        mi.getseq(
                            r.rid as u32,
                            (t_off + len - 2) as u32,
                            (t_off + len) as u32,
                            &mut acceptor,
                        );
                    } else {
                        mi.getseq(
                            r.rid as u32,
                            t_off as u32,
                            (t_off + 2) as u32,
                            &mut acceptor,
                        );
                        mi.getseq(
                            r.rid as u32,
                            (t_off + len - 2) as u32,
                            (t_off + len) as u32,
                            &mut donor,
                        );
                        revcomp_splice(&mut donor);
                        revcomp_splice(&mut acceptor);
                    }
                    let score1 = match (donor[0], donor[1]) {
                        (2, 3) => 3,
                        (2, 1) => 2,
                        (0, 3) => 1,
                        _ => 0,
                    };
                    let score2 = match (acceptor[0], acceptor[1]) {
                        (0, 2) => 3,
                        (0, 1) => 1,
                        _ => 0,
                    };
                    lines.push(format!(
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        mi.seqs[r.rid as usize].name,
                        t_off,
                        t_off + len,
                        qname,
                        score1 + score2,
                        if rev { '-' } else { '+' },
                    ));
                }
                t_off += len;
            }
            _ => {}
        }
    }
    lines
}

fn revcomp_splice(s: &mut [u8; 2]) {
    let c = if s[1] < 4 { 3 - s[1] } else { 4 };
    s[1] = if s[0] < 4 { 3 - s[0] } else { 4 };
    s[0] = c;
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
    use crate::flags::CigarOp;
    use crate::types::{AlignExtra, Cigar};

    fn make_test_data() -> (MmIdx, MapOpt) {
        let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let mi =
            MmIdx::build_from_str(10, 15, false, 14, &[ref_seq as &[u8]], Some(&["ref1"])).unwrap();
        let mut io = crate::options::IdxOpt::default();
        let mut mo = MapOpt::default();
        crate::options::set_opt(None, &mut io, &mut mo).unwrap();
        crate::options::mapopt_update(&mut mo, &mi);
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
        let lines = format_paf(&mi, &opt, "query1", query, &result);
        assert!(!lines.is_empty());
        // Check PAF format: should have at least 12 tab-separated fields
        let fields: Vec<&str> = lines[0].split('\t').collect();
        assert!(
            fields.len() >= 12,
            "PAF should have >=12 fields, got {}",
            fields.len()
        );
        assert_eq!(fields[0], "query1");
        assert_eq!(fields[5], "ref1");
    }

    #[test]
    fn test_format_paf_copy_comment_after_tags() {
        let (mi, mut opt) = make_test_data();
        opt.flag |= MapFlags::CIGAR | MapFlags::OUT_CG | MapFlags::OUT_CS | MapFlags::COPY_COMMENT;
        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let result = map_query(&mi, &opt, "query1", query);
        let lines =
            format_paf_with_comment(&mi, &opt, "query1", query, Some("comment text"), &result);
        assert!(!lines.is_empty());
        assert!(lines[0].contains("\tcs:Z:"));
        assert!(lines[0].ends_with("\tcomment text"));
    }

    #[test]
    fn test_format_write_junc() {
        let mi = MmIdx::build_from_str(
            5,
            3,
            false,
            14,
            &[b"ACGTGTCCAGACGT" as &[u8]],
            Some(&["ref1"]),
        )
        .unwrap();
        let mut opt = MapOpt::default();
        opt.flag |= MapFlags::OUT_JUNC | MapFlags::CIGAR;
        let mut r = AlignReg::default();
        r.rid = 0;
        r.rs = 0;
        r.re = 14;
        r.qs = 0;
        r.qe = 8;
        r.id = 0;
        r.parent = 0;
        r.mapq = 60;
        r.is_spliced = true;
        let mut extra = AlignExtra::default();
        extra.trans_strand = 1;
        extra.cigar = Cigar(vec![
            (4u32 << 4) | CigarOp::Match as u32,
            (6u32 << 4) | CigarOp::NSkip as u32,
            (4u32 << 4) | CigarOp::Match as u32,
        ]);
        r.extra = Some(Box::new(extra));
        let result = MapResult {
            regs: vec![r],
            rep_len: 0,
        };

        let lines = format_paf(&mi, &opt, "read1", b"ACGTACGT", &result);
        assert_eq!(lines, vec!["ref1\t4\t10\tread1\t6\t+".to_string()]);
    }

    #[test]
    fn test_dust_minimizers_removes_low_complexity_hits() {
        let mut minimizers = vec![
            Mm128::new((123u64 << 8) | 15, 20u64 << 1),
            Mm128::new((456u64 << 8) | 15, 80u64 << 1),
        ];
        let before = minimizers.len();
        dust_minimizers(
            &mut minimizers,
            b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            20,
        );
        assert!(minimizers.len() < before);
    }

    #[test]
    fn test_map_with_preset() {
        let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let mut io = crate::options::IdxOpt::default();
        let mut mo = MapOpt::default();
        crate::options::set_opt(Some("map-ont"), &mut io, &mut mo).unwrap();

        let mi = MmIdx::build_from_str(
            io.w as i32,
            io.k as i32,
            false,
            io.bucket_bits as i32,
            &[ref_seq as &[u8]],
            Some(&["chr1"]),
        )
        .unwrap();
        crate::options::mapopt_update(&mut mo, &mi);

        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let result = map_query(&mi, &mo, "read1", query);
        assert!(!result.regs.is_empty());
    }
}
