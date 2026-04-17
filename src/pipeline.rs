use crate::bseq::{BseqFile, BseqRecord};
use crate::flags::MapFlags;
use crate::format::sam;
use crate::index::split::{self, SplitQueryRecord};
use crate::index::MmIdx;
use crate::map;
use crate::options::{self, MapOpt};
use crate::pe;
use crate::seq;
use rayon::prelude::*;
use std::io::{self, BufWriter, Write};

fn add_rg_tag(line: &mut String, rg_id: Option<&str>) {
    let Some(rg_id) = rg_id else {
        return;
    };
    let tag = format!("\tRG:Z:{}", rg_id);
    let mut tabs_seen = 0usize;
    for (pos, ch) in line.char_indices() {
        if ch == '\t' {
            tabs_seen += 1;
            if tabs_seen == 11 {
                line.insert_str(pos, &tag);
                return;
            }
        }
    }
    line.push_str(&tag);
}

/// Map a FASTA/FASTQ file against the index and write PAF output to stdout.
pub fn map_file_paf(mi: &MmIdx, opt: &MapOpt, path: &str, n_threads: usize) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    // Set up rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let batch = fp.read_batch(opt.mini_batch_size, false)?;
        if batch.is_empty() {
            break;
        }

        // Map in parallel
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|rec| {
                    let result = map::map_query(mi, opt, &rec.name, &rec.seq);
                    map::format_paf_with_comment(
                        mi,
                        opt,
                        &rec.name,
                        &rec.seq,
                        Some(&rec.comment),
                        &result,
                    )
                })
                .collect()
        });

        // Write results sequentially
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

pub fn map_file_paf_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let batch = fp.read_batch(opt.mini_batch_size, false)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|rec| {
                    let result =
                        map_split_query(parts, &part_opts, &rid_shifts, opt, &rec.name, &rec.seq);
                    map::format_paf_with_comment(
                        mi,
                        opt,
                        &rec.name,
                        &rec.seq,
                        Some(&rec.comment),
                        &result,
                    )
                })
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

/// Map a FASTA/FASTQ file and write SAM output to stdout.
pub fn map_file_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    // Write SAM header
    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let with_qual = !opt.flag.contains(MapFlags::NO_QUAL);
    loop {
        let batch = fp.read_batch(opt.mini_batch_size, with_qual)?;
        if batch.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|rec| {
                    let result = map::map_query(mi, opt, &rec.name, &rec.seq);
                    let mut lines = Vec::new();
                    if result.regs.is_empty() {
                        if !opt.flag.contains(MapFlags::SAM_HIT_ONLY) {
                            let mut line = sam::write_sam_record_with_comment(
                                mi,
                                &rec.name,
                                &rec.seq,
                                &rec.qual,
                                None,
                                0,
                                &[],
                                opt.flag,
                                result.rep_len,
                                Some(&rec.comment),
                            );
                            add_rg_tag(&mut line, rg_id.as_deref());
                            lines.push(line);
                        }
                    } else {
                        for r in result.regs.iter() {
                            if opt.flag.contains(MapFlags::NO_PRINT_2ND) && r.id != r.parent {
                                continue;
                            }
                            let mut line = sam::write_sam_record_with_comment(
                                mi,
                                &rec.name,
                                &rec.seq,
                                &rec.qual,
                                Some(r),
                                result.regs.len(),
                                &result.regs,
                                opt.flag,
                                result.rep_len,
                                Some(&rec.comment),
                            );
                            add_rg_tag(&mut line, rg_id.as_deref());
                            lines.push(line);
                        }
                    }
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

pub fn map_file_sam_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let with_qual = !opt.flag.contains(MapFlags::NO_QUAL);
    loop {
        let batch = fp.read_batch(opt.mini_batch_size, with_qual)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|rec| {
                    let result =
                        map_split_query(parts, &part_opts, &rid_shifts, opt, &rec.name, &rec.seq);
                    let mut lines = Vec::new();
                    if result.regs.is_empty() {
                        if !opt.flag.contains(MapFlags::SAM_HIT_ONLY) {
                            let mut line = sam::write_sam_record_with_comment(
                                mi,
                                &rec.name,
                                &rec.seq,
                                &rec.qual,
                                None,
                                0,
                                &[],
                                opt.flag,
                                result.rep_len,
                                Some(&rec.comment),
                            );
                            add_rg_tag(&mut line, rg_id.as_deref());
                            lines.push(line);
                        }
                    } else {
                        for r in result.regs.iter() {
                            if opt.flag.contains(MapFlags::NO_PRINT_2ND) && r.id != r.parent {
                                continue;
                            }
                            let mut line = sam::write_sam_record_with_comment(
                                mi,
                                &rec.name,
                                &rec.seq,
                                &rec.qual,
                                Some(r),
                                result.regs.len(),
                                &result.regs,
                                opt.flag,
                                result.rep_len,
                                Some(&rec.comment),
                            );
                            add_rg_tag(&mut line, rg_id.as_deref());
                            lines.push(line);
                        }
                    }
                    lines
                })
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

struct Fragment {
    records: Vec<BseqRecord>,
}

fn read_fragment_batch(
    fp: &mut BseqFile,
    pending: &mut Option<BseqRecord>,
    chunk_size: i64,
    with_qual: bool,
) -> io::Result<Vec<Fragment>> {
    let mut fragments = Vec::new();
    let mut total_len = 0i64;
    loop {
        let first = match pending.take() {
            Some(rec) => rec,
            None => match fp.read_record()? {
                Some(rec) => rec,
                None => break,
            },
        };
        let mut records = vec![first];
        loop {
            match fp.read_record()? {
                Some(next) if seq::qname_same(records[0].name.as_bytes(), next.name.as_bytes()) => {
                    records.push(next);
                }
                Some(next) => {
                    *pending = Some(next);
                    break;
                }
                None => break,
            }
        }
        for rec in &mut records {
            if !with_qual {
                rec.qual.clear();
            }
            total_len += rec.l_seq as i64;
        }
        fragments.push(Fragment { records });
        if total_len >= chunk_size {
            break;
        }
    }
    Ok(fragments)
}

pub fn map_file_frag_paf(mi: &MmIdx, opt: &MapOpt, path: &str, n_threads: usize) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let mut pending = None;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let batch = read_fragment_batch(&mut fp, &mut pending, opt.mini_batch_size, false)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|frag| format_fragment_paf(mi, opt, None, frag))
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

pub fn map_file_frag_paf_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let mut pending = None;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let part_opts = prepare_part_opts(parts, opt);
    let shifts = rid_shifts(parts);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let batch = read_fragment_batch(&mut fp, &mut pending, opt.mini_batch_size, false)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|frag| format_fragment_paf(mi, opt, Some((parts, &part_opts, &shifts)), frag))
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

pub fn map_file_frag_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let mut pending = None;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let with_qual = !opt.flag.contains(MapFlags::NO_QUAL);
    loop {
        let batch = read_fragment_batch(&mut fp, &mut pending, opt.mini_batch_size, with_qual)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|frag| format_fragment_sam(mi, opt, None, frag, rg_id.as_deref()))
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

pub fn map_file_frag_sam_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let mut pending = None;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let part_opts = prepare_part_opts(parts, opt);
    let shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let with_qual = !opt.flag.contains(MapFlags::NO_QUAL);
    loop {
        let batch = read_fragment_batch(&mut fp, &mut pending, opt.mini_batch_size, with_qual)?;
        if batch.is_empty() {
            break;
        }
        let results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .map(|frag| {
                    format_fragment_sam(
                        mi,
                        opt,
                        Some((parts, &part_opts, &shifts)),
                        frag,
                        rg_id.as_deref(),
                    )
                })
                .collect()
        });
        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

type SplitRefs<'a> = (&'a [MmIdx], &'a [MapOpt], &'a [u32]);

fn map_fragment_results(
    mi: &MmIdx,
    opt: &MapOpt,
    split_refs: Option<SplitRefs<'_>>,
    frag: &Fragment,
) -> Vec<map::MapResult> {
    if frag.records.len() == 1 {
        let rec = &frag.records[0];
        return vec![match split_refs {
            Some((parts, part_opts, shifts)) => {
                map_split_query(parts, part_opts, shifts, opt, &rec.name, &rec.seq)
            }
            None => map::map_query(mi, opt, &rec.name, &rec.seq),
        }];
    }

    let mut oriented = Vec::with_capacity(frag.records.len());
    let mut revcomped = Vec::with_capacity(frag.records.len());
    for (i, rec) in frag.records.iter().enumerate() {
        let rev = if i == 0 {
            (opt.pe_ori >> 1) & 1 != 0
        } else if i == 1 {
            opt.pe_ori & 1 != 0
        } else {
            false
        };
        let mut seq = rec.seq.clone();
        if rev {
            seq::revcomp_ascii(&mut seq);
        }
        oriented.push(seq);
        revcomped.push(rev);
    }
    let qseqs: Vec<&[u8]> = oriented.iter().map(Vec::as_slice).collect();
    let qname = strip_pe_suffix(&frag.records[0].name);
    let used_split_refs = split_refs.is_some();
    let mut results = match split_refs {
        Some((parts, part_opts, shifts)) => {
            map_split_fragment_queries(parts, part_opts, shifts, opt, qname, &qseqs)
        }
        None => map::map_frag_queries(mi, opt, qname, &qseqs),
    };
    if results.len() == 2 {
        let (left, right) = results.split_at_mut(1);
        if used_split_refs {
            let split_merge_gap = right[0].frag_gap;
            pair_results_with_gap(
                opt,
                split_merge_gap,
                frag.records[0].l_seq as i32,
                frag.records[1].l_seq as i32,
                &mut left[0],
                &mut right[0],
            );
        } else {
            pair_results(
                opt,
                frag.records[0].l_seq as i32,
                frag.records[1].l_seq as i32,
                &mut left[0],
                &mut right[0],
            );
        }
    }
    for (result, (rec, rev)) in results.iter_mut().zip(frag.records.iter().zip(revcomped)) {
        restore_pair_orientation(result, rec.l_seq as i32, rev);
    }
    results
}

fn format_fragment_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    split_refs: Option<SplitRefs<'_>>,
    frag: &Fragment,
) -> Vec<String> {
    let results = map_fragment_results(mi, opt, split_refs, frag);
    let n_seg = frag.records.len() as i32;
    let mut lines = Vec::new();
    for (seg_idx, (rec, result)) in frag.records.iter().zip(&results).enumerate() {
        lines.extend(map::format_paf_segment_with_comment(
            mi,
            opt,
            &rec.name,
            &rec.seq,
            Some(&rec.comment),
            result,
            n_seg,
            seg_idx as i32,
        ));
    }
    lines
}

fn format_fragment_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    split_refs: Option<SplitRefs<'_>>,
    frag: &Fragment,
    rg_id: Option<&str>,
) -> Vec<String> {
    let results = map_fragment_results(mi, opt, split_refs, frag);
    let mut lines = Vec::new();
    if frag.records.len() == 2 {
        format_pe_sam_records(
            mi,
            opt,
            &frag.records[0],
            &results[0],
            &results[1],
            true,
            rg_id,
            &mut lines,
        );
        format_pe_sam_records(
            mi,
            opt,
            &frag.records[1],
            &results[1],
            &results[0],
            false,
            rg_id,
            &mut lines,
        );
    } else {
        for (rec, result) in frag.records.iter().zip(&results) {
            format_single_sam_records(mi, opt, rec, result, rg_id, &mut lines);
        }
    }
    lines
}

fn format_single_sam_records(
    mi: &MmIdx,
    opt: &MapOpt,
    rec: &BseqRecord,
    result: &map::MapResult,
    rg_id: Option<&str>,
    lines: &mut Vec<String>,
) {
    if result.regs.is_empty() {
        if !opt.flag.contains(MapFlags::SAM_HIT_ONLY) {
            let mut line = sam::write_sam_record_with_comment(
                mi,
                &rec.name,
                &rec.seq,
                &rec.qual,
                None,
                0,
                &[],
                opt.flag,
                result.rep_len,
                Some(&rec.comment),
            );
            add_rg_tag(&mut line, rg_id);
            lines.push(line);
        }
    } else {
        for r in result.regs.iter() {
            if opt.flag.contains(MapFlags::NO_PRINT_2ND) && r.id != r.parent {
                continue;
            }
            let mut line = sam::write_sam_record_with_comment(
                mi,
                &rec.name,
                &rec.seq,
                &rec.qual,
                Some(r),
                result.regs.len(),
                &result.regs,
                opt.flag,
                result.rep_len,
                Some(&rec.comment),
            );
            add_rg_tag(&mut line, rg_id);
            lines.push(line);
        }
    }
}

fn prepare_part_opts(parts: &[MmIdx], opt: &MapOpt) -> Vec<MapOpt> {
    parts
        .iter()
        .map(|part| {
            let mut part_opt = opt.clone();
            options::mapopt_update(&mut part_opt, part);
            part_opt
        })
        .collect()
}

fn rid_shifts(parts: &[MmIdx]) -> Vec<u32> {
    let mut shifts = Vec::with_capacity(parts.len());
    let mut shift = 0u32;
    for part in parts {
        shifts.push(shift);
        shift += part.seqs.len() as u32;
    }
    shifts
}

fn map_split_query(
    parts: &[MmIdx],
    part_opts: &[MapOpt],
    rid_shifts: &[u32],
    opt: &MapOpt,
    qname: &str,
    qseq: &[u8],
) -> map::MapResult {
    let records: Vec<SplitQueryRecord> = parts
        .iter()
        .zip(part_opts)
        .map(|(part, part_opt)| {
            let result = map::map_query(part, part_opt, qname, qseq);
            let record = SplitQueryRecord {
                n_reg: result.regs.len() as i32,
                rep_len: result.rep_len,
                frag_gap: result.frag_gap,
                regs: result.regs,
            };
            let mut buf = Vec::new();
            split::write_split_query_record(&mut buf, &record, opt.flag.contains(MapFlags::CIGAR))
                .expect("in-memory split record write cannot fail");
            split::read_split_query_record(&mut &buf[..], opt.flag.contains(MapFlags::CIGAR))
                .expect("in-memory split record read cannot fail")
        })
        .collect();
    let idx_k = parts.first().map(|part| part.k).unwrap_or(0);
    let merged =
        split::merge_split_query_records(&records, rid_shifts, opt, idx_k, qseq.len() as i32);
    map::MapResult {
        regs: merged.regs,
        rep_len: merged.rep_len,
        frag_gap: merged.frag_gap,
    }
}

fn map_split_fragment_queries(
    parts: &[MmIdx],
    part_opts: &[MapOpt],
    rid_shifts: &[u32],
    opt: &MapOpt,
    qname: &str,
    qseqs: &[&[u8]],
) -> Vec<map::MapResult> {
    let n_segs = qseqs.len();
    let mut per_seg_records: Vec<Vec<SplitQueryRecord>> = (0..n_segs)
        .map(|_| Vec::with_capacity(parts.len()))
        .collect();
    for (part, part_opt) in parts.iter().zip(part_opts) {
        let mut frag_results = map::map_frag_queries(part, part_opt, qname, qseqs);
        if frag_results.len() == 2
            && part_opt.pe_ori >= 0
            && part_opt.flag.contains(MapFlags::CIGAR)
        {
            let (left, right) = frag_results.split_at_mut(1);
            pair_results(
                part_opt,
                qseqs[0].len() as i32,
                qseqs[1].len() as i32,
                &mut left[0],
                &mut right[0],
            );
        }
        for (seg, result) in frag_results.into_iter().enumerate() {
            let record = SplitQueryRecord {
                n_reg: result.regs.len() as i32,
                rep_len: result.rep_len,
                frag_gap: result.frag_gap,
                regs: result.regs,
            };
            let mut buf = Vec::new();
            split::write_split_query_record(&mut buf, &record, opt.flag.contains(MapFlags::CIGAR))
                .expect("in-memory split record write cannot fail");
            per_seg_records[seg].push(
                split::read_split_query_record(&mut &buf[..], opt.flag.contains(MapFlags::CIGAR))
                    .expect("in-memory split record read cannot fail"),
            );
        }
    }
    per_seg_records
        .into_iter()
        .zip(qseqs)
        .map(|(records, qseq)| {
            let idx_k = parts.first().map(|part| part.k).unwrap_or(0);
            let merged = split::merge_split_query_records(
                &records,
                rid_shifts,
                opt,
                idx_k,
                qseq.len() as i32,
            );
            map::MapResult {
                regs: merged.regs,
                rep_len: merged.rep_len,
                frag_gap: merged.frag_gap,
            }
        })
        .collect()
}

/// Map interleaved paired-end reads from a single file.
///
/// Reads are consumed in pairs (R1, R2, R1, R2, ...).
pub fn map_file_interleaved_pe_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        // Read pairs from interleaved file
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp.read_record()?;
            let r2 = fp.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let base_name = strip_pe_suffix(&rec1.name);
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results =
                        map::map_frag_queries(mi, opt, base_name, &[&seq1, &seq2]);
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    pair_results(
                        opt,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);

                    let mut lines = Vec::new();
                    // Use base name without /1 /2 suffix
                    format_pe_sam_records(
                        mi,
                        opt,
                        &BseqRecord {
                            name: base_name.to_string(),
                            ..rec1.clone()
                        },
                        &res1,
                        &res2,
                        true,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    format_pe_sam_records(
                        mi,
                        opt,
                        &BseqRecord {
                            name: base_name.to_string(),
                            ..rec2.clone()
                        },
                        &res2,
                        &res1,
                        false,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

pub fn map_file_interleaved_pe_sam_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp.read_record()?;
            let r2 = fp.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let base_name = strip_pe_suffix(&rec1.name);
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results = map_split_fragment_queries(
                        parts,
                        &part_opts,
                        &rid_shifts,
                        opt,
                        base_name,
                        &[&seq1, &seq2],
                    );
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);
                    let split_merge_gap = res2.frag_gap;
                    pair_results_with_gap(
                        opt,
                        split_merge_gap,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );

                    let mut lines = Vec::new();
                    format_pe_sam_records(
                        mi,
                        opt,
                        &BseqRecord {
                            name: base_name.to_string(),
                            ..rec1.clone()
                        },
                        &res1,
                        &res2,
                        true,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    format_pe_sam_records(
                        mi,
                        opt,
                        &BseqRecord {
                            name: base_name.to_string(),
                            ..rec2.clone()
                        },
                        &res2,
                        &res1,
                        false,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

/// Strip /1 or /2 suffix from read name for PE output.
fn strip_pe_suffix(name: &str) -> &str {
    if name.len() >= 2 {
        let bytes = name.as_bytes();
        if bytes[bytes.len() - 2] == b'/'
            && (bytes[bytes.len() - 1] == b'1' || bytes[bytes.len() - 1] == b'2')
        {
            return &name[..name.len() - 2];
        }
    }
    name
}

/// Map paired-end FASTQ files against the index and write SAM output.
///
/// Reads pairs from two files simultaneously. Each pair is mapped independently,
/// then paired to set proper-pair flags and adjust MAPQ.
pub fn map_file_pe_sam(
    mi: &MmIdx,
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        // Read pairs
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp1.read_record()?;
            let r2 = fp2.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        // Map pairs in parallel
        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results =
                        map::map_frag_queries(mi, opt, &rec1.name, &[&seq1, &seq2]);
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    pair_results(
                        opt,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);

                    // Format SAM records
                    let mut lines = Vec::new();
                    // Read 1
                    format_pe_sam_records(
                        mi,
                        opt,
                        rec1,
                        &res1,
                        &res2,
                        true,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    // Read 2
                    format_pe_sam_records(
                        mi,
                        opt,
                        rec2,
                        &res2,
                        &res1,
                        false,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

pub fn map_file_pe_sam_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
    rg: Option<&str>,
    args: &[String],
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let hdr = sam::write_sam_hdr(mi, rg, args);
    writeln!(out, "{}", hdr)?;
    let rg_id = rg.and_then(sam::read_group_id);

    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp1.read_record()?;
            let r2 = fp2.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results = map_split_fragment_queries(
                        parts,
                        &part_opts,
                        &rid_shifts,
                        opt,
                        &rec1.name,
                        &[&seq1, &seq2],
                    );
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);
                    let split_merge_gap = res2.frag_gap;
                    pair_results_with_gap(
                        opt,
                        split_merge_gap,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );

                    let mut lines = Vec::new();
                    format_pe_sam_records(
                        mi,
                        opt,
                        rec1,
                        &res1,
                        &res2,
                        true,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    format_pe_sam_records(
                        mi,
                        opt,
                        rec2,
                        &res2,
                        &res1,
                        false,
                        rg_id.as_deref(),
                        &mut lines,
                    );
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

fn pair_results(
    opt: &MapOpt,
    qlen1: i32,
    qlen2: i32,
    res1: &mut map::MapResult,
    res2: &mut map::MapResult,
) {
    pair_results_with_gap(
        opt,
        res1.frag_gap.max(res2.frag_gap),
        qlen1,
        qlen2,
        res1,
        res2,
    );
}

fn pair_results_with_gap(
    opt: &MapOpt,
    max_gap_ref: i32,
    qlen1: i32,
    qlen2: i32,
    res1: &mut map::MapResult,
    res2: &mut map::MapResult,
) {
    if !res1.regs.is_empty()
        && !res2.regs.is_empty()
        && res1.regs[0].extra.is_some()
        && res2.regs[0].extra.is_some()
    {
        let qlens = [qlen1, qlen2];
        let mut n_regs = [res1.regs.len(), res2.regs.len()];
        let mut regs_pair = [
            std::mem::take(&mut res1.regs),
            std::mem::take(&mut res2.regs),
        ];
        pe::pair(
            max_gap_ref,
            opt.pe_bonus,
            opt.a * 2 + opt.b,
            opt.a,
            &qlens,
            &mut n_regs,
            &mut regs_pair,
        );
        res1.regs = regs_pair[0].clone();
        res2.regs = regs_pair[1].clone();
    }
}

/// Format SAM records for one read of a pair.
fn format_pe_sam_records(
    mi: &MmIdx,
    opt: &MapOpt,
    rec: &BseqRecord,
    result: &map::MapResult,
    mate_result: &map::MapResult,
    is_read1: bool,
    rg_id: Option<&str>,
    lines: &mut Vec<String>,
) {
    if result.regs.is_empty() {
        if opt.flag.contains(MapFlags::SAM_HIT_ONLY) {
            return;
        }
        let mut line = sam::write_sam_record_with_comment(
            mi,
            &rec.name,
            &rec.seq,
            &rec.qual,
            None,
            0,
            &[],
            opt.flag,
            result.rep_len,
            Some(&rec.comment),
        );
        // Add PE flags
        add_pe_flags(&mut line, is_read1, true, mate_result, None);
        set_unmapped_record_mate_position(mi, &mut line, mate_result);
        add_rg_tag(&mut line, rg_id);
        lines.push(line);
    } else {
        for r in result.regs.iter() {
            if opt.flag.contains(MapFlags::NO_PRINT_2ND) && r.id != r.parent {
                continue;
            }
            let mut line = sam::write_sam_record_with_comment(
                mi,
                &rec.name,
                &rec.seq,
                &rec.qual,
                Some(r),
                result.regs.len(),
                &result.regs,
                opt.flag,
                result.rep_len,
                Some(&rec.comment),
            );
            add_pe_flags(&mut line, is_read1, false, mate_result, Some(r));
            set_mapped_record_mate_fields(mi, &mut line, r, mate_result, is_read1);
            add_rg_tag(&mut line, rg_id);
            lines.push(line);
        }
    }
}

/// Modify SAM flag field to include PE flags.
fn add_pe_flags(
    line: &mut String,
    is_read1: bool,
    unmapped: bool,
    mate_result: &map::MapResult,
    current: Option<&crate::types::AlignReg>,
) {
    // Parse existing flag from SAM line and modify it
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 2 {
        return;
    }
    let mut flag: u16 = fields[1].parse().unwrap_or(0);

    flag |= 0x1; // paired
    if is_read1 {
        flag |= 0x40;
    } else {
        flag |= 0x80;
    }

    // Mate unmapped
    if mate_result.regs.is_empty() {
        flag |= 0x8;
    } else if let Some(mate) = primary_reg(mate_result) {
        // Mate reverse strand
        if mate.rev {
            flag |= 0x20;
        }
        // Proper pair
        if !unmapped && current.is_some_and(|r| r.proper_frag) {
            flag |= 0x2;
        }
    }

    // Rebuild line with updated flag
    let mut new_line = String::with_capacity(line.len());
    new_line.push_str(fields[0]); // QNAME
    new_line.push('\t');
    new_line.push_str(&flag.to_string());
    for field in &fields[2..] {
        new_line.push('\t');
        new_line.push_str(field);
    }
    *line = new_line;
}

fn primary_reg(result: &map::MapResult) -> Option<&crate::types::AlignReg> {
    result
        .regs
        .iter()
        .find(|r| r.sam_pri)
        .or_else(|| result.regs.iter().find(|r| r.id == r.parent))
        .or_else(|| result.regs.first())
}

fn set_unmapped_record_mate_position(mi: &MmIdx, line: &mut String, mate_result: &map::MapResult) {
    let Some(mate) = primary_reg(mate_result) else {
        return;
    };
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 4 {
        return;
    }
    let rname = mi
        .seqs
        .get(mate.rid as usize)
        .map(|seq| seq.name.as_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("*");
    let mut new_line = String::with_capacity(line.len() + rname.len() + 16);
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            new_line.push('\t');
        }
        match i {
            2 => new_line.push_str(rname),
            3 => new_line.push_str(&(mate.rs + 1).to_string()),
            6 => new_line.push('='),
            7 => new_line.push_str(&(mate.rs + 1).to_string()),
            _ => new_line.push_str(field),
        }
    }
    *line = new_line;
}

fn set_mapped_record_mate_fields(
    mi: &MmIdx,
    line: &mut String,
    current: &crate::types::AlignReg,
    mate_result: &map::MapResult,
    is_read1: bool,
) {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 9 {
        return;
    }

    let (rnext, pnext, tlen) = if let Some(mate) = primary_reg(mate_result) {
        let rnext = if current.rid == mate.rid {
            "=".to_string()
        } else {
            mi.seqs
                .get(mate.rid as usize)
                .map(|seq| seq.name.clone())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| "*".to_string())
        };
        let pnext = (mate.rs + 1).to_string();
        let tlen = if current.rid == mate.rid {
            inferred_template_len(current, mate, is_read1)
        } else {
            0
        };
        (rnext, pnext, tlen)
    } else {
        ("=".to_string(), (current.rs + 1).to_string(), 0)
    };
    let tlen = tlen.to_string();

    let mut new_line = String::with_capacity(line.len() + rnext.len() + pnext.len() + tlen.len());
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            new_line.push('\t');
        }
        match i {
            6 => new_line.push_str(&rnext),
            7 => new_line.push_str(&pnext),
            8 => new_line.push_str(&tlen),
            _ => new_line.push_str(field),
        }
    }
    *line = new_line;
}

fn inferred_template_len(
    current: &crate::types::AlignReg,
    mate: &crate::types::AlignReg,
    _is_read1: bool,
) -> i32 {
    let this_pos5 = if current.rev {
        current.re - 1
    } else {
        current.rs
    };
    let next_pos5 = if mate.rev { mate.re - 1 } else { mate.rs };
    let mut tlen = next_pos5 - this_pos5;
    if tlen > 0 {
        tlen += 1;
    } else if tlen < 0 {
        tlen -= 1;
    }
    tlen
}

/// Map paired-end files, writing PAF output.
pub fn map_file_pe_paf(
    mi: &MmIdx,
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            match (fp1.read_record()?, fp2.read_record()?) {
                (Some(r1), Some(r2)) => {
                    total_len += r1.l_seq as i64 + r2.l_seq as i64;
                    pairs.push((r1, r2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results =
                        map::map_frag_queries(mi, opt, &rec1.name, &[&seq1, &seq2]);
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    pair_results(
                        opt,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);
                    let name1 = format!("{}/1", rec1.name);
                    let name2 = format!("{}/2", rec2.name);
                    let mut lines = map::format_paf_with_comment(
                        mi,
                        opt,
                        &name1,
                        &rec1.seq,
                        Some(&rec1.comment),
                        &res1,
                    );
                    lines.extend(map::format_paf_with_comment(
                        mi,
                        opt,
                        &name2,
                        &rec2.seq,
                        Some(&rec2.comment),
                        &res2,
                    ));
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

pub fn map_file_pe_paf_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path1: &str,
    path2: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp1 = BseqFile::open(path1)?;
    let mut fp2 = BseqFile::open(path2)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            match (fp1.read_record()?, fp2.read_record()?) {
                (Some(r1), Some(r2)) => {
                    total_len += r1.l_seq as i64 + r2.l_seq as i64;
                    pairs.push((r1, r2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results = map_split_fragment_queries(
                        parts,
                        &part_opts,
                        &rid_shifts,
                        opt,
                        &rec1.name,
                        &[&seq1, &seq2],
                    );
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);
                    let split_merge_gap = res2.frag_gap;
                    pair_results_with_gap(
                        opt,
                        split_merge_gap,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );
                    let name1 = format!("{}/1", rec1.name);
                    let name2 = format!("{}/2", rec2.name);
                    let mut lines = map::format_paf_with_comment(
                        mi,
                        opt,
                        &name1,
                        &rec1.seq,
                        Some(&rec1.comment),
                        &res1,
                    );
                    lines.extend(map::format_paf_with_comment(
                        mi,
                        opt,
                        &name2,
                        &rec2.seq,
                        Some(&rec2.comment),
                        &res2,
                    ));
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

pub fn map_file_interleaved_pe_paf_split(
    mi: &MmIdx,
    parts: &[MmIdx],
    opt: &MapOpt,
    path: &str,
    n_threads: usize,
) -> io::Result<()> {
    let mut fp = BseqFile::open(path)?;
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let part_opts = prepare_part_opts(parts, opt);
    let rid_shifts = rid_shifts(parts);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    loop {
        let mut pairs: Vec<(BseqRecord, BseqRecord)> = Vec::new();
        let mut total_len = 0i64;
        loop {
            let r1 = fp.read_record()?;
            let r2 = fp.read_record()?;
            match (r1, r2) {
                (Some(rec1), Some(rec2)) => {
                    total_len += rec1.l_seq as i64 + rec2.l_seq as i64;
                    pairs.push((rec1, rec2));
                    if total_len >= opt.mini_batch_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        if pairs.is_empty() {
            break;
        }

        let results: Vec<_> = pool.install(|| {
            pairs
                .par_iter()
                .map(|(rec1, rec2)| {
                    let base_name = strip_pe_suffix(&rec1.name);
                    let (seq1, seq2, rev1, rev2) =
                        oriented_pair_sequences(opt, &rec1.seq, &rec2.seq);
                    let mut frag_results = map_split_fragment_queries(
                        parts,
                        &part_opts,
                        &rid_shifts,
                        opt,
                        base_name,
                        &[&seq1, &seq2],
                    );
                    let mut res1 = frag_results.remove(0);
                    let mut res2 = frag_results.remove(0);
                    restore_pair_orientation(&mut res1, rec1.l_seq as i32, rev1);
                    restore_pair_orientation(&mut res2, rec2.l_seq as i32, rev2);
                    let split_merge_gap = res2.frag_gap;
                    pair_results_with_gap(
                        opt,
                        split_merge_gap,
                        rec1.l_seq as i32,
                        rec2.l_seq as i32,
                        &mut res1,
                        &mut res2,
                    );
                    let name1 = format!("{}/1", base_name);
                    let name2 = format!("{}/2", base_name);
                    let mut lines = map::format_paf_with_comment(
                        mi,
                        opt,
                        &name1,
                        &rec1.seq,
                        Some(&rec1.comment),
                        &res1,
                    );
                    lines.extend(map::format_paf_with_comment(
                        mi,
                        opt,
                        &name2,
                        &rec2.seq,
                        Some(&rec2.comment),
                        &res2,
                    ));
                    lines
                })
                .collect()
        });

        for lines in &results {
            for line in lines {
                writeln!(out, "{}", line)?;
            }
        }
    }
    out.flush()
}

fn oriented_pair_sequences(
    opt: &MapOpt,
    seq1: &[u8],
    seq2: &[u8],
) -> (Vec<u8>, Vec<u8>, bool, bool) {
    let rev1 = (opt.pe_ori >> 1) & 1 != 0;
    let rev2 = opt.pe_ori & 1 != 0;
    let mut s1 = seq1.to_vec();
    let mut s2 = seq2.to_vec();
    if rev1 {
        seq::revcomp_ascii(&mut s1);
    }
    if rev2 {
        seq::revcomp_ascii(&mut s2);
    }
    (s1, s2, rev1, rev2)
}

fn restore_pair_orientation(result: &mut map::MapResult, qlen: i32, was_revcomped: bool) {
    if !was_revcomped {
        return;
    }
    for r in &mut result.regs {
        let old_qs = r.qs;
        r.qs = qlen - r.qe;
        r.qe = qlen - old_qs;
        r.rev = !r.rev;
        if let Some(ref mut p) = r.extra {
            if p.trans_strand == 1 {
                p.trans_strand = 2;
            } else if p.trans_strand == 2 {
                p.trans_strand = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    #[test]
    fn test_map_file_paf_basic() {
        // Create a temp reference and query
        let mut ref_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(ref_file, ">ref1").unwrap();
        writeln!(
            ref_file,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        ref_file.flush().unwrap();

        let mi = MmIdx::build_from_file(
            ref_file.path().to_str().unwrap(),
            10,
            15,
            14,
            crate::flags::IdxFlags::empty(),
            50_000_000,
            u64::MAX,
        )
        .unwrap()
        .unwrap();

        let mut query_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(query_file, ">read1").unwrap();
        writeln!(
            query_file,
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        .unwrap();
        query_file.flush().unwrap();

        let mut opt = MapOpt::default();
        crate::options::mapopt_update(&mut opt, &mi);
        // Test the core mapping
        let result = map::map_query(
            &mi,
            &opt,
            "read1",
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        );
        assert!(!result.regs.is_empty());
    }

    #[test]
    fn test_split_fragment_queries_matches_unsplit_pair() {
        let mut ref_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(ref_file, ">t2").unwrap();
        writeln!(
            ref_file,
            "GGACATCCCGATGGTGCAGGTGCTATTAAAGGTTCGTTTGTTCAACGATTAAAGTCCTACCTGTACGAAAGGAC"
        )
        .unwrap();
        ref_file.flush().unwrap();

        let mut io = crate::options::IdxOpt::default();
        let mut opt = crate::options::MapOpt::default();
        crate::options::set_opt(Some("sr"), &mut io, &mut opt).unwrap();
        opt.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
        let full = MmIdx::build_from_file(
            ref_file.path().to_str().unwrap(),
            io.w as i32,
            io.k as i32,
            io.bucket_bits,
            io.flag,
            io.mini_batch_size,
            u64::MAX,
        )
        .unwrap()
        .unwrap();
        crate::options::mapopt_update(&mut opt, &full);
        let parts = MmIdx::build_parts_from_file(
            ref_file.path().to_str().unwrap(),
            io.w as i32,
            io.k as i32,
            io.bucket_bits,
            io.flag,
            40,
        )
        .unwrap();
        let part_opts = prepare_part_opts(&parts, &opt);
        let shifts = rid_shifts(&parts);
        let r1 = b"GGACATCCCGATGGTGCAGGTGCTATTAAAGGTTCGTTTG";
        let r2 = b"TCAACGATTAAAGTCCTACCTGTACGAAAGGAC";

        let split =
            map_split_fragment_queries(&parts, &part_opts, &shifts, &opt, "pair", &[r1, r2]);
        let full_res = map::map_frag_queries(&full, &opt, "pair", &[r1, r2]);

        assert_eq!(split.len(), 2);
        assert_eq!(split[0].regs.is_empty(), full_res[0].regs.is_empty());
        assert_eq!(split[1].regs.is_empty(), full_res[1].regs.is_empty());
        if !split[0].regs.is_empty() && !full_res[0].regs.is_empty() {
            assert_eq!(split[0].regs[0].rid, full_res[0].regs[0].rid);
        }
        if !split[1].regs.is_empty() && !full_res[1].regs.is_empty() {
            assert_eq!(split[1].regs[0].rid, full_res[1].regs[0].rid);
        }
    }
}
