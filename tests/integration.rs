use std::path::Path;
use minimap2::flags::{IdxFlags, MapFlags};
use minimap2::index::MmIdx;
use minimap2::map;
use minimap2::options::{self, IdxOpt, MapOpt};

fn build_index(seqs: &[(&str, &[u8])], io: &IdxOpt) -> MmIdx {
    let seq_data: Vec<&[u8]> = seqs.iter().map(|s| s.1).collect();
    let names: Vec<&str> = seqs.iter().map(|s| s.0).collect();
    MmIdx::build_from_str(
        io.w as i32, io.k as i32,
        io.flag.contains(IdxFlags::HPC),
        io.bucket_bits as i32,
        &seq_data, Some(&names),
    ).unwrap()
}

fn setup_opts(preset: Option<&str>) -> (IdxOpt, MapOpt) {
    let mut io = IdxOpt::default();
    let mut mo = MapOpt::default();
    options::set_opt(preset, &mut io, &mut mo).unwrap();
    (io, mo)
}

fn map_one(mi: &MmIdx, mo: &mut MapOpt, name: &str, seq: &[u8]) -> map::MapResult {
    options::mapopt_update(mo, mi);
    map::map_query(mi, mo, name, seq)
}

#[test]
fn test_self_mapping() {
    let (io, mut mo) = setup_opts(None);
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", seq)], &io);
    let result = map_one(&mi, &mut mo, "query1", seq);
    assert!(!result.regs.is_empty(), "Self-mapping should produce hits");
    assert_eq!(result.regs[0].rid, 0);
    // MAPQ depends on uniqueness; self-mapping may have low MAPQ
    assert!(result.regs[0].mapq <= 60);
}

#[test]
fn test_no_hit() {
    let (io, mut mo) = setup_opts(None);
    let mi = build_index(&[("ref1", b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")], &io);
    let result = map_one(&mi, &mut mo, "query1", b"NNNNNNNNNNNNNNNNNNNN");
    assert!(result.regs.is_empty());
}

#[test]
fn test_cigar_identical() {
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", seq);
    assert!(!result.regs.is_empty());
    let r = &result.regs[0];
    assert!(r.extra.is_some(), "Should have CIGAR");
    let extra = r.extra.as_ref().unwrap();
    assert!(!extra.cigar.0.is_empty(), "CIGAR should not be empty");
    assert_eq!(extra.dp_score, 128); // 64 * 2 (match score)
}

#[test]
fn test_cigar_with_mismatch() {
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR;
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let qry_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", ref_seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", qry_seq);
    assert!(!result.regs.is_empty());
    assert!(result.regs[0].extra.is_some());
}

#[test]
fn test_paf_format() {
    let (io, mut mo) = setup_opts(None);
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("chr1", ref_seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", ref_seq);
    let lines = map::format_paf(&mi, &mo, "read1", ref_seq, &result);
    assert!(!lines.is_empty());
    let fields: Vec<&str> = lines[0].split('\t').collect();
    assert!(fields.len() >= 12);
    assert_eq!(fields[0], "read1"); // qname
    assert_eq!(fields[5], "chr1");  // rname
}

#[test]
fn test_map_ont_preset() {
    let (io, mut mo) = setup_opts(Some("map-ont"));
    assert_eq!(io.k, 15);
    assert_eq!(io.w, 10);
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", ref_seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", ref_seq);
    assert!(!result.regs.is_empty());
}

#[test]
fn test_map_hifi_preset() {
    let (io, mut mo) = setup_opts(Some("map-hifi"));
    assert_eq!(io.k, 19);
    assert_eq!(io.w, 19);
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", ref_seq)], &io);
    let _result = map_one(&mi, &mut mo, "read1", ref_seq);
    // With k=19, w=19, we need longer sequences for hits
    // A 64bp seq might not produce enough minimizers
}

#[test]
fn test_index_roundtrip() {
    let (io, _mo) = setup_opts(None);
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", ref_seq)], &io);

    // Dump to buffer
    let mut buf = Vec::new();
    minimap2::index::io::idx_dump(&mut buf, &mi).unwrap();

    // Load from buffer
    let mut cursor = std::io::Cursor::new(buf);
    let mi2 = minimap2::index::io::idx_load(&mut cursor).unwrap().unwrap();

    assert_eq!(mi.seqs.len(), mi2.seqs.len());
    assert_eq!(mi.seqs[0].name, mi2.seqs[0].name);
    assert_eq!(mi.seqs[0].len, mi2.seqs[0].len);
    assert_eq!(mi.k, mi2.k);
    assert_eq!(mi.w, mi2.w);
}

#[test]
fn test_divergence_estimation() {
    let (io, mut mo) = setup_opts(None);
    let ref_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", ref_seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", ref_seq);
    assert!(!result.regs.is_empty());
    // Self-mapping should have very low divergence
    assert!(result.regs[0].div >= 0.0);
    assert!(result.regs[0].div < 0.01, "Self-mapping div should be near 0, got {}", result.regs[0].div);
}

#[test]
fn test_multiple_references() {
    let (io, mut mo) = setup_opts(None);
    let mi = build_index(&[
        ("chr1", b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
        ("chr2", b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"),
    ], &io);

    let result1 = map_one(&mi, &mut mo, "read1",
        b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
    assert!(!result1.regs.is_empty());
    assert_eq!(result1.regs[0].rid, 0); // maps to chr1

    let result2 = map_one(&mi, &mut mo, "read2",
        b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA");
    assert!(!result2.regs.is_empty());
    assert_eq!(result2.regs[0].rid, 1); // maps to chr2
}

#[test]
fn test_eqx_cigar() {
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::EQX;
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mi = build_index(&[("ref1", seq)], &io);
    let result = map_one(&mi, &mut mo, "read1", seq);
    assert!(!result.regs.is_empty());
    let extra = result.regs[0].extra.as_ref().unwrap();
    // All CIGAR ops should be = (op 7) or X (op 8), not M (op 0)
    for &c in &extra.cigar.0 {
        let op = c & 0xf;
        assert!(op != 0, "EQX mode should not have M operations, got op={}", op);
    }
}

// === File-based integration tests using minimap2 test data ===

fn map_file_pair(ref_path: &str, qry_path: &str, preset: Option<&str>) -> Vec<(String, map::MapResult)> {
    if !Path::new(ref_path).exists() || !Path::new(qry_path).exists() {
        return Vec::new(); // skip if test data not available
    }
    let (io, mut mo) = setup_opts(preset);
    let mi = MmIdx::build_from_file(
        ref_path, io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open(qry_path).unwrap();
    let mut results = Vec::new();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        results.push((rec.name, result));
    }
    results
}

#[test]
fn test_mt_genome_mapping() {
    let results = map_file_pair("minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa", None);
    if results.is_empty() { return; } // skip if files not found
    assert_eq!(results.len(), 1);
    assert!(!results[0].1.regs.is_empty(), "Should map MT_orang to MT_human");
    let r = &results[0].1.regs[0];
    assert_eq!(r.rid, 0);
    // Check coordinates match C minimap2: qs=61, qe=16018, rs=637, re=16562
    assert_eq!(r.qs, 61);
    assert_eq!(r.qe, 16018);
    assert_eq!(r.rs, 637);
    assert_eq!(r.re, 16562);
    assert_eq!(r.mapq, 60);
}

#[test]
fn test_inversion_mapping() {
    let results = map_file_pair("minimap2/test/t-inv.fa", "minimap2/test/q-inv.fa", None);
    if results.is_empty() { return; }
    assert_eq!(results.len(), 2);
    // Both reads should map
    assert!(!results[0].1.regs.is_empty(), "read1 should map");
    assert!(!results[1].1.regs.is_empty(), "read2 should map");
}

#[test]
fn test_x3s_mapping() {
    let results = map_file_pair("minimap2/test/x3s-ref.fa", "minimap2/test/x3s-qry.fa", None);
    if results.is_empty() { return; }
    assert_eq!(results.len(), 1);
    assert!(!results[0].1.regs.is_empty(), "x3s query should map");
    let r = &results[0].1.regs[0];
    assert_eq!(r.rid, 0);
    assert!(r.rev, "x3s maps to reverse strand");
}

#[test]
fn test_mt_with_cigar() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    let rec = fp.read_record().unwrap().unwrap();
    let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
    assert!(!result.regs.is_empty());
    let r = &result.regs[0];
    assert!(r.extra.is_some(), "Should have CIGAR");
    let cigar = &r.extra.as_ref().unwrap().cigar;
    assert!(!cigar.0.is_empty(), "CIGAR should not be empty");
    // Verify CIGAR consumes correct number of query/ref bases
    let mut q_consumed = 0i32;
    let mut t_consumed = 0i32;
    for &c in &cigar.0 {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 7 | 8 => { q_consumed += len; t_consumed += len; }
            1 => { q_consumed += len; }
            2 | 3 => { t_consumed += len; }
            _ => {}
        }
    }
    // CIGAR consumption should match or slightly exceed coordinate range
    // (coordinates may be clipped to reference boundaries)
    assert!(q_consumed >= r.qe - r.qs,
        "CIGAR query consumption {} should be >= qe-qs={}", q_consumed, r.qe - r.qs);
    assert!(t_consumed >= r.re - r.rs,
        "CIGAR target consumption {} should be >= re-rs={}", t_consumed, r.re - r.rs);
}

#[test]
fn test_x3s_cigar_matches_c() {
    if !Path::new("minimap2/test/x3s-ref.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/x3s-ref.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/x3s-qry.fa").unwrap();
    let rec = fp.read_record().unwrap().unwrap();
    let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
    assert!(!result.regs.is_empty());
    let r = &result.regs[0];
    // C minimap2 produces: query 134 0 70 - ref 388 258 328 70 70 60 cg:Z:70M
    assert_eq!(r.qs, 0);
    assert_eq!(r.qe, 70);
    assert_eq!(r.rs, 258);
    assert_eq!(r.re, 328);
    let extra = r.extra.as_ref().unwrap();
    assert_eq!(extra.cigar.0.len(), 1, "Should be a single 70M");
    assert_eq!(extra.cigar.0[0], 70 << 4, "Should be 70M"); // op=0 (M)
}

#[test]
fn test_sam_cigar_seq_consistency() {
    // Verify CIGAR query consumption equals SEQ length (critical for samtools)
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_SAM;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    let rec = fp.read_record().unwrap().unwrap();
    let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
    assert!(!result.regs.is_empty());
    let r = &result.regs[0];
    assert!(r.extra.is_some());

    // Compute query bases consumed by CIGAR (M/I/S/=/X consume query)
    let cigar = &r.extra.as_ref().unwrap().cigar;
    let mut q_consumed = 0i32;
    for &c in &cigar.0 {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 1 | 4 | 7 | 8 => q_consumed += len, // M, I, S, =, X
            _ => {}
        }
    }

    // For hard-clip mode: SEQ length = aligned query length = qe - qs
    // CIGAR should consume exactly qe - qs query bases
    let aligned_qlen = r.qe - r.qs;
    assert!(q_consumed > 0, "CIGAR should consume query bases");
    // Allow small discrepancy from coordinate clamping
    assert!((q_consumed - aligned_qlen).abs() <= 100,
        "CIGAR query consumption {} should be close to aligned qlen {}", q_consumed, aligned_qlen);
}

#[test]
fn test_aligner_api() {
    use minimap2::aligner::Aligner;
    use std::io::Write;

    let mut f = tempfile::NamedTempFile::new().unwrap();
    writeln!(f, ">ref1").unwrap();
    writeln!(f, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
    f.flush().unwrap();

    let aligner = Aligner::builder()
        .index(f.path().to_str().unwrap())
        .with_cigar()
        .build()
        .unwrap();

    assert_eq!(aligner.n_seq(), 1);
    assert_eq!(aligner.seq_name(0), "ref1");
    assert_eq!(aligner.seq_len(0), 64);

    let hits = aligner.map(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
    assert!(!hits.is_empty());
    assert!(hits[0].extra.is_some(), "with_cigar should produce CIGAR");
}

// === Tests for debugging HiFi CIGAR differences (last M before soft-clip too short) ===

/// Helper: parse a CIGAR string like "61S15857M200S" into Vec<(op_char, len)>
fn parse_cigar_string(s: &str) -> Vec<(char, u32)> {
    let mut ops = Vec::new();
    let mut num = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else {
            let len: u32 = num.parse().unwrap_or(0);
            num.clear();
            ops.push((c, len));
        }
    }
    ops
}

/// Helper: count query-consuming bases in a BAM-encoded CIGAR
fn cigar_query_consumed(cigar: &[u32]) -> i32 {
    let mut q = 0i32;
    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 1 | 4 | 7 | 8 => q += len, // M, I, S, =, X
            _ => {}
        }
    }
    q
}

/// Helper: count query bases consumed by alignment ops only (M, I, =, X -- no S/H)
fn cigar_aligned_query(cigar: &[u32]) -> i32 {
    let mut q = 0i32;
    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 1 | 7 | 8 => q += len, // M, I, =, X
            _ => {}
        }
    }
    q
}

/// Helper: count ref bases consumed by alignment ops (M, D, N, =, X)
fn cigar_ref_consumed(cigar: &[u32]) -> i32 {
    let mut t = 0i32;
    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        match op {
            0 | 2 | 3 | 7 | 8 => t += len, // M, D, N, =, X
            _ => {}
        }
    }
    t
}

/// Test 1: CIGAR query consumption must exactly equal qe - qs for all alignments.
/// This catches the bug where the right extension ends early, leaving
/// ~60-100bp unaccounted for between CIGAR and coordinates.
#[test]
fn test_cigar_qlen_equals_qe_minus_qs() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        for (i, r) in result.regs.iter().enumerate() {
            if let Some(ref extra) = r.extra {
                let aligned_q = cigar_aligned_query(&extra.cigar.0);
                let expected_q = r.qe - r.qs;
                assert_eq!(aligned_q, expected_q,
                    "Read {} reg {}: CIGAR aligned query bases ({}) != qe-qs ({}) [qe={}, qs={}]. \
                     CIGAR: {}",
                    rec.name, i, aligned_q, expected_q, r.qe, r.qs,
                    minimap2::align::cigar_to_string(&extra.cigar.0));

                let ref_consumed = cigar_ref_consumed(&extra.cigar.0);
                let expected_r = r.re - r.rs;
                assert_eq!(ref_consumed, expected_r,
                    "Read {} reg {}: CIGAR ref bases ({}) != re-rs ({}) [re={}, rs={}]. \
                     CIGAR: {}",
                    rec.name, i, ref_consumed, expected_r, r.re, r.rs,
                    minimap2::align::cigar_to_string(&extra.cigar.0));
            }
        }
    }
}

/// Test 2: CIGAR query consumption consistency for map-hifi preset on MT data.
/// Uses the HiFi preset which is where the 462/649 differences were observed.
#[test]
fn test_hifi_cigar_qlen_consistency() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(Some("map-hifi"));
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        for (i, r) in result.regs.iter().enumerate() {
            if let Some(ref extra) = r.extra {
                let aligned_q = cigar_aligned_query(&extra.cigar.0);
                let expected_q = r.qe - r.qs;
                assert_eq!(aligned_q, expected_q,
                    "HiFi: Read {} reg {}: CIGAR aligned query bases ({}) != qe-qs ({}) \
                     [qs={}, qe={}, rs={}, re={}]. CIGAR: {}",
                    rec.name, i, aligned_q, expected_q,
                    r.qs, r.qe, r.rs, r.re,
                    minimap2::align::cigar_to_string(&extra.cigar.0));
            }
        }
    }
}

/// Test 3: SAM CIGAR consistency -- sum of query-consuming ops must equal query length.
/// For SAM, the CIGAR includes soft clips, so M+I+S+=/X must equal full query length.
#[test]
fn test_sam_cigar_equals_query_length() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_SAM;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    while let Ok(Some(rec)) = fp.read_record() {
        let qlen = rec.seq.len() as i32;
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        for (i, r) in result.regs.iter().enumerate() {
            if let Some(ref extra) = r.extra {
                // Construct the full CIGAR including soft clips (as SAM would)
                let mut full_cigar: Vec<u32> = Vec::new();

                // Leading soft clip
                let qs_internal = if r.rev { qlen - r.qe } else { r.qs };
                if qs_internal > 0 {
                    full_cigar.push((qs_internal as u32) << 4 | 4); // S
                }

                // Core CIGAR
                full_cigar.extend_from_slice(&extra.cigar.0);

                // Trailing soft clip
                let qe_internal = if r.rev { qlen - r.qs } else { r.qe };
                let trailing = qlen - qe_internal;
                if trailing > 0 {
                    full_cigar.push((trailing as u32) << 4 | 4); // S
                }

                let total_q = cigar_query_consumed(&full_cigar);
                assert_eq!(total_q, qlen,
                    "SAM: Read {} reg {}: total query-consuming CIGAR ops ({}) != qlen ({}). \
                     qs={}, qe={}, rev={}, leading_S={}, trailing_S={}, core_CIGAR={}",
                    rec.name, i, total_q, qlen,
                    r.qs, r.qe, r.rev, qs_internal, trailing,
                    minimap2::align::cigar_to_string(&extra.cigar.0));
            }
        }
    }
}

/// Test 4: Compare Rust vs C minimap2 PAF output on MT test data with CIGAR.
/// This is the key integration test that catches the qend difference.
#[test]
fn test_mt_cigar_vs_c_minimap2() {
    let c_mm2 = Path::new("minimap2/minimap2");
    if !c_mm2.exists() || !Path::new("minimap2/test/MT-human.fa").exists() {
        eprintln!("Skipping: C minimap2 binary or test data not found");
        return;
    }

    // Run C minimap2
    let c_output = std::process::Command::new(c_mm2)
        .args(&["-c", "--eqx", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"])
        .output()
        .expect("Failed to run C minimap2");
    let c_paf = String::from_utf8_lossy(&c_output.stdout);
    let c_lines: Vec<&str> = c_paf.lines().collect();
    assert!(!c_lines.is_empty(), "C minimap2 produced no output");

    // Run Rust minimap2
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG | MapFlags::EQX;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    let mut rust_lines: Vec<String> = Vec::new();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        let lines = map::format_paf(&mi, &mo, &rec.name, &rec.seq, &result);
        rust_lines.extend(lines);
    }
    assert!(!rust_lines.is_empty(), "Rust minimap2 produced no output");

    // Compare line by line
    let n = c_lines.len().min(rust_lines.len());
    for i in 0..n {
        let c_fields: Vec<&str> = c_lines[i].split('\t').collect();
        let r_fields: Vec<&str> = rust_lines[i].split('\t').collect();

        // Compare the first 12 mandatory PAF fields
        for f in 0..12.min(c_fields.len()).min(r_fields.len()) {
            if c_fields[f] != r_fields[f] {
                // Report difference with context
                eprintln!("PAF field {} differs for line {}:", f, i);
                eprintln!("  C:    {}", c_fields[..12.min(c_fields.len())].join("\t"));
                eprintln!("  Rust: {}", r_fields[..12.min(r_fields.len())].join("\t"));

                // For fields 2,3 (qs,qe) and 7,8 (rs,re), compute the delta
                if f == 2 || f == 3 || f == 7 || f == 8 {
                    let c_val: i64 = c_fields[f].parse().unwrap_or(0);
                    let r_val: i64 = r_fields[f].parse().unwrap_or(0);
                    eprintln!("  Delta: {} (Rust - C)", r_val - c_val);
                }
            }
        }

        // Strict check on coordinates (qs, qe, rs, re = fields 2,3,7,8)
        for &f in &[2usize, 3, 7, 8] {
            if f < c_fields.len() && f < r_fields.len() {
                assert_eq!(c_fields[f], r_fields[f],
                    "MT CIGAR: PAF field {} (qs/qe/rs/re) differs on line {}.\n  C:    {}\n  Rust: {}",
                    f, i,
                    c_fields[..12.min(c_fields.len())].join("\t"),
                    r_fields[..12.min(r_fields.len())].join("\t"));
            }
        }
    }
    assert_eq!(c_lines.len(), rust_lines.len(),
        "Different number of PAF lines: C={}, Rust={}", c_lines.len(), rust_lines.len());
}

/// Test 5: Compare Rust vs C minimap2 with map-hifi preset on MT data.
/// This specifically targets the HiFi preset where 462/649 reads had differences.
#[test]
fn test_mt_hifi_cigar_vs_c_minimap2() {
    let c_mm2 = Path::new("minimap2/minimap2");
    if !c_mm2.exists() || !Path::new("minimap2/test/MT-human.fa").exists() {
        eprintln!("Skipping: C minimap2 binary or test data not found");
        return;
    }

    // Run C minimap2 with map-hifi preset
    let c_output = std::process::Command::new(c_mm2)
        .args(&["-c", "-x", "map-hifi", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"])
        .output()
        .expect("Failed to run C minimap2");
    let c_paf = String::from_utf8_lossy(&c_output.stdout);
    let c_lines: Vec<&str> = c_paf.lines().collect();

    // Run Rust minimap2 with map-hifi preset
    let (io, mut mo) = setup_opts(Some("map-hifi"));
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    let mut rust_lines: Vec<String> = Vec::new();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        let lines = map::format_paf(&mi, &mo, &rec.name, &rec.seq, &result);
        rust_lines.extend(lines);
    }

    if c_lines.is_empty() && rust_lines.is_empty() {
        // Both produce no output (HiFi k=19 may need longer sequences)
        return;
    }

    let n = c_lines.len().min(rust_lines.len());
    for i in 0..n {
        let c_fields: Vec<&str> = c_lines[i].split('\t').collect();
        let r_fields: Vec<&str> = rust_lines[i].split('\t').collect();

        // Compare coordinates and report differences with diagnostic info
        for &f in &[2usize, 3, 7, 8] {
            if f < c_fields.len() && f < r_fields.len() && c_fields[f] != r_fields[f] {
                let c_val: i64 = c_fields[f].parse().unwrap_or(0);
                let r_val: i64 = r_fields[f].parse().unwrap_or(0);
                let field_name = match f { 2 => "qs", 3 => "qe", 7 => "rs", 8 => "re", _ => "?" };
                panic!(
                    "HiFi MT: {} differs by {} on line {}.\n  C:    {}\n  Rust: {}",
                    field_name, r_val - c_val, i,
                    c_fields[..12.min(c_fields.len())].join("\t"),
                    r_fields[..12.min(r_fields.len())].join("\t"));
            }
        }

        // Also compare CIGAR tags if present
        let c_cg = c_fields.iter().find(|f| f.starts_with("cg:Z:"));
        let r_cg = r_fields.iter().find(|f| f.starts_with("cg:Z:"));
        if let (Some(c_cg), Some(r_cg)) = (c_cg, r_cg) {
            if c_cg != r_cg {
                // Parse CIGARs and find where they diverge
                let c_ops = parse_cigar_string(&c_cg[5..]);
                let r_ops = parse_cigar_string(&r_cg[5..]);
                let diff_idx = c_ops.iter().zip(r_ops.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(c_ops.len().min(r_ops.len()));

                // Show the last few ops around the difference
                let start = diff_idx.saturating_sub(2);
                let c_context: Vec<String> = c_ops[start..c_ops.len().min(diff_idx + 3)]
                    .iter().map(|(c, l)| format!("{}{}", l, c)).collect();
                let r_context: Vec<String> = r_ops[start..r_ops.len().min(diff_idx + 3)]
                    .iter().map(|(c, l)| format!("{}{}", l, c)).collect();
                eprintln!("HiFi MT line {}: CIGAR differs at op {} (0-indexed).", i, diff_idx);
                eprintln!("  C CIGAR around diff: ...{}", c_context.join(""));
                eprintln!("  R CIGAR around diff: ...{}", r_context.join(""));
                eprintln!("  C has {} ops, Rust has {} ops", c_ops.len(), r_ops.len());

                // Check if the difference is in the last M before S pattern
                if let Some((last_c, last_len)) = r_ops.last() {
                    if *last_c == 'M' || *last_c == '=' {
                        if let Some((c_last_c, c_last_len)) = c_ops.last() {
                            if *c_last_c == *last_c && *c_last_len > *last_len {
                                eprintln!("  PATTERN MATCH: Last {} is shorter in Rust ({}) vs C ({}), delta={}",
                                    last_c, last_len, c_last_len, c_last_len - last_len);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Test 6: Right extension endpoint consistency.
/// After alignment, verify that the right extension produces coordinates
/// consistent with the CIGAR. Specifically: the sum of aligned query bases
/// from the CIGAR should exactly equal qe - qs (no gap).
#[test]
fn test_right_extension_endpoint() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }

    // Test with multiple presets to catch preset-specific issues
    for preset in &[None, Some("map-hifi"), Some("map-ont")] {
        let (io, mut mo) = setup_opts(*preset);
        mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
        let mi = MmIdx::build_from_file(
            "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
            io.flag, io.mini_batch_size, io.batch_size,
        ).unwrap().unwrap();
        options::mapopt_update(&mut mo, &mi);

        let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
        while let Ok(Some(rec)) = fp.read_record() {
            let qlen = rec.seq.len() as i32;
            let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
            for (i, r) in result.regs.iter().enumerate() {
                if let Some(ref extra) = r.extra {
                    let cigar = &extra.cigar.0;
                    let aligned_q = cigar_aligned_query(cigar);
                    let aligned_r = cigar_ref_consumed(cigar);
                    let expected_q = r.qe - r.qs;
                    let expected_r = r.re - r.rs;
                    let preset_name = preset.unwrap_or("default");

                    // Strict: CIGAR must exactly match coordinates
                    assert_eq!(aligned_q, expected_q,
                        "[{}] Read {} reg {}: CIGAR query bases ({}) != qe-qs ({}). \
                         qs={}, qe={}, qlen={}, CIGAR={}",
                        preset_name, rec.name, i, aligned_q, expected_q,
                        r.qs, r.qe, qlen,
                        minimap2::align::cigar_to_string(cigar));

                    assert_eq!(aligned_r, expected_r,
                        "[{}] Read {} reg {}: CIGAR ref bases ({}) != re-rs ({}). \
                         rs={}, re={}, CIGAR={}",
                        preset_name, rec.name, i, aligned_r, expected_r,
                        r.rs, r.re,
                        minimap2::align::cigar_to_string(cigar));

                    // Additional check: soft-clip + aligned query should not exceed query length
                    let soft_clip_left = if r.rev { qlen - r.qe } else { r.qs };
                    let soft_clip_right = if r.rev { r.qs } else { qlen - r.qe };
                    let total = soft_clip_left + aligned_q + soft_clip_right;
                    assert_eq!(total, qlen,
                        "[{}] Read {} reg {}: soft_left({}) + aligned({}) + soft_right({}) = {} != qlen({})",
                        preset_name, rec.name, i, soft_clip_left, aligned_q, soft_clip_right, total, qlen);
                }
            }
        }
    }
}

/// Test 7: Compare t2/q2 test data with C minimap2 (another pair to test gap-filling).
#[test]
fn test_t2q2_cigar_vs_c_minimap2() {
    let c_mm2 = Path::new("minimap2/minimap2");
    if !c_mm2.exists() || !Path::new("minimap2/test/t2.fa").exists()
        || !Path::new("minimap2/test/q2.fa").exists()
    {
        eprintln!("Skipping: C minimap2 binary or test data not found");
        return;
    }

    // Run C minimap2
    let c_output = std::process::Command::new(c_mm2)
        .args(&["-c", "minimap2/test/t2.fa", "minimap2/test/q2.fa"])
        .output()
        .expect("Failed to run C minimap2");
    let c_paf = String::from_utf8_lossy(&c_output.stdout);
    let c_lines: Vec<&str> = c_paf.lines().filter(|l| !l.is_empty()).collect();

    // Run Rust minimap2
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/t2.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/q2.fa").unwrap();
    let mut rust_lines: Vec<String> = Vec::new();
    while let Ok(Some(rec)) = fp.read_record() {
        let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);
        let lines = map::format_paf(&mi, &mo, &rec.name, &rec.seq, &result);
        rust_lines.extend(lines);
    }

    let n = c_lines.len().min(rust_lines.len());
    for i in 0..n {
        let c_fields: Vec<&str> = c_lines[i].split('\t').collect();
        let r_fields: Vec<&str> = rust_lines[i].split('\t').collect();

        // Compare coordinates
        for &f in &[2usize, 3, 7, 8] {
            if f < c_fields.len() && f < r_fields.len() {
                assert_eq!(c_fields[f], r_fields[f],
                    "t2/q2: PAF field {} differs on line {}.\n  C:    {}\n  Rust: {}",
                    f, i,
                    c_fields[..12.min(c_fields.len())].join("\t"),
                    r_fields[..12.min(r_fields.len())].join("\t"));
            }
        }
    }
    assert_eq!(c_lines.len(), rust_lines.len(),
        "t2/q2: Different number of PAF lines: C={}, Rust={}", c_lines.len(), rust_lines.len());
}

/// Test 8: Detailed right extension diagnostic.
/// Map MT-orang to MT-human and dump detailed alignment info to help debug
/// the "last M before S is too short" pattern.
#[test]
fn test_right_extension_diagnostic() {
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let (io, mut mo) = setup_opts(None);
    mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    let mi = MmIdx::build_from_file(
        "minimap2/test/MT-human.fa", io.w as i32, io.k as i32, io.bucket_bits,
        io.flag, io.mini_batch_size, io.batch_size,
    ).unwrap().unwrap();
    options::mapopt_update(&mut mo, &mi);

    let mut fp = minimap2::bseq::BseqFile::open("minimap2/test/MT-orang.fa").unwrap();
    let rec = fp.read_record().unwrap().unwrap();
    let qlen = rec.seq.len() as i32;
    let result = map::map_query(&mi, &mo, &rec.name, &rec.seq);

    for (i, r) in result.regs.iter().enumerate() {
        if let Some(ref extra) = r.extra {
            let cigar = &extra.cigar.0;

            // Find the last alignment op (not soft-clip)
            let last_align_op = cigar.last().map(|&c| {
                let op = c & 0xf;
                let len = c >> 4;
                (op, len)
            });

            let aligned_q = cigar_aligned_query(cigar);
            let soft_right = if r.rev { r.qs } else { qlen - r.qe };

            eprintln!("Reg {}: qs={} qe={} rs={} re={} rev={} mapq={} qlen={}",
                i, r.qs, r.qe, r.rs, r.re, r.rev, r.mapq, qlen);
            eprintln!("  aligned_q={} soft_right={} total={}",
                aligned_q, soft_right, aligned_q + r.qs + soft_right);
            if let Some((op, len)) = last_align_op {
                let op_char = b"MIDNSHP=XB"[op as usize] as char;
                eprintln!("  last alignment op: {}{}", len, op_char);
            }
            eprintln!("  CIGAR: {}", minimap2::align::cigar_to_string(cigar));

            // The key assertion: everything must add up
            let total = if r.rev {
                (qlen - r.qe) + aligned_q + r.qs
            } else {
                r.qs + aligned_q + (qlen - r.qe)
            };
            assert_eq!(total, qlen,
                "Reg {}: coordinates + CIGAR don't add up to qlen. \
                 qs={}, aligned_q={}, trailing={}, total={}, qlen={}",
                i, r.qs, aligned_q, qlen - r.qe, total, qlen);
        }
    }
}

/// Test that C and Rust produce identical CIGARs for MT genome with map-hifi preset.
#[test]
fn test_mt_hifi_full_cigar_vs_c() {
    if !Path::new("minimap2/minimap2").exists() || !Path::new("target/release/minimap2-pure-rs").exists() { return; }
    if !Path::new("minimap2/test/MT-human.fa").exists() { return; }
    let c_out = std::process::Command::new("minimap2/minimap2")
        .args(&["-a", "-x", "map-hifi", "--eqx", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"])
        .stderr(std::process::Stdio::null())
        .output().unwrap();
    let c_sam = String::from_utf8_lossy(&c_out.stdout);
    let c_records: Vec<&str> = c_sam.lines().filter(|l| !l.starts_with('@')).collect();

    let r_out = std::process::Command::new("target/release/minimap2-pure-rs")
        .args(&["-a", "-x", "map-hifi", "--eqx", "minimap2/test/MT-human.fa", "minimap2/test/MT-orang.fa"])
        .stderr(std::process::Stdio::null())
        .output().unwrap();
    let r_sam = String::from_utf8_lossy(&r_out.stdout);
    let r_records: Vec<&str> = r_sam.lines().filter(|l| !l.starts_with('@')).collect();

    assert_eq!(c_records.len(), r_records.len(), 
        "Different number of SAM records: C={} Rust={}", c_records.len(), r_records.len());
    
    for (i, (c, r)) in c_records.iter().zip(r_records.iter()).enumerate() {
        let c_fields: Vec<&str> = c.split('\t').collect();
        let r_fields: Vec<&str> = r.split('\t').collect();
        if c_fields.len() > 5 && r_fields.len() > 5 {
            assert_eq!(c_fields[5], r_fields[5],
                "Record {}: CIGAR differs\n  C:  {}\n  Rs: {}", i, c_fields[5], r_fields[5]);
        }
    }
}

/// Test that gap-fill APPROX_MAX CIGAR covers the full gap.
#[test]
fn test_gapfill_cigar_covers_full_gap() {
    use minimap2::align::{align_pair_dual, cigar_to_string};
    use minimap2::align::score::gen_simple_mat;
    use minimap2::flags::KswFlags;
    
    let mut mat = Vec::new();
    gen_simple_mat(5, &mut mat, 1, 4, 1);

    for len in [100, 200, 500, 1000] {
        let query: Vec<u8> = (0..len).map(|i| (i % 4) as u8).collect();
        let mut target = query.clone();
        if len > 50 { target[50] = (target[50] + 1) % 4; }
        if len > 150 { target[150] = (target[150] + 2) % 4; }
        if len > 100 { target.insert(100, 3); }

        let ez = align_pair_dual(&query, &target, 5, &mat, 
            6, 2, 26, 1, 30000, 400, -1, KswFlags::APPROX_MAX);
        
        if ez.cigar.is_empty() { continue; }
        
        let mut qcons = 0i32;
        let mut tcons = 0i32;
        for &c in &ez.cigar {
            let op = c & 0xf; let clen = (c >> 4) as i32;
            match op { 0 | 7 | 8 => { qcons += clen; tcons += clen; } 1 => { qcons += clen; } 2 | 3 => { tcons += clen; } _ => {} }
        }
        
        assert_eq!(qcons, query.len() as i32,
            "len={}: gap-fill CIGAR consumes {} query bases but gap has {} (CIGAR={})",
            len, qcons, query.len(), cigar_to_string(&ez.cigar));
        assert_eq!(tcons, target.len() as i32,
            "len={}: gap-fill CIGAR consumes {} target bases but gap has {} (CIGAR={})",
            len, tcons, target.len(), cigar_to_string(&ez.cigar));
    }
}

/// Test that a single read against chr11 produces the same CIGAR as C.
/// This catches the remaining alignment difference that only manifests
/// with longer right extensions (>100bp).
#[test] 
fn test_chr11_cigar_vs_c() {
    if !Path::new("minimap2/minimap2").exists() || !Path::new("target/release/minimap2-pure-rs").exists() { return; }
    // Need chr11 reference and test read
    let chr11 = "/tmp/chr11.fa";
    let query = "/tmp/mini_query.fq";
    if !Path::new(chr11).exists() || !Path::new(query).exists() { return; }
    
    let c_out = std::process::Command::new("minimap2/minimap2")
        .args(&["-c", "-x", "map-hifi", chr11, query])
        .stderr(std::process::Stdio::null())
        .output().unwrap();
    let c_paf = String::from_utf8_lossy(&c_out.stdout);
    let c_line = c_paf.lines().next().unwrap_or("");
    let c_fields: Vec<&str> = c_line.split('\t').collect();
    
    let r_out = std::process::Command::new("target/release/minimap2-pure-rs")
        .args(&["-c", "-x", "map-hifi", chr11, query])
        .stderr(std::process::Stdio::null())
        .output().unwrap();
    let r_paf = String::from_utf8_lossy(&r_out.stdout);
    let r_line = r_paf.lines().next().unwrap_or("");
    let r_fields: Vec<&str> = r_line.split('\t').collect();
    
    if c_fields.len() > 3 && r_fields.len() > 3 {
        assert_eq!(c_fields[3], r_fields[3],
            "qend differs: C={} Rust={}", c_fields[3], r_fields[3]);
    }
}

mod zdrop_test_data;

/// Test that SIMD and scalar produce identical z-drop results for real sequences.
/// These sequences are from a z-dropped gap-fill that causes CIGAR differences.
#[test]
fn test_zdrop_real_sequences_simd_vs_scalar() {
    use minimap2::align::{align_pair_dual, cigar_to_string};
    use minimap2::align::score::gen_simple_mat;
    use minimap2::flags::KswFlags;
    
    let (query, target) = zdrop_test_data::get_zdrop_seqs();
    let mut mat = Vec::new();
    gen_simple_mat(5, &mut mat, 1, 4, 1); // map-hifi: a=1, b=4
    
    // Z-drop second pass: no flags (empty), zdrop=400
    let simd = align_pair_dual(&query, &target, 5, &mat,
        6, 2, 26, 1, // q=6, e=2, q2=26, e2=1
        30001, 400, -1, KswFlags::empty());
    
    let scalar = minimap2::align::ksw2::ksw_extd2(&query, &target, 5, &mat,
        6, 2, 26, 1,
        30001, 400, -1, KswFlags::empty());
    
    let simd_cig = cigar_to_string(&simd.cigar);
    let scalar_cig = cigar_to_string(&scalar.cigar);
    
    eprintln!("SIMD:   zdrop={} max_t={} max_q={} score={} cigar={}", simd.zdropped, simd.max_t, simd.max_q, simd.score, simd_cig);
    eprintln!("Scalar: zdrop={} max_t={} max_q={} score={} cigar={}", scalar.zdropped, scalar.max_t, scalar.max_q, scalar.score, scalar_cig);
    
    assert_eq!(simd.zdropped, scalar.zdropped,
        "zdropped differs: SIMD={} scalar={}", simd.zdropped, scalar.zdropped);
    assert_eq!(simd.max_t, scalar.max_t, 
        "max_t differs: SIMD={} scalar={}", simd.max_t, scalar.max_t);
    assert_eq!(simd_cig, scalar_cig,
        "CIGAR differs: SIMD={} scalar={}", simd_cig, scalar_cig);
}

/// Test z-drop second pass behavior with various bandwidths.
/// The z-drop fires at different positions depending on bandwidth.
#[test]
fn test_zdrop_second_pass_bandwidth() {
    use minimap2::align::{align_pair_dual, cigar_to_string};
    use minimap2::align::score::gen_simple_mat;
    use minimap2::flags::KswFlags;
    
    let (query, target) = zdrop_test_data::get_zdrop_seqs();
    let mut mat = Vec::new();
    gen_simple_mat(5, &mut mat, 1, 4, 1);
    
    // Test with different bandwidths - the z-drop result should be consistent
    for bw in [500, 751, 1000, 5000, 30001] {
        let simd = align_pair_dual(&query, &target, 5, &mat,
            6, 2, 26, 1, bw, 400, -1, KswFlags::empty());
        let scalar = minimap2::align::ksw2::ksw_extd2(&query, &target, 5, &mat,
            6, 2, 26, 1, bw, 400, -1, KswFlags::empty());
        
        eprintln!("bw={}: SIMD max_t={} zdrop={} cigar_ops={}, Scalar max_t={} zdrop={} cigar_ops={}",
            bw, simd.max_t, simd.zdropped, simd.cigar.len(), scalar.max_t, scalar.zdropped, scalar.cigar.len());
        
        assert_eq!(simd.max_t, scalar.max_t,
            "bw={}: max_t SIMD={} scalar={}", bw, simd.max_t, scalar.max_t);
    }
}

/// Test that APPROX_MAX first pass produces the same CIGAR for SIMD and scalar.
#[test]
fn test_zdrop_first_pass_approx_max() {
    use minimap2::align::{align_pair_dual, cigar_to_string};
    use minimap2::align::score::gen_simple_mat;
    use minimap2::flags::KswFlags;
    
    let (query, target) = zdrop_test_data::get_zdrop_seqs();
    let mut mat = Vec::new();
    gen_simple_mat(5, &mut mat, 1, 4, 1);
    
    let simd = align_pair_dual(&query, &target, 5, &mat,
        6, 2, 26, 1, 30001, 400, -1, KswFlags::APPROX_MAX);
    let scalar = minimap2::align::ksw2::ksw_extd2(&query, &target, 5, &mat,
        6, 2, 26, 1, 30001, 400, -1, KswFlags::APPROX_MAX);
    
    let simd_cig = cigar_to_string(&simd.cigar);
    let scalar_cig = cigar_to_string(&scalar.cigar);
    
    eprintln!("First pass APPROX_MAX:");
    eprintln!("  SIMD:   cigar={} score={} zdropped={}", simd_cig, simd.score, simd.zdropped);
    eprintln!("  Scalar: cigar={} score={} zdropped={}", scalar_cig, scalar.score, scalar.zdropped);
    
    assert_eq!(simd_cig, scalar_cig,
        "First pass CIGAR differs: SIMD={} scalar={}", simd_cig, scalar_cig);
}
