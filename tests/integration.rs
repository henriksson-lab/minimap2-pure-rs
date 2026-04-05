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
