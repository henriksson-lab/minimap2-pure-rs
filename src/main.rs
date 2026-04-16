use clap::Parser;
use minimap2::flags::MapFlags;
use minimap2::index::{self, MmIdx};
use minimap2::options::{self, IdxOpt, MapOpt};
use minimap2::pipeline;
use minimap2::types::MM_VERSION;

#[derive(Parser)]
#[command(name = "minimap2-pure-rs", version = MM_VERSION, about = "Pure Rust minimap2 sequence aligner")]
struct Cli {
    /// Reference FASTA/index file
    target: String,

    /// Query FASTA/FASTQ file(s)
    query: Vec<String>,

    /// Preset (map-ont, map-pb, map-hifi, sr, splice, asm5, asm10, asm20, ava-ont, ava-pb)
    #[arg(short = 'x', long = "preset")]
    preset: Option<String>,

    /// k-mer size
    #[arg(short = 'k', default_value_t = 0)]
    kmer: i16,

    /// Minimizer window size
    #[arg(short = 'w', default_value_t = 0)]
    window: i16,

    /// Number of threads
    #[arg(short = 't', long = "threads", default_value_t = 3)]
    threads: usize,

    /// Output SAM
    #[arg(short = 'a', long = "sam")]
    sam: bool,

    /// Output CIGAR in PAF
    #[arg(short = 'c')]
    cigar: bool,

    /// Output cs tag
    #[arg(long = "cs")]
    cs: bool,

    /// Output MD tag
    #[arg(long = "MD")]
    md: bool,

    /// Output ds tag
    #[arg(long = "ds")]
    ds: bool,

    /// Output to FILE instead of stdout
    #[arg(short = 'o', value_name = "FILE")]
    output: Option<String>,

    /// Matching score
    #[arg(short = 'A', default_value_t = 0)]
    match_score: i32,

    /// Mismatch penalty
    #[arg(short = 'B', default_value_t = 0)]
    mismatch: i32,

    /// Filter out top FLOAT fraction of repetitive minimizers
    #[arg(short = 'f', default_value_t = -1.0)]
    mid_occ_frac: f32,

    /// Gap open penalty
    #[arg(short = 'O', value_name = "INT[,INT]")]
    gap_open: Option<String>,

    /// Gap extension penalty
    #[arg(short = 'E', value_name = "INT[,INT]")]
    gap_ext: Option<String>,

    /// Bandwidth
    #[arg(short = 'r', value_name = "INT[,INT]")]
    bandwidth: Option<String>,

    /// Top N secondary alignments
    #[arg(short = 'N', default_value_t = 0)]
    best_n: i32,

    /// Don't output secondary alignments
    #[arg(long = "secondary", value_name = "yes|no")]
    secondary: Option<String>,

    /// Min peak DP alignment score
    #[arg(short = 's', default_value_t = 0)]
    min_dp_score: i32,

    /// Z-drop score [,inversion Z-drop]
    #[arg(short = 'z', value_name = "INT[,INT]")]
    zdrop: Option<String>,

    /// Stop chain elongation if no minimizers in NUM bp
    #[arg(short = 'g', default_value_t = 0)]
    max_gap: i32,

    /// Max fragment length
    #[arg(short = 'F', default_value_t = 0)]
    max_frag_len: i32,

    /// Enable fragment mode for adjacent reads with the same name
    #[arg(long = "frag", value_name = "yes|no", default_missing_value = "yes", num_args = 0..=1)]
    frag: Option<String>,

    /// Min number of minimizers on a chain
    #[arg(short = 'n', default_value_t = 0)]
    min_cnt: i32,

    /// Min chaining score
    #[arg(short = 'm', default_value_t = 0)]
    min_chain_score: i32,

    /// Min secondary-to-primary score ratio
    #[arg(short = 'p', default_value_t = -1.0)]
    pri_ratio: f32,

    /// Skip self and dual mappings (all-vs-all mode)
    #[arg(short = 'X')]
    skip_self: bool,

    /// Suppress PAF output for unmapped
    #[arg(long = "paf-no-hit")]
    paf_no_hit: bool,

    /// Dump index to file
    #[arg(short = 'd', value_name = "FILE")]
    dump_index: Option<String>,

    /// Soft clipping
    #[arg(short = 'Y')]
    softclip: bool,

    /// Long CIGAR in CG tag for BAM compatibility
    #[arg(short = 'L')]
    long_cigar: bool,

    /// Use =/X instead of M in CIGAR
    #[arg(long = "eqx")]
    eqx: bool,

    /// Only map to forward strand
    #[arg(long = "for-only")]
    for_only: bool,

    /// Only map to reverse strand
    #[arg(long = "rev-only")]
    rev_only: bool,

    /// Preserve query-strand coordinates for reverse-strand PAF output
    #[arg(long = "qstrand")]
    qstrand: bool,

    /// Output all chains (no primary/secondary selection)
    #[arg(short = 'P')]
    all_chains: bool,

    /// Copy FASTA/FASTQ comments to output
    #[arg(long = "copy-comment")]
    copy_comment: bool,

    /// Do not output base qualities in SAM
    #[arg(short = 'Q')]
    no_qual: bool,

    /// Only output SAM records with hits
    #[arg(long = "sam-hit-only")]
    sam_hit_only: bool,

    /// Output SEQ/QUAL for secondary alignments with hard clipping
    #[arg(long = "secondary-seq")]
    secondary_seq: bool,

    /// HPC mode (homopolymer compression)
    #[arg(short = 'H')]
    hpc: bool,

    /// Build an index without target sequences
    #[arg(long = "idx-no-seq")]
    idx_no_seq: bool,

    /// Splice junction BED12 file
    #[arg(long = "junc-bed", value_name = "FILE")]
    junc_bed: Option<String>,

    /// Write splice junctions extracted from spliced alignments
    #[arg(long = "write-junc")]
    write_junc: bool,

    /// Split index prefix for large genomes (multi-pass mapping)
    #[arg(long = "split-prefix", value_name = "PREFIX")]
    split_prefix: Option<String>,

    /// SDUST threshold (0 to disable)
    #[arg(short = 'T', long = "sdust-thres", default_value_t = 0)]
    sdust_thres: i32,

    /// Read group line
    #[arg(short = 'R', value_name = "STR")]
    rg: Option<String>,

    /// Splice junction score bonus
    #[arg(long = "junc-bonus", default_value_t = i32::MIN)]
    junc_bonus: i32,

    /// Splice junction penalty
    #[arg(long = "junc-pen", default_value_t = i32::MIN)]
    junc_pen: i32,

    /// Splice model: 0 for old, 1 for new
    #[arg(short = 'J', default_value_t = -1)]
    splice_model: i32,

    /// Splice strand: f, r, b, or n
    #[arg(short = 'u', value_name = "CHAR")]
    splice_strand: Option<String>,

    /// ALT contig score drop fraction
    #[arg(long = "alt-drop", default_value_t = -1.0)]
    alt_drop: f32,

    /// ALT contig name list
    #[arg(long = "alt", value_name = "FILE")]
    alt: Option<String>,

    /// Max intron length (for splice)
    #[arg(short = 'G', long = "max-intron-len", default_value_t = 0)]
    max_intron: i32,

    /// Split index for every NUM input bases
    #[arg(short = 'I', value_name = "NUM")]
    batch_size: Option<String>,

    /// Mini-batch size for mapping (in bases)
    #[arg(short = 'K', value_name = "NUM")]
    mini_batch: Option<String>,

    /// Maximum number of secondary alignments to output
    #[arg(long = "max-occ", default_value_t = 0)]
    max_occ: i32,
}

/// Check that the CPU supports the instruction set this binary was compiled for.
/// With `-C target-cpu=native`, the compiler may emit AVX2/AVX-512 instructions
/// in non-SIMD code. Without this check, the binary would SIGILL on older CPUs.
fn check_cpu_features() {
    #[cfg(target_arch = "x86_64")]
    {
        let mut missing = Vec::new();
        // Check features that target-cpu=native may have enabled.
        // These are the common ones for Skylake+ (the build machine).
        if cfg!(target_feature = "avx2") && !is_x86_feature_detected!("avx2") {
            missing.push("AVX2");
        }
        if cfg!(target_feature = "avx") && !is_x86_feature_detected!("avx") {
            missing.push("AVX");
        }
        if cfg!(target_feature = "sse4.2") && !is_x86_feature_detected!("sse4.2") {
            missing.push("SSE4.2");
        }
        if cfg!(target_feature = "sse4.1") && !is_x86_feature_detected!("sse4.1") {
            missing.push("SSE4.1");
        }
        if cfg!(target_feature = "bmi2") && !is_x86_feature_detected!("bmi2") {
            missing.push("BMI2");
        }
        if cfg!(target_feature = "fma") && !is_x86_feature_detected!("fma") {
            missing.push("FMA");
        }
        if !missing.is_empty() {
            eprintln!(
                "ERROR: This binary was compiled for a CPU with {} support,",
                missing.join(", ")
            );
            eprintln!("but this CPU does not support: {}", missing.join(", "));
            eprintln!();
            eprintln!("Rebuild without -C target-cpu=native for a portable binary:");
            eprintln!("  Remove .cargo/config.toml or set rustflags = []");
            eprintln!("  Then: cargo build --release");
            std::process::exit(1);
        }
    }
}

fn main() {
    check_cpu_features();
    env_logger::init();
    let t_start = std::time::Instant::now();
    let cli = Cli::parse();
    let args: Vec<String> = std::env::args().collect();

    // Initialize options
    let mut io = IdxOpt::default();
    let mut mo = MapOpt::default();

    // Apply preset first
    if let Some(ref preset) = cli.preset {
        if let Err(e) = options::set_opt(Some(preset), &mut io, &mut mo) {
            eprintln!("[ERROR] unknown preset: {}", e);
            std::process::exit(1);
        }
    } else {
        options::set_opt(None, &mut io, &mut mo).unwrap();
    }

    // Override with CLI options
    if cli.kmer > 0 {
        io.k = cli.kmer;
    }
    if cli.window > 0 {
        io.w = cli.window;
    }
    if cli.match_score > 0 {
        mo.a = cli.match_score;
    }
    if cli.mismatch > 0 {
        mo.b = cli.mismatch;
    }
    if cli.mid_occ_frac >= 0.0 {
        mo.mid_occ_frac = cli.mid_occ_frac;
    }
    if let Some(ref s) = cli.gap_open {
        let parts: Vec<&str> = s.split(',').collect();
        mo.q = parts[0].parse().unwrap_or(mo.q);
        if parts.len() > 1 {
            mo.q2 = parts[1].parse().unwrap_or(mo.q2);
        }
    }
    if let Some(ref s) = cli.gap_ext {
        let parts: Vec<&str> = s.split(',').collect();
        mo.e = parts[0].parse().unwrap_or(mo.e);
        if parts.len() > 1 {
            mo.e2 = parts[1].parse().unwrap_or(mo.e2);
        }
    }
    if let Some(ref s) = cli.bandwidth {
        let parts: Vec<&str> = s.split(',').collect();
        mo.bw = parts[0].parse().unwrap_or(mo.bw);
        if parts.len() > 1 {
            mo.bw_long = parts[1].parse().unwrap_or(mo.bw_long);
        }
    }
    if cli.best_n > 0 {
        mo.best_n = cli.best_n;
    }
    if cli.min_dp_score > 0 {
        mo.min_dp_max = cli.min_dp_score;
    }
    if cli.sam {
        mo.flag |= MapFlags::OUT_SAM | MapFlags::CIGAR;
    }
    if cli.cigar {
        mo.flag |= MapFlags::CIGAR | MapFlags::OUT_CG;
    }
    if cli.cs {
        mo.flag |= MapFlags::OUT_CS;
    }
    if cli.md {
        mo.flag |= MapFlags::OUT_MD;
    }
    if cli.ds {
        mo.flag |= MapFlags::OUT_DS;
    }
    if cli.paf_no_hit {
        mo.flag |= MapFlags::PAF_NO_HIT;
    }
    if cli.for_only {
        mo.flag |= MapFlags::FOR_ONLY;
    }
    if cli.rev_only {
        mo.flag |= MapFlags::REV_ONLY;
    }
    if cli.qstrand {
        mo.flag |= MapFlags::QSTRAND | MapFlags::NO_INV;
    }
    if cli.all_chains {
        mo.flag |= MapFlags::ALL_CHAINS;
    }
    if cli.copy_comment {
        mo.flag |= MapFlags::COPY_COMMENT;
    }
    if cli.no_qual {
        mo.flag |= MapFlags::NO_QUAL;
    }
    if cli.sam_hit_only {
        mo.flag |= MapFlags::SAM_HIT_ONLY;
    }
    if cli.secondary_seq {
        mo.flag |= MapFlags::SECONDARY_SEQ;
    }
    if cli.hpc {
        io.flag |= minimap2::flags::IdxFlags::HPC;
    }
    if cli.idx_no_seq {
        io.flag |= minimap2::flags::IdxFlags::NO_SEQ;
    }
    if cli.write_junc {
        mo.flag |= MapFlags::OUT_JUNC | MapFlags::CIGAR;
    }
    if cli.sdust_thres > 0 {
        mo.sdust_thres = cli.sdust_thres;
    }
    if cli.softclip {
        mo.flag |= MapFlags::SOFTCLIP;
    }
    if cli.long_cigar {
        mo.flag |= MapFlags::LONG_CIGAR;
    }
    if cli.eqx {
        mo.flag |= MapFlags::EQX;
    }
    if let Some(ref sec) = cli.secondary {
        if sec == "no" {
            mo.flag |= MapFlags::NO_PRINT_2ND;
        } else if sec == "yes" {
            mo.flag.remove(MapFlags::NO_PRINT_2ND);
        }
    }
    if cli.max_intron > 0 {
        mo.max_gap_ref = cli.max_intron;
        mo.bw = cli.max_intron;
        mo.bw_long = cli.max_intron;
    }
    if let Some(ref s) = cli.zdrop {
        let parts: Vec<&str> = s.split(',').collect();
        mo.zdrop = parts[0].parse().unwrap_or(mo.zdrop);
        if parts.len() > 1 {
            mo.zdrop_inv = parts[1].parse().unwrap_or(mo.zdrop_inv);
        }
    }
    if cli.max_gap > 0 {
        mo.max_gap = cli.max_gap;
    }
    if cli.max_frag_len > 0 {
        mo.max_frag_len = cli.max_frag_len;
    }
    if let Some(ref frag) = cli.frag {
        if frag == "yes" {
            mo.flag |= MapFlags::FRAG_MODE;
        } else if frag == "no" {
            mo.flag.remove(MapFlags::FRAG_MODE);
        } else {
            eprintln!("[ERROR] --frag expects yes or no");
            std::process::exit(1);
        }
    }
    if cli.min_cnt > 0 {
        mo.min_cnt = cli.min_cnt;
    }
    if cli.min_chain_score > 0 {
        mo.min_chain_score = cli.min_chain_score;
    }
    if cli.pri_ratio >= 0.0 {
        mo.pri_ratio = cli.pri_ratio;
    }
    if cli.junc_bonus != i32::MIN {
        mo.junc_bonus = cli.junc_bonus;
    }
    if cli.junc_pen != i32::MIN {
        mo.junc_pen = cli.junc_pen;
    }
    if cli.splice_model == 0 {
        mo.flag |= MapFlags::SPLICE_OLD;
    } else if cli.splice_model == 1 {
        mo.flag.remove(MapFlags::SPLICE_OLD);
    }
    if let Some(ref strand) = cli.splice_strand {
        match strand.as_bytes().first().copied() {
            Some(b'b') => mo.flag |= MapFlags::SPLICE_FOR | MapFlags::SPLICE_REV,
            Some(b'f') => {
                mo.flag |= MapFlags::SPLICE_FOR;
                mo.flag.remove(MapFlags::SPLICE_REV);
            }
            Some(b'r') => {
                mo.flag |= MapFlags::SPLICE_REV;
                mo.flag.remove(MapFlags::SPLICE_FOR);
            }
            Some(b'n') => mo.flag.remove(MapFlags::SPLICE_FOR | MapFlags::SPLICE_REV),
            _ => {}
        }
    }
    if cli.alt_drop >= 0.0 {
        mo.alt_drop = cli.alt_drop;
    }
    if cli.skip_self {
        mo.flag |= MapFlags::NO_DIAG | MapFlags::NO_DUAL;
    }
    if let Some(ref prefix) = cli.split_prefix {
        mo.split_prefix = Some(prefix.clone());
    }
    if let Some(ref kb) = cli.mini_batch {
        if let Ok(v) = parse_num(kb) {
            mo.mini_batch_size = v;
        }
    }
    if let Some(ref ib) = cli.batch_size {
        if let Ok(v) = parse_num(ib) {
            io.batch_size = v as u64;
        }
    }
    if cli.max_occ > 0 {
        mo.max_occ = cli.max_occ;
    }

    // Validate options
    if let Err(e) = options::check_opt(&io, &mo) {
        eprintln!("[ERROR] {}", e);
        std::process::exit(1);
    }

    // Build or load index
    let is_idx = index::io::is_idx_file(&cli.target).unwrap_or(false);
    let mut split_parts: Option<Vec<MmIdx>> = None;
    let mut mi = if is_idx {
        eprintln!("[M::main] loading index from {}", cli.target);
        let mut f = std::fs::File::open(&cli.target).unwrap();
        match index::io::idx_load(&mut f) {
            Ok(Some(mi)) => mi,
            _ => {
                eprintln!("[ERROR] failed to load index");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!(
            "[M::main] building index for {} (k={}, w={})",
            cli.target, io.k, io.w
        );
        if let Some(ref prefix) = cli.split_prefix {
            if let Ok(parts) = MmIdx::build_parts_from_file(
                &cli.target,
                io.w as i32,
                io.k as i32,
                io.bucket_bits,
                io.flag,
                io.batch_size,
            ) {
                for (part_idx, part) in parts.iter().enumerate() {
                    if let Err(e) = index::split::create_split_tmp(prefix, part_idx, part) {
                        eprintln!("[WARNING] failed to write split temp header: {}", e);
                        break;
                    }
                }
                let _ = index::split::remove_split_tmps(prefix, parts.len());
                split_parts = Some(parts);
            }
        }
        let build_batch_size = if cli.split_prefix.is_some() {
            u64::MAX
        } else {
            io.batch_size
        };
        match MmIdx::build_from_file(
            &cli.target,
            io.w as i32,
            io.k as i32,
            io.bucket_bits,
            io.flag,
            io.mini_batch_size,
            build_batch_size,
        ) {
            Ok(Some(mi)) => mi,
            Ok(None) => {
                eprintln!("[ERROR] failed to build index (empty file?)");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("[ERROR] failed to open {}: {}", cli.target, e);
                std::process::exit(1);
            }
        }
    };

    mi.stat();

    // Load junction annotations if provided
    if let Some(ref junc_path) = cli.junc_bed {
        match minimap2::junc::read_junc_bed(&mut mi, junc_path) {
            Ok(n) => eprintln!("[M::main] loaded {} junctions from {}", n, junc_path),
            Err(e) => eprintln!("[WARNING] failed to read junctions: {}", e),
        }
    }

    if let Some(ref alt_path) = cli.alt {
        match mi.read_alt_file(alt_path) {
            Ok(n) => eprintln!("[M::main] found {} ALT contigs", n),
            Err(e) => eprintln!("[WARNING] failed to read ALT contigs: {}", e),
        }
    }

    // Dump index if requested
    if let Some(ref out_path) = cli.dump_index {
        eprintln!("[M::main] writing index to {}", out_path);
        let mut f = std::fs::File::create(out_path).unwrap();
        index::io::idx_dump(&mut f, &mi).unwrap();
    }

    if cli.split_prefix.is_some() {
        eprintln!(
            "[WARNING] --split-prefix uses Rust split-index merging for single-end, grouped fragment, and two-file paired-end mapping"
        );
    }

    // Redirect output to file if -o specified
    #[cfg(unix)]
    if let Some(ref path) = cli.output {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("[ERROR] failed to create output file {}: {}", path, e);
            std::process::exit(1);
        });
        unsafe {
            libc_dup2(file.as_raw_fd(), 1);
        }
    }

    // Map query files
    if cli.query.is_empty() {
        // No query files — index-only mode
        return;
    }

    // Update mapping options based on index
    options::mapopt_update(&mut mo, &mi);
    eprintln!("[M::main] mid_occ = {}", mo.mid_occ);
    let split_parts_ref = split_parts.as_deref();

    if cli.query.len() == 2 {
        // Paired-end mode: two query files
        eprintln!("[M::main] mapping PE: {} + {}", cli.query[0], cli.query[1]);
        let result = if mo.flag.contains(MapFlags::OUT_SAM) {
            if let Some(parts) = split_parts_ref {
                pipeline::map_file_pe_sam_split(
                    &mi,
                    parts,
                    &mo,
                    &cli.query[0],
                    &cli.query[1],
                    cli.threads,
                    cli.rg.as_deref(),
                    &args,
                )
            } else {
                pipeline::map_file_pe_sam(
                    &mi,
                    &mo,
                    &cli.query[0],
                    &cli.query[1],
                    cli.threads,
                    cli.rg.as_deref(),
                    &args,
                )
            }
        } else {
            if let Some(parts) = split_parts_ref {
                pipeline::map_file_pe_paf_split(
                    &mi,
                    parts,
                    &mo,
                    &cli.query[0],
                    &cli.query[1],
                    cli.threads,
                )
            } else {
                pipeline::map_file_pe_paf(&mi, &mo, &cli.query[0], &cli.query[1], cli.threads)
            }
        };
        if let Err(e) = result {
            eprintln!("[ERROR] PE mapping failed: {}", e);
            std::process::exit(1);
        }
    } else {
        for qpath in &cli.query {
            eprintln!("[M::main] mapping {}", qpath);
            let result = if mo.flag.contains(MapFlags::OUT_SAM) {
                if let Some(parts) = split_parts_ref {
                    if mo.flag.contains(MapFlags::FRAG_MODE) {
                        pipeline::map_file_frag_sam_split(
                            &mi,
                            parts,
                            &mo,
                            qpath,
                            cli.threads,
                            cli.rg.as_deref(),
                            &args,
                        )
                    } else {
                        pipeline::map_file_sam_split(
                            &mi,
                            parts,
                            &mo,
                            qpath,
                            cli.threads,
                            cli.rg.as_deref(),
                            &args,
                        )
                    }
                } else {
                    if mo.flag.contains(MapFlags::FRAG_MODE) {
                        pipeline::map_file_frag_sam(
                            &mi,
                            &mo,
                            qpath,
                            cli.threads,
                            cli.rg.as_deref(),
                            &args,
                        )
                    } else {
                        pipeline::map_file_sam(
                            &mi,
                            &mo,
                            qpath,
                            cli.threads,
                            cli.rg.as_deref(),
                            &args,
                        )
                    }
                }
            } else {
                if let Some(parts) = split_parts_ref {
                    if mo.flag.contains(MapFlags::FRAG_MODE) {
                        pipeline::map_file_frag_paf_split(&mi, parts, &mo, qpath, cli.threads)
                    } else {
                        pipeline::map_file_paf_split(&mi, parts, &mo, qpath, cli.threads)
                    }
                } else {
                    if mo.flag.contains(MapFlags::FRAG_MODE) {
                        pipeline::map_file_frag_paf(&mi, &mo, qpath, cli.threads)
                    } else {
                        pipeline::map_file_paf(&mi, &mo, qpath, cli.threads)
                    }
                }
            };
            if let Err(e) = result {
                eprintln!("[ERROR] mapping failed: {}", e);
                std::process::exit(1);
            }
        }
    }
    let elapsed = t_start.elapsed();
    eprintln!("[M::main] Version: {}, pairwise mapping", MM_VERSION);
    eprintln!("[M::main] CMD: {}", args.join(" "));
    eprintln!(
        "[M::main] Real time: {:.3} sec; Peak RSS: {:.3} GB",
        elapsed.as_secs_f64(),
        peak_rss_gb()
    );
}

#[cfg(unix)]
unsafe fn libc_dup2(oldfd: i32, newfd: i32) -> i32 {
    extern "C" {
        fn dup2(oldfd: i32, newfd: i32) -> i32;
    }
    dup2(oldfd, newfd)
}

/// Parse a number with optional K/M/G suffix.
fn parse_num(s: &str) -> Result<i64, std::num::ParseIntError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    let (num_part, multiplier) = match s.as_bytes().last() {
        Some(b'k' | b'K') => (&s[..s.len() - 1], 1_000i64),
        Some(b'm' | b'M') => (&s[..s.len() - 1], 1_000_000i64),
        Some(b'g' | b'G') => (&s[..s.len() - 1], 1_000_000_000i64),
        _ => (s, 1i64),
    };
    Ok(num_part.parse::<i64>()? * multiplier)
}

fn peak_rss_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmHWM:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb.parse::<f64>() {
                            return kb / 1_048_576.0;
                        }
                    }
                }
            }
        }
    }
    0.0
}
