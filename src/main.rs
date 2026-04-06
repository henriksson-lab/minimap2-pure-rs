use clap::Parser;
use minimap2::flags::MapFlags;
use minimap2::index::{self, MmIdx};
use minimap2::options::{self, IdxOpt, MapOpt};
use minimap2::pipeline;
use minimap2::types::MM_VERSION;


#[derive(Parser)]
#[command(name = "minimap2-rs", version = MM_VERSION, about = "Pure Rust minimap2 sequence aligner")]
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

    /// Output to FILE instead of stdout
    #[arg(short = 'o', value_name = "FILE")]
    output: Option<String>,

    /// Matching score
    #[arg(short = 'A', default_value_t = 0)]
    match_score: i32,

    /// Mismatch penalty
    #[arg(short = 'B', default_value_t = 0)]
    mismatch: i32,

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

    /// Use =/X instead of M in CIGAR
    #[arg(long = "eqx")]
    eqx: bool,

    /// Only map to forward strand
    #[arg(long = "for-only")]
    for_only: bool,

    /// Only map to reverse strand
    #[arg(long = "rev-only")]
    rev_only: bool,

    /// Output all chains (no primary/secondary selection)
    #[arg(short = 'P')]
    all_chains: bool,

    /// Copy FASTA/FASTQ comments to output
    #[arg(long = "copy-comment")]
    copy_comment: bool,

    /// HPC mode (homopolymer compression)
    #[arg(short = 'H')]
    hpc: bool,

    /// Splice junction BED12 file
    #[arg(long = "junc-bed", value_name = "FILE")]
    junc_bed: Option<String>,

    /// Split index prefix for large genomes (multi-pass mapping)
    #[arg(long = "split-prefix", value_name = "PREFIX")]
    split_prefix: Option<String>,

    /// SDUST threshold (0 to disable)
    #[arg(long = "sdust-thres", default_value_t = 0)]
    sdust_thres: i32,

    /// Read group line
    #[arg(short = 'R', value_name = "STR")]
    rg: Option<String>,

    /// Max intron length (for splice)
    #[arg(short = 'G', long = "max-intron-len", default_value_t = 0)]
    max_intron: i32,

    /// Mini-batch size for mapping (in bases)
    #[arg(short = 'K', value_name = "NUM")]
    mini_batch: Option<String>,

    /// Maximum number of secondary alignments to output
    #[arg(long = "max-occ", default_value_t = 0)]
    max_occ: i32,
}

fn main() {
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
    if cli.kmer > 0 { io.k = cli.kmer; }
    if cli.window > 0 { io.w = cli.window; }
    if cli.match_score > 0 { mo.a = cli.match_score; }
    if cli.mismatch > 0 { mo.b = cli.mismatch; }
    if let Some(ref s) = cli.gap_open {
        let parts: Vec<&str> = s.split(',').collect();
        mo.q = parts[0].parse().unwrap_or(mo.q);
        if parts.len() > 1 { mo.q2 = parts[1].parse().unwrap_or(mo.q2); }
    }
    if let Some(ref s) = cli.gap_ext {
        let parts: Vec<&str> = s.split(',').collect();
        mo.e = parts[0].parse().unwrap_or(mo.e);
        if parts.len() > 1 { mo.e2 = parts[1].parse().unwrap_or(mo.e2); }
    }
    if let Some(ref s) = cli.bandwidth {
        let parts: Vec<&str> = s.split(',').collect();
        mo.bw = parts[0].parse().unwrap_or(mo.bw);
        if parts.len() > 1 { mo.bw_long = parts[1].parse().unwrap_or(mo.bw_long); }
    }
    if cli.best_n > 0 { mo.best_n = cli.best_n; }
    if cli.min_dp_score > 0 { mo.min_dp_max = cli.min_dp_score; }
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
    if cli.paf_no_hit {
        mo.flag |= MapFlags::PAF_NO_HIT;
    }
    if cli.for_only { mo.flag |= MapFlags::FOR_ONLY; }
    if cli.rev_only { mo.flag |= MapFlags::REV_ONLY; }
    if cli.all_chains { mo.flag |= MapFlags::ALL_CHAINS; }
    if cli.copy_comment { mo.flag |= MapFlags::COPY_COMMENT; }
    if cli.hpc { io.flag |= minimap2::flags::IdxFlags::HPC; }
    if cli.sdust_thres > 0 { mo.sdust_thres = cli.sdust_thres; }
    if cli.softclip {
        mo.flag |= MapFlags::SOFTCLIP;
    }
    if cli.eqx {
        mo.flag |= MapFlags::EQX;
    }
    if let Some(ref sec) = cli.secondary {
        if sec == "no" {
            mo.flag |= MapFlags::NO_PRINT_2ND;
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
        if parts.len() > 1 { mo.zdrop_inv = parts[1].parse().unwrap_or(mo.zdrop_inv); }
    }
    if cli.max_gap > 0 { mo.max_gap = cli.max_gap; }
    if cli.min_cnt > 0 { mo.min_cnt = cli.min_cnt; }
    if cli.min_chain_score > 0 { mo.min_chain_score = cli.min_chain_score; }
    if cli.pri_ratio >= 0.0 { mo.pri_ratio = cli.pri_ratio; }
    if cli.skip_self { mo.flag |= MapFlags::NO_DIAG | MapFlags::NO_DUAL; }
    if let Some(ref prefix) = cli.split_prefix {
        mo.split_prefix = Some(prefix.clone());
    }
    if let Some(ref kb) = cli.mini_batch {
        if let Ok(v) = parse_num(kb) { mo.mini_batch_size = v; }
    }
    if cli.max_occ > 0 { mo.max_occ = cli.max_occ; }

    // Validate options
    if let Err(e) = options::check_opt(&io, &mo) {
        eprintln!("[ERROR] {}", e);
        std::process::exit(1);
    }

    // Build or load index
    let is_idx = index::io::is_idx_file(&cli.target).unwrap_or(false);
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
        eprintln!("[M::main] building index for {} (k={}, w={})", cli.target, io.k, io.w);
        match MmIdx::build_from_file(
            &cli.target, io.w as i32, io.k as i32, io.bucket_bits,
            io.flag, io.mini_batch_size, io.batch_size,
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

    // Dump index if requested
    if let Some(ref out_path) = cli.dump_index {
        eprintln!("[M::main] writing index to {}", out_path);
        let mut f = std::fs::File::create(out_path).unwrap();
        index::io::idx_dump(&mut f, &mi).unwrap();
    }

    if cli.split_prefix.is_some() {
        eprintln!("[WARNING] --split-prefix is not fully implemented; results may be incomplete for very large genomes");
    }

    // Redirect output to file if -o specified
    #[cfg(unix)]
    if let Some(ref path) = cli.output {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("[ERROR] failed to create output file {}: {}", path, e);
            std::process::exit(1);
        });
        unsafe { libc_dup2(file.as_raw_fd(), 1); }
    }

    // Map query files
    if cli.query.is_empty() {
        // No query files — index-only mode
        return;
    }

    // Update mapping options based on index
    options::mapopt_update(&mut mo, &mi);
    eprintln!("[M::main] mid_occ = {}", mo.mid_occ);

    if cli.query.len() == 2 {
        // Paired-end mode: two query files
        eprintln!("[M::main] mapping PE: {} + {}", cli.query[0], cli.query[1]);
        let result = if mo.flag.contains(MapFlags::OUT_SAM) {
            pipeline::map_file_pe_sam(&mi, &mo, &cli.query[0], &cli.query[1], cli.threads, cli.rg.as_deref(), &args)
        } else {
            pipeline::map_file_pe_paf(&mi, &mo, &cli.query[0], &cli.query[1], cli.threads)
        };
        if let Err(e) = result {
            eprintln!("[ERROR] PE mapping failed: {}", e);
            std::process::exit(1);
        }
    } else if cli.query.len() == 1 && mo.flag.contains(MapFlags::FRAG_MODE) {
        // Interleaved paired-end mode: single file with alternating R1/R2
        eprintln!("[M::main] mapping interleaved PE: {}", cli.query[0]);
        let result = if mo.flag.contains(MapFlags::OUT_SAM) {
            pipeline::map_file_interleaved_pe_sam(&mi, &mo, &cli.query[0], cli.threads, cli.rg.as_deref(), &args)
        } else {
            pipeline::map_file_paf(&mi, &mo, &cli.query[0], cli.threads)
        };
        if let Err(e) = result {
            eprintln!("[ERROR] mapping failed: {}", e);
            std::process::exit(1);
        }
    } else {
        for qpath in &cli.query {
            eprintln!("[M::main] mapping {}", qpath);
            let result = if mo.flag.contains(MapFlags::OUT_SAM) {
                pipeline::map_file_sam(&mi, &mo, qpath, cli.threads, cli.rg.as_deref(), &args)
            } else {
                pipeline::map_file_paf(&mi, &mo, qpath, cli.threads)
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
    eprintln!("[M::main] Real time: {:.3} sec; Peak RSS: {:.3} GB",
        elapsed.as_secs_f64(), peak_rss_gb());
}

#[cfg(unix)]
unsafe fn libc_dup2(oldfd: i32, newfd: i32) -> i32 {
    extern "C" { fn dup2(oldfd: i32, newfd: i32) -> i32; }
    dup2(oldfd, newfd)
}

/// Parse a number with optional K/M/G suffix.
fn parse_num(s: &str) -> Result<i64, std::num::ParseIntError> {
    let s = s.trim();
    if s.is_empty() { return Ok(0); }
    let (num_part, multiplier) = match s.as_bytes().last() {
        Some(b'k' | b'K') => (&s[..s.len()-1], 1_000i64),
        Some(b'm' | b'M') => (&s[..s.len()-1], 1_000_000i64),
        Some(b'g' | b'G') => (&s[..s.len()-1], 1_000_000_000i64),
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
