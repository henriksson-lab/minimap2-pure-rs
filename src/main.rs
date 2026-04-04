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

    /// Min CIGAR score for primary
    #[arg(short = 's', default_value_t = 0)]
    min_dp_score: i32,

    /// Suppress PAF output for unmapped
    #[arg(long = "paf-no-hit")]
    paf_no_hit: bool,

    /// Dump index to file
    #[arg(short = 'd', value_name = "FILE")]
    dump_index: Option<String>,

    /// Soft clipping
    #[arg(short = 'Y')]
    softclip: bool,

    /// Read group line
    #[arg(short = 'R', value_name = "STR")]
    rg: Option<String>,

    /// Max intron length (for splice)
    #[arg(short = 'G', long = "max-intron-len", default_value_t = 0)]
    max_intron: i32,
}

fn main() {
    env_logger::init();
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
    if cli.softclip {
        mo.flag |= MapFlags::SOFTCLIP;
    }
    if let Some(ref sec) = cli.secondary {
        if sec == "no" {
            mo.flag |= MapFlags::NO_PRINT_2ND;
        }
    }
    if cli.max_intron > 0 {
        options::MapOpt::default(); // just to reference
        mo.max_gap_ref = cli.max_intron;
        mo.bw = cli.max_intron;
        mo.bw_long = cli.max_intron;
    }

    // Validate options
    if let Err(e) = options::check_opt(&io, &mo) {
        eprintln!("[ERROR] {}", e);
        std::process::exit(1);
    }

    // Build or load index
    let is_idx = index::io::is_idx_file(&cli.target).unwrap_or(false);
    let mi = if is_idx {
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

    // Dump index if requested
    if let Some(ref out_path) = cli.dump_index {
        eprintln!("[M::main] writing index to {}", out_path);
        let mut f = std::fs::File::create(out_path).unwrap();
        index::io::idx_dump(&mut f, &mi).unwrap();
    }

    // Map query files
    if cli.query.is_empty() {
        // No query files — index-only mode
        return;
    }

    // Update mid_occ based on index
    if mo.mid_occ <= 0 {
        mo.mid_occ = mi.cal_max_occ(mo.mid_occ_frac);
        if mo.mid_occ < mo.min_mid_occ { mo.mid_occ = mo.min_mid_occ; }
        if mo.max_mid_occ > mo.min_mid_occ && mo.mid_occ > mo.max_mid_occ { mo.mid_occ = mo.max_mid_occ; }
        eprintln!("[M::main] mid_occ = {}", mo.mid_occ);
    }

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
