use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use lbc_miner::{config, hip, opencl, pow, stratum};

const DEFAULT_POOL: &str = "stratum+tcp://lbrypool.net:3334";
const DEFAULT_BACKEND: Backend = Backend::Opencl;
const DEFAULT_WORKER: &str = "rust";
const DEFAULT_MINE_SECONDS: u64 = 60;
const DEFAULT_PLATFORM: usize = 0;
const DEFAULT_DEVICE: usize = 0;
const DEFAULT_WORK_SIZE: usize = 256;
const DEFAULT_BATCH_SIZE: usize = 1_048_576;
const DEFAULT_START_NONCE: u32 = 0;
const DEFAULT_CONNECT_TIMEOUT: u64 = 20;
const DEFAULT_RECONNECT_DELAY: u64 = 5;
const DEFAULT_UI_MODE: UiMode = UiMode::Log;
const EVENT_LOG_LIMIT: usize = 12;
const HASHES_PER_DIFF_ONE_SHARE: f64 = 16_777_216.0;

#[derive(Debug, Parser)]
#[command(name = "lbc-miner")]
#[command(about = "Experimental Rust/OpenCL LBRY miner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Connect to a Stratum pool, subscribe, authorize, and print incoming jobs.
    StratumSmoke {
        #[arg(long, default_value = DEFAULT_POOL)]
        url: String,

        #[arg(long)]
        wallet: String,

        #[arg(
            long,
            alias = "password",
            value_name = "PASSWORD",
            default_value = "rust"
        )]
        worker: String,

        #[arg(long, default_value_t = 20)]
        seconds: u64,
    },

    /// Connect to Stratum and assemble one LBRY work header without mining.
    WorkSmoke {
        #[arg(long, default_value = DEFAULT_POOL)]
        url: String,

        #[arg(long)]
        wallet: String,

        #[arg(
            long,
            alias = "password",
            value_name = "PASSWORD",
            default_value = "rust"
        )]
        worker: String,

        #[arg(long, default_value_t = 20)]
        seconds: u64,
    },

    /// Hash a serialized 112-byte LBRY block header.
    HashHeader {
        #[arg(long)]
        header_hex: String,
    },

    /// Suggest pool difficulty for a target share interval.
    SuggestDiff {
        #[arg(long)]
        mhs: f64,

        #[arg(long, default_value_t = 45.0)]
        share_seconds: f64,

        #[arg(long)]
        difficulty: Option<f64>,
    },

    /// List visible OpenCL devices when built with --features opencl.
    ListOpencl,

    /// Show HIP/ROCm toolchain and runtime diagnostics.
    ListHip {
        #[arg(long)]
        hip_arch: Option<String>,
    },

    /// Run a small HIPRTC scanner self-test without connecting to a pool.
    HipSelfTest {
        #[arg(long, default_value_t = 0)]
        device: usize,

        #[arg(long, default_value = "gfx1201")]
        hip_arch: String,

        #[arg(long, default_value_t = 256)]
        work_size: usize,

        #[arg(long, default_value_t = 16)]
        batch_size: usize,

        #[arg(long, default_value_t = 0)]
        start_nonce: u32,
    },

    /// Run a tiny OpenCL LBRY kernel self-test without connecting to a pool.
    GpuSelfTest {
        #[arg(long, default_value_t = 0)]
        platform: usize,

        #[arg(long, default_value_t = 0)]
        device: usize,

        #[arg(long, default_value_t = 1)]
        work_size: usize,

        #[arg(long, default_value_t = 16)]
        batch_size: usize,

        #[arg(long, default_value_t = 0)]
        start_nonce: u32,
    },

    /// Assemble one pool job, run one GPU batch, locally validate candidates, and exit.
    GpuWorkSmoke {
        #[arg(long, default_value = DEFAULT_POOL)]
        url: String,

        #[arg(long)]
        wallet: String,

        #[arg(
            long,
            alias = "password",
            value_name = "PASSWORD",
            default_value = "rust"
        )]
        worker: String,

        #[arg(long, default_value_t = 20)]
        seconds: u64,

        #[arg(long, default_value_t = 0)]
        platform: usize,

        #[arg(long, default_value_t = 0)]
        device: usize,

        #[arg(long, default_value_t = 256)]
        work_size: usize,

        #[arg(long, default_value_t = 4096)]
        batch_size: usize,

        #[arg(long, default_value_t = 0)]
        start_nonce: u32,
    },

    /// Mine LBRY with OpenCL. Use --seconds 0 to run until Ctrl+C.
    Mine {
        #[arg(long)]
        config: Option<PathBuf>,

        #[arg(long)]
        backend: Option<String>,

        #[arg(long)]
        url: Option<String>,

        #[arg(long)]
        wallet: Option<String>,

        #[arg(long, alias = "password", value_name = "PASSWORD")]
        worker: Option<String>,

        #[arg(long)]
        seconds: Option<u64>,

        #[arg(long)]
        platform: Option<usize>,

        #[arg(long)]
        device: Option<usize>,

        #[arg(long)]
        hip_arch: Option<String>,

        #[arg(long)]
        work_size: Option<usize>,

        #[arg(long)]
        batch_size: Option<usize>,

        #[arg(long)]
        start_nonce: Option<u32>,

        #[arg(long)]
        dry_run: bool,

        #[arg(long)]
        connect_timeout: Option<u64>,

        #[arg(long)]
        reconnect_delay: Option<u64>,

        #[arg(long, value_name = "MODE")]
        ui: Option<String>,

        #[arg(long)]
        target_mhs: Option<f64>,

        #[arg(long)]
        batch_sleep_ms: Option<u64>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::StratumSmoke {
            url,
            wallet,
            worker,
            seconds,
        } => {
            let client = stratum::Client::connect(&url)?;
            client.smoke(&wallet, &worker, Duration::from_secs(seconds))?;
        }
        Command::WorkSmoke {
            url,
            wallet,
            worker,
            seconds,
        } => {
            let client = stratum::Client::connect(&url)?;
            client.work_smoke(&wallet, &worker, Duration::from_secs(seconds))?;
        }
        Command::HashHeader { header_hex } => {
            let header = pow::HeaderBytes::from_hex(&header_hex)?;
            let hash = pow::lbry_hash(header.as_bytes());
            println!("{}", hex::encode(hash));
        }
        Command::SuggestDiff {
            mhs,
            share_seconds,
            difficulty,
        } => {
            let suggested = suggested_pool_difficulty(mhs, share_seconds);
            println!(
                "suggested difficulty: {} for {:.3} MH/s at one share every {}",
                format_difficulty(suggested),
                mhs,
                format_optional_seconds(Some(share_seconds))
            );
            if let Some(difficulty) = difficulty {
                println!(
                    "difficulty {} expected share interval: {}",
                    format_difficulty(difficulty),
                    format_optional_seconds(expected_share_interval_secs(mhs, difficulty))
                );
            }
        }
        Command::ListOpencl => {
            opencl::list_devices()?;
        }
        Command::ListHip { hip_arch } => {
            hip::list_devices(hip_arch.as_deref())?;
        }
        Command::HipSelfTest {
            device,
            hip_arch,
            work_size,
            batch_size,
            start_nonce,
        } => {
            hip::gpu_self_test(device, &hip_arch, work_size, batch_size, start_nonce)?;
        }
        Command::GpuSelfTest {
            platform,
            device,
            work_size,
            batch_size,
            start_nonce,
        } => {
            opencl::gpu_self_test(platform, device, work_size, batch_size, start_nonce)?;
        }
        Command::GpuWorkSmoke {
            url,
            wallet,
            worker,
            seconds,
            platform,
            device,
            work_size,
            batch_size,
            start_nonce,
        } => {
            let first = stratum::Client::connect(&url)?.first_work(
                &wallet,
                &worker,
                Duration::from_secs(seconds),
                true,
            )?;
            let difficulty = first.difficulty.unwrap_or(1.0);
            let target = pow::lbry_share_target(difficulty)?;
            let target_tail = pow::target_tail64(&target);
            let config = opencl::SearchConfig {
                platform_index: platform,
                device_index: device,
                work_size,
                batch_size,
            };

            println!("target: {}", hex::encode(target));
            println!("target_tail: {target_tail:#018x}");
            let candidates =
                opencl::scan_batch(&config, &first.work.header, start_nonce, target_tail)?;

            let mut valid = 0usize;
            for nonce in &candidates {
                let header = first.work.with_nonce(*nonce);
                let hash = pow::lbry_work_hash(&header);
                if pow::hash_meets_target(&hash, &target) {
                    valid += 1;
                    println!(
                        "valid candidate nonce={} nonce_hex={} hash={}",
                        nonce,
                        stratum::StratumWork::submit_nonce_hex(*nonce),
                        hex::encode(hash)
                    );
                }
            }

            println!(
                "gpu work smoke complete: candidates={} valid={} batch_size={}",
                candidates.len(),
                valid,
                batch_size
            );
        }
        Command::Mine {
            config: config_path,
            backend,
            url,
            wallet,
            worker,
            seconds,
            platform,
            device,
            hip_arch,
            work_size,
            batch_size,
            start_nonce,
            dry_run,
            connect_timeout,
            reconnect_delay,
            ui,
            target_mhs,
            batch_sleep_ms,
        } => {
            let file_config = load_mine_config(config_path.as_deref())?;
            let options = resolve_mine_options(
                MineCliOverrides {
                    backend,
                    url,
                    wallet,
                    worker,
                    seconds,
                    platform,
                    device,
                    hip_arch,
                    work_size,
                    batch_size,
                    start_nonce,
                    dry_run,
                    connect_timeout,
                    reconnect_delay,
                    ui,
                    target_mhs,
                    batch_sleep_ms,
                },
                file_config,
            )?;
            run_miner(options)?;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Backend {
    Opencl,
    Hip,
}

impl Backend {
    fn parse(input: &str) -> Result<Self> {
        match input.to_ascii_lowercase().as_str() {
            "opencl" => Ok(Self::Opencl),
            "hip" | "rocm" => Ok(Self::Hip),
            other => bail!("unsupported backend '{other}'; expected opencl or hip"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Opencl => "opencl",
            Self::Hip => "hip",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum UiMode {
    Log,
    Tui,
}

impl UiMode {
    fn parse(input: &str) -> Result<Self> {
        match input.to_ascii_lowercase().as_str() {
            "log" | "text" => Ok(Self::Log),
            "tui" | "dashboard" => Ok(Self::Tui),
            other => bail!("unsupported ui '{other}'; expected log or tui"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Tui => "tui",
        }
    }

    fn is_tui(self) -> bool {
        matches!(self, Self::Tui)
    }
}

#[derive(Debug, Default)]
struct MineCliOverrides {
    backend: Option<String>,
    url: Option<String>,
    wallet: Option<String>,
    worker: Option<String>,
    seconds: Option<u64>,
    platform: Option<usize>,
    device: Option<usize>,
    hip_arch: Option<String>,
    work_size: Option<usize>,
    batch_size: Option<usize>,
    start_nonce: Option<u32>,
    dry_run: bool,
    connect_timeout: Option<u64>,
    reconnect_delay: Option<u64>,
    ui: Option<String>,
    target_mhs: Option<f64>,
    batch_sleep_ms: Option<u64>,
}

fn load_mine_config(path: Option<&Path>) -> Result<config::MineConfig> {
    let Some(path) = path else {
        return Ok(config::MineConfig::default());
    };

    let config = config::ConfigFile::from_path(path)?;
    println!("loaded config: {}", path.display());
    Ok(config.mine)
}

fn resolve_mine_options(
    cli: MineCliOverrides,
    file_config: config::MineConfig,
) -> Result<MineOptions> {
    let backend = cli
        .backend
        .or(file_config.backend)
        .map(|backend| Backend::parse(&backend))
        .transpose()?
        .unwrap_or(DEFAULT_BACKEND);

    let url = cli
        .url
        .or(file_config.url)
        .unwrap_or_else(|| DEFAULT_POOL.to_owned());
    let wallet = cli
        .wallet
        .or(file_config.wallet)
        .context("wallet is required; pass --wallet or set mine.wallet in config")?;
    let worker = cli
        .worker
        .or(file_config.worker)
        .unwrap_or_else(|| DEFAULT_WORKER.to_owned());
    let seconds = cli
        .seconds
        .or(file_config.seconds)
        .unwrap_or(DEFAULT_MINE_SECONDS);
    let platform = cli
        .platform
        .or(file_config.platform)
        .unwrap_or(DEFAULT_PLATFORM);
    let device = cli.device.or(file_config.device).unwrap_or(DEFAULT_DEVICE);
    let hip_arch = cli.hip_arch.or(file_config.hip_arch);
    let work_size = cli
        .work_size
        .or(file_config.work_size)
        .unwrap_or(DEFAULT_WORK_SIZE);
    let batch_size = cli
        .batch_size
        .or(file_config.batch_size)
        .unwrap_or(DEFAULT_BATCH_SIZE);
    let start_nonce = cli
        .start_nonce
        .or(file_config.start_nonce)
        .unwrap_or(DEFAULT_START_NONCE);
    let dry_run = cli.dry_run || file_config.dry_run.unwrap_or(false);
    let connect_timeout = cli
        .connect_timeout
        .or(file_config.connect_timeout)
        .unwrap_or(DEFAULT_CONNECT_TIMEOUT);
    let reconnect_delay = cli
        .reconnect_delay
        .or(file_config.reconnect_delay)
        .unwrap_or(DEFAULT_RECONNECT_DELAY);
    let ui = cli
        .ui
        .or(file_config.ui)
        .map(|ui| UiMode::parse(&ui))
        .transpose()?
        .unwrap_or(DEFAULT_UI_MODE);
    let target_mhs = cli.target_mhs.or(file_config.target_mhs);
    if let Some(target_mhs) = target_mhs {
        if !target_mhs.is_finite() || target_mhs <= 0.0 {
            bail!("target-mhs must be a positive finite number");
        }
    }
    let batch_sleep = Duration::from_millis(
        cli.batch_sleep_ms
            .or(file_config.batch_sleep_ms)
            .unwrap_or(0),
    );

    let batch_step =
        u32::try_from(batch_size).context("batch-size must fit in the 32-bit LBRY nonce space")?;
    let search_config = opencl::SearchConfig {
        platform_index: platform,
        device_index: device,
        work_size,
        batch_size,
    };

    Ok(MineOptions {
        backend,
        url,
        wallet,
        worker,
        seconds,
        config: search_config,
        hip_arch,
        batch_step,
        start_nonce,
        dry_run,
        ui,
        target_mhs,
        batch_sleep,
        connect_timeout: Duration::from_secs(connect_timeout),
        reconnect_delay: Duration::from_secs(reconnect_delay),
    })
}

#[derive(Debug)]
struct MineOptions {
    backend: Backend,
    url: String,
    wallet: String,
    worker: String,
    seconds: u64,
    config: opencl::SearchConfig,
    hip_arch: Option<String>,
    batch_step: u32,
    start_nonce: u32,
    dry_run: bool,
    ui: UiMode,
    target_mhs: Option<f64>,
    batch_sleep: Duration,
    connect_timeout: Duration,
    reconnect_delay: Duration,
}

#[derive(Debug, Default)]
struct MineStats {
    hashes: u64,
    candidates_seen: u64,
    valid_seen: u64,
    submitted: u64,
    accepted: u64,
    rejected: u64,
    reconnects: u64,
    new_jobs: u64,
    new_blocks: u64,
    stale_candidates: u64,
    last_scan_mhs: f64,
    best_scan_mhs: f64,
    last_batch_ms: f64,
    last_throttle_ms: f64,
    total_throttle_ms: f64,
    last_share_diff: Option<f64>,
    best_share_diff: f64,
    last_share_nonce: Option<u32>,
    last_job: Option<String>,
}

struct MinerUi {
    mode: UiMode,
    events: VecDeque<String>,
    last_render: Instant,
}

impl MinerUi {
    fn new(mode: UiMode) -> Self {
        Self {
            mode,
            events: VecDeque::with_capacity(EVENT_LOG_LIMIT),
            last_render: Instant::now(),
        }
    }

    fn is_log(&self) -> bool {
        self.mode == UiMode::Log
    }

    fn event(&mut self, message: impl Into<String>) {
        let message = message.into();
        if self.is_log() {
            println!("{message}");
            return;
        }

        if self.events.len() == EVENT_LOG_LIMIT {
            self.events.pop_front();
        }
        self.events.push_back(message);
    }

    fn maybe_render(
        &mut self,
        options: &MineOptions,
        stats: &MineStats,
        started: Instant,
        difficulty: f64,
        next_nonce: u32,
    ) {
        if self.mode.is_tui() && self.last_render.elapsed() >= Duration::from_secs(1) {
            self.render(options, stats, started, difficulty, next_nonce);
        }
    }

    fn render(
        &mut self,
        options: &MineOptions,
        stats: &MineStats,
        started: Instant,
        difficulty: f64,
        next_nonce: u32,
    ) {
        if !self.mode.is_tui() {
            return;
        }

        let elapsed = started.elapsed().as_secs_f64().max(0.001);
        let mhs = stats.hashes as f64 / elapsed / 1_000_000.0;
        let accept_rate = if stats.submitted == 0 {
            0.0
        } else {
            stats.accepted as f64 * 100.0 / stats.submitted as f64
        };
        let share_interval = expected_share_interval_secs(mhs, difficulty);
        let suggested_30s = suggested_pool_difficulty(mhs, 30.0);
        let suggested_60s = suggested_pool_difficulty(mhs, 60.0);
        let shares_per_hour = share_interval.map_or(0.0, |seconds| 3600.0 / seconds);

        print!("\x1b[2J\x1b[H");
        println!("LBRY Rust Miner");
        println!("================");
        println!(
            "Backend: {:<6} Pool: {}",
            options.backend.as_str(),
            options.url
        );
        println!(
            "Worker: {}  Runtime: {}",
            options.worker,
            format_duration(started.elapsed())
        );
        println!();
        println!(
            "[Speed] avg={:.3} MH/s  batch={:.3} MH/s  best_batch={:.3} MH/s",
            mhs, stats.last_scan_mhs, stats.best_scan_mhs
        );
        println!(
            "        batch_ms={:.2} throttle_ms={:.2} total_throttle={} target={}",
            stats.last_batch_ms,
            stats.last_throttle_ms,
            format_duration(Duration::from_millis(stats.total_throttle_ms as u64)),
            format_optional_mhs(options.target_mhs)
        );
        println!(
            "[Shares] submitted={} accepted={} rejected={} accept={:.1}%  rate={:.2}/hr",
            stats.submitted, stats.accepted, stats.rejected, accept_rate, shares_per_hour
        );
        println!(
            "         last_diff={} best_diff={} last_nonce={}",
            format_optional_difficulty(stats.last_share_diff),
            format_difficulty(stats.best_share_diff),
            stats
                .last_share_nonce
                .map(|nonce| format!("{nonce}"))
                .unwrap_or_else(|| "<none>".to_owned())
        );
        println!(
            "[Diff] pool={}  expected_share={}  suggested: 30s={} 60s={}",
            difficulty,
            format_optional_seconds(share_interval),
            format_difficulty(suggested_30s),
            format_difficulty(suggested_60s)
        );
        println!(
            "[Work] jobs={} blocks={} stale_candidates={} last_job={}",
            stats.new_jobs,
            stats.new_blocks,
            stats.stale_candidates,
            stats.last_job.as_deref().unwrap_or("<none>")
        );
        println!(
            "[GPU] batch={} work_size={} next_nonce={} dry_run={} reconnects={} fixed_sleep={}ms",
            options.config.batch_size,
            options.config.work_size,
            next_nonce,
            options.dry_run,
            stats.reconnects,
            options.batch_sleep.as_millis()
        );
        println!(
            "[Workload] hashes={} candidates={} valid={}",
            stats.hashes, stats.candidates_seen, stats.valid_seen
        );
        println!();
        println!("Events:");
        for event in &self.events {
            println!("  {event}");
        }
        println!();
        println!("Press Ctrl+C to stop.");
        let _ = io::stdout().flush();
        self.last_render = Instant::now();
    }
}

enum MineStepError {
    Reconnect(anyhow::Error),
    Fatal(anyhow::Error),
}

enum SearchBackend {
    Opencl(opencl::Scanner),
    Hip(hip::Scanner),
}

impl SearchBackend {
    fn new(
        backend: Backend,
        config: &opencl::SearchConfig,
        hip_arch: Option<&str>,
    ) -> Result<Self> {
        match backend {
            Backend::Opencl => Ok(Self::Opencl(opencl::Scanner::new(config)?)),
            Backend::Hip => Ok(Self::Hip(hip::Scanner::new(config, hip_arch)?)),
        }
    }

    fn search_batch(
        &mut self,
        header: &[u8; pow::LBRY_HEADER_LEN],
        start_nonce: u32,
        target_tail: u64,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Opencl(scanner) => scanner.search_batch(header, start_nonce, target_tail),
            Self::Hip(scanner) => scanner.search_batch(header, start_nonce, target_tail),
        }
    }
}

fn run_miner(options: MineOptions) -> Result<()> {
    let mut scanner = SearchBackend::new(
        options.backend,
        &options.config,
        options.hip_arch.as_deref(),
    )?;
    let started = Instant::now();
    let deadline = if options.seconds == 0 {
        None
    } else {
        Some(started + Duration::from_secs(options.seconds))
    };
    let mut last_report = started;
    let mut next_nonce = options.start_nonce;
    let mut stats = MineStats::default();
    let mut session = None;
    let mut ui = MinerUi::new(options.ui);
    let mut current_difficulty = 0.0;

    ui.event(format!(
        "mining started: backend={} batch_size={} work_size={} hip_arch={} dry_run={} reconnect_delay={}s ui={} target={} fixed_sleep={}ms",
        options.backend.as_str(),
        options.config.batch_size,
        options.config.work_size,
        options.hip_arch.as_deref().unwrap_or("<unset>"),
        options.dry_run,
        options.reconnect_delay.as_secs(),
        options.ui.as_str(),
        format_optional_mhs(options.target_mhs),
        options.batch_sleep.as_millis()
    ));
    ui.render(&options, &stats, started, current_difficulty, next_nonce);

    loop {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }

        if session.is_none() {
            match connect_mining_session(&options, ui.is_log()) {
                Ok(new_session) => {
                    next_nonce = options.start_nonce;
                    current_difficulty = new_session.difficulty();
                    if let Ok(work) = new_session.work() {
                        stats.new_jobs += 1;
                        stats.last_job = Some(work.job_id.clone());
                        ui.event(format!(
                            "initial work: job={} nonce reset to {next_nonce}",
                            work.job_id
                        ));
                    }
                    session = Some(new_session);
                    if !ui.is_log() {
                        ui.event(format!("connected to {}", options.url));
                    }
                    if stats.reconnects > 0 {
                        ui.event(format!(
                            "reconnected to pool; nonce reset to {} (reconnects={})",
                            next_nonce, stats.reconnects
                        ));
                    }
                }
                Err(err) => {
                    stats.reconnects += 1;
                    ui.event(format!(
                        "connect failed: {err:#}; reconnecting in {}s",
                        options.reconnect_delay.as_secs()
                    ));
                    if !sleep_until_retry_or_deadline(options.reconnect_delay, deadline) {
                        break;
                    }
                    continue;
                }
            }
        }

        let step = {
            let active_session = session
                .as_mut()
                .expect("session is established before mining step");
            mine_one_step(
                active_session,
                &mut scanner,
                &options,
                &mut stats,
                &mut next_nonce,
                &mut ui,
            )
        };

        match step {
            Ok(difficulty) => {
                current_difficulty = difficulty;
                if last_report.elapsed() >= Duration::from_secs(5) {
                    if ui.is_log() {
                        print_stats(&stats, started, difficulty);
                    }
                    last_report = Instant::now();
                }
                ui.maybe_render(&options, &stats, started, current_difficulty, next_nonce);
            }
            Err(MineStepError::Reconnect(err)) => {
                stats.reconnects += 1;
                session = None;
                ui.event(format!(
                    "stratum session lost: {err:#}; reconnecting in {}s",
                    options.reconnect_delay.as_secs()
                ));
                if !sleep_until_retry_or_deadline(options.reconnect_delay, deadline) {
                    break;
                }
            }
            Err(MineStepError::Fatal(err)) => return Err(err),
        }
    }

    if let Some(active_session) = session.as_mut() {
        let drain_until = Instant::now() + Duration::from_secs(2);
        while Instant::now() < drain_until {
            match active_session.poll_updates(Duration::from_millis(200), false) {
                Ok(update) => {
                    apply_update(
                        update,
                        &mut stats,
                        &mut next_nonce,
                        options.start_nonce,
                        &mut ui,
                    );
                }
                Err(err) => {
                    ui.event(format!("final reply drain stopped: {err:#}"));
                    break;
                }
            }
        }
    }

    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    ui.event(format!(
        "mining stopped: {:.3} MH/s hashes={} candidates={} valid={} submitted={} accepted={} rejected={} reconnects={}",
        stats.hashes as f64 / elapsed / 1_000_000.0,
        stats.hashes,
        stats.candidates_seen,
        stats.valid_seen,
        stats.submitted,
        stats.accepted,
        stats.rejected,
        stats.reconnects
    ));
    ui.render(&options, &stats, started, current_difficulty, next_nonce);

    Ok(())
}

fn connect_mining_session(options: &MineOptions, verbose: bool) -> Result<stratum::MiningSession> {
    stratum::Client::connect(&options.url)?
        .mining_session(
            &options.wallet,
            &options.worker,
            options.connect_timeout,
            verbose,
        )
        .context("failed to establish Stratum mining session")
}

fn mine_one_step(
    session: &mut stratum::MiningSession,
    scanner: &mut SearchBackend,
    options: &MineOptions,
    stats: &mut MineStats,
    next_nonce: &mut u32,
    ui: &mut MinerUi,
) -> std::result::Result<f64, MineStepError> {
    let update = session
        .poll_updates(Duration::ZERO, false)
        .map_err(MineStepError::Reconnect)?;
    apply_update(update, stats, next_nonce, options.start_nonce, ui);

    let work = session.work().map_err(MineStepError::Reconnect)?.clone();
    let difficulty = session.difficulty();
    let target = pow::lbry_share_target(difficulty).map_err(MineStepError::Fatal)?;
    let target_tail = pow::target_tail64(&target);
    let batch_start = *next_nonce;
    let scan_started = Instant::now();
    let candidates = scanner
        .search_batch(&work.header, batch_start, target_tail)
        .map_err(MineStepError::Fatal)?;
    let scan_elapsed = scan_started.elapsed().as_secs_f64().max(0.000_001);
    stats.last_batch_ms = scan_elapsed * 1000.0;
    stats.last_scan_mhs = options.config.batch_size as f64 / scan_elapsed / 1_000_000.0;
    stats.best_scan_mhs = stats.best_scan_mhs.max(stats.last_scan_mhs);
    stats.last_throttle_ms = 0.0;
    stats.candidates_seen += candidates.len() as u64;
    stats.hashes += options.config.batch_size as u64;

    let post_scan_update = session
        .poll_updates(Duration::ZERO, false)
        .map_err(MineStepError::Reconnect)?;
    let job_changed = post_scan_update.new_work;
    apply_update(post_scan_update, stats, next_nonce, options.start_nonce, ui);
    if job_changed {
        if !candidates.is_empty() {
            stats.stale_candidates += candidates.len() as u64;
            ui.event(format!(
                "discarded {} candidate(s) from stale work",
                candidates.len()
            ));
        }
        return Ok(difficulty);
    }

    for nonce in &candidates {
        let header = work.with_nonce(*nonce);
        let hash = pow::lbry_work_hash(&header);
        if !pow::hash_meets_target(&hash, &target) {
            continue;
        }

        stats.valid_seen += 1;
        let share_diff = pow::lbry_hash_difficulty(&hash);
        stats.last_share_diff = Some(share_diff);
        stats.best_share_diff = stats.best_share_diff.max(share_diff);
        stats.last_share_nonce = Some(*nonce);
        if options.dry_run {
            ui.event(format!(
                "dry-run valid share nonce={} diff={} nonce_hex={} hash={}",
                nonce,
                format_difficulty(share_diff),
                stratum::StratumWork::submit_nonce_hex(*nonce),
                hex::encode(hash)
            ));
        } else {
            let id = session
                .submit_share(&work, *nonce)
                .map_err(MineStepError::Reconnect)?;
            stats.submitted += 1;
            ui.event(format!(
                "submitted share id={id} nonce={} diff={} nonce_hex={} hash={}",
                nonce,
                format_difficulty(share_diff),
                stratum::StratumWork::submit_nonce_hex(*nonce),
                hex::encode(hash)
            ));
        }
    }

    let previous_nonce = *next_nonce;
    *next_nonce = next_nonce.wrapping_add(options.batch_step);
    if *next_nonce < previous_nonce {
        session.roll_work().map_err(MineStepError::Fatal)?;
        *next_nonce = options.start_nonce;
        ui.event("nonce space wrapped; rolled extranonce2");
    }

    apply_batch_throttle(scan_elapsed, options, stats);

    Ok(difficulty)
}

fn apply_update(
    update: stratum::SessionUpdate,
    stats: &mut MineStats,
    next_nonce: &mut u32,
    start_nonce: u32,
    ui: &mut MinerUi,
) {
    stats.accepted += update.accepted;
    stats.rejected += update.rejected;
    for id in update.accepted_ids {
        ui.event(format!("share accepted: id={id}"));
    }
    for (id, error) in update.rejected_shares {
        ui.event(format!("share rejected: id={id} error={error}"));
    }
    if update.new_work {
        stats.new_jobs += 1;
        if update.new_block {
            stats.new_blocks += 1;
        }
        if let Some(job_id) = update.job_id {
            stats.last_job = Some(job_id.clone());
            if update.new_block {
                ui.event(format!(
                    "new block work: job={job_id} clean={} nonce reset to {start_nonce}",
                    update.clean
                ));
            } else {
                ui.event(format!(
                    "new work: job={job_id} clean={} nonce reset to {start_nonce}",
                    update.clean
                ));
            }
        } else {
            ui.event(format!("new work received; nonce reset to {start_nonce}"));
        }
        *next_nonce = start_nonce;
    }
}

fn print_stats(stats: &MineStats, started: Instant, difficulty: f64) {
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    println!(
        "stats: {:.3} MH/s batch={:.3} MH/s throttle_ms={:.2} hashes={} candidates={} valid={} submitted={} accepted={} rejected={} reconnects={} jobs={} blocks={} stale={} diff={}",
        stats.hashes as f64 / elapsed / 1_000_000.0,
        stats.last_scan_mhs,
        stats.last_throttle_ms,
        stats.hashes,
        stats.candidates_seen,
        stats.valid_seen,
        stats.submitted,
        stats.accepted,
        stats.rejected,
        stats.reconnects,
        stats.new_jobs,
        stats.new_blocks,
        stats.stale_candidates,
        difficulty
    );
}

fn apply_batch_throttle(scan_elapsed: f64, options: &MineOptions, stats: &mut MineStats) {
    let mut sleep_for = options.batch_sleep;
    if let Some(target_mhs) = options.target_mhs {
        let desired = options.config.batch_size as f64 / (target_mhs * 1_000_000.0);
        if desired > scan_elapsed {
            sleep_for += Duration::from_secs_f64(desired - scan_elapsed);
        }
    }

    if sleep_for.is_zero() {
        return;
    }

    stats.last_throttle_ms = sleep_for.as_secs_f64() * 1000.0;
    stats.total_throttle_ms += stats.last_throttle_ms;
    sleep(sleep_for);
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn expected_share_interval_secs(mhs: f64, difficulty: f64) -> Option<f64> {
    if mhs <= 0.0 || !mhs.is_finite() || difficulty <= 0.0 || !difficulty.is_finite() {
        return None;
    }

    Some(difficulty * HASHES_PER_DIFF_ONE_SHARE / (mhs * 1_000_000.0))
}

fn suggested_pool_difficulty(mhs: f64, target_seconds: f64) -> f64 {
    if mhs <= 0.0 || !mhs.is_finite() || target_seconds <= 0.0 || !target_seconds.is_finite() {
        return 0.0;
    }

    mhs * 1_000_000.0 * target_seconds / HASHES_PER_DIFF_ONE_SHARE
}

fn format_optional_seconds(seconds: Option<f64>) -> String {
    let Some(seconds) = seconds else {
        return "<warming>".to_owned();
    };

    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{:.1}m", seconds / 60.0)
    }
}

fn format_optional_difficulty(difficulty: Option<f64>) -> String {
    difficulty
        .map(format_difficulty)
        .unwrap_or_else(|| "<none>".to_owned())
}

fn format_optional_mhs(mhs: Option<f64>) -> String {
    mhs.map(|mhs| format!("{mhs:.1} MH/s"))
        .unwrap_or_else(|| "<none>".to_owned())
}

fn format_difficulty(difficulty: f64) -> String {
    if !difficulty.is_finite() || difficulty <= 0.0 {
        return "<none>".to_owned();
    }

    if difficulty >= 1000.0 {
        format!("{difficulty:.0}")
    } else if difficulty >= 10.0 {
        format!("{difficulty:.1}")
    } else {
        format!("{difficulty:.3}")
    }
}

fn sleep_until_retry_or_deadline(delay: Duration, deadline: Option<Instant>) -> bool {
    let sleep_for = if let Some(deadline) = deadline {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        let remaining = deadline.duration_since(now);
        if delay < remaining { delay } else { remaining }
    } else {
        delay
    };

    sleep(sleep_for);
    deadline.is_none_or(|deadline| Instant::now() < deadline)
}
