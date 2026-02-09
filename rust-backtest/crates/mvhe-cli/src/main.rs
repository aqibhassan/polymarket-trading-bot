use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use mvhe_core::config::BacktestConfig;
use mvhe_core::window::compute_windows;
use mvhe_core::CandleStore;
use mvhe_engine::parallel::{FullResult, ParallelRunner};
use mvhe_engine::{BacktestEngine, FillSimulator, StatisticalValidator};
use mvhe_strategy::false_sentiment::FalseSentiment;
use mvhe_strategy::momentum::MomentumConfirmation;
use mvhe_strategy::singularity::Singularity;
use mvhe_strategy::traits::Strategy;

#[derive(Parser, Debug)]
#[command(name = "mvhe-backtest", about = "MVHE Rust backtesting engine")]
struct Cli {
    /// Path to CSV candle data file
    #[arg(long)]
    candles: PathBuf,

    /// Path to TOML config file(s), comma-separated for merge
    #[arg(long, default_value = "config/default.toml")]
    config: String,

    /// Strategies to run (comma-separated)
    #[arg(long, default_value = "momentum_confirmation,false_sentiment,singularity")]
    strategies: String,

    /// Output format
    #[arg(long, default_value = "json")]
    output: String,

    /// Output file path (stdout if not specified)
    #[arg(long)]
    output_file: Option<PathBuf>,

    /// Number of bootstrap resamples
    #[arg(long, default_value = "1000")]
    n_bootstrap: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Initial balance
    #[arg(long)]
    initial_balance: Option<f64>,
}

/// JSON output structure matching Python BacktestReporter format.
#[derive(Debug, Serialize)]
struct OutputReport {
    meta: OutputMeta,
    results: Vec<StrategyReport>,
}

#[derive(Debug, Serialize)]
struct OutputMeta {
    candle_file: String,
    total_candles: usize,
    total_windows: usize,
    strategies_run: Vec<String>,
    elapsed_ms: u128,
}

#[derive(Debug, Serialize)]
struct StrategyReport {
    strategy_id: String,
    backtest: BacktestSummary,
    validation: ValidationSummary,
}

#[derive(Debug, Serialize)]
struct BacktestSummary {
    total_trades: usize,
    total_pnl: f64,
    total_return_pct: f64,
    sharpe_ratio: f64,
    sortino_ratio: f64,
    max_drawdown_pct: f64,
    calmar_ratio: f64,
    win_rate: f64,
    profit_factor: f64,
    avg_win_loss_ratio: f64,
}

#[derive(Debug, Serialize)]
struct ValidationSummary {
    sharpe_ci_95_lower: f64,
    sharpe_ci_95_upper: f64,
    p_value: f64,
    is_significant: bool,
    overfitting_warning: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let start = Instant::now();

    // Load config
    let config_paths: Vec<PathBuf> = cli.config.split(',').map(PathBuf::from).collect();
    let config_refs: Vec<&std::path::Path> = config_paths.iter().map(|p| p.as_path()).collect();
    let config = match BacktestConfig::from_toml_files(&config_refs) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            std::process::exit(1);
        }
    };

    // Load candles
    eprintln!("Loading candles from {:?}...", cli.candles);
    let load_start = Instant::now();
    let candles = match CandleStore::from_csv(&cli.candles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading candles: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "Loaded {} candles in {:.1}ms",
        candles.len(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    // Compute 15-minute windows
    let windows = compute_windows(&candles);
    eprintln!("Computed {} 15-minute windows", windows.len());

    // Build strategies
    let strategy_names: Vec<&str> = cli.strategies.split(',').map(str::trim).collect();
    let strategies: Vec<Box<dyn Strategy>> = strategy_names
        .iter()
        .filter_map(|name| match *name {
            "momentum_confirmation" | "reversal_catcher" => {
                Some(Box::new(MomentumConfirmation::new(
                    &config.strategy.momentum_confirmation,
                )) as Box<dyn Strategy>)
            }
            "false_sentiment" => {
                Some(Box::new(FalseSentiment::new(
                    &config.strategy.false_sentiment,
                )) as Box<dyn Strategy>)
            }
            "singularity" => {
                Some(Box::new(Singularity::new(
                    &config.strategy.singularity,
                )) as Box<dyn Strategy>)
            }
            unknown => {
                eprintln!("Unknown strategy: {}", unknown);
                None
            }
        })
        .collect();

    if strategies.is_empty() {
        eprintln!("No valid strategies specified");
        std::process::exit(1);
    }

    // Build engine
    let initial_balance = cli
        .initial_balance
        .unwrap_or(config.paper.initial_balance);
    let fill_sim = FillSimulator::new(
        config.risk.fees.slippage_bps,
        config.risk.fees.fee_constant,
    );
    let engine = BacktestEngine::new(
        initial_balance,
        config.risk.max_position_pct,
        config.risk.kelly_multiplier,
        fill_sim,
    );
    let validator = StatisticalValidator::new(cli.n_bootstrap, cli.n_bootstrap, 0.05, cli.seed);

    // Run parallel backtest
    eprintln!(
        "Running {} strategies across {} windows...",
        strategies.len(),
        windows.len()
    );
    let run_start = Instant::now();
    let runner = ParallelRunner::new(engine, validator);
    let results = runner.run_all(&candles, &windows, &strategies);
    eprintln!(
        "Backtest complete in {:.1}ms",
        run_start.elapsed().as_secs_f64() * 1000.0
    );

    // Build output
    let elapsed = start.elapsed();
    let report = build_report(&cli, &candles, &windows, &strategy_names, &results, elapsed.as_millis());

    // Print human-readable summary to stderr
    print_summary(&report);

    // Output JSON
    let json = serde_json::to_string_pretty(&report).expect("JSON serialization failed");

    if let Some(output_path) = &cli.output_file {
        std::fs::write(output_path, &json).expect("Failed to write output file");
        eprintln!("Results written to {:?}", output_path);
    } else {
        println!("{}", json);
    }

    eprintln!("\nTotal elapsed: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
}

fn build_report(
    cli: &Cli,
    candles: &CandleStore,
    windows: &[mvhe_core::Window],
    strategy_names: &[&str],
    results: &[FullResult],
    elapsed_ms: u128,
) -> OutputReport {
    let strategy_reports: Vec<StrategyReport> = results
        .iter()
        .map(|r| StrategyReport {
            strategy_id: r.backtest.strategy_id.clone(),
            backtest: BacktestSummary {
                total_trades: r.backtest.metrics.total_trades,
                total_pnl: r.backtest.metrics.total_pnl,
                total_return_pct: r.backtest.metrics.total_return_pct,
                sharpe_ratio: r.backtest.metrics.sharpe_ratio,
                sortino_ratio: r.backtest.metrics.sortino_ratio,
                max_drawdown_pct: r.backtest.metrics.max_drawdown_pct,
                calmar_ratio: r.backtest.metrics.calmar_ratio,
                win_rate: r.backtest.metrics.win_rate,
                profit_factor: r.backtest.metrics.profit_factor,
                avg_win_loss_ratio: r.backtest.metrics.avg_win_loss_ratio,
            },
            validation: ValidationSummary {
                sharpe_ci_95_lower: r.validation.sharpe_ci_95.0,
                sharpe_ci_95_upper: r.validation.sharpe_ci_95.1,
                p_value: r.validation.p_value,
                is_significant: r.validation.is_significant,
                overfitting_warning: r.validation.overfitting_warning.clone(),
            },
        })
        .collect();

    OutputReport {
        meta: OutputMeta {
            candle_file: cli.candles.display().to_string(),
            total_candles: candles.len(),
            total_windows: windows.len(),
            strategies_run: strategy_names.iter().map(|s| s.to_string()).collect(),
            elapsed_ms,
        },
        results: strategy_reports,
    }
}

fn print_summary(report: &OutputReport) {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("MVHE Rust Backtest Results");
    eprintln!("{}", "=".repeat(80));
    eprintln!(
        "Candles: {} | Windows: {} | Elapsed: {}ms",
        report.meta.total_candles, report.meta.total_windows, report.meta.elapsed_ms
    );
    eprintln!("{}", "-".repeat(80));
    eprintln!(
        "{:<25} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Strategy", "Trades", "WinRate", "Sharpe", "MaxDD", "PnL", "PF"
    );
    eprintln!("{}", "-".repeat(80));

    for r in &report.results {
        eprintln!(
            "{:<25} {:>8} {:>7.1}% {:>8.2} {:>7.2}% {:>8.2} {:>8.2}",
            r.strategy_id,
            r.backtest.total_trades,
            r.backtest.win_rate * 100.0,
            r.backtest.sharpe_ratio,
            r.backtest.max_drawdown_pct * 100.0,
            r.backtest.total_pnl,
            r.backtest.profit_factor,
        );
    }

    eprintln!("{}", "-".repeat(80));
    eprintln!("Validation:");
    for r in &report.results {
        eprintln!(
            "  {}: Sharpe CI [{:.2}, {:.2}], p={:.4}, sig={}{}",
            r.strategy_id,
            r.validation.sharpe_ci_95_lower,
            r.validation.sharpe_ci_95_upper,
            r.validation.p_value,
            r.validation.is_significant,
            r.validation
                .overfitting_warning
                .as_ref()
                .map(|w| format!(" [WARN: {}]", w))
                .unwrap_or_default(),
        );
    }
    eprintln!("{}", "=".repeat(80));
}
