use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mvhe_core::config::{MomentumConfig, FalseSentimentConfig, SingularityConfig};
use mvhe_core::window::compute_windows;
use mvhe_core::CandleStore;
use mvhe_engine::{BacktestEngine, FillSimulator, StatisticalValidator};
use mvhe_engine::parallel::ParallelRunner;
use mvhe_strategy::false_sentiment::FalseSentiment;
use mvhe_strategy::momentum::MomentumConfirmation;
use mvhe_strategy::singularity::Singularity;
use mvhe_strategy::traits::Strategy;

fn make_candles(n: usize) -> CandleStore {
    let mut store = CandleStore::with_capacity(n);
    let base_ts: i64 = 1735689600;
    for i in 0..n {
        let ts = base_ts + (i as i64) * 60;
        // Simulate trending price with some noise
        let trend = (i as f64) * 0.001;
        let noise = ((i as f64) * 0.1).sin() * 0.05;
        let price = 100.0 + trend + noise;
        store.push(ts, price, price + 0.02, price - 0.02, price + 0.01, 1000.0 + (i as f64));
    }
    store
}

fn bench_single_strategy(c: &mut Criterion) {
    let candles = make_candles(10_000);
    let windows = compute_windows(&candles);
    let config = MomentumConfig::default();
    let strategy = MomentumConfirmation::new(&config);
    let engine = BacktestEngine::default();

    c.bench_function("single_strategy_10k", |b| {
        b.iter(|| {
            let result = engine.run(black_box(&candles), black_box(&windows), &strategy);
            black_box(result);
        });
    });
}

fn bench_all_strategies_parallel(c: &mut Criterion) {
    let candles = make_candles(10_000);
    let windows = compute_windows(&candles);

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(MomentumConfirmation::new(&MomentumConfig::default())),
        Box::new(FalseSentiment::new(&FalseSentimentConfig::default())),
        Box::new(Singularity::new(&SingularityConfig::default())),
    ];

    let runner = ParallelRunner::new(
        BacktestEngine::default(),
        StatisticalValidator::new(100, 100, 0.05, 42),
    );

    c.bench_function("all_strategies_parallel_10k", |b| {
        b.iter(|| {
            let results =
                runner.run_all(black_box(&candles), black_box(&windows), black_box(&strategies));
            black_box(results);
        });
    });
}

fn bench_compute_windows(c: &mut Criterion) {
    let candles = make_candles(100_000);

    c.bench_function("compute_windows_100k", |b| {
        b.iter(|| {
            let windows = compute_windows(black_box(&candles));
            black_box(windows);
        });
    });
}

criterion_group!(
    benches,
    bench_single_strategy,
    bench_all_strategies_parallel,
    bench_compute_windows,
);
criterion_main!(benches);
