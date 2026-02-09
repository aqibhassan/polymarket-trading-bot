use rayon::prelude::*;
use serde::Serialize;

use mvhe_core::{CandleStore, Window};
use mvhe_strategy::traits::Strategy;

use crate::engine::{BacktestEngine, BacktestResult};
use crate::validator::{StatisticalValidator, ValidationResult};

/// Full result: backtest + statistical validation for a single strategy.
#[derive(Debug, Clone, Serialize)]
pub struct FullResult {
    pub backtest: BacktestResult,
    pub validation: ValidationResult,
}

/// Orchestrate parallel backtesting across multiple strategies.
///
/// Level 1: All strategies in parallel via `par_iter`
/// Level 2: Validation (bootstrap + permutation) already parallelized internally
pub struct ParallelRunner {
    engine: BacktestEngine,
    validator: StatisticalValidator,
}

impl ParallelRunner {
    pub fn new(engine: BacktestEngine, validator: StatisticalValidator) -> Self {
        Self { engine, validator }
    }

    /// Run all strategies in parallel and return full results.
    pub fn run_all(
        &self,
        candles: &CandleStore,
        windows: &[Window],
        strategies: &[Box<dyn Strategy>],
    ) -> Vec<FullResult> {
        strategies
            .par_iter()
            .map(|strategy| {
                let backtest = self.engine.run(candles, windows, strategy.as_ref());

                let trade_pnls: Vec<f64> = backtest.trades.iter().map(|t| t.pnl).collect();
                let validation = self.validator.validate(&backtest.equity_curve, &trade_pnls);

                FullResult {
                    backtest,
                    validation,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvhe_core::config::{MomentumConfig, FalseSentimentConfig, SingularityConfig};
    use mvhe_core::window::compute_windows;
    use mvhe_strategy::false_sentiment::FalseSentiment;
    use mvhe_strategy::momentum::MomentumConfirmation;
    use mvhe_strategy::singularity::Singularity;

    fn make_candles(n: usize) -> CandleStore {
        let mut store = CandleStore::with_capacity(n);
        let base_ts: i64 = 1735689600;
        for i in 0..n {
            let ts = base_ts + (i as i64) * 60;
            let price = 100.0 + (i as f64 % 30.0) * 0.02;
            store.push(ts, price, price + 0.01, price - 0.01, price + 0.005, 1000.0);
        }
        store
    }

    #[test]
    fn test_parallel_runner_all_strategies() {
        let candles = make_candles(150); // 10 windows
        let windows = compute_windows(&candles);

        let strategies: Vec<Box<dyn Strategy>> = vec![
            Box::new(MomentumConfirmation::new(&MomentumConfig::default())),
            Box::new(FalseSentiment::new(&FalseSentimentConfig::default())),
            Box::new(Singularity::new(&SingularityConfig::default())),
        ];

        let runner = ParallelRunner::new(
            BacktestEngine::default(),
            StatisticalValidator::new(50, 50, 0.05, 42),
        );

        let results = runner.run_all(&candles, &windows, &strategies);
        assert_eq!(results.len(), 3);

        for result in &results {
            assert!(!result.backtest.equity_curve.is_empty());
        }
    }
}
