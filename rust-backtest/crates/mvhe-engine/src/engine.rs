use serde::Serialize;

use mvhe_core::signal::{ExitReason, Side};
use mvhe_core::{CandleStore, Window};
use mvhe_strategy::traits::Strategy;

use crate::fill_sim::FillSimulator;
use crate::metrics::MetricsCalculator;

/// A single completed trade.
#[derive(Debug, Clone, Serialize)]
pub struct Trade {
    pub strategy_id: String,
    pub direction: Side,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub entry_ts: i64,
    pub exit_ts: i64,
    pub exit_reason: ExitReason,
    pub fee: f64,
    pub slippage: f64,
}

/// Full backtest result for a single strategy.
#[derive(Debug, Clone, Serialize)]
pub struct BacktestResult {
    pub strategy_id: String,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub metrics: crate::metrics::Metrics,
}

/// Event-driven backtesting engine for binary prediction markets.
///
/// Iterates through pre-computed 15-minute windows:
///   1. Evaluates strategy on window candles
///   2. Simulates fill with slippage + Polymarket dynamic fee
///   3. Settles at window close: winning side -> $1, losing -> $0
///   4. Tracks equity curve and all trades
pub struct BacktestEngine {
    initial_balance: f64,
    max_position_pct: f64,
    kelly_multiplier: f64,
    fill_sim: FillSimulator,
    book_depth: f64,
}

impl BacktestEngine {
    pub fn new(
        initial_balance: f64,
        max_position_pct: f64,
        kelly_multiplier: f64,
        fill_sim: FillSimulator,
    ) -> Self {
        Self {
            initial_balance,
            max_position_pct,
            kelly_multiplier,
            fill_sim,
            // Default book depth for backtesting (no real orderbook)
            book_depth: 100_000.0,
        }
    }

    /// Run a full backtest of a strategy across all windows.
    pub fn run(
        &self,
        candles: &CandleStore,
        windows: &[Window],
        strategy: &dyn Strategy,
    ) -> BacktestResult {
        let mut balance = self.initial_balance;
        let mut trades = Vec::with_capacity(windows.len() / 2);
        let mut equity_curve = Vec::with_capacity(windows.len() + 1);
        equity_curve.push(balance);

        for window in windows {
            // Evaluate strategy
            let signal = match strategy.evaluate_window(candles, window) {
                Some(s) => s,
                None => {
                    equity_curve.push(balance);
                    continue;
                }
            };

            // Position sizing: quarter-Kelly for binary markets
            // f* = (win_prob - entry_price) / (entry_price * (1 - entry_price))
            let p = signal.entry_price.clamp(0.01, 0.99);
            let win_prob = strategy.estimated_win_prob();
            let kelly_raw = (win_prob - p) / (p * (1.0 - p));
            let kelly_fraction = (kelly_raw * self.kelly_multiplier).max(0.0);
            let position_pct = kelly_fraction.min(self.max_position_pct);

            let size = balance * position_pct;
            if size <= 0.0 {
                equity_curve.push(balance);
                continue;
            }

            // Simulate fill
            let fill = self.fill_sim.simulate_fill(p, size, self.book_depth);

            if fill.net_cost > balance {
                equity_curve.push(balance);
                continue;
            }

            // Deduct cost
            balance -= fill.net_cost;

            // Binary settlement: determine if trade won
            let won = match signal.direction {
                Side::Yes => window.settled_green(),
                Side::No => !window.settled_green(),
            };

            // Settlement payoff: winning side -> $1 per unit, losing -> $0
            let exit_price = if won { 1.0 } else { 0.0 };
            let settlement_value = size * exit_price;
            balance += settlement_value;

            let pnl = settlement_value - fill.net_cost;

            trades.push(Trade {
                strategy_id: strategy.id().to_string(),
                direction: signal.direction,
                entry_price: fill.fill_price,
                exit_price,
                size,
                pnl,
                entry_ts: window.start_ts,
                exit_ts: window.end_ts,
                exit_reason: ExitReason::ResolutionGuard,
                fee: fill.fee,
                slippage: fill.slippage,
            });

            equity_curve.push(balance);
        }

        let metrics = MetricsCalculator::calculate(&equity_curve, &trades);

        BacktestResult {
            strategy_id: strategy.id().to_string(),
            trades,
            equity_curve,
            metrics,
        }
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(10_000.0, 0.02, 0.25, FillSimulator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvhe_core::config::MomentumConfig;
    use mvhe_core::window::compute_windows;
    use mvhe_strategy::momentum::MomentumConfirmation;

    fn make_trending_candles(n: usize) -> CandleStore {
        let mut store = CandleStore::with_capacity(n);
        let base_ts: i64 = 1735689600;
        for i in 0..n {
            let ts = base_ts + (i as i64) * 60;
            let price = 100.0 + (i as f64) * 0.02;
            store.push(ts, price, price + 0.01, price - 0.01, price + 0.015, 1000.0);
        }
        store
    }

    #[test]
    fn test_engine_runs_without_panic() {
        let candles = make_trending_candles(60);
        let windows = compute_windows(&candles);
        let config = MomentumConfig::default();
        let strategy = MomentumConfirmation::new(&config);
        let engine = BacktestEngine::default();

        let result = engine.run(&candles, &windows, &strategy);

        assert!(!result.equity_curve.is_empty());
        assert_eq!(result.strategy_id, "momentum_confirmation");
    }

    #[test]
    fn test_engine_equity_curve_starts_at_initial_balance() {
        let candles = make_trending_candles(15);
        let windows = compute_windows(&candles);
        let config = MomentumConfig::default();
        let strategy = MomentumConfirmation::new(&config);
        let engine = BacktestEngine::default();

        let result = engine.run(&candles, &windows, &strategy);
        assert!((result.equity_curve[0] - 10_000.0).abs() < 1e-10);
    }
}
