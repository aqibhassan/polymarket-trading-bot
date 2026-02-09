use mvhe_core::{CandleStore, Signal, Window};

/// Unique strategy identifier.
pub type StrategyId = &'static str;

/// Pure-function strategy interface for backtesting.
///
/// Strategies evaluate a 15-minute window of 1-minute candles and optionally
/// produce a trading signal. All strategies must be Send + Sync for Rayon
/// parallelism.
pub trait Strategy: Send + Sync {
    /// Evaluate a single 15-minute window and optionally produce a signal.
    ///
    /// Arguments:
    /// - `candles`: Full candle store (use window indices to access relevant candles)
    /// - `window`: The pre-computed 15-minute window being evaluated
    ///
    /// Returns `Some(signal)` if entry conditions are met, `None` otherwise.
    fn evaluate_window(&self, candles: &CandleStore, window: &Window) -> Option<Signal>;

    /// Return the strategy's unique identifier.
    fn id(&self) -> StrategyId;

    /// Return the strategy's estimated win probability (from config/backtest validation).
    ///
    /// Used by the engine for Kelly position sizing: `f* = (win_prob - p) / (p * (1-p))`.
    fn estimated_win_prob(&self) -> f64;
}
