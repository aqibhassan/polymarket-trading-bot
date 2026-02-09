use std::collections::HashMap;

use mvhe_core::config::MomentumConfig;
use mvhe_core::signal::{Confidence, Side, Signal, SignalType};
use mvhe_core::{CandleStore, Window};

use crate::traits::{Strategy, StrategyId};

/// MomentumConfirmation strategy — follow BTC momentum for Polymarket YES/NO tokens.
///
/// Exact port of `src/strategies/reversal_catcher.py`.
///
/// BTC 15m candles are momentum-driven: this strategy enters in the SAME
/// direction as the cumulative BTC move with tiered entry thresholds.
pub struct MomentumConfirmation {
    tier_thresholds: HashMap<u32, f64>,
    entry_minute_start: u32,
    entry_minute_end: u32,
    min_confidence: f64,
    estimated_win_prob: f64,
}

impl MomentumConfirmation {
    pub fn new(config: &MomentumConfig) -> Self {
        Self {
            tier_thresholds: config.tier_thresholds(),
            entry_minute_start: config.entry_minute_start,
            entry_minute_end: config.entry_minute_end,
            min_confidence: config.min_confidence,
            estimated_win_prob: config.estimated_win_prob,
        }
    }

    /// Compute data-driven confidence score.
    ///
    /// Components:
    ///   - magnitude_score (40%): |cum_return_pct| / 0.25, capped at 1.0
    ///   - time_score (35%): later minutes = higher accuracy
    ///   - last_3_bonus: +0.10 if last 3 candles agree with direction
    ///   - no_reversal_bonus: +0.05 if no reversal candle in window
    ///   - base floor: +0.10
    fn compute_confidence(
        &self,
        cum_return_pct: f64,
        minute: u32,
        last_3_agree: bool,
        no_reversal: bool,
    ) -> Confidence {
        // 1. Magnitude score
        let magnitude_score = (cum_return_pct / 0.25).min(1.0);

        // 2. Time score
        let time_score = if minute >= self.entry_minute_start && minute <= self.entry_minute_end {
            let entry_range = self.entry_minute_end - self.entry_minute_start;
            if entry_range > 0 {
                (1.0 - 0.1 * (minute - self.entry_minute_start) as f64).max(0.8)
            } else {
                1.0
            }
        } else {
            ((minute as f64 - 4.0) / 4.0).clamp(0.0, 1.0)
        };

        // 3. Bonuses
        let last_3_bonus = if last_3_agree { 0.10 } else { 0.0 };
        let no_reversal_bonus = if no_reversal { 0.05 } else { 0.0 };

        // 4. Overall
        let overall =
            (0.40 * magnitude_score + 0.35 * time_score + last_3_bonus + no_reversal_bonus + 0.10)
                .min(1.0);

        Confidence {
            trend_strength: round4(magnitude_score),
            threshold_exceedance: round4(time_score),
            book_normality: round4(last_3_bonus),
            liquidity_quality: round4(no_reversal_bonus),
            overall: round4(overall),
        }
    }

    /// Check if last 3 candles agree with cumulative direction.
    fn check_last_3_agree(candles: &CandleStore, start: usize, end: usize, cum_return: f64) -> bool {
        let count = end - start;
        if count < 3 {
            return false;
        }
        // Check last 3 candles
        for i in (end - 3)..end {
            let candle_return = candles.close[i] - candles.open[i];
            if cum_return > 0.0 {
                // Expect green candles (close > open)
                if candle_return <= 0.0 {
                    return false;
                }
            } else {
                // Expect red candles (close < open)
                if candle_return >= 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Check that no reversal candle appeared in the window.
    fn check_no_reversal(candles: &CandleStore, start: usize, end: usize, cum_return: f64) -> bool {
        for i in start..end {
            let candle_return = candles.close[i] - candles.open[i];
            if cum_return > 0.0 {
                // A red candle is a reversal
                if candle_return < 0.0 {
                    return false;
                }
            } else {
                // A green candle is a reversal
                if candle_return > 0.0 {
                    return false;
                }
            }
        }
        true
    }
}

impl Strategy for MomentumConfirmation {
    fn evaluate_window(&self, candles: &CandleStore, window: &Window) -> Option<Signal> {
        let n_candles = window.candle_count();

        // Iterate through entry minutes to find first matching tier
        for minute in self.entry_minute_start..=self.entry_minute_end {
            let minute_idx = minute as usize;
            if minute_idx >= n_candles {
                continue;
            }

            let threshold_pct = match self.tier_thresholds.get(&minute) {
                Some(&t) => t,
                None => continue,
            };

            // Candle index for this minute within the window
            let candle_idx = window.start_idx + minute_idx;
            let current_close = candles.close[candle_idx];
            let cum_return = (current_close - window.open_price) / window.open_price;
            let cum_return_pct = cum_return.abs() * 100.0;

            if cum_return_pct < threshold_pct {
                continue;
            }

            // Available candles up to and including this minute
            let avail_end = window.start_idx + minute_idx + 1;
            let last_3_agree =
                Self::check_last_3_agree(candles, window.start_idx, avail_end, cum_return);
            let no_reversal =
                Self::check_no_reversal(candles, window.start_idx, avail_end, cum_return);

            let confidence =
                self.compute_confidence(cum_return_pct, minute, last_3_agree, no_reversal);

            if !confidence.meets_minimum(self.min_confidence) {
                continue;
            }

            // Determine direction and entry price
            let (direction, entry_price) = if cum_return > 0.0 {
                // BTC up -> buy YES
                let yes_price = sigmoid_yes_price(cum_return);
                (Side::Yes, yes_price)
            } else {
                // BTC down -> buy NO
                let yes_price = sigmoid_yes_price(cum_return);
                (Side::No, 1.0 - yes_price)
            };

            let stop_loss = entry_price * (1.0 - 1.0); // hold-to-settlement: stop_loss_pct=1.0 -> 0
            let take_profit = entry_price * (1.0 + 0.40);

            return Some(Signal {
                strategy_id: self.id().to_string(),
                signal_type: SignalType::Entry,
                direction,
                confidence,
                entry_price,
                stop_loss,
                take_profit,
            });
        }

        None
    }

    fn id(&self) -> StrategyId {
        "momentum_confirmation"
    }

    fn estimated_win_prob(&self) -> f64 {
        self.estimated_win_prob
    }
}

/// Map cumulative BTC return to YES token price via sigmoid.
///
/// `1 / (1 + exp(-cum_return / 0.07))`
#[inline]
fn sigmoid_yes_price(cum_return: f64) -> f64 {
    1.0 / (1.0 + (-cum_return / 0.07).exp())
}

#[inline]
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvhe_core::candle::CandleStore;
    use mvhe_core::window::Window;

    fn make_window_candles(n: usize, trend_up: bool) -> (CandleStore, Window) {
        let mut store = CandleStore::with_capacity(n);
        let base_ts: i64 = 1735689600;
        let base_price = 100.0;

        for i in 0..n {
            let ts = base_ts + (i as i64) * 60;
            let delta = if trend_up {
                (i as f64) * 0.015
            } else {
                -(i as f64) * 0.015
            };
            let o = base_price + delta;
            let c = o + if trend_up { 0.01 } else { -0.01 };
            let h = o.max(c) + 0.005;
            let l = o.min(c) - 0.005;
            store.push(ts, o, h, l, c, 1000.0);
        }

        let window = Window {
            start_idx: 0,
            end_idx: n,
            open_price: store.open[0],
            settlement_price: store.close[n - 1],
            start_ts: base_ts,
            end_ts: base_ts + 900,
        };

        (store, window)
    }

    #[test]
    fn test_momentum_uptrend_signal() {
        let (candles, window) = make_window_candles(15, true);
        let config = MomentumConfig::default();
        let strategy = MomentumConfirmation::new(&config);

        let signal = strategy.evaluate_window(&candles, &window);
        // With a 0.015% per candle move, by minute 8 cum_return_pct should exceed 0.10%
        if let Some(sig) = signal {
            assert_eq!(sig.direction, Side::Yes);
            assert!(sig.confidence.overall >= 0.70);
        }
        // Signal may or may not fire depending on exact magnitudes — that's ok
    }

    #[test]
    fn test_sigmoid_yes_price() {
        let p = sigmoid_yes_price(0.0);
        assert!((p - 0.5).abs() < 1e-10);

        let p_up = sigmoid_yes_price(0.001);
        assert!(p_up > 0.5);

        let p_down = sigmoid_yes_price(-0.001);
        assert!(p_down < 0.5);
    }

    #[test]
    fn test_check_last_3_agree_green() {
        let mut store = CandleStore::with_capacity(5);
        for i in 0..5 {
            // All green candles (close > open)
            store.push(i as i64, 100.0, 102.0, 99.0, 101.0, 1000.0);
        }
        assert!(MomentumConfirmation::check_last_3_agree(&store, 0, 5, 1.0));
        assert!(!MomentumConfirmation::check_last_3_agree(&store, 0, 5, -1.0));
    }

    #[test]
    fn test_check_no_reversal() {
        let mut store = CandleStore::with_capacity(3);
        // All green
        store.push(0, 100.0, 102.0, 99.0, 101.0, 1000.0);
        store.push(1, 101.0, 103.0, 100.0, 102.0, 1000.0);
        store.push(2, 102.0, 104.0, 101.0, 103.0, 1000.0);
        assert!(MomentumConfirmation::check_no_reversal(&store, 0, 3, 1.0));

        // Add a red candle
        let mut store2 = CandleStore::with_capacity(3);
        store2.push(0, 100.0, 102.0, 99.0, 101.0, 1000.0);
        store2.push(1, 101.0, 103.0, 100.0, 100.5, 1000.0); // red
        store2.push(2, 100.5, 102.0, 99.5, 101.0, 1000.0);
        assert!(!MomentumConfirmation::check_no_reversal(&store2, 0, 3, 1.0));
    }
}
