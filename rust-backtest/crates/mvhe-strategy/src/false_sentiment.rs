use mvhe_core::config::FalseSentimentConfig;
use mvhe_core::signal::{Confidence, Side, Signal, SignalType};
use mvhe_core::{CandleStore, Window};

use crate::traits::{Strategy, StrategyId};

/// FalseSentiment strategy — contrarian BTC 15m candle signal detection.
///
/// Port of `src/strategies/false_sentiment.py`.
///
/// Detects when BTC price action diverges from the dominant trend direction
/// and enters contrarian positions. Without live orderbook data, uses
/// conservative proxies for book normality and liquidity.
pub struct FalseSentiment {
    entry_threshold_base: f64,
    threshold_time_scaling: f64,
    lookback_candles: u32,
    min_confidence: f64,
    no_entry_after_minute: u32,
    #[allow(dead_code)]
    force_exit_minute: u32,
}

impl FalseSentiment {
    pub fn new(config: &FalseSentimentConfig) -> Self {
        Self {
            entry_threshold_base: config.entry_threshold_base,
            threshold_time_scaling: config.threshold_time_scaling,
            lookback_candles: config.lookback_candles,
            min_confidence: config.min_confidence,
            no_entry_after_minute: config.no_entry_after_minute,
            force_exit_minute: config.force_exit_minute,
        }
    }

    /// Dynamic entry threshold: base + (scaling * minute / 15)
    fn dynamic_threshold(&self, minute: u32) -> f64 {
        self.entry_threshold_base + (self.threshold_time_scaling * minute as f64 / 15.0)
    }

    /// Analyze BTC trend from candles.
    ///
    /// Returns (trend_strength, dominant_direction):
    /// - trend_strength: 0-1, how strong the trend is
    /// - dominant_direction: "up" or "down"
    fn analyze_trend(
        candles: &CandleStore,
        start: usize,
        end: usize,
        lookback: usize,
    ) -> (f64, bool) {
        let lookback_start = if end > lookback { end - lookback } else { start };
        let actual_lookback = end - lookback_start;
        if actual_lookback < 2 {
            return (0.0, true);
        }

        let first_close = candles.close[lookback_start];
        let last_close = candles.close[end - 1];

        if first_close == 0.0 {
            return (0.0, true);
        }

        let cum_return = (last_close - first_close) / first_close;
        let is_up = cum_return > 0.0;

        // Count candles in the dominant direction
        let mut agree_count = 0u32;
        for i in lookback_start..end {
            let candle_dir = candles.close[i] > candles.open[i];
            if candle_dir == is_up {
                agree_count += 1;
            }
        }

        let trend_strength = agree_count as f64 / actual_lookback as f64;
        (trend_strength, is_up)
    }

    /// Compute confidence for false sentiment detection.
    ///
    /// Components (matching Python):
    ///   - trend_strength (35%): strength of the detected trend
    ///   - book_normality (25%): proxy=0.5 without live orderbook
    ///   - liquidity_quality (20%): proxy=0.8 without live data
    ///   - threshold_exceedance (20%): how much the price exceeds the threshold
    fn compute_confidence(
        trend_strength: f64,
        threshold_exceedance: f64,
    ) -> Confidence {
        // Without live orderbook data, use conservative proxies
        let book_normality = 0.5;
        let liquidity_quality = 0.8;

        let overall = (0.35 * trend_strength
            + 0.25 * book_normality
            + 0.20 * liquidity_quality
            + 0.20 * threshold_exceedance.min(1.0))
            .min(1.0);

        Confidence {
            trend_strength: round4(trend_strength),
            threshold_exceedance: round4(threshold_exceedance.min(1.0)),
            book_normality: round4(book_normality),
            liquidity_quality: round4(liquidity_quality),
            overall: round4(overall),
        }
    }
}

impl Strategy for FalseSentiment {
    fn evaluate_window(&self, candles: &CandleStore, window: &Window) -> Option<Signal> {
        let n_candles = window.candle_count();

        // Evaluate at each minute within the entry window
        for minute in 0..=self.no_entry_after_minute {
            let minute_idx = minute as usize;
            if minute_idx >= n_candles {
                break;
            }

            let avail_end = window.start_idx + minute_idx + 1;

            // Dynamic threshold check
            let threshold = self.dynamic_threshold(minute);

            // Analyze trend from available candles
            let (trend_strength, is_up) = Self::analyze_trend(
                candles,
                window.start_idx,
                avail_end,
                self.lookback_candles as usize,
            );

            // Check if price divergence exceeds threshold
            let current_close = candles.close[avail_end - 1];
            let cum_return = (current_close - window.open_price) / window.open_price;
            let cum_return_abs = cum_return.abs();

            // The "false sentiment" trigger: strong trend that exceeds threshold
            if cum_return_abs < (threshold / 100.0) {
                continue;
            }

            if trend_strength < 0.5 {
                continue;
            }

            let threshold_exceedance = cum_return_abs / (threshold / 100.0) - 1.0;
            let confidence = Self::compute_confidence(trend_strength, threshold_exceedance);

            if !confidence.meets_minimum(self.min_confidence) {
                continue;
            }

            // Contrarian direction: opposite of the dominant trend
            let (direction, entry_price) = if is_up {
                // Trend is up, contrarian says buy NO
                let yes_price = 1.0 / (1.0 + (-cum_return / 0.07).exp());
                (Side::No, 1.0 - yes_price)
            } else {
                // Trend is down, contrarian says buy YES
                let yes_price = 1.0 / (1.0 + (-cum_return / 0.07).exp());
                (Side::Yes, yes_price)
            };

            let stop_loss = entry_price * (1.0 - 0.04);
            let take_profit = entry_price * (1.0 + 0.05);

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
        "false_sentiment"
    }

    fn estimated_win_prob(&self) -> f64 {
        // Conservative estimate — contrarian strategy has lower base accuracy
        0.65
    }
}

#[inline]
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_threshold() {
        let config = FalseSentimentConfig::default();
        let strategy = FalseSentiment::new(&config);

        let t0 = strategy.dynamic_threshold(0);
        assert!((t0 - 0.59).abs() < 1e-10);

        let t8 = strategy.dynamic_threshold(8);
        assert!(t8 > t0);
    }

    #[test]
    fn test_analyze_trend_up() {
        let mut store = CandleStore::with_capacity(5);
        for i in 0..5 {
            let price = 100.0 + i as f64;
            store.push(i as i64, price, price + 0.5, price - 0.5, price + 0.8, 1000.0);
        }
        let (strength, is_up) = FalseSentiment::analyze_trend(&store, 0, 5, 5);
        assert!(is_up);
        assert!(strength > 0.5);
    }
}
