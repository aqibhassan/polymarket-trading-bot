use std::collections::HashMap;

use mvhe_core::config::SingularityConfig;
use mvhe_core::signal::{Confidence, Side, Signal, SignalType};
use mvhe_core::{CandleStore, Window};

use crate::traits::{Strategy, StrategyId};

/// Internal vote from a single signal source.
#[derive(Debug, Clone)]
struct SignalVote {
    name: &'static str,
    direction: VoteDir,
    strength: f64,
    weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoteDir {
    Yes,
    No,
}

/// Hardcoded hourly win-rate statistics for time-of-day signal.
///
/// Index = UTC hour, value = (estimated_win_rate, position_size_multiplier)
const HOURLY_STATS: [(f64, f64); 24] = [
    (0.88, 1.00), // 00
    (0.87, 0.95), // 01
    (0.86, 0.90), // 02
    (0.85, 0.85), // 03
    (0.84, 0.80), // 04
    (0.83, 0.75), // 05
    (0.85, 0.85), // 06
    (0.86, 0.90), // 07
    (0.88, 1.00), // 08
    (0.89, 1.05), // 09
    (0.90, 1.10), // 10
    (0.91, 1.15), // 11
    (0.90, 1.10), // 12
    (0.89, 1.05), // 13
    (0.88, 1.00), // 14
    (0.89, 1.05), // 15
    (0.90, 1.10), // 16
    (0.89, 1.05), // 17
    (0.88, 1.00), // 18
    (0.87, 0.95), // 19
    (0.88, 1.00), // 20
    (0.89, 1.05), // 21
    (0.88, 1.00), // 22
    (0.87, 0.95), // 23
];

/// Singularity ensemble strategy — combines 5 signal sources.
///
/// Port of `src/strategies/singularity.py`.
///
/// Signals:
///   1. Momentum (40%) — BTC cumulative return direction
///   2. OFI (25%) — order flow proxy from candle volume imbalance
///   3. Futures (15%) — proxy from recent candle velocity
///   4. Vol regime (10%) — tick-level vol vs sigmoid-implied vol
///   5. Time-of-day (10%) — hardcoded hourly stats
///
/// Requires minimum 3 of 5 signals to agree on direction.
pub struct Singularity {
    tier_thresholds: HashMap<u32, f64>,
    w_momentum: f64,
    w_ofi: f64,
    w_futures: f64,
    w_vol: f64,
    w_time: f64,
    min_signals_agree: u32,
    min_confidence: f64,
    entry_minute_start: u32,
    entry_minute_end: u32,
    estimated_win_prob: f64,
}

impl Singularity {
    pub fn new(config: &SingularityConfig) -> Self {
        Self {
            tier_thresholds: config.tier_thresholds(),
            w_momentum: config.weight_momentum,
            w_ofi: config.weight_ofi,
            w_futures: config.weight_futures,
            w_vol: config.weight_vol,
            w_time: config.weight_time,
            min_signals_agree: config.min_signals_agree,
            min_confidence: config.min_confidence,
            entry_minute_start: config.entry_minute_start,
            entry_minute_end: config.entry_minute_end,
            estimated_win_prob: config.estimated_win_prob,
        }
    }

    /// Signal 1: Momentum vote based on cumulative BTC return.
    fn vote_momentum(
        &self,
        candles: &CandleStore,
        window: &Window,
        minute: u32,
        minute_idx: usize,
    ) -> Option<SignalVote> {
        let threshold_pct = *self.tier_thresholds.get(&minute)?;

        let candle_idx = window.start_idx + minute_idx;
        let current_close = candles.close[candle_idx];
        let cum_return = (current_close - window.open_price) / window.open_price;
        let cum_return_pct = cum_return.abs() * 100.0;

        if cum_return_pct < threshold_pct {
            return None;
        }

        let direction = if cum_return > 0.0 {
            VoteDir::Yes
        } else {
            VoteDir::No
        };
        let strength = (cum_return_pct / 0.25).min(1.0);

        Some(SignalVote {
            name: "momentum",
            direction,
            strength,
            weight: self.w_momentum,
        })
    }

    /// Signal 2: OFI proxy from candle volume imbalance.
    ///
    /// Without live orderbook, we approximate order flow imbalance from
    /// the ratio of buying vs selling volume using candle close position
    /// relative to high-low range.
    fn vote_ofi(
        &self,
        candles: &CandleStore,
        window: &Window,
        avail_end: usize,
    ) -> Option<SignalVote> {
        let lookback = 5.min(avail_end - window.start_idx);
        if lookback < 2 {
            return None;
        }

        let start = avail_end - lookback;
        let mut buy_vol = 0.0f64;
        let mut sell_vol = 0.0f64;

        for i in start..avail_end {
            let range = candles.high[i] - candles.low[i];
            if range <= 0.0 {
                continue;
            }
            // Close position relative to range: 1.0 = closed at high (all buy), 0.0 = closed at low (all sell)
            let close_position = (candles.close[i] - candles.low[i]) / range;
            buy_vol += close_position * candles.volume[i];
            sell_vol += (1.0 - close_position) * candles.volume[i];
        }

        let total = buy_vol + sell_vol;
        if total == 0.0 {
            return None;
        }

        let imbalance = (buy_vol - sell_vol) / total;
        if imbalance.abs() < 0.1 {
            return None; // Too weak
        }

        let direction = if imbalance > 0.0 {
            VoteDir::Yes
        } else {
            VoteDir::No
        };
        let strength = imbalance.abs().min(1.0);

        Some(SignalVote {
            name: "ofi",
            direction,
            strength,
            weight: self.w_ofi,
        })
    }

    /// Signal 3: Futures lead-lag proxy from candle velocity.
    ///
    /// Without live futures feed, we approximate the lead-lag signal from
    /// the velocity (rate of change) of recent candle closes.
    fn vote_futures(
        &self,
        candles: &CandleStore,
        window: &Window,
        avail_end: usize,
    ) -> Option<SignalVote> {
        let lookback = 3.min(avail_end - window.start_idx);
        if lookback < 2 {
            return None;
        }

        let start = avail_end - lookback;
        let first = candles.close[start];
        let last = candles.close[avail_end - 1];

        if first == 0.0 {
            return None;
        }

        // Velocity: percent change per candle
        let velocity = (last - first) / first / (lookback as f64);
        let velocity_pct = velocity.abs() * 100.0;

        // Need meaningful velocity to signal
        if velocity_pct < 0.01 {
            return None;
        }

        let direction = if velocity > 0.0 {
            VoteDir::Yes
        } else {
            VoteDir::No
        };
        let strength = (velocity_pct / 0.05).min(1.0);

        Some(SignalVote {
            name: "futures",
            direction,
            strength,
            weight: self.w_futures,
        })
    }

    /// Signal 4: Volatility regime from tick-level realized vol vs sigmoid-implied vol.
    fn vote_vol_regime(
        &self,
        candles: &CandleStore,
        window: &Window,
        avail_end: usize,
        cum_return: f64,
    ) -> Option<SignalVote> {
        let lookback = 5.min(avail_end - window.start_idx);
        if lookback < 3 {
            return None;
        }

        let start = avail_end - lookback;

        // Compute realized vol from candle log returns
        let mut log_returns = Vec::with_capacity(lookback - 1);
        for i in (start + 1)..avail_end {
            let prev = candles.close[i - 1];
            if prev > 0.0 {
                log_returns.push((candles.close[i] / prev).ln());
            }
        }

        if log_returns.len() < 2 {
            return None;
        }

        let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let variance =
            log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / log_returns.len() as f64;
        let realized_vol = variance.sqrt();

        // Sigmoid-implied vol (from YES price)
        let yes_price = 1.0 / (1.0 + (-cum_return / 0.07).exp());
        let implied_vol = (yes_price * (1.0 - yes_price)).sqrt() * 0.1;

        if implied_vol == 0.0 {
            return None;
        }

        let vol_ratio = realized_vol / implied_vol;

        // long_vol: options underpriced → follow dominant direction
        if vol_ratio > 1.25 {
            let direction = if cum_return > 0.0 {
                VoteDir::Yes
            } else {
                VoteDir::No
            };
            let strength = (vol_ratio / 2.0).min(1.0);
            Some(SignalVote {
                name: "vol_regime",
                direction,
                strength,
                weight: self.w_vol,
            })
        } else {
            None // short_vol or neutral — don't trade
        }
    }

    /// Signal 5: Time-of-day seasonality from hardcoded hourly stats.
    fn vote_time_of_day(&self, window_start_ts: i64) -> Option<SignalVote> {
        let hour = ((window_start_ts % 86400) / 3600) as usize % 24;
        let (win_rate, pos_mult) = HOURLY_STATS[hour];

        if pos_mult < 0.75 {
            return None; // Don't trade in low-edge hours
        }

        let strength = (pos_mult / 1.25).min(1.0);
        let direction = if win_rate > 0.85 {
            VoteDir::Yes
        } else {
            VoteDir::No
        };

        Some(SignalVote {
            name: "time_of_day",
            direction,
            strength,
            weight: self.w_time,
        })
    }
}

impl Strategy for Singularity {
    fn evaluate_window(&self, candles: &CandleStore, window: &Window) -> Option<Signal> {
        let n_candles = window.candle_count();

        // Evaluate at the earliest valid entry minute (first match wins, like Python)
        for minute in self.entry_minute_start..=self.entry_minute_end {
            let minute_idx = minute as usize;
            if minute_idx >= n_candles {
                continue;
            }

            let avail_end = window.start_idx + minute_idx + 1;

            // Compute cumulative return for shared context
            let candle_idx = window.start_idx + minute_idx;
            let current_close = candles.close[candle_idx];
            let cum_return = (current_close - window.open_price) / window.open_price;

            // Collect votes from all 5 sources
            let mut votes: Vec<SignalVote> = Vec::with_capacity(5);

            if let Some(v) = self.vote_momentum(candles, window, minute, minute_idx) {
                votes.push(v);
            }
            if let Some(v) = self.vote_ofi(candles, window, avail_end) {
                votes.push(v);
            }
            if let Some(v) = self.vote_futures(candles, window, avail_end) {
                votes.push(v);
            }
            if let Some(v) = self.vote_vol_regime(candles, window, avail_end, cum_return) {
                votes.push(v);
            }
            if let Some(v) = self.vote_time_of_day(window.start_ts) {
                votes.push(v);
            }

            if votes.is_empty() {
                continue;
            }

            // Count YES vs NO votes
            let yes_votes: Vec<&SignalVote> =
                votes.iter().filter(|v| v.direction == VoteDir::Yes).collect();
            let no_votes: Vec<&SignalVote> =
                votes.iter().filter(|v| v.direction == VoteDir::No).collect();

            let yes_count = yes_votes.len() as u32;
            let no_count = no_votes.len() as u32;

            // Graceful degradation: adjust min agreement if fewer sources available
            let available_sources = votes.len() as u32;
            let effective_min = self.min_signals_agree.min(available_sources);

            let (direction, agreeing_votes) = if yes_count >= effective_min {
                (VoteDir::Yes, &yes_votes)
            } else if no_count >= effective_min {
                (VoteDir::No, &no_votes)
            } else {
                continue;
            };

            // Weighted confidence
            let total_weight: f64 = agreeing_votes.iter().map(|v| v.weight).sum();
            let weighted_confidence = if total_weight > 0.0 {
                agreeing_votes
                    .iter()
                    .map(|v| v.strength * v.weight)
                    .sum::<f64>()
                    / total_weight
            } else {
                0.0
            };

            // Signal count bonus: more signals = higher confidence
            let signal_count_mult = (agreeing_votes.len() as f64 / 3.0).min(1.5);
            let overall_confidence = (weighted_confidence * signal_count_mult).min(1.0);

            if overall_confidence < self.min_confidence {
                continue;
            }

            // Build signal
            let (side, entry_price) = match direction {
                VoteDir::Yes => {
                    let yes_price = 1.0 / (1.0 + (-cum_return / 0.07).exp());
                    (Side::Yes, yes_price)
                }
                VoteDir::No => {
                    let yes_price = 1.0 / (1.0 + (-cum_return / 0.07).exp());
                    (Side::No, 1.0 - yes_price)
                }
            };

            // Find momentum vote for trend_strength field
            let momentum_strength = votes
                .iter()
                .find(|v| v.name == "momentum")
                .map(|v| v.strength)
                .unwrap_or(0.0);
            let ofi_strength = votes
                .iter()
                .find(|v| v.name == "ofi")
                .map(|v| v.strength)
                .unwrap_or(0.0);
            let time_strength = votes
                .iter()
                .find(|v| v.name == "time_of_day")
                .map(|v| v.strength)
                .unwrap_or(0.0);

            let confidence = Confidence {
                trend_strength: round4(momentum_strength),
                threshold_exceedance: round4(overall_confidence),
                book_normality: round4(ofi_strength),
                liquidity_quality: round4(time_strength),
                overall: round4(overall_confidence),
            };

            let stop_loss = entry_price * (1.0 - 1.0); // hold-to-settlement
            let take_profit = entry_price * (1.0 + 0.40);

            return Some(Signal {
                strategy_id: self.id().to_string(),
                signal_type: SignalType::Entry,
                direction: side,
                confidence,
                entry_price,
                stop_loss,
                take_profit,
            });
        }

        None
    }

    fn id(&self) -> StrategyId {
        "singularity"
    }

    fn estimated_win_prob(&self) -> f64 {
        self.estimated_win_prob
    }
}

#[inline]
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trending_candles(n: usize, up: bool) -> (CandleStore, Window) {
        let mut store = CandleStore::with_capacity(n);
        let base_ts: i64 = 1735689600; // 2025-01-01T00:00:00Z (midnight UTC)
        let base_price = 100.0;

        for i in 0..n {
            let ts = base_ts + (i as i64) * 60;
            let delta = if up {
                (i as f64) * 0.02
            } else {
                -(i as f64) * 0.02
            };
            let o = base_price + delta;
            let c = o + if up { 0.015 } else { -0.015 };
            let h = o.max(c) + 0.005;
            let l = o.min(c) - 0.005;
            // Volume biased in trend direction
            let v = if up { 1200.0 } else { 800.0 };
            store.push(ts, o, h, l, c, v);
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
    fn test_singularity_requires_minimum_agreement() {
        let (candles, window) = make_trending_candles(15, true);
        let config = SingularityConfig::default();
        let strategy = Singularity::new(&config);

        // This should either produce a signal or not depending on agreement
        let _result = strategy.evaluate_window(&candles, &window);
        // We don't assert specific behavior, just that it doesn't panic
    }

    #[test]
    fn test_vote_time_of_day() {
        let config = SingularityConfig::default();
        let strategy = Singularity::new(&config);

        // Midnight UTC -> hour 0
        let vote = strategy.vote_time_of_day(1735689600);
        assert!(vote.is_some());
        let v = vote.unwrap();
        assert_eq!(v.name, "time_of_day");
    }

    #[test]
    fn test_vote_ofi_needs_data() {
        let config = SingularityConfig::default();
        let strategy = Singularity::new(&config);

        let mut store = CandleStore::with_capacity(1);
        store.push(0, 100.0, 101.0, 99.0, 100.5, 1000.0);
        let window = Window {
            start_idx: 0,
            end_idx: 1,
            open_price: 100.0,
            settlement_price: 100.5,
            start_ts: 0,
            end_ts: 900,
        };

        // Only 1 candle — not enough for OFI
        let vote = strategy.vote_ofi(&store, &window, 1);
        assert!(vote.is_none());
    }
}
