use serde::{Deserialize, Serialize};

use crate::candle::CandleStore;

/// A 15-minute window over 1-minute candle data.
///
/// Pre-computed for O(1) lookups during strategy evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Window {
    /// Index of the first 1m candle in this window (inclusive).
    pub start_idx: usize,
    /// Index past the last 1m candle in this window (exclusive).
    pub end_idx: usize,
    /// BTC price at the start of this window (open of first candle).
    pub open_price: f64,
    /// BTC price at settlement (close of last candle).
    pub settlement_price: f64,
    /// Timestamp of window start.
    pub start_ts: i64,
    /// Timestamp of window end.
    pub end_ts: i64,
}

impl Window {
    /// Number of 1-minute candles in this window.
    #[inline]
    pub fn candle_count(&self) -> usize {
        self.end_idx - self.start_idx
    }

    /// Whether the settlement was green (close > open).
    #[inline]
    pub fn settled_green(&self) -> bool {
        self.settlement_price > self.open_price
    }
}

/// Compute non-overlapping 15-minute windows from sorted 1-minute candles.
///
/// Each window contains up to 15 consecutive candles. Windows are aligned to
/// 15-minute boundaries based on the first candle's timestamp.
pub fn compute_windows(candles: &CandleStore) -> Vec<Window> {
    if candles.is_empty() {
        return Vec::new();
    }

    let window_seconds: i64 = 15 * 60; // 900 seconds
    let mut windows = Vec::new();

    // Align to 15-minute boundary
    let base_ts = candles.timestamps[0];
    let aligned_start = base_ts - (base_ts % window_seconds);

    let mut window_start_ts = aligned_start;
    let mut i = 0;
    let n = candles.len();

    while i < n {
        let window_end_ts = window_start_ts + window_seconds;

        // Find all candles in this window
        let start_idx = i;
        while i < n && candles.timestamps[i] < window_end_ts {
            i += 1;
        }

        if i > start_idx {
            windows.push(Window {
                start_idx,
                end_idx: i,
                open_price: candles.open[start_idx],
                settlement_price: candles.close[i - 1],
                start_ts: window_start_ts,
                end_ts: window_end_ts,
            });
        }

        window_start_ts = window_end_ts;
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles(n: usize) -> CandleStore {
        let mut store = CandleStore::with_capacity(n);
        let base_ts: i64 = 1735689600; // 2025-01-01T00:00:00Z
        for i in 0..n {
            let ts = base_ts + (i as i64) * 60;
            let price = 100.0 + (i as f64) * 0.1;
            store.push(ts, price, price + 0.5, price - 0.5, price + 0.2, 1000.0);
        }
        store
    }

    #[test]
    fn test_compute_windows_exact_15() {
        let candles = make_test_candles(15);
        let windows = compute_windows(&candles);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].candle_count(), 15);
        assert_eq!(windows[0].open_price, candles.open[0]);
        assert_eq!(windows[0].settlement_price, candles.close[14]);
    }

    #[test]
    fn test_compute_windows_multiple() {
        let candles = make_test_candles(45);
        let windows = compute_windows(&candles);
        assert_eq!(windows.len(), 3);
        for w in &windows {
            assert_eq!(w.candle_count(), 15);
        }
    }

    #[test]
    fn test_compute_windows_partial() {
        let candles = make_test_candles(20);
        let windows = compute_windows(&candles);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].candle_count(), 15);
        assert_eq!(windows[1].candle_count(), 5);
    }

    #[test]
    fn test_window_settled_green() {
        let w = Window {
            start_idx: 0,
            end_idx: 15,
            open_price: 100.0,
            settlement_price: 101.0,
            start_ts: 0,
            end_ts: 900,
        };
        assert!(w.settled_green());

        let w2 = Window {
            start_idx: 0,
            end_idx: 15,
            open_price: 101.0,
            settlement_price: 100.0,
            start_ts: 0,
            end_ts: 900,
        };
        assert!(!w2.settled_green());
    }
}
