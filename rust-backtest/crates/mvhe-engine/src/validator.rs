use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::metrics::sharpe_from_returns;

/// Result of statistical validation.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    pub sharpe_ci_95: (f64, f64),
    pub p_value: f64,
    pub is_significant: bool,
    pub overfitting_warning: Option<String>,
}

/// Statistical validator for backtest results.
///
/// Validates that backtest performance is statistically significant using:
/// - Bootstrap 95% CI on Sharpe ratio (parallelized with Rayon)
/// - Permutation test on trade PnLs (parallelized with Rayon)
/// - IS/OOS overfitting check (70/30 split)
pub struct StatisticalValidator {
    n_bootstrap: usize,
    n_permutations: usize,
    significance_level: f64,
    seed: u64,
}

impl StatisticalValidator {
    pub fn new(
        n_bootstrap: usize,
        n_permutations: usize,
        significance_level: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_bootstrap,
            n_permutations,
            significance_level,
            seed,
        }
    }

    /// Run all validation checks.
    pub fn validate(
        &self,
        equity_curve: &[f64],
        trade_pnls: &[f64],
    ) -> ValidationResult {
        let returns = Self::equity_to_returns(equity_curve);

        let sharpe_ci = self.bootstrap_sharpe_ci(&returns);
        let p_value = self.permutation_test(trade_pnls);
        let is_significant = p_value < self.significance_level;

        // Overfitting check
        let overfitting_warning = if equity_curve.len() >= 20 {
            let split_idx = (equity_curve.len() as f64 * 0.7) as usize;
            let is_returns = Self::equity_to_returns(&equity_curve[..split_idx]);
            let oos_returns = Self::equity_to_returns(&equity_curve[split_idx..]);
            if is_returns.len() >= 2 && oos_returns.len() >= 2 {
                let is_sharpe = sharpe_from_returns(&is_returns);
                let oos_sharpe = sharpe_from_returns(&oos_returns);
                Self::check_overfitting(is_sharpe, oos_sharpe)
            } else {
                None
            }
        } else {
            None
        };

        ValidationResult {
            sharpe_ci_95: sharpe_ci,
            p_value,
            is_significant,
            overfitting_warning,
        }
    }

    /// Bootstrap 95% CI on Sharpe ratio using Rayon parallelism.
    fn bootstrap_sharpe_ci(&self, returns: &[f64]) -> (f64, f64) {
        if returns.len() < 2 {
            return (0.0, 0.0);
        }

        let n = returns.len();
        let seed = self.seed;
        let n_bootstrap = self.n_bootstrap;

        // Each bootstrap iteration uses an independent RNG seeded deterministically
        let mut bootstrapped_sharpes: Vec<f64> = (0..n_bootstrap)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                let sample: Vec<f64> = (0..n).map(|_| returns[rng.gen_range(0..n)]).collect();
                sharpe_from_returns(&sample)
            })
            .collect();

        bootstrapped_sharpes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = (n_bootstrap as f64 * 0.025) as usize;
        let upper_idx = (n_bootstrap as f64 * 0.975) as usize;

        (
            bootstrapped_sharpes[lower_idx],
            bootstrapped_sharpes[upper_idx.min(bootstrapped_sharpes.len() - 1)],
        )
    }

    /// Monte Carlo permutation test: randomly flip PnL signs.
    fn permutation_test(&self, pnls: &[f64]) -> f64 {
        if pnls.is_empty() {
            return 1.0;
        }

        let observed_total: f64 = pnls.iter().sum();
        let seed = self.seed;
        let n_permutations = self.n_permutations;

        let count_gte: usize = (0..n_permutations)
            .into_par_iter()
            .filter(|&i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(10000 + i as u64));
                let shuffled_total: f64 = pnls
                    .iter()
                    .map(|&p| {
                        if rng.gen_bool(0.5) {
                            p
                        } else {
                            -p
                        }
                    })
                    .sum();
                shuffled_total >= observed_total
            })
            .count();

        count_gte as f64 / n_permutations as f64
    }

    fn check_overfitting(is_sharpe: f64, oos_sharpe: f64) -> Option<String> {
        if oos_sharpe <= 0.0 {
            return Some(format!(
                "Out-of-sample Sharpe ({:.2}) is non-positive. In-sample was {:.2}. Likely overfitting.",
                oos_sharpe, is_sharpe
            ));
        }
        let ratio = is_sharpe / oos_sharpe;
        if ratio > 2.0 {
            return Some(format!(
                "IS/OOS Sharpe ratio is {:.2}x (IS={:.2}, OOS={:.2}). Possible overfitting.",
                ratio, is_sharpe, oos_sharpe
            ));
        }
        None
    }

    fn equity_to_returns(equity_curve: &[f64]) -> Vec<f64> {
        if equity_curve.len() < 2 {
            return Vec::new();
        }
        equity_curve
            .windows(2)
            .map(|w| {
                if w[0] == 0.0 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect()
    }
}

impl Default for StatisticalValidator {
    fn default() -> Self {
        Self::new(1000, 1000, 0.05, 42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_ci_empty() {
        let validator = StatisticalValidator::default();
        let (lo, hi) = validator.bootstrap_sharpe_ci(&[]);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn test_permutation_test_zero_pnl() {
        let validator = StatisticalValidator::default();
        let pnls = vec![0.0, 0.0, 0.0];
        let p = validator.permutation_test(&pnls);
        // With all zeros, every permutation matches -> p_value = 1.0
        assert!((p - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_permutation_test_significant() {
        let validator = StatisticalValidator::new(100, 1000, 0.05, 42);
        // Very profitable strategy
        let pnls = vec![10.0; 50];
        let p = validator.permutation_test(&pnls);
        assert!(p < 0.05, "Expected significant p-value, got {}", p);
    }

    #[test]
    fn test_check_overfitting_none() {
        let result = StatisticalValidator::check_overfitting(2.0, 1.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_overfitting_detected() {
        let result = StatisticalValidator::check_overfitting(4.0, 1.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_validate_full() {
        let validator = StatisticalValidator::new(100, 100, 0.05, 42);
        let equity: Vec<f64> = (0..100).map(|i| 10000.0 + i as f64 * 10.0).collect();
        let pnls: Vec<f64> = (0..50).map(|_| 20.0).collect();

        let result = validator.validate(&equity, &pnls);
        assert!(result.sharpe_ci_95.0 <= result.sharpe_ci_95.1);
    }
}
