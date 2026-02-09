use serde::Serialize;

use crate::engine::Trade;

const ANNUALIZATION_FACTOR: f64 = 15.874507866; // sqrt(252)

/// Computed performance metrics.
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub total_return_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown_pct: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win_loss_ratio: f64,
    pub total_trades: usize,
    pub total_pnl: f64,
}

pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate all performance metrics from equity curve and trades.
    pub fn calculate(equity_curve: &[f64], trades: &[Trade]) -> Metrics {
        let returns = Self::compute_returns(equity_curve);
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();

        Metrics {
            total_return_pct: Self::total_return(equity_curve),
            sharpe_ratio: Self::sharpe_ratio(&returns),
            sortino_ratio: Self::sortino_ratio(&returns),
            max_drawdown_pct: Self::max_drawdown(equity_curve),
            calmar_ratio: Self::calmar_ratio(&returns, Self::max_drawdown(equity_curve)),
            win_rate: Self::win_rate(&pnls),
            profit_factor: Self::profit_factor(&pnls),
            avg_win_loss_ratio: Self::avg_win_loss_ratio(&pnls),
            total_trades: trades.len(),
            total_pnl: pnls.iter().sum(),
        }
    }

    fn compute_returns(equity_curve: &[f64]) -> Vec<f64> {
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

    fn sharpe_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = std_dev(returns);
        if std == 0.0 {
            return 0.0;
        }
        mean / std * ANNUALIZATION_FACTOR
    }

    fn sortino_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().copied().filter(|&r| r < 0.0).collect();
        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }
        let downside_std = std_dev(&downside);
        if downside_std == 0.0 {
            return 0.0;
        }
        mean / downside_std * ANNUALIZATION_FACTOR
    }

    fn max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.len() < 2 {
            return 0.0;
        }
        let mut peak = equity_curve[0];
        let mut max_dd = 0.0f64;
        for &value in &equity_curve[1..] {
            if value > peak {
                peak = value;
            }
            if peak > 0.0 {
                let dd = (peak - value) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }
        max_dd
    }

    fn calmar_ratio(returns: &[f64], max_dd: f64) -> f64 {
        if max_dd == 0.0 || returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let annualized = mean * 252.0;
        annualized / max_dd
    }

    fn total_return(equity_curve: &[f64]) -> f64 {
        if equity_curve.len() < 2 || equity_curve[0] == 0.0 {
            return 0.0;
        }
        (equity_curve.last().unwrap() - equity_curve[0]) / equity_curve[0]
    }

    fn win_rate(pnls: &[f64]) -> f64 {
        if pnls.is_empty() {
            return 0.0;
        }
        let wins = pnls.iter().filter(|&&p| p > 0.0).count();
        wins as f64 / pnls.len() as f64
    }

    fn profit_factor(pnls: &[f64]) -> f64 {
        let gross_profit: f64 = pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        if gross_loss == 0.0 {
            return if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        }
        gross_profit / gross_loss
    }

    fn avg_win_loss_ratio(pnls: &[f64]) -> f64 {
        let wins: Vec<f64> = pnls.iter().copied().filter(|&p| p > 0.0).collect();
        let losses: Vec<f64> = pnls
            .iter()
            .copied()
            .filter(|&p| p < 0.0)
            .map(f64::abs)
            .collect();
        if wins.is_empty() || losses.is_empty() {
            return 0.0;
        }
        let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        if avg_loss == 0.0 {
            return 0.0;
        }
        avg_win / avg_loss
    }
}

/// Population standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Public Sharpe calculation for use by validator.
pub fn sharpe_from_returns(returns: &[f64]) -> f64 {
    MetricsCalculator::sharpe_ratio(returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvhe_core::signal::Side;

    fn make_trades(pnls: &[f64]) -> Vec<Trade> {
        pnls.iter()
            .map(|&pnl| Trade {
                strategy_id: "test".into(),
                direction: Side::Yes,
                entry_price: 0.5,
                exit_price: if pnl > 0.0 { 1.0 } else { 0.0 },
                size: 100.0,
                pnl,
                entry_ts: 0,
                exit_ts: 900,
                exit_reason: mvhe_core::signal::ExitReason::ResolutionGuard,
                fee: 0.0,
                slippage: 0.0,
            })
            .collect()
    }

    #[test]
    fn test_win_rate() {
        let pnls = vec![10.0, -5.0, 20.0, -3.0, 15.0];
        assert!((MetricsCalculator::win_rate(&pnls) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_profit_factor() {
        let pnls = vec![10.0, -5.0, 20.0, -3.0];
        let pf = MetricsCalculator::profit_factor(&pnls);
        assert!((pf - 30.0 / 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0];
        let dd = MetricsCalculator::max_drawdown(&equity);
        // Peak 115, trough 100 -> dd = 15/115
        assert!((dd - 15.0 / 115.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_return() {
        let equity = vec![100.0, 110.0, 120.0];
        let ret = MetricsCalculator::total_return(&equity);
        assert!((ret - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_zero_std() {
        let returns = vec![0.01, 0.01, 0.01]; // constant returns
        let sharpe = MetricsCalculator::sharpe_ratio(&returns);
        assert_eq!(sharpe, 0.0); // std = 0
    }

    #[test]
    fn test_metrics_full() {
        let pnls = vec![10.0, -5.0, 20.0, -3.0, 15.0];
        let trades = make_trades(&pnls);
        let equity: Vec<f64> = std::iter::once(100.0)
            .chain(pnls.iter().scan(100.0, |acc, &pnl| {
                *acc += pnl;
                Some(*acc)
            }))
            .collect();

        let metrics = MetricsCalculator::calculate(&equity, &trades);
        assert_eq!(metrics.total_trades, 5);
        assert!((metrics.win_rate - 0.6).abs() < 1e-10);
        assert!(metrics.total_pnl > 0.0);
    }
}
