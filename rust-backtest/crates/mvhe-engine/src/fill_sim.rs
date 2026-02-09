use serde::Serialize;

const DEFAULT_BASE_SLIPPAGE: f64 = 0.0005; // 5 bps
const LARGE_ORDER_THRESHOLD: f64 = 0.05; // 5% of book depth

/// Result of a simulated fill.
#[derive(Debug, Clone, Serialize)]
pub struct SimulatedFill {
    pub fill_price: f64,
    pub slippage: f64,
    pub fee: f64,
    pub net_cost: f64,
}

/// Simulate realistic order fills with slippage and Polymarket dynamic fees.
pub struct FillSimulator {
    base_slippage: f64,
    fee_constant: f64,
}

impl FillSimulator {
    pub fn new(base_slippage_bps: u32, fee_constant: f64) -> Self {
        Self {
            base_slippage: base_slippage_bps as f64 / 10000.0,
            fee_constant,
        }
    }

    /// Simulate a fill with slippage and Polymarket dynamic fee.
    ///
    /// Slippage model:
    ///   - Base: 5 bps (configurable)
    ///   - If order < 5% of book depth: linear scaling
    ///   - If order >= 5%: sqrt scaling for large orders
    ///
    /// Fee model (Polymarket dynamic):
    ///   `fee = notional * fee_constant * p^2 * (1-p)^2`
    ///   where p = entry_price (treated as probability)
    pub fn simulate_fill(
        &self,
        order_price: f64,
        order_size: f64,
        book_depth: f64,
    ) -> SimulatedFill {
        // Slippage calculation
        let slippage_pct = if book_depth <= 0.0 {
            self.base_slippage
        } else {
            let size_ratio = order_size / book_depth;
            if size_ratio < LARGE_ORDER_THRESHOLD {
                size_ratio * self.base_slippage
            } else {
                size_ratio.sqrt() * self.base_slippage
            }
        };

        let slippage_amount = order_price * slippage_pct;
        let fill_price = order_price + slippage_amount;
        let notional = fill_price * order_size;

        // Polymarket dynamic fee: notional * fee_constant * p^2 * (1-p)^2
        let p = order_price.clamp(0.01, 0.99);
        let fee = notional * self.fee_constant * p * p * (1.0 - p) * (1.0 - p);

        let net_cost = notional + fee;

        SimulatedFill {
            fill_price,
            slippage: slippage_amount,
            fee,
            net_cost,
        }
    }
}

impl Default for FillSimulator {
    fn default() -> Self {
        Self {
            base_slippage: DEFAULT_BASE_SLIPPAGE,
            fee_constant: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_basic() {
        let sim = FillSimulator::default();
        let fill = sim.simulate_fill(0.50, 100.0, 10000.0);

        assert!(fill.fill_price > 0.50);
        assert!(fill.slippage > 0.0);
        assert!(fill.fee > 0.0);
        assert!(fill.net_cost > fill.fill_price * 100.0);
    }

    #[test]
    fn test_fill_no_book_depth() {
        let sim = FillSimulator::default();
        let fill = sim.simulate_fill(0.50, 100.0, 0.0);

        // Should use base slippage
        let expected_slippage = 0.50 * DEFAULT_BASE_SLIPPAGE;
        assert!((fill.slippage - expected_slippage).abs() < 1e-10);
    }

    #[test]
    fn test_polymarket_dynamic_fee() {
        let sim = FillSimulator::new(5, 0.25);

        // Fee peaks at p=0.5: notional * 0.25 * 0.5^2 * 0.5^2 = notional * 0.015625
        let fill = sim.simulate_fill(0.50, 100.0, 100000.0);
        let expected_fee_ratio = 0.25 * 0.5 * 0.5 * 0.5 * 0.5; // ~0.015625
        let actual_ratio = fill.fee / (fill.fill_price * 100.0);
        assert!((actual_ratio - expected_fee_ratio).abs() < 1e-6);

        // Fee drops at extremes
        let fill_extreme = sim.simulate_fill(0.90, 100.0, 100000.0);
        assert!(fill_extreme.fee < fill.fee);
    }
}
