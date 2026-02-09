use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    Entry,
    Exit,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    ResolutionGuard,
    ProfitTarget,
    HardStopLoss,
    MaxTime,
    TrailingStop,
    BacktestEnd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Confidence {
    pub trend_strength: f64,
    pub threshold_exceedance: f64,
    pub book_normality: f64,
    pub liquidity_quality: f64,
    pub overall: f64,
}

impl Confidence {
    pub fn meets_minimum(&self, min_confidence: f64) -> bool {
        self.overall >= min_confidence
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub strategy_id: String,
    pub signal_type: SignalType,
    pub direction: Side,
    pub confidence: Confidence,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
}
