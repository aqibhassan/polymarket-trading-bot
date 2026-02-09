use serde::Deserialize;
use std::path::Path;

/// Top-level backtest config, parsed from the same TOML files as Python.
#[derive(Debug, Clone, Deserialize)]
pub struct BacktestConfig {
    #[serde(default)]
    pub strategy: StrategySection,
    #[serde(default)]
    pub risk: RiskConfig,
    #[serde(default)]
    pub exit: ExitConfig,
    #[serde(default)]
    pub paper: PaperConfig,
}

impl BacktestConfig {
    /// Load config from a TOML file path.
    pub fn from_toml(path: &Path) -> Result<Self, ConfigError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;
        Self::from_toml_str(&content)
    }

    /// Parse config from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        toml::from_str(s).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    /// Load and merge multiple TOML files (later files override earlier).
    pub fn from_toml_files(paths: &[&Path]) -> Result<Self, ConfigError> {
        if paths.is_empty() {
            return Err(ConfigError::Parse("no config files provided".into()));
        }
        // Start from first file
        let mut content =
            std::fs::read_to_string(paths[0]).map_err(|e| ConfigError::Io(e.to_string()))?;

        // For merging, we parse each file as a toml::Value and merge
        let mut base: toml::Value =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        for path in &paths[1..] {
            content =
                std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;
            let overlay: toml::Value =
                toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;
            merge_toml(&mut base, overlay);
        }

        let merged_str =
            toml::to_string(&base).map_err(|e| ConfigError::Parse(e.to_string()))?;
        Self::from_toml_str(&merged_str)
    }
}

fn merge_toml(base: &mut toml::Value, overlay: toml::Value) {
    if let (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) =
        (base, overlay)
    {
        for (key, value) in overlay_table {
            if let Some(base_value) = base_table.get_mut(&key) {
                if base_value.is_table() && value.is_table() {
                    merge_toml(base_value, value);
                    continue;
                }
            }
            base_table.insert(key, value);
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StrategySection {
    #[serde(default)]
    pub momentum_confirmation: MomentumConfig,
    #[serde(default)]
    pub false_sentiment: FalseSentimentConfig,
    #[serde(default)]
    pub singularity: SingularityConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EntryTier {
    pub minute: u32,
    pub threshold_pct: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MomentumConfig {
    #[serde(default = "default_8")]
    pub entry_minute_start: u32,
    #[serde(default = "default_10")]
    pub entry_minute_end: u32,
    #[serde(default = "default_true")]
    pub hold_to_settlement: bool,
    #[serde(default = "default_0_40")]
    pub profit_target_pct: f64,
    #[serde(default = "default_1_00")]
    pub stop_loss_pct: f64,
    #[serde(default = "default_10")]
    pub max_hold_minutes: u32,
    #[serde(default = "default_15")]
    pub resolution_guard_minute: u32,
    #[serde(default = "default_0_70")]
    pub min_confidence: f64,
    #[serde(default)]
    pub entry_tiers: Vec<EntryTier>,
    #[serde(default = "default_0_10")]
    pub entry_threshold: f64,
    #[serde(default = "default_0_884")]
    pub estimated_win_prob: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            entry_minute_start: 8,
            entry_minute_end: 10,
            hold_to_settlement: true,
            profit_target_pct: 0.40,
            stop_loss_pct: 1.00,
            max_hold_minutes: 10,
            resolution_guard_minute: 15,
            min_confidence: 0.70,
            entry_tiers: vec![
                EntryTier { minute: 8, threshold_pct: 0.10 },
                EntryTier { minute: 9, threshold_pct: 0.08 },
                EntryTier { minute: 10, threshold_pct: 0.05 },
            ],
            entry_threshold: 0.10,
            estimated_win_prob: 0.884,
        }
    }
}

impl MomentumConfig {
    /// Build a minute -> threshold lookup map.
    pub fn tier_thresholds(&self) -> std::collections::HashMap<u32, f64> {
        if self.entry_tiers.is_empty() {
            // Flat threshold fallback
            (self.entry_minute_start..=self.entry_minute_end)
                .map(|m| (m, self.entry_threshold))
                .collect()
        } else {
            self.entry_tiers
                .iter()
                .map(|t| (t.minute, t.threshold_pct))
                .collect()
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FalseSentimentConfig {
    #[serde(default = "default_0_59")]
    pub entry_threshold_base: f64,
    #[serde(default = "default_0_15")]
    pub threshold_time_scaling: f64,
    #[serde(default = "default_5")]
    pub lookback_candles: u32,
    #[serde(default = "default_0_60")]
    pub min_confidence: f64,
    #[serde(default = "default_7")]
    pub max_hold_minutes: u32,
    #[serde(default = "default_11")]
    pub force_exit_minute: u32,
    #[serde(default = "default_8")]
    pub no_entry_after_minute: u32,
}

impl Default for FalseSentimentConfig {
    fn default() -> Self {
        Self {
            entry_threshold_base: 0.59,
            threshold_time_scaling: 0.15,
            lookback_candles: 5,
            min_confidence: 0.60,
            max_hold_minutes: 7,
            force_exit_minute: 11,
            no_entry_after_minute: 8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SingularityConfig {
    #[serde(default = "default_0_40")]
    pub weight_momentum: f64,
    #[serde(default = "default_0_25")]
    pub weight_ofi: f64,
    #[serde(default = "default_0_15")]
    pub weight_futures: f64,
    #[serde(default = "default_0_10")]
    pub weight_vol: f64,
    #[serde(default = "default_0_10")]
    pub weight_time: f64,
    #[serde(default = "default_3")]
    pub min_signals_agree: u32,
    #[serde(default = "default_0_72")]
    pub min_confidence: f64,
    #[serde(default = "default_6")]
    pub entry_minute_start: u32,
    #[serde(default = "default_10")]
    pub entry_minute_end: u32,
    #[serde(default)]
    pub entry_tiers: Vec<EntryTier>,
    #[serde(default = "default_12")]
    pub resolution_guard_minute: u32,
    #[serde(default = "default_2")]
    pub exit_reversal_count: u32,
    #[serde(default = "default_0_035")]
    pub max_position_pct: f64,
    #[serde(default = "default_1_00")]
    pub stop_loss_pct: f64,
    #[serde(default = "default_0_40")]
    pub profit_target_pct: f64,
    #[serde(default = "default_0_90")]
    pub estimated_win_prob: f64,
}

impl Default for SingularityConfig {
    fn default() -> Self {
        Self {
            weight_momentum: 0.40,
            weight_ofi: 0.25,
            weight_futures: 0.15,
            weight_vol: 0.10,
            weight_time: 0.10,
            min_signals_agree: 3,
            min_confidence: 0.72,
            entry_minute_start: 6,
            entry_minute_end: 10,
            entry_tiers: vec![
                EntryTier { minute: 6, threshold_pct: 0.12 },
                EntryTier { minute: 7, threshold_pct: 0.10 },
                EntryTier { minute: 8, threshold_pct: 0.08 },
                EntryTier { minute: 9, threshold_pct: 0.06 },
                EntryTier { minute: 10, threshold_pct: 0.05 },
            ],
            resolution_guard_minute: 12,
            exit_reversal_count: 2,
            max_position_pct: 0.035,
            stop_loss_pct: 1.00,
            profit_target_pct: 0.40,
            estimated_win_prob: 0.90,
        }
    }
}

impl SingularityConfig {
    pub fn tier_thresholds(&self) -> std::collections::HashMap<u32, f64> {
        if self.entry_tiers.is_empty() {
            [(6, 0.12), (7, 0.10), (8, 0.08), (9, 0.06), (10, 0.05)]
                .into_iter()
                .collect()
        } else {
            self.entry_tiers
                .iter()
                .map(|t| (t.minute, t.threshold_pct))
                .collect()
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    #[serde(default = "default_0_02")]
    pub max_position_pct: f64,
    #[serde(default = "default_0_05")]
    pub max_daily_drawdown_pct: f64,
    #[serde(default = "default_0_25")]
    pub kelly_multiplier: f64,
    #[serde(default = "default_10000")]
    pub min_starting_balance: f64,
    #[serde(default)]
    pub fees: FeeConfig,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_pct: 0.02,
            max_daily_drawdown_pct: 0.05,
            kelly_multiplier: 0.25,
            min_starting_balance: 10000.0,
            fees: FeeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeeConfig {
    #[serde(default = "default_polymarket_dynamic")]
    pub fee_model: String,
    #[serde(default = "default_0_25")]
    pub fee_constant: f64,
    #[serde(default = "default_5")]
    pub slippage_bps: u32,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            fee_model: "polymarket_dynamic".into(),
            fee_constant: 0.25,
            slippage_bps: 5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExitConfig {
    #[serde(default = "default_0_05")]
    pub profit_target_pct: f64,
    #[serde(default = "default_0_03")]
    pub trailing_stop_pct: f64,
    #[serde(default = "default_0_04")]
    pub hard_stop_loss_pct: f64,
    #[serde(default = "default_420")]
    pub max_hold_seconds: u32,
}

impl Default for ExitConfig {
    fn default() -> Self {
        Self {
            profit_target_pct: 0.05,
            trailing_stop_pct: 0.03,
            hard_stop_loss_pct: 0.04,
            max_hold_seconds: 420,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PaperConfig {
    #[serde(default = "default_10000")]
    pub initial_balance: f64,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            initial_balance: 10000.0,
        }
    }
}

// Default value helpers
fn default_true() -> bool { true }
fn default_2() -> u32 { 2 }
fn default_3() -> u32 { 3 }
fn default_5() -> u32 { 5 }
fn default_6() -> u32 { 6 }
fn default_7() -> u32 { 7 }
fn default_8() -> u32 { 8 }
fn default_10() -> u32 { 10 }
fn default_11() -> u32 { 11 }
fn default_12() -> u32 { 12 }
fn default_15() -> u32 { 15 }
fn default_420() -> u32 { 420 }
fn default_0_02() -> f64 { 0.02 }
fn default_0_03() -> f64 { 0.03 }
fn default_0_035() -> f64 { 0.035 }
fn default_0_04() -> f64 { 0.04 }
fn default_0_05() -> f64 { 0.05 }
fn default_0_10() -> f64 { 0.10 }
fn default_0_15() -> f64 { 0.15 }
fn default_0_25() -> f64 { 0.25 }
fn default_0_40() -> f64 { 0.40 }
fn default_0_59() -> f64 { 0.59 }
fn default_0_60() -> f64 { 0.60 }
fn default_0_70() -> f64 { 0.70 }
fn default_0_72() -> f64 { 0.72 }
fn default_0_884() -> f64 { 0.884 }
fn default_0_90() -> f64 { 0.90 }
fn default_1_00() -> f64 { 1.00 }
fn default_10000() -> f64 { 10000.0 }
fn default_polymarket_dynamic() -> String { "polymarket_dynamic".into() }

#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config I/O error: {}", e),
            ConfigError::Parse(e) => write!(f, "config parse error: {}", e),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_default_config() {
        let toml_str = r#"
[strategy.momentum_confirmation]
entry_minute_start = 8
entry_minute_end = 10
min_confidence = 0.70

[[strategy.momentum_confirmation.entry_tiers]]
minute = 8
threshold_pct = 0.10

[[strategy.momentum_confirmation.entry_tiers]]
minute = 9
threshold_pct = 0.08

[risk]
max_position_pct = 0.02
kelly_multiplier = 0.25
"#;

        let config = BacktestConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.strategy.momentum_confirmation.entry_minute_start, 8);
        assert_eq!(config.strategy.momentum_confirmation.entry_tiers.len(), 2);
        assert!((config.risk.kelly_multiplier - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tier_thresholds() {
        let config = MomentumConfig::default();
        let thresholds = config.tier_thresholds();
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[&8] - 0.10).abs() < 1e-10);
        assert!((thresholds[&9] - 0.08).abs() < 1e-10);
        assert!((thresholds[&10] - 0.05).abs() < 1e-10);
    }
}
