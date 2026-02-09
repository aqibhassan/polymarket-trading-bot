pub mod candle;
pub mod config;
pub mod signal;
pub mod window;

pub use candle::CandleStore;
pub use config::{BacktestConfig, FalseSentimentConfig, MomentumConfig, RiskConfig, SingularityConfig};
pub use signal::{Confidence, ExitReason, Side, Signal, SignalType};
pub use window::Window;
