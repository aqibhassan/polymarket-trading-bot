pub mod engine;
pub mod fill_sim;
pub mod metrics;
pub mod parallel;
pub mod validator;

pub use engine::{BacktestEngine, BacktestResult, Trade};
pub use fill_sim::{FillSimulator, SimulatedFill};
pub use metrics::MetricsCalculator;
pub use parallel::ParallelRunner;
pub use validator::{StatisticalValidator, ValidationResult};
