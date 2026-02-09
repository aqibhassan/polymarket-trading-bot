"""Backtesting framework for MVHE — event-driven simulation and analysis."""

from __future__ import annotations

from src.backtesting.data_loader import DataLoader
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.fill_simulator import FillSimulator, SimulatedFill
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.reporter import BacktestReporter
from src.backtesting.validator import StatisticalValidator, ValidationResult

__all__ = [
    "BacktestEngine",
    "BacktestReporter",
    "BacktestResult",
    "DataLoader",
    "FillSimulator",
    "MetricsCalculator",
    "SimulatedFill",
    "StatisticalValidator",
    "ValidationResult",
]
