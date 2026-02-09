"""Tests for DataLoader — CSV parsing, gap detection, duplicate removal."""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path  # noqa: TCH003

import pytest

from src.backtesting.data_loader import DataLoader, DataLoadError


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class TestCsvLoading:
    def test_load_valid_csv(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T00:15:00,105,115,100,110,1200
            2024-01-01T00:30:00,110,120,105,115,900
        """)
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        assert len(candles) == 3
        assert candles[0].open == Decimal("100")
        assert candles[0].close == Decimal("105")
        assert candles[1].volume == Decimal("1200")

    def test_sorted_output(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:30:00,110,120,105,115,900
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T00:15:00,105,115,100,110,1200
        """)
        csv_file = tmp_path / "unsorted.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        for i in range(1, len(candles)):
            assert candles[i].timestamp >= candles[i - 1].timestamp

    def test_missing_file_raises(self, loader: DataLoader) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            loader.load_csv("/nonexistent/path.csv")

    def test_missing_columns_raises(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = "timestamp,open,high\n2024-01-01,100,110\n"
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(DataLoadError, match="missing required"):
            loader.load_csv(csv_file)

    def test_optional_columns(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume,exchange,symbol
            2024-01-01T00:00:00,100,110,95,105,1000,binance,BTC/USDT
        """)
        csv_file = tmp_path / "extra.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        assert candles[0].exchange == "binance"
        assert candles[0].symbol == "BTC/USDT"

    def test_epoch_timestamp(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            1704067200,100,110,95,105,1000
        """)
        csv_file = tmp_path / "epoch.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        assert len(candles) == 1
        assert candles[0].timestamp.year == 2024

    def test_bad_row_skipped(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            bad_timestamp,abc,110,95,105,1000
            2024-01-01T00:15:00,105,115,100,110,1200
        """)
        csv_file = tmp_path / "partial.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        assert len(candles) == 2


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_duplicates_removed(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T00:00:00,101,111,96,106,1100
            2024-01-01T00:15:00,105,115,100,110,1200
        """)
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        assert len(candles) == 2
        # First occurrence kept
        assert candles[0].open == Decimal("100")


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

class TestGapDetection:
    def test_no_gaps(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T00:15:00,105,115,100,110,1200
            2024-01-01T00:30:00,110,120,105,115,900
        """)
        csv_file = tmp_path / "nogap.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        gaps = loader.detect_gaps(candles)
        assert gaps == []

    def test_gap_detected(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T01:00:00,105,115,100,110,1200
        """)
        csv_file = tmp_path / "gap.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        gaps = loader.detect_gaps(candles)
        assert len(gaps) == 1

    def test_custom_interval(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
            2024-01-01T01:00:00,105,115,100,110,1200
            2024-01-01T03:00:00,110,120,105,115,900
        """)
        csv_file = tmp_path / "hourly.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        # 1h interval -> gap between 01:00 and 03:00 (2h > 1.5h tolerance)
        gaps = loader.detect_gaps(candles, expected_interval=timedelta(hours=1))
        assert len(gaps) == 1

        # 2h interval -> no gap (2h delta < 3h tolerance)
        gaps = loader.detect_gaps(candles, expected_interval=timedelta(hours=2))
        assert len(gaps) == 0

    def test_single_candle_no_gaps(self, loader: DataLoader, tmp_path: Path) -> None:
        csv_content = textwrap.dedent("""\
            timestamp,open,high,low,close,volume
            2024-01-01T00:00:00,100,110,95,105,1000
        """)
        csv_file = tmp_path / "single.csv"
        csv_file.write_text(csv_content)

        candles = loader.load_csv(csv_file)
        gaps = loader.detect_gaps(candles)
        assert gaps == []


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------

class TestOHLCVValidation:
    def test_high_lt_low_rejected_by_candle_model(self) -> None:
        """The Candle model itself rejects high < low."""
        from src.models.market import Candle

        with pytest.raises(ValueError, match="high must be >= low"):
            Candle(
                exchange="test",
                symbol="TEST",
                open=Decimal("100"),
                high=Decimal("90"),
                low=Decimal("110"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                timestamp=datetime(2024, 1, 1),
            )
