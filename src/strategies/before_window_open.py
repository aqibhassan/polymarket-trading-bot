"""Before Window Open (BWO) Strategy — enter pre-window at ~$0.50 fair value.

Unlike MomentumConfirmation which enters at minutes 8-10 at $0.65-$0.80,
BWO enters BEFORE minute 0 when price is near $0.50 (or up to $0.60 early).
At $0.50 entry, break-even is ~52% after fees.

Uses pre-window BTC signals (prior candle momentum, multi-TF alignment,
short-term carry, volatility regime, candle patterns, volume profile)
to predict next 15m candle direction.

Supports two signal modes:
  1. Ensemble (default): hand-crafted weighted vote of pre-window BTC features.
  2. ML model: trained classifier loaded from joblib, uses all 31 features
     (pre-window + early-window + TA indicators) to predict direction.

Hold-to-settlement exit (same as momentum strategy).
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.logging import get_logger
from src.models.signal import Confidence, ExitReason, Signal, SignalType
from src.strategies.base import BaseStrategy
from src.strategies.registry import register

if TYPE_CHECKING:
    from src.config.loader import ConfigLoader
    from src.models.market import MarketState, OrderBookSnapshot, Position
    from src.models.order import Fill

logger = get_logger(__name__)

# Trading session UTC hour ranges (must match backtest_before_window.py)
SESSIONS: dict[str, range] = {
    "asia": range(0, 8),
    "europe": range(8, 14),
    "us": range(14, 22),
    "late": range(22, 24),
}


def _prior_window_momentum(
    prior_candles: list[Any],
) -> tuple[float, float]:
    """Prior 15m candle direction and magnitude.

    Returns (direction: -1/0/+1, magnitude: abs return).
    """
    if not prior_candles or len(prior_candles) < 2:
        return 0.0, 0.0
    open_p = float(prior_candles[0].open)
    close_p = float(prior_candles[-1].close)
    if open_p == 0:
        return 0.0, 0.0
    ret = (close_p - open_p) / open_p
    direction = 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)
    return direction, abs(ret)


def _multi_tf_alignment(
    recent_candles: list[Any],
) -> float:
    """Multi-timeframe alignment score from recent 1m candles.

    Computes 15m, 1h, 4h trend directions and returns alignment (-3 to +3).
    """
    if len(recent_candles) < 60:
        return 0.0

    def _direction(candles: list[Any]) -> float:
        o = float(candles[0].open)
        c = float(candles[-1].close)
        if o == 0:
            return 0.0
        ret = (c - o) / o
        return 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)

    d15 = _direction(recent_candles[-15:])
    d60 = _direction(recent_candles[-60:])
    d240 = _direction(recent_candles[-240:]) if len(recent_candles) >= 240 else _direction(recent_candles)
    return d15 + d60 + d240


def _short_term_momentum(
    recent_candles: list[Any],
) -> tuple[float, float]:
    """Last 5 1m candles direction and strength.

    Returns (direction: -1/0/+1, strength: proportion green 0-1).
    """
    if len(recent_candles) < 5:
        return 0.0, 0.0
    last5 = recent_candles[-5:]
    o = float(last5[0].open)
    c = float(last5[-1].close)
    if o == 0:
        return 0.0, 0.0
    ret = (c - o) / o
    direction = 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)
    green_count = sum(1 for cn in last5 if float(cn.close) > float(cn.open))
    return direction, green_count / 5.0


def _volatility_regime(
    recent_candles: list[Any],
) -> float:
    """Realized vol percentile. Returns 1.0 (high), -1.0 (low), or 0.0."""
    if len(recent_candles) < 60:
        return 0.0
    returns = []
    for i in range(1, len(recent_candles)):
        prev_c = float(recent_candles[i - 1].close)
        if prev_c > 0:
            r = (float(recent_candles[i].close) - prev_c) / prev_c
            returns.append(r)
    if len(returns) < 30:
        return 0.0
    recent_r = returns[-15:]
    recent_vol = math.sqrt(sum(r * r for r in recent_r) / len(recent_r))
    window_size = 15
    vol_values = []
    for i in range(window_size, len(returns)):
        chunk = returns[i - window_size:i]
        v = math.sqrt(sum(r * r for r in chunk) / len(chunk))
        vol_values.append(v)
    if not vol_values:
        return 0.0
    pct = sum(1 for v in vol_values if v <= recent_vol) / len(vol_values)
    if pct > 0.75:
        return 1.0
    if pct < 0.25:
        return -1.0
    return 0.0


def _candle_streak(recent_candles: list[Any]) -> tuple[float, float]:
    """Consecutive same-direction streak from end. Returns (length, direction)."""
    if len(recent_candles) < 2:
        return 0.0, 0.0
    streak = 1
    last_dir = 1.0 if float(recent_candles[-1].close) > float(recent_candles[-1].open) else -1.0
    for i in range(len(recent_candles) - 2, max(len(recent_candles) - 11, -1), -1):
        c = recent_candles[i]
        d = 1.0 if float(c.close) > float(c.open) else -1.0
        if d == last_dir:
            streak += 1
        else:
            break
    return float(streak), last_dir


def _compute_early_window_features(
    early_candles: list[Any],
) -> dict[str, float]:
    """Compute early-window features from candles 0..entry_minute.

    Returns dict with keys: early_cum_return, early_direction, early_magnitude,
    early_green_ratio, early_vol, early_max_move.
    """
    features: dict[str, float] = {}
    if not early_candles:
        return {
            "early_cum_return": 0.0, "early_direction": 0.0,
            "early_magnitude": 0.0, "early_green_ratio": 0.0,
            "early_vol": 0.0, "early_max_move": 0.0,
        }

    open_price = float(early_candles[0].open)
    close_price = float(early_candles[-1].close)
    if open_price != 0:
        cum_return = (close_price - open_price) / open_price
    else:
        cum_return = 0.0

    features["early_cum_return"] = cum_return
    features["early_direction"] = 1.0 if cum_return > 0 else (-1.0 if cum_return < 0 else 0.0)
    features["early_magnitude"] = abs(cum_return)

    green_count = sum(1 for c in early_candles if float(c.close) > float(c.open))
    features["early_green_ratio"] = green_count / len(early_candles)

    # Realized volatility of early candles
    if len(early_candles) >= 2:
        sq_sum = 0.0
        for j in range(1, len(early_candles)):
            prev_c = float(early_candles[j - 1].close)
            if prev_c > 0:
                r = (float(early_candles[j].close) - prev_c) / prev_c
                sq_sum += r * r
        features["early_vol"] = math.sqrt(sq_sum / (len(early_candles) - 1)) if sq_sum > 0 else 0.0
    else:
        features["early_vol"] = 0.0

    # Max single-candle move
    max_move = 0.0
    for c in early_candles:
        o = float(c.open)
        if o != 0:
            move = abs((float(c.close) - o) / o)
            if move > max_move:
                max_move = move
    features["early_max_move"] = max_move

    return features


def _compute_ta_features(recent_candles: list[Any]) -> dict[str, float]:
    """Compute TA indicator features from pre-window 1m candle history.

    Returns dict with keys: rsi_14, macd_histogram_sign, bb_pct_b,
    atr_14, mean_reversion_z, price_vs_vwap.
    """
    features: dict[str, float] = {}
    hist_len = len(recent_candles)

    # --- RSI(14) ---
    if hist_len >= 15:
        gains = 0.0
        losses = 0.0
        for j in range(hist_len - 14, hist_len):
            delta = float(recent_candles[j].close) - float(recent_candles[j - 1].close)
            if delta > 0:
                gains += delta
            else:
                losses += abs(delta)
        avg_gain = gains / 14.0
        avg_loss = losses / 14.0
        if avg_loss == 0:
            features["rsi_14"] = 100.0
        else:
            rs = avg_gain / avg_loss
            features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    else:
        features["rsi_14"] = 50.0

    # --- MACD(12,26) histogram sign ---
    if hist_len >= 34:
        closes = [float(c.close) for c in recent_candles]

        def _ema(data: list[float], period: int) -> float:
            if len(data) < period:
                return data[-1] if data else 0.0
            k = 2.0 / (period + 1)
            ema_val = sum(data[:period]) / period
            for val in data[period:]:
                ema_val = val * k + ema_val * (1 - k)
            return ema_val

        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = ema12 - ema26
        features["macd_histogram_sign"] = 1.0 if macd_line > 0 else (-1.0 if macd_line < 0 else 0.0)
    else:
        features["macd_histogram_sign"] = 0.0

    # --- Bollinger %B(20,2) ---
    if hist_len >= 20:
        bb_closes = [float(recent_candles[j].close) for j in range(hist_len - 20, hist_len)]
        bb_mean = sum(bb_closes) / 20.0
        bb_var = sum((c - bb_mean) ** 2 for c in bb_closes) / 20.0
        bb_std = math.sqrt(bb_var) if bb_var > 0 else 0.001
        upper_bb = bb_mean + 2 * bb_std
        lower_bb = bb_mean - 2 * bb_std
        band_width = upper_bb - lower_bb
        if band_width > 0:
            features["bb_pct_b"] = (bb_closes[-1] - lower_bb) / band_width
        else:
            features["bb_pct_b"] = 0.5
    else:
        features["bb_pct_b"] = 0.5

    # --- ATR(14) ---
    if hist_len >= 15:
        atr_sum = 0.0
        for j in range(hist_len - 14, hist_len):
            c = recent_candles[j]
            prev_close = float(recent_candles[j - 1].close)
            high = float(c.high)
            low = float(c.low)
            close = float(c.close)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            atr_sum += tr
        features["atr_14"] = atr_sum / 14.0
    else:
        features["atr_14"] = 0.0

    # --- Mean reversion z-score (vs 60-candle SMA) ---
    if hist_len >= 60:
        mr_closes = [float(recent_candles[j].close) for j in range(hist_len - 60, hist_len)]
        mr_mean = sum(mr_closes) / 60.0
        mr_var = sum((c - mr_mean) ** 2 for c in mr_closes) / 60.0
        mr_std = math.sqrt(mr_var) if mr_var > 0 else 0.001
        features["mean_reversion_z"] = (mr_closes[-1] - mr_mean) / mr_std
    else:
        features["mean_reversion_z"] = 0.0

    # --- Price vs VWAP (60-candle) ---
    if hist_len >= 60:
        vwap_pv = 0.0
        vwap_vol = 0.0
        for j in range(hist_len - 60, hist_len):
            c = recent_candles[j]
            typical = (float(c.high) + float(c.low) + float(c.close)) / 3.0
            vwap_pv += typical * float(c.volume)
            vwap_vol += float(c.volume)
        if vwap_vol > 0:
            vwap = vwap_pv / vwap_vol
            current_close = float(recent_candles[-1].close)
            features["price_vs_vwap"] = (current_close - vwap) / vwap if vwap != 0 else 0.0
        else:
            features["price_vs_vwap"] = 0.0
    else:
        features["price_vs_vwap"] = 0.0

    return features


def _compute_vol_profile_features(recent_candles: list[Any]) -> dict[str, float]:
    """Compute volume profile features: vol_ratio and vol_dir_align.

    Mirrors the backtest Feature 7 logic using recent_1m_candles.
    """
    hist_len = len(recent_candles)
    if hist_len < 60:
        return {"vol_ratio": 1.0, "vol_dir_align": 0.0}

    total_vol = sum(float(c.volume) for c in recent_candles)
    avg_vol = total_vol / hist_len if hist_len > 0 else 1.0
    if avg_vol == 0:
        avg_vol = 1.0

    recent_15 = recent_candles[-15:]
    recent_vol_sum = sum(float(c.volume) for c in recent_15)
    up_vol = sum(float(c.volume) for c in recent_15 if float(c.close) > float(c.open))
    down_vol = sum(float(c.volume) for c in recent_15 if float(c.close) < float(c.open))
    recent_avg = recent_vol_sum / 15.0

    total_dir_vol = up_vol + down_vol
    vol_dir_align = (up_vol - down_vol) / total_dir_vol if total_dir_vol > 0 else 0.0

    return {
        "vol_ratio": recent_avg / avg_vol,
        "vol_dir_align": vol_dir_align,
    }


def _compute_vol_percentile(recent_candles: list[Any]) -> float:
    """Compute vol_percentile feature: min(recent_vol / full_vol / 2.0, 1.0)."""
    hist_len = len(recent_candles)
    if hist_len < 60:
        return 0.5

    # Recent 14-candle volatility
    recent_sq = 0.0
    for j in range(hist_len - 14, hist_len):
        prev_c = float(recent_candles[j - 1].close)
        if prev_c > 0:
            r = (float(recent_candles[j].close) - prev_c) / prev_c
            recent_sq += r * r
    recent_vol = math.sqrt(recent_sq / 14) if recent_sq > 0 else 0.0

    full_sq = 0.0
    full_n = 0
    for j in range(1, hist_len):
        prev_c = float(recent_candles[j - 1].close)
        if prev_c > 0:
            r = (float(recent_candles[j].close) - prev_c) / prev_c
            full_sq += r * r
            full_n += 1
    full_vol = math.sqrt(full_sq / full_n) if full_n > 0 else 0.001

    vol_ratio = recent_vol / full_vol if full_vol > 0 else 1.0
    return min(vol_ratio / 2.0, 1.0)


def _compute_mtf_individual(recent_candles: list[Any]) -> dict[str, float]:
    """Compute individual MTF directions: mtf_15m, mtf_1h, mtf_4h."""
    hist_len = len(recent_candles)
    if hist_len < 60:
        return {"mtf_15m": 0.0, "mtf_1h": 0.0, "mtf_4h": 0.0}

    def _direction(candles: list[Any]) -> float:
        o = float(candles[0].open)
        c = float(candles[-1].close)
        if o == 0:
            return 0.0
        ret = (c - o) / o
        return 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)

    d15 = _direction(recent_candles[-15:])
    d60 = _direction(recent_candles[-60:])
    d240 = _direction(recent_candles[-240:]) if hist_len >= 240 else _direction(recent_candles)
    return {"mtf_15m": d15, "mtf_1h": d60, "mtf_4h": d240}


@register("before_window_open")
class BeforeWindowOpenStrategy(BaseStrategy):
    """BWO: Enter pre-window at fair value using ensemble of pre-window BTC signals.

    Supports two signal modes:
      - Ensemble (default): hand-crafted weighted vote of pre-window features.
      - ML model: trained classifier loaded from joblib; uses all 31 features.

    Entry: Before minute 0 (or early minutes 0-3/5 if price <= max_entry_price).
    Exit: Hold to settlement (binary $0 or $1).
    """

    REQUIRED_PARAMS: ClassVar[list[str]] = [
        "strategy.before_window_open.min_confidence",
    ]

    # Feature names in the exact order expected by the ML model.
    # Must match compute_all_features() output order in backtest_before_window.py.
    FEATURE_NAMES: ClassVar[list[str]] = [
        "prior_dir", "prior_mag",
        "mtf_score", "mtf_15m", "mtf_1h", "mtf_4h",
        "stm_dir", "stm_strength",
        "vol_regime", "vol_percentile",
        "tod_hour", "tod_asia", "tod_europe", "tod_us", "tod_late",
        "streak_len", "streak_dir",
        "vol_ratio", "vol_dir_align",
        "early_cum_return", "early_direction", "early_magnitude",
        "early_green_ratio", "early_vol", "early_max_move",
        "rsi_14", "macd_histogram_sign", "bb_pct_b",
        "atr_14", "mean_reversion_z", "price_vs_vwap",
    ]

    def __init__(self, config: ConfigLoader, strategy_id: str | None = None) -> None:
        super().__init__(config, strategy_id or "before_window_open")

        # Signal weights (ensemble mode)
        self._w_prior = float(config.get("strategy.before_window_open.weight_prior_momentum", 0.25))
        self._w_mtf = float(config.get("strategy.before_window_open.weight_multi_tf", 0.20))
        self._w_stm = float(config.get("strategy.before_window_open.weight_short_momentum", 0.20))
        self._w_pattern = float(config.get("strategy.before_window_open.weight_candle_pattern", 0.15))
        self._w_volume = float(config.get("strategy.before_window_open.weight_volume", 0.10))
        self._w_vol_regime = float(config.get("strategy.before_window_open.weight_vol_regime", 0.10))

        # Entry parameters (ensemble mode)
        self._min_confidence = float(config.get("strategy.before_window_open.min_confidence", 0.40))
        self._max_entry_price = float(config.get("strategy.before_window_open.max_entry_price", 0.60))
        self._entry_minute_end = int(config.get("strategy.before_window_open.entry_minute_end", 3))

        # Ensemble voting threshold
        self._min_vote_weight = float(config.get("strategy.before_window_open.min_vote_weight", 0.40))

        # MTF alignment threshold (abs score must be >= this)
        self._mtf_min_alignment = float(config.get("strategy.before_window_open.mtf_min_alignment", 2.0))

        # Short-term momentum strength threshold
        self._stm_min_strength = float(config.get("strategy.before_window_open.stm_min_strength", 0.6))

        # Candle pattern min streak
        self._pattern_min_streak = int(config.get("strategy.before_window_open.pattern_min_streak", 3))

        # Volume directional alignment threshold
        self._volume_min_alignment = float(config.get("strategy.before_window_open.volume_min_alignment", 0.3))

        # Exit
        self._resolution_guard_minute = int(
            config.get("strategy.before_window_open.resolution_guard_minute", 14)
        )
        self._hold_to_settlement = bool(
            config.get("strategy.before_window_open.hold_to_settlement", True)
        )
        self._max_position_pct = float(
            config.get("strategy.before_window_open.max_position_pct", 0.05)
        )

        # --- ML model configuration ---
        self._use_ml_model = bool(
            config.get("strategy.before_window_open.use_ml_model", False)
        )
        self._model_path = str(
            config.get("strategy.before_window_open.model_path", "models/bwo_model_min3.joblib")
        )
        self._ml_entry_minute = int(
            config.get("strategy.before_window_open.entry_minute", 3)
        )
        self._ml_confidence_threshold = float(
            config.get("strategy.before_window_open.ml_confidence_threshold", 0.70)
        )
        self._ml_price_sensitivity = float(
            config.get("strategy.before_window_open.ml_price_sensitivity", 0.5)
        )
        self._ml_max_entry_price = float(
            config.get("strategy.before_window_open.ml_max_entry_price", 0.70)
        )

        # Load ML model if enabled
        self._ml_model: Any = None
        if self._use_ml_model:
            self._ml_model = self._load_ml_model()

        # Runtime state
        self._last_evaluation: dict[str, Any] = {}

    def _load_ml_model(self) -> Any:
        """Load trained ML model from joblib file.

        Returns the model object, or None if loading fails.
        """
        model_file = Path(self._model_path)
        if not model_file.is_absolute():
            # Resolve relative to project root (two levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            model_file = project_root / self._model_path

        if not model_file.exists():
            logger.warning(
                "ml_model_not_found",
                path=str(model_file),
                fallback="ensemble",
            )
            return None

        try:
            import joblib  # noqa: PLC0415 — lazy import to avoid hard dependency

            model = joblib.load(model_file)
            logger.info(
                "ml_model_loaded",
                path=str(model_file),
                model_type=type(model).__name__,
            )
            return model
        except ImportError:
            logger.warning("joblib_not_installed", fallback="ensemble")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ml_model_load_error",
                path=str(model_file),
                error=str(exc),
                fallback="ensemble",
            )
            return None

    def generate_signals(
        self,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
        context: dict[str, Any],
    ) -> list[Signal]:
        """Generate BWO signals using pre-window BTC data.

        When ``use_ml_model`` is enabled and the model is loaded, uses the ML
        prediction path. Otherwise falls back to the hand-crafted ensemble.

        Expected context keys:
            prior_window_candles: list of 1m candles from the previous 15m window
            recent_1m_candles: list of recent 1m candles (up to 1440 for history)
            early_window_candles: list of 1m candles from minute 0..current within
                the current 15m window (used by ML mode for early-window features)
            minute_in_window: current minute within the 15m window (0-14)
            yes_price: current Polymarket YES token price
            window_timestamp: datetime of window start (used for time-of-day features)
        """
        signals: list[Signal] = []
        market_id = market_state.market_id
        minute_in_window = int(context.get("minute_in_window", 0))
        yes_price = float(context.get("yes_price", 0.5))

        # --- Exit checks for open positions ---
        for pos in self.open_positions:
            exit_signal = self._check_exit(pos, minute_in_window)
            if exit_signal is not None:
                signals.append(exit_signal)

        # --- Route to ML or ensemble signal generation ---
        if self._ml_model is not None:
            ml_signals = self._generate_ml_signals(
                market_state, context, signals, market_id,
                minute_in_window, yes_price,
            )
            return ml_signals

        return self._generate_ensemble_signals(
            context, signals, market_id, minute_in_window, yes_price,
        )

    # ------------------------------------------------------------------
    # ML signal generation
    # ------------------------------------------------------------------

    def _generate_ml_signals(
        self,
        market_state: MarketState,
        context: dict[str, Any],
        signals: list[Signal],
        market_id: str,
        minute_in_window: int,
        yes_price: float,
    ) -> list[Signal]:
        """Generate entry signal using the trained ML model.

        Only fires at the exact ``entry_minute`` configured for the model.
        """
        # ML mode: only trigger at the configured entry minute (not before, not after)
        if minute_in_window != self._ml_entry_minute:
            self._last_evaluation = {
                "outcome": "skip", "reason": "ml_not_entry_minute",
                "minute": minute_in_window, "market_id": market_id,
                "ml_entry_minute": self._ml_entry_minute,
                "mode": "ml",
            }
            return signals

        # Price gate: use ML-specific max entry price
        if yes_price > self._ml_max_entry_price and (1.0 - yes_price) > self._ml_max_entry_price:
            self._last_evaluation = {
                "outcome": "skip", "reason": "price_too_high",
                "minute": minute_in_window, "market_id": market_id,
                "yes_price": yes_price,
                "mode": "ml",
            }
            return signals

        # Build feature vector
        feature_dict = self._compute_ml_features(context, minute_in_window)
        if feature_dict is None:
            self._last_evaluation = {
                "outcome": "skip", "reason": "ml_feature_error",
                "minute": minute_in_window, "market_id": market_id,
                "mode": "ml",
            }
            return signals

        try:
            import numpy as np  # noqa: PLC0415

            feature_vector = np.array(
                [[feature_dict.get(name, 0.0) for name in self.FEATURE_NAMES]]
            )

            # Get class probabilities
            proba = self._ml_model.predict_proba(feature_vector)[0]
            # Assume model classes are [0, 1] where 1 = "Up"
            classes = list(self._ml_model.classes_)
            if 1 in classes:
                up_idx = classes.index(1)
            else:
                up_idx = 0
            confidence_up = float(proba[up_idx])
            confidence_down = 1.0 - confidence_up

            # Direction from early BTC movement
            early_dir = feature_dict.get("early_direction", 0.0)
            if early_dir > 0:
                ml_direction = "Up"
                ml_confidence = confidence_up
            elif early_dir < 0:
                ml_direction = "Down"
                ml_confidence = confidence_down
            else:
                # Flat early movement — use model's stronger side
                if confidence_up >= confidence_down:
                    ml_direction = "Up"
                    ml_confidence = confidence_up
                else:
                    ml_direction = "Down"
                    ml_confidence = confidence_down

            logger.info(
                "bwo_ml_prediction",
                market_id=market_id,
                direction=ml_direction,
                confidence=round(ml_confidence, 4),
                threshold=self._ml_confidence_threshold,
                minute=minute_in_window,
                early_direction=early_dir,
            )

            # Confidence gate
            if ml_confidence < self._ml_confidence_threshold:
                self._last_evaluation = {
                    "outcome": "skip", "reason": "ml_low_confidence",
                    "minute": minute_in_window, "market_id": market_id,
                    "ml_confidence": round(ml_confidence, 4),
                    "threshold": self._ml_confidence_threshold,
                    "mode": "ml",
                }
                return signals

            # Entry price adjustment based on early-window return magnitude
            early_cum_return = feature_dict.get("early_cum_return", 0.0)
            raw_entry = 0.50 + abs(early_cum_return) * self._ml_price_sensitivity
            adjusted_entry = min(raw_entry, self._ml_max_entry_price)

            from src.models.market import Side

            if ml_direction == "Up":
                direction = Side.YES
                entry_price = Decimal(str(round(adjusted_entry, 4)))
            else:
                direction = Side.NO
                entry_price = Decimal(str(round(adjusted_entry, 4)))

            confidence_obj = Confidence(
                trend_strength=round(feature_dict.get("prior_mag", 0.0), 4),
                threshold_exceedance=round(ml_confidence, 4),
                overall=round(ml_confidence, 4),
            )

            self._last_evaluation = {
                "outcome": "entry", "direction": ml_direction,
                "minute": minute_in_window, "market_id": market_id,
                "ml_confidence": round(ml_confidence, 4),
                "entry_price": float(entry_price),
                "mode": "ml",
                "features": {k: round(v, 6) for k, v in feature_dict.items()},
            }

            logger.info(
                "bwo_ml_entry",
                market_id=market_id,
                direction=ml_direction,
                confidence=round(ml_confidence, 4),
                minute=minute_in_window,
                entry_price=float(entry_price),
            )

            signals.append(Signal(
                strategy_id=self.strategy_id,
                market_id=market_id,
                signal_type=SignalType.ENTRY,
                direction=direction,
                strength=Decimal(str(round(ml_confidence, 4))),
                confidence=confidence_obj,
                entry_price=entry_price,
                stop_loss=Decimal("0"),
                take_profit=Decimal("1"),
                metadata={
                    "direction": ml_direction.lower(),
                    "ml_confidence": str(round(ml_confidence, 4)),
                    "entry_minute": str(minute_in_window),
                    "strategy_type": "before_window_open",
                    "mode": "ml",
                },
            ))
            return signals

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bwo_ml_predict_error",
                market_id=market_id,
                error=str(exc),
                fallback="ensemble",
            )
            # Fall back to ensemble on ML prediction error
            return self._generate_ensemble_signals(
                context, signals, market_id, minute_in_window, yes_price,
            )

    def _compute_ml_features(
        self,
        context: dict[str, Any],
        minute_in_window: int,
    ) -> dict[str, float] | None:
        """Build the full feature dict for ML prediction.

        Mirrors compute_all_features() from the backtest script, adapted to the
        live context data structures.

        Returns None if essential data is missing.
        """
        prior_candles = context.get("prior_window_candles", [])
        recent_candles = context.get("recent_1m_candles", [])
        early_candles = context.get("early_window_candles", [])
        window_ts = context.get("window_timestamp")

        features: dict[str, float] = {}

        # --- Pre-window features ---

        # Feature 1: Prior window momentum
        prior_dir, prior_mag = _prior_window_momentum(prior_candles)
        features["prior_dir"] = prior_dir
        features["prior_mag"] = prior_mag

        # Feature 2: Multi-timeframe alignment
        mtf_score = _multi_tf_alignment(recent_candles)
        features["mtf_score"] = mtf_score
        mtf_indiv = _compute_mtf_individual(recent_candles)
        features.update(mtf_indiv)

        # Feature 3: Short-term momentum
        stm_dir, stm_strength = _short_term_momentum(recent_candles)
        features["stm_dir"] = stm_dir
        features["stm_strength"] = stm_strength

        # Feature 4: Volatility regime
        vol_regime = _volatility_regime(recent_candles)
        features["vol_regime"] = vol_regime
        features["vol_percentile"] = _compute_vol_percentile(recent_candles)

        # Feature 5: Time of day
        if window_ts is not None:
            hour = window_ts.hour
        else:
            hour = 12  # default fallback
        features["tod_hour"] = float(hour)
        features["tod_asia"] = 1.0 if hour in SESSIONS["asia"] else 0.0
        features["tod_europe"] = 1.0 if hour in SESSIONS["europe"] else 0.0
        features["tod_us"] = 1.0 if hour in SESSIONS["us"] else 0.0
        features["tod_late"] = 1.0 if hour in SESSIONS["late"] else 0.0

        # Feature 6: Candle pattern (streak)
        streak_len, streak_dir = _candle_streak(recent_candles)
        features["streak_len"] = streak_len
        features["streak_dir"] = streak_dir

        # Feature 7: Volume profile
        vol_features = _compute_vol_profile_features(recent_candles)
        features.update(vol_features)

        # Feature 8: Early-window features
        early_features = _compute_early_window_features(early_candles)
        features.update(early_features)

        # Feature 9-14: TA indicators
        ta_features = _compute_ta_features(recent_candles)
        features.update(ta_features)

        return features

    # ------------------------------------------------------------------
    # Ensemble signal generation (original logic, unchanged)
    # ------------------------------------------------------------------

    def _generate_ensemble_signals(
        self,
        context: dict[str, Any],
        signals: list[Signal],
        market_id: str,
        minute_in_window: int,
        yes_price: float,
    ) -> list[Signal]:
        """Generate entry signals using the hand-crafted ensemble vote.

        This is the original BWO signal logic, extracted into its own method
        so it can be called as the default path or as a fallback from ML mode.
        """
        # --- Entry gate: only trade before/at early window minutes ---
        if minute_in_window > self._entry_minute_end:
            self._last_evaluation = {
                "outcome": "skip", "reason": "past_entry_window",
                "minute": minute_in_window, "market_id": market_id,
            }
            return signals

        # Price gate: don't buy tokens above max entry price
        if yes_price > self._max_entry_price and (1.0 - yes_price) > self._max_entry_price:
            self._last_evaluation = {
                "outcome": "skip", "reason": "price_too_high",
                "minute": minute_in_window, "market_id": market_id,
                "yes_price": yes_price,
            }
            return signals

        prior_candles = context.get("prior_window_candles", [])
        recent_candles = context.get("recent_1m_candles", [])

        # --- Compute features ---
        prior_dir, prior_mag = _prior_window_momentum(prior_candles)
        mtf_score = _multi_tf_alignment(recent_candles)
        stm_dir, stm_strength = _short_term_momentum(recent_candles)
        vol_regime = _volatility_regime(recent_candles)
        streak_len, streak_dir = _candle_streak(recent_candles)

        # Volume alignment from recent candles
        vol_align = 0.0
        if len(recent_candles) >= 15:
            recent_15 = recent_candles[-15:]
            up_vol = sum(float(c.volume) for c in recent_15 if float(c.close) > float(c.open))
            down_vol = sum(float(c.volume) for c in recent_15 if float(c.close) < float(c.open))
            total_dir_vol = up_vol + down_vol
            if total_dir_vol > 0:
                vol_align = (up_vol - down_vol) / total_dir_vol

        # --- Ensemble weighted vote ---
        votes: dict[str, float] = {}

        # Prior momentum
        if prior_dir != 0:
            d = "Up" if prior_dir > 0 else "Down"
            votes[d] = votes.get(d, 0.0) + self._w_prior

        # Multi-TF alignment
        if abs(mtf_score) >= self._mtf_min_alignment:
            d = "Up" if mtf_score > 0 else "Down"
            votes[d] = votes.get(d, 0.0) + self._w_mtf

        # Short-term momentum
        if stm_dir != 0 and (
            (stm_dir > 0 and stm_strength >= self._stm_min_strength)
            or (stm_dir < 0 and (1.0 - stm_strength) >= self._stm_min_strength)
        ):
            d = "Up" if stm_dir > 0 else "Down"
            votes[d] = votes.get(d, 0.0) + self._w_stm

        # Candle pattern
        if streak_dir != 0 and streak_len >= self._pattern_min_streak:
            d = "Up" if streak_dir > 0 else "Down"
            votes[d] = votes.get(d, 0.0) + self._w_pattern

        # Volume alignment
        if abs(vol_align) > self._volume_min_alignment:
            d = "Up" if vol_align > 0 else "Down"
            votes[d] = votes.get(d, 0.0) + self._w_volume

        # Vol regime boost
        if vol_regime == 1.0 and votes:
            leading = max(votes, key=votes.get)  # type: ignore[arg-type]
            votes[leading] = votes.get(leading, 0.0) + self._w_vol_regime

        if not votes:
            self._last_evaluation = {
                "outcome": "skip", "reason": "no_signals",
                "minute": minute_in_window, "market_id": market_id,
            }
            return signals

        leading_dir = max(votes, key=votes.get)  # type: ignore[arg-type]
        vote_weight = votes[leading_dir]

        if vote_weight < self._min_vote_weight:
            self._last_evaluation = {
                "outcome": "skip", "reason": "low_conviction",
                "minute": minute_in_window, "market_id": market_id,
                "vote_weight": round(vote_weight, 4),
            }
            return signals

        # --- Generate entry signal ---
        from src.models.market import Side

        if leading_dir == "Up":
            direction = Side.YES
            entry_price = Decimal(str(yes_price))
        else:
            direction = Side.NO
            entry_price = Decimal(str(1.0 - yes_price))

        confidence_val = min(vote_weight, 1.0)
        if confidence_val < self._min_confidence:
            self._last_evaluation = {
                "outcome": "skip", "reason": "low_confidence",
                "minute": minute_in_window, "market_id": market_id,
                "confidence": round(confidence_val, 4),
            }
            return signals

        confidence_obj = Confidence(
            trend_strength=round(prior_mag, 4),
            threshold_exceedance=round(abs(mtf_score) / 3.0, 4),
            overall=round(confidence_val, 4),
        )

        self._last_evaluation = {
            "outcome": "entry", "direction": leading_dir,
            "minute": minute_in_window, "market_id": market_id,
            "vote_weight": round(vote_weight, 4),
            "features": {
                "prior_dir": prior_dir, "mtf_score": mtf_score,
                "stm_dir": stm_dir, "vol_regime": vol_regime,
                "streak_len": streak_len, "vol_align": round(vol_align, 4),
            },
        }

        logger.info(
            "bwo_entry",
            market_id=market_id,
            direction=leading_dir,
            vote_weight=round(vote_weight, 4),
            minute=minute_in_window,
            entry_price=float(entry_price),
        )

        signals.append(Signal(
            strategy_id=self.strategy_id,
            market_id=market_id,
            signal_type=SignalType.ENTRY,
            direction=direction,
            strength=Decimal(str(round(confidence_val, 4))),
            confidence=confidence_obj,
            entry_price=entry_price,
            stop_loss=Decimal("0"),
            take_profit=Decimal("1"),
            metadata={
                "direction": leading_dir.lower(),
                "vote_weight": str(round(vote_weight, 4)),
                "entry_minute": str(minute_in_window),
                "strategy_type": "before_window_open",
            },
        ))
        return signals

    def _check_exit(self, position: Position, minute_in_window: int) -> Signal | None:
        """Check exit conditions. BWO holds to settlement by default."""
        if minute_in_window >= self._resolution_guard_minute:
            logger.info(
                "bwo_exit_resolution_guard",
                market_id=position.market_id,
                minute=minute_in_window,
            )
            return Signal(
                strategy_id=self.strategy_id,
                market_id=position.market_id,
                signal_type=SignalType.EXIT,
                direction=position.side,
                exit_reason=ExitReason.RESOLUTION_GUARD,
                metadata={"minute": str(minute_in_window)},
            )
        return None

    def on_fill(self, fill: Fill, position: Position) -> None:
        logger.info(
            "bwo_fill",
            strategy=self.strategy_id,
            order_id=str(fill.order_id),
            price=float(fill.price),
            size=float(fill.size),
            market_id=position.market_id,
        )

    def on_cancel(self, order_id: str, reason: str) -> None:
        logger.info(
            "bwo_cancel",
            strategy=self.strategy_id,
            order_id=order_id,
            reason=reason,
        )
