"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  R_100 ENSEMBLE BOT  v2  —  Rise / Fall  —  Railway                        ║
║                                                                              ║
║  ARCHITECTURE                                                                ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Multi-timeframe candle engine  (1m / 2m / 3m / 5m)                        ║
║  Three independent strategies:                                               ║
║    1. Trend        — MACD histogram + velocity + ATR expansion              ║
║    2. Mean Revert  — RSI extremes (no TA-Lib required)                      ║
║    3. Breakout     — Price compression → expansion                          ║
║  Regime detector   — TRENDING / RANGING / CALM / CHAOS                     ║
║  Volatility guard  — blocks trades in chaotic / balanced markets            ║
║  Ensemble combiner — regime-weighted vote across all three strategies       ║
║                                                                              ║
║  NEW IN v2 — DISTRIBUTION INTELLIGENCE                                      ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  FeatureCalibrator — live adaptive regime thresholds (candle-based)        ║
║    • Samples atr_slope, |velocity|, |acceleration| from each 1m candle     ║
║    • After warmup (50 candles) sets CHAOS/TRENDING/CALM boundaries from    ║
║      observed percentiles of R_100's actual feature distributions          ║
║    • Re-calibrates every 50 candles (~50 min) — aggressive, stays current  ║
║    • Works on any symbol: R_10, R_25, R_100, 1HZ10V etc.                  ║
║    • Replaces all hardcoded thresholds in RegimeDetector +                 ║
║      VolatilityGuardian — both now read from the calibrator                ║
║                                                                              ║
║  StrategyPerformanceTracker — online adaptive ensemble weights             ║
║    • Tracks win/loss per strategy (Trend/MeanRev/Breakout) per regime      ║
║    • Blends learned win rates with static REGIME_WEIGHTS (BLEND=0.35)     ║
║    • Only activates per strategy after 5+ qualifying trades (data guard)   ║
║    • Memory resets every 50 trades — stays responsive to regime shifts     ║
║    • Outcome is recorded after every settlement and fed back to weights    ║
║                                                                              ║
║  RISK                                                                        ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Martingale: $0.35 × 1.6×, max 4 steps, reset on win or step limit        ║
║  Ladder: $0.35 → $0.56 → $0.90 → $1.43 → $2.29 → reset                  ║
║  Max cycle risk: $5.53  |  Breakeven WR: 51.3% (95% ROI payout)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import math
import os
import sys
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy not found — using pure Python fallbacks", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


API_TOKEN  = _env("DERIV_API_TOKEN", "iCCn0vuMCzLcq1J")
APP_ID     = _env("DERIV_APP_ID",    1089)
SYMBOL     = _env("SYMBOL",          "R_100")

# Signal
MIN_CONFIDENCE  = _env("MIN_CONFIDENCE",   0.62)
MIN_SCORE_ALIGN = _env("MIN_SCORE_ALIGN",  0.25)   # abs(ensemble score) threshold
DECISION_COOLDOWN = _env("DECISION_COOLDOWN", 8)   # seconds between evaluations

# Candle history per granularity
CANDLE_HISTORY  = _env("CANDLE_HISTORY",   300)    # candles to keep per timeframe
WARMUP_CANDLES  = _env("WARMUP_CANDLES",   50)     # 1m candles before trading

# Regime weights: [trend_w, mean_rev_w, breakout_w]
REGIME_WEIGHTS  = {
    "TRENDING": [0.50, 0.20, 0.30],
    "RANGING":  [0.20, 0.60, 0.20],
    "CALM":     [0.30, 0.30, 0.40],
}

# Expiry per regime (minutes)
EXPIRY_MAP = {"CALM": 2, "RANGING": 3, "TRENDING": 5}

# Risk
BASE_STAKE      = _env("BASE_STAKE",      1.0)
MARTI_MULT      = _env("MARTI_MULT",      1.89)
MAX_MARTI_STEPS = _env("MAX_MARTI_STEPS", 4)
MAX_RISK_PCT    = _env("MAX_RISK_PCT",    0.03)
TARGET_PROFIT   = _env("TARGET_PROFIT",   50.0)
STOP_LOSS       = _env("STOP_LOSS",       20.0)

# ── Distribution intelligence ────────────────────────────────────────────────
# FeatureCalibrator: learns regime boundaries from observed 1m-candle features.
# Thresholds are set as percentiles of the rolling feature sample window so
# they automatically scale to whatever symbol / volatility you are trading.
RECAL_CANDLES    = _env("RECAL_CANDLES",    50)    # re-calibrate every N 1m candles
RECAL_WINDOW     = _env("RECAL_WINDOW",     200)   # rolling sample window (candles)
CHAOS_PCT        = _env("CHAOS_PCT",        0.92)  # p92 atr_slope  → CHAOS gate
TRENDING_PCT     = _env("TRENDING_PCT",     0.65)  # p65 |velocity| → TRENDING gate
CALM_PCT         = _env("CALM_PCT",         0.30)  # p30 atr_slope  → CALM gate
ACCEL_PCT        = _env("ACCEL_PCT",        0.92)  # p92 |accel|    → CHAOS co-gate

# StrategyPerformanceTracker: blends static regime weights with learned win
# rates per strategy per regime. PERF_BLEND=0 means always use static weights.
PERF_BLEND          = _env("PERF_BLEND",          0.35)  # 0=static, 1=fully learned
WEIGHT_RESET_TRADES = _env("WEIGHT_RESET_TRADES", 50)    # reset memory every N trades
MIN_STRAT_TRADES    = _env("MIN_STRAT_TRADES",    5)     # min trades before using learned rate

# Resilience
RECONNECT_MIN   = _env("RECONNECT_MIN",   2)
RECONNECT_MAX   = _env("RECONNECT_MAX",   60)
WS_PING         = _env("WS_PING",         30)
BUY_RETRIES     = _env("BUY_RETRIES",     8)
LOCK_TIMEOUT    = _env("LOCK_TIMEOUT",    350)   # 10 min max for 5m expiry

PORT = _env("PORT", 8080)

# Granularities to track
GRANULARITIES = [60, 120, 180, 300]


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def info(msg):  print(f"[{_ts()}] [INFO ] {msg}", flush=True)
def warn(msg):  print(f"[{_ts()}] [WARN ] {msg}", flush=True)
def err(msg):   print(f"[{_ts()}] [ERROR] {msg}", flush=True)
def tlog(msg):  print(f"[{_ts()}] [TRADE] {msg}", flush=True)
def jlog(obj):  print(json.dumps(obj), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Candle:
    __slots__ = ("open", "high", "low", "close", "epoch")
    def __init__(self, epoch: int, price: float):
        self.epoch = epoch
        self.open  = price
        self.high  = price
        self.low   = price
        self.close = price

    def update(self, price: float):
        self.high  = max(self.high, price)
        self.low   = min(self.low,  price)
        self.close = price

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "open": self.open,
                "high": self.high,   "low":  self.low,
                "close": self.close}


class CandleEngine:
    """
    Builds OHLC candles from raw ticks for multiple granularities.

    Fix vs original: uses epoch % gran to floor timestamp correctly.
    Original used ts.second % gran which only worked for gran=60
    and produced broken candles for 120s, 180s, 300s.
    """

    def __init__(self):
        # {gran: deque of Candle}
        self.candles: Dict[int, deque] = {
            g: deque(maxlen=CANDLE_HISTORY) for g in GRANULARITIES
        }

    def ingest(self, epoch: int, price: float):
        for gran in GRANULARITIES:
            floored = epoch - (epoch % gran)
            buf     = self.candles[gran]

            if buf and buf[-1].epoch == floored:
                buf[-1].update(price)
            else:
                buf.append(Candle(floored, price))

    def load_history(self, gran: int, raw_candles: list):
        """Seed from ticks_history API response."""
        buf = self.candles[gran]
        buf.clear()
        for c in raw_candles:
            candle = Candle(int(c["epoch"]), float(c["open"]))
            candle.high  = float(c["high"])
            candle.low   = float(c["low"])
            candle.close = float(c["close"])
            buf.append(candle)
        info(f"Loaded {len(buf)} historical candles for {gran}s")

    def get(self, gran: int) -> List[Candle]:
        return list(self.candles[gran])

    def ready(self, min_candles: int = WARMUP_CANDLES) -> bool:
        return len(self.candles[60]) >= min_candles


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (pure Python — no TA-Lib required)
# ─────────────────────────────────────────────────────────────────────────────

def _closes(candles: List[Candle]) -> List[float]:
    return [c.close for c in candles]

def _highs(candles: List[Candle]) -> List[float]:
    return [c.high for c in candles]

def _lows(candles: List[Candle]) -> List[float]:
    return [c.low for c in candles]


def calc_ema(values: List[float], period: int) -> List[float]:
    """Exponential moving average (full series)."""
    if len(values) < period:
        return [values[-1]] * len(values)
    k   = 2.0 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def calc_macd(closes: List[float],
              fast: int = 12, slow: int = 26,
              signal_p: int = 9) -> dict:
    """MACD histogram, signal line, and histogram."""
    if len(closes) < slow + signal_p:
        return {"hist": 0.0, "macd": 0.0, "signal": 0.0}
    ema_fast   = calc_ema(closes, fast)
    ema_slow   = calc_ema(closes, slow)
    macd_line  = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal     = calc_ema(macd_line, signal_p)
    hist       = [m - s for m, s in zip(macd_line, signal)]
    return {"hist": hist[-1], "macd": macd_line[-1], "signal": signal[-1]}


def calc_rsi(closes: List[float], period: int = 14) -> float:
    """Wilder RSI."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(candles: List[Candle], period: int = 14) -> float:
    """Wilder ATR — same formula as TA-Lib."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i-1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def calc_atr_series(candles: List[Candle], period: int = 14) -> List[float]:
    """ATR as a series for slope calculation."""
    if len(candles) < period + 2:
        return [0.0]
    result = []
    for i in range(period, len(candles)):
        result.append(calc_atr(candles[:i+1], period))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a unified feature dict from the 1m candle series.
    All computations are pure Python — no TA-Lib required.
    """

    def extract(self, engine: CandleEngine) -> Optional[dict]:
        candles_1m = engine.get(60)
        if len(candles_1m) < WARMUP_CANDLES:
            return None

        closes = _closes(candles_1m)
        n      = len(closes)

        # ── MACD ──────────────────────────────────────────────────────────────
        macd = calc_macd(closes)

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = calc_rsi(closes)

        # ── ATR + slope ────────────────────────────────────────────────────────
        atr_now  = calc_atr(candles_1m)
        atr_prev = calc_atr(candles_1m[:-3]) if len(candles_1m) > 3 else atr_now
        atr_slope = (atr_now - atr_prev) / 3.0 if atr_prev != 0 else 0.0

        # ── Velocity & acceleration ────────────────────────────────────────────
        velocity = (closes[-1] - closes[-10]) / 10.0 if n >= 10 else 0.0
        prev_vel = (closes[-11] - closes[-21]) / 10.0 if n >= 21 else velocity
        acceleration = velocity - prev_vel

        # ── Tick imbalance ─────────────────────────────────────────────────────
        diffs = [closes[i] - closes[i-1] for i in range(max(1, n-20), n)]
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
        tick_imbalance = sum(signs) / len(signs) if signs else 0.0

        # ── Multi-TF trend alignment ───────────────────────────────────────────
        tf_emas = {}
        for gran in GRANULARITIES:
            tf_candles = engine.get(gran)
            if len(tf_candles) >= 10:
                cl = _closes(tf_candles)
                e7  = calc_ema(cl, 7)[-1]
                e21 = calc_ema(cl, 21)[-1]
                tf_emas[gran] = e7 - e21
        tf_bullish = sum(1 for v in tf_emas.values() if v > 0)
        tf_bearish = sum(1 for v in tf_emas.values() if v < 0)
        tf_align   = tf_bullish if tf_bullish >= tf_bearish else -tf_bearish

        # ── Recent range ───────────────────────────────────────────────────────
        recent_20 = closes[-20:] if n >= 20 else closes
        price_range = max(recent_20) - min(recent_20)

        return {
            "velocity":       velocity,
            "acceleration":   acceleration,
            "atr":            atr_now,
            "atr_slope":      atr_slope,
            "rsi":            rsi,
            "macd_hist":      macd["hist"],
            "macd_line":      macd["macd"],
            "tick_imbalance": tick_imbalance,
            "price_range":    price_range,
            "tf_align":       tf_align,         # -4 to +4 (multi-TF agreement)
            "last_close":     closes[-1],
        }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CALIBRATOR  —  live adaptive regime thresholds
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCalibrator:
    """
    Learns the live distribution of market features extracted from completed
    1m candles and derives regime thresholds proportional to R_100's actual
    behaviour — not hardcoded numbers that may be totally wrong for this symbol.

    What it samples (one observation per completed 1m candle)
    ─────────────────────────────────────────────────────────
      atr_slope   — rate of ATR change; drives CHAOS and CALM gates
      |velocity|  — absolute price velocity; drives TRENDING gate
      |accel|     — absolute acceleration; CHAOS co-signal

    How thresholds are derived
    ──────────────────────────
    After WARMUP_CANDLES observations, and then every RECAL_CANDLES thereafter,
    the calibrator sorts its rolling window and picks percentile cuts:

      chaos_thresh    = p(CHAOS_PCT)    of atr_slope samples   [default p92]
      trending_thresh = p(TRENDING_PCT) of |velocity| samples  [default p65]
      calm_thresh     = p(CALM_PCT)     of atr_slope samples   [default p30]
      accel_thresh    = p(ACCEL_PCT)    of |accel| samples     [default p92]

    RegimeDetector and VolatilityGuardian both accept a calibrator instance
    and read these thresholds on every call — no restart needed.
    """

    def __init__(self):
        self._atr_slopes : deque = deque(maxlen=int(RECAL_WINDOW))
        self._velocities : deque = deque(maxlen=int(RECAL_WINDOW))
        self._accels     : deque = deque(maxlen=int(RECAL_WINDOW))
        self._candle_count = 0
        self.calibrated    = False
        self._history: list = []

        # Live thresholds — start at original hardcoded defaults so the bot
        # behaves sensibly before the first calibration fires at warmup.
        self.chaos_thresh    = 0.080
        self.trending_thresh = 0.350
        self.calm_thresh     = 0.010
        self.accel_thresh    = 0.500
        # Derivative thresholds used by strategies (scaled from primary ones)
        self.ranging_vel_thresh  = 0.150   # v < this → RANGING candidate
        self.ranging_atr_thresh  = 0.012   # as_ < this → RANGING candidate
        self.breakout_atr_thresh = 0.012   # as_ > this → breakout expansion
        self.guardian_ti_thresh  = 0.120   # tick_imbalance min for guardian

    def record(self, feat: dict):
        """
        Call once per completed 1m candle with the extracted feature dict.
        Handles initial calibration at warmup and periodic re-calibration.
        """
        self._atr_slopes.append(abs(feat["atr_slope"]))
        self._velocities.append(abs(feat["velocity"]))
        self._accels.append(abs(feat["acceleration"]))
        self._candle_count += 1

        if not self.calibrated and self._candle_count >= WARMUP_CANDLES:
            self._calibrate("initial-warmup")
        elif self.calibrated and self._candle_count % int(RECAL_CANDLES) == 0:
            self._calibrate(f"scheduled (candle {self._candle_count})")

    @staticmethod
    def _pct(data: list, p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[max(0, min(len(s) - 1, int(len(s) * p)))]

    def _calibrate(self, reason: str):
        if len(self._atr_slopes) < 10:
            return

        prev = (self.chaos_thresh, self.trending_thresh,
                self.calm_thresh, self.accel_thresh)

        asl = list(self._atr_slopes)
        vel = list(self._velocities)
        acc = list(self._accels)

        self.chaos_thresh    = round(self._pct(asl, CHAOS_PCT),    6)
        self.trending_thresh = round(self._pct(vel, TRENDING_PCT), 6)
        self.calm_thresh     = round(self._pct(asl, CALM_PCT),     6)
        self.accel_thresh    = round(self._pct(acc, ACCEL_PCT),    6)

        # Derivative thresholds scaled from primary ones so strategies stay
        # proportional — these replace the hardcoded values in the strategies.
        self.ranging_vel_thresh  = round(self.trending_thresh * 0.43, 6)  # ~p28
        self.ranging_atr_thresh  = round(self.calm_thresh     * 1.20, 6)  # just above calm
        self.breakout_atr_thresh = round(self.calm_thresh     * 1.20, 6)  # same as ranging
        # Guardian tick-imbalance stays dimensionless (0–1), keep at 0.12
        self.calibrated = True

        event = {
            "candle":         self._candle_count,
            "reason":         reason,
            "n":              len(asl),
            "chaos_thresh":   self.chaos_thresh,
            "trending_thresh":self.trending_thresh,
            "calm_thresh":    self.calm_thresh,
            "accel_thresh":   self.accel_thresh,
            "d_chaos":    round(self.chaos_thresh    - prev[0], 6),
            "d_trending": round(self.trending_thresh - prev[1], 6),
            "d_calm":     round(self.calm_thresh     - prev[2], 6),
        }
        self._history.append(event)

        info("─" * 64)
        info(f"[RECAL] {reason}  |  candle={self._candle_count}  n={len(asl)}")
        info(f"  CHAOS    : atr_slope > {self.chaos_thresh:.6f}"
             f"  (p{int(CHAOS_PCT*100)})  Δ={event['d_chaos']:+.6f}")
        info(f"  TRENDING : |velocity| > {self.trending_thresh:.6f}"
             f"  (p{int(TRENDING_PCT*100)})  Δ={event['d_trending']:+.6f}")
        info(f"  CALM     : atr_slope < {self.calm_thresh:.6f}"
             f"  (p{int(CALM_PCT*100)})  Δ={event['d_calm']:+.6f}")
        info(f"  ACCEL    : |accel| > {self.accel_thresh:.6f}"
             f"  (p{int(ACCEL_PCT*100)})")
        info(f"  RANGING  : vel < {self.ranging_vel_thresh:.6f}"
             f"  atr < {self.ranging_atr_thresh:.6f}")
        info("─" * 64)

    def summary(self) -> str:
        if not self._history:
            return f"pre-warmup ({self._candle_count}/{WARMUP_CANDLES} candles)"
        h = self._history[-1]
        return (f"recals={len(self._history)}  last={h['candle']}  "
                f"chaos={self.chaos_thresh:.5f}  "
                f"trending={self.trending_thresh:.5f}  "
                f"calm={self.calm_thresh:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY PERFORMANCE TRACKER  —  online adaptive ensemble weights
# ─────────────────────────────────────────────────────────────────────────────

class StrategyPerformanceTracker:
    """
    Tracks per-strategy, per-regime win rates from live trade outcomes and
    blends them with the static REGIME_WEIGHTS so the ensemble slowly adapts
    toward whichever strategy is actually working in the current conditions.

    Blend formula
    ─────────────
    For each strategy i in {trend, mean_rev, breakout}:
      if strategy i has >= MIN_STRAT_TRADES outcomes in this regime:
        learned_w[i] = win_rate[i] (normalised across all three)
        blended_w[i] = (1-PERF_BLEND)*static_w[i] + PERF_BLEND*learned_w[i]
      else:
        blended_w[i] = static_w[i]   # not enough data — keep static

    Memory resets every WEIGHT_RESET_TRADES total trades so stale data from
    a past regime doesn't permanently distort weights.

    Usage
    ─────
    1. Call record_signal(ensemble_result) just before placing every trade.
    2. Call record_outcome(won, regime) after every settlement.
    3. Pass the tracker to EnsembleCombiner — it calls adjusted_weights().
    """

    STRATS = ["trend", "mean_rev", "breakout"]
    REGIMES = ["TRENDING", "RANGING", "CALM"]

    def __init__(self):
        self._reset_stats()
        self._trade_count  = 0
        self._last_signal: Optional[dict] = None

    def _reset_stats(self):
        # {regime: {strategy: [wins, total]}}
        self._stats: Dict[str, Dict[str, list]] = {
            r: {s: [0, 0] for s in self.STRATS}
            for r in self.REGIMES
        }

    def record_signal(self, ensemble_result: dict):
        """Store the signal that triggered the last trade."""
        self._last_signal = ensemble_result

    def record_outcome(self, won: bool, regime: str):
        """
        Call after every settlement. Updates win/loss counters for each
        strategy that agreed with the direction of the trade taken.
        """
        if not self._last_signal or regime not in self.REGIMES:
            return

        direction = self._last_signal.get("direction", "NONE")
        strats    = self._last_signal.get("strategies", {})

        for s in self.STRATS:
            sig = strats.get(s, {}).get("signal", "NONE")
            if _dir_val(sig) == _dir_val(direction):   # agreed with trade
                self._stats[regime][s][1] += 1
                if won:
                    self._stats[regime][s][0] += 1

        self._trade_count += 1
        if self._trade_count % int(WEIGHT_RESET_TRADES) == 0:
            info(f"[PERF] Resetting weight memory at trade {self._trade_count}")
            self._reset_stats()

    def adjusted_weights(self, regime: str) -> list:
        """
        Returns [trend_w, mean_rev_w, breakout_w] blended between static
        REGIME_WEIGHTS and learned win rates for the given regime.
        Falls back per-strategy to static weight when data is insufficient.
        """
        static = REGIME_WEIGHTS.get(regime, [1/3, 1/3, 1/3])
        reg_stats = self._stats.get(regime, {})

        learned_rates = []
        for s in self.STRATS:
            wins, total = reg_stats.get(s, [0, 0])
            if total >= int(MIN_STRAT_TRADES):
                learned_rates.append(wins / total if total > 0 else 0.0)
            else:
                learned_rates.append(None)   # insufficient data

        # Normalise the learned rates that are available
        available = [(i, v) for i, v in enumerate(learned_rates) if v is not None]
        if len(available) == 3:
            total_rate = sum(v for _, v in available) or 1.0
            normed = [v / total_rate for v in learned_rates]
            blended = [
                round((1 - PERF_BLEND) * static[i] + PERF_BLEND * normed[i], 5)
                for i in range(3)
            ]
            # Re-normalise to sum=1
            s_total = sum(blended) or 1.0
            return [round(w / s_total, 5) for w in blended]

        # Partial data: blend only where available, keep static for the rest
        result = list(static)
        available_idx = [i for i, v in enumerate(learned_rates) if v is not None]
        if available_idx:
            av_total = sum(learned_rates[i] for i in available_idx) or 1.0
            for i in available_idx:
                normed_i = learned_rates[i] / av_total
                result[i] = round(
                    (1 - PERF_BLEND) * static[i] + PERF_BLEND * normed_i, 5
                )
            s_total = sum(result) or 1.0
            result = [round(w / s_total, 5) for w in result]
        return result

    def status(self, regime: str) -> str:
        parts = []
        for s in self.STRATS:
            wins, total = self._stats.get(regime, {}).get(s, [0, 0])
            wr = f"{wins/total*100:.0f}%" if total >= int(MIN_STRAT_TRADES) else "n/a"
            parts.append(f"{s}:{wr}({total}t)")
        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Classifies market into TRENDING / RANGING / CALM / CHAOS.

    v2: All numeric thresholds are read from FeatureCalibrator so they
    automatically scale to R_100's (or any symbol's) actual feature
    distributions rather than being hardcoded. Falls back to the original
    hardcoded values via the calibrator's defaults before first calibration.
    """

    def detect(self, feat: dict, cal: "FeatureCalibrator") -> str:
        v   = abs(feat["velocity"])
        as_ = abs(feat["atr_slope"])
        mh  = abs(feat["macd_hist"])
        rsi = feat["rsi"]
        ti  = abs(feat["tick_imbalance"])
        ac  = abs(feat["acceleration"])

        # CHAOS: atr_slope or acceleration in top percentile
        if as_ > cal.chaos_thresh or ac > cal.accel_thresh:
            return "CHAOS"

        # TRENDING: velocity in top percentile + MACD + tick confirmation
        if (v > cal.trending_thresh and
                as_ > cal.calm_thresh * 2 and
                mh > 0.005 and ti > 0.25):
            return "TRENDING"

        # RANGING: RSI mid-band + low velocity + stable ATR
        if (30 < rsi < 70 and
                v < cal.ranging_vel_thresh and
                as_ < cal.ranging_atr_thresh):
            return "RANGING"

        # CALM: very low ATR slope and low velocity
        if as_ < cal.calm_thresh and v < cal.ranging_vel_thresh:
            return "CALM"

        return "RANGING"   # default


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY GUARDIAN
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityGuardian:
    """
    Pre-trade gate that blocks entries during chaotic or flat markets.

    v2: CHAOS thresholds read from FeatureCalibrator (same source as
    RegimeDetector) so guardian and regime detection stay in sync.
    Tick-imbalance threshold stays dimensionless at 0.12.
    """

    def check(self, feat: dict, cal: "FeatureCalibrator") -> tuple:
        """Returns (allow: bool, reason: str)."""
        # Block if ATR slope is in chaos territory
        if abs(feat["atr_slope"]) > cal.chaos_thresh:
            return False, (f"ATR slope chaotic "
                           f"({feat['atr_slope']:.5f} > {cal.chaos_thresh:.5f})")

        # Block if acceleration is extreme
        if abs(feat["acceleration"]) > cal.accel_thresh:
            return False, (f"Acceleration chaotic "
                           f"({feat['acceleration']:.5f} > {cal.accel_thresh:.5f})")

        # Block if tick imbalance too neutral (dimensionless — no calibration needed)
        if abs(feat["tick_imbalance"]) < cal.guardian_ti_thresh:
            return False, (f"Tick imbalance too neutral "
                           f"({feat['tick_imbalance']:.3f} < {cal.guardian_ti_thresh:.3f})")

        return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

class StrategyResult:
    __slots__ = ("signal", "confidence", "reason")
    def __init__(self, signal: str, confidence: float, reason: str):
        self.signal     = signal       # "BUY_RISE" | "BUY_FALL" | "NONE"
        self.confidence = confidence
        self.reason     = reason


class TrendStrategy:
    """
    MACD histogram + velocity + ATR expansion.
    v2: signal thresholds scaled from calibrated trending/calm values so the
    strategy fires on the right proportion of candles regardless of symbol.
    """

    def compute(self, feat: dict, cal: "FeatureCalibrator") -> StrategyResult:
        mh  = feat["macd_hist"]
        v   = feat["velocity"]
        as_ = feat["atr_slope"]
        tfa = feat["tf_align"]

        # Use 15% of trending threshold as the minimum signal level
        v_min  = cal.trending_thresh * 0.15
        as_min = cal.calm_thresh     * 0.50

        bull = mh > 0.002 and v > v_min  and as_ > as_min
        bear = mh < -0.002 and v < -v_min and as_ > as_min

        if not bull and not bear:
            return StrategyResult("NONE", 0.0, "No momentum confluence")

        # Confidence: normalised against calibrated scale
        macd_score = min(1.0, abs(mh) / 0.02)
        vel_score  = min(1.0, abs(v)  / max(cal.trending_thresh, 1e-9))
        atr_score  = min(1.0, abs(as_) / max(cal.chaos_thresh * 0.5, 1e-9))
        tf_score   = abs(tfa) / 4.0

        conf = (0.35 * macd_score +
                0.30 * vel_score  +
                0.20 * atr_score  +
                0.15 * tf_score)
        conf = min(0.90, 0.50 + conf * 0.40)

        signal = "BUY_RISE" if bull else "BUY_FALL"
        return StrategyResult(signal, conf,
                              f"MACD={mh:.4f} vel={v:.4f} "
                              f"atr_slope={as_:.4f} tf_align={tfa} "
                              f"v_min={v_min:.5f}")


class MeanReversionStrategy:
    """
    RSI extremes with tick imbalance confirmation.
    v2: RSI thresholds unchanged (dimensionless 0-100 scale). Tick imbalance
    minimum read from calibrator (guardian_ti_thresh) for consistency.
    """

    def compute(self, feat: dict, cal: "FeatureCalibrator") -> StrategyResult:
        rsi    = feat["rsi"]
        ti     = feat["tick_imbalance"]
        ti_min = cal.guardian_ti_thresh   # coherent with guardian gate

        # Oversold: RSI < 32, ticks still selling (confirming extreme)
        if rsi < 32 and ti < -ti_min:
            conf = min(0.88, 0.65 + (32 - rsi) / 32 * 0.20)
            return StrategyResult("BUY_RISE", conf,
                                  f"RSI oversold={rsi:.1f} ti={ti:.3f}")

        # Overbought: RSI > 68, ticks still buying
        if rsi > 68 and ti > ti_min:
            conf = min(0.88, 0.65 + (rsi - 68) / 32 * 0.20)
            return StrategyResult("BUY_FALL", conf,
                                  f"RSI overbought={rsi:.1f} ti={ti:.3f}")

        return StrategyResult("NONE", 0.0, f"RSI={rsi:.1f} not extreme")


class BreakoutStrategy:
    """
    Price range compression → expansion.
    v2: Compression and expansion thresholds read from calibrator so they
    scale correctly to R_100's actual ATR and velocity distributions.
    """

    def compute(self, feat: dict, cal: "FeatureCalibrator") -> StrategyResult:
        as_  = feat["atr_slope"]
        v    = feat["velocity"]
        rng  = feat["price_range"]

        compress_thresh = cal.calm_thresh * 0.50      # tighter than calm
        expand_thresh   = cal.breakout_atr_thresh      # calibrated expansion gate
        vel_thresh      = cal.trending_thresh * 0.12  # modest directional move

        # Compression: ATR slope very low AND tight range
        if abs(as_) < compress_thresh:
            return StrategyResult("NONE", 0.0,
                                  f"Compression atr={as_:.5f} < {compress_thresh:.5f}")

        # Expansion with direction
        if abs(as_) > expand_thresh and abs(v) > vel_thresh:
            # Confidence normalised against calibrated scale
            atr_score = min(1.0, abs(as_) / max(cal.chaos_thresh * 0.5, 1e-9))
            vel_score = min(1.0, abs(v)   / max(cal.trending_thresh, 1e-9))
            conf = min(0.82, 0.55 + atr_score * 0.15 + vel_score * 0.12)
            signal = "BUY_RISE" if v > 0 else "BUY_FALL"
            return StrategyResult(signal, conf,
                                  f"Breakout atr={as_:.5f}>{expand_thresh:.5f} "
                                  f"vel={v:.5f}>{vel_thresh:.5f}")

        return StrategyResult("NONE", 0.0,
                              f"Insufficient expansion atr={as_:.5f} v={v:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE COMBINER
# ─────────────────────────────────────────────────────────────────────────────

def _dir_val(signal: str) -> int:
    if signal == "BUY_RISE": return  1
    if signal == "BUY_FALL": return -1
    return 0


class EnsembleCombiner:
    """
    Combines three strategy results using regime-specific weights.

    v2: Accepts FeatureCalibrator (passed to each strategy) and
    StrategyPerformanceTracker (provides adaptive blended weights).
    Weights start static, drift toward learned win rates over time.
    """

    def __init__(self):
        self.trend    = TrendStrategy()
        self.mean_rev = MeanReversionStrategy()
        self.breakout = BreakoutStrategy()

    def evaluate(self, feat: dict, regime: str,
                 cal: "FeatureCalibrator",
                 perf: "StrategyPerformanceTracker") -> dict:

        # Each strategy now receives the calibrator for scaled thresholds
        t  = self.trend.compute(feat, cal)
        mr = self.mean_rev.compute(feat, cal)
        br = self.breakout.compute(feat, cal)

        # Adaptive weights: blended static + learned win rates
        weights = perf.adjusted_weights(regime)
        results = [t, mr, br]

        # Weighted directional score
        score = sum(
            weights[i] * results[i].confidence * _dir_val(results[i].signal)
            for i in range(3)
        )

        # Determine direction
        if score >= MIN_SCORE_ALIGN:
            direction = "BUY_RISE"
        elif score <= -MIN_SCORE_ALIGN:
            direction = "BUY_FALL"
        else:
            return self._no_trade("Score below threshold", t, mr, br,
                                  score, regime, weights)

        # Confidence = weighted average of agreeing strategies
        agree = [i for i in range(3)
                 if _dir_val(results[i].signal) == _dir_val(direction)]
        if not agree:
            return self._no_trade("No strategies agree", t, mr, br,
                                  score, regime, weights)

        conf = sum(weights[i] * results[i].confidence for i in agree)
        conf_total = sum(weights[i] for i in agree)
        conf = conf / conf_total if conf_total > 0 else 0.0

        # Agreement boost: all 3 strategies same direction → +8%
        if len(agree) == 3:
            conf = min(1.0, conf * 1.08)

        # Multi-timeframe alignment boost
        tfa = feat["tf_align"]
        if direction == "BUY_RISE" and tfa >= 3:
            conf = min(1.0, conf * 1.04)
        elif direction == "BUY_FALL" and tfa <= -3:
            conf = min(1.0, conf * 1.04)

        return {
            "trade":      conf >= MIN_CONFIDENCE,
            "direction":  direction,
            "confidence": round(conf, 4),
            "score":      round(score, 4),
            "regime":     regime,
            "weights":    [round(w, 4) for w in weights],
            "strategies": {
                "trend":    {"signal": t.signal,  "conf": round(t.confidence, 3),
                             "reason": t.reason},
                "mean_rev": {"signal": mr.signal, "conf": round(mr.confidence, 3),
                             "reason": mr.reason},
                "breakout": {"signal": br.signal, "conf": round(br.confidence, 3),
                             "reason": br.reason},
            },
            "agree_count": len(agree),
        }

    def _no_trade(self, reason, t, mr, br, score, regime, weights) -> dict:
        return {
            "trade":      False,
            "direction":  "NONE",
            "confidence": 0.0,
            "score":      round(score, 4),
            "regime":     regime,
            "reason":     reason,
            "weights":    [round(w, 4) for w in weights],
            "strategies": {
                "trend":    {"signal": t.signal,  "conf": round(t.confidence, 3)},
                "mean_rev": {"signal": mr.signal, "conf": round(mr.confidence, 3)},
                "breakout": {"signal": br.signal, "conf": round(br.confidence, 3)},
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.stake       = BASE_STAKE
        self.marti_step  = 0
        self.balance     = 1000.0
        self.session_pnl = 0.0
        self.wins        = 0
        self.losses      = 0

    def get_stake(self, confidence: float) -> float:
        stake = round(BASE_STAKE * (MARTI_MULT ** self.marti_step), 2)
        # Confidence scaling
        if confidence >= 0.82:
            stake = round(stake * 1.20, 2)
        elif confidence >= 0.77:
            stake = round(stake * 1.10, 2)
        # Hard cap: MAX_RISK_PCT of balance
        stake = min(stake, round(self.balance * MAX_RISK_PCT, 2))
        stake = max(stake, 0.35)   # minimum $0.35
        return stake

    def record_win(self, profit: float):
        self.wins        += 1
        self.session_pnl += profit
        self.marti_step   = 0
        self.stake        = BASE_STAKE
        tlog(f"WIN  +${profit:.4f}  |  streak reset  |  P&L ${self.session_pnl:+.4f}")
        self._stats()

    def record_loss(self, amount: float):
        self.losses      += 1
        self.session_pnl -= amount
        if self.marti_step < MAX_MARTI_STEPS:
            self.marti_step += 1
        else:
            tlog(f"Max martingale steps reached → reset")
            self.marti_step = 0
        tlog(f"LOSS -${amount:.2f}  |  step={self.marti_step}  |  "
             f"P&L ${self.session_pnl:+.4f}")
        self._stats()

    def can_trade(self) -> bool:
        if self.session_pnl >= TARGET_PROFIT:
            info(f"Target profit hit (${self.session_pnl:.4f}) — stopping")
            return False
        if self.session_pnl <= -STOP_LOSS:
            warn(f"Stop-loss hit (${self.session_pnl:.4f}) — stopping")
            return False
        return True

    def _stats(self):
        total = self.wins + self.losses
        wr    = self.wins / total * 100 if total else 0
        info(f"Trades:{total}  W:{self.wins}  L:{self.losses}  "
             f"WR:{wr:.1f}%  P&L:${self.session_pnl:.4f}  "
             f"Balance:${self.balance:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    def __init__(self):
        self.endpoint   = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
        self.ws         = None
        self._send_q    = None
        self._inbox     = None
        self._send_task = None
        self._recv_task = None

    async def connect(self) -> bool:
        try:
            info(f"Connecting → {self.endpoint}")
            self.ws = await websockets.connect(
                self.endpoint,
                ping_interval=WS_PING,
                ping_timeout=20,
                close_timeout=10,
            )
            self._send_q = asyncio.Queue()
            self._inbox  = asyncio.Queue()
            self._start_io()
            await asyncio.sleep(0)   # yield so pump tasks are scheduled first
            await self._send({"authorize": API_TOKEN})
            resp = await self._recv_type("authorize", timeout=15)
            if not resp or "error" in resp:
                msg = (resp or {}).get("error", {}).get("message", "timeout")
                err(f"Auth failed: {msg}")
                return False
            auth = resp.get("authorize", {})
            info(f"Auth OK  |  {auth.get('loginid')}  |  "
                 f"Balance: ${auth.get('balance', 0):.2f}")
            return True
        except Exception as e:
            err(f"Connect error: {e}")
            traceback.print_exc(file=sys.stdout)
            return False

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump(), name="send")
        self._recv_task = asyncio.create_task(self._recv_pump(), name="recv")

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"Recv pump: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send(self, data: dict):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_q.put((data, fut))
        await fut

    async def receive(self, timeout: float = 60) -> dict:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def _recv_type(self, msg_type: str, timeout: float = 10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def subscribe_ticks(self) -> bool:
        await self._send({"ticks": SYMBOL, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            msg = (resp or {}).get("error", {}).get("message", "timeout")
            err(f"Tick subscribe failed: {msg}")
            return False
        info(f"Subscribed to {SYMBOL} ticks")
        return True

    async def subscribe_balance(self) -> bool:
        await self._send({"balance": 1, "subscribe": 1})
        resp = await self._recv_type("balance", timeout=10)
        if not resp or "error" in resp:
            warn("Balance subscribe failed — using placeholder balance")
            return False
        info(f"Balance subscribed")
        return True

    async def load_history(self, engine: CandleEngine):
        """Load historical candles for all granularities."""
        for gran in GRANULARITIES:
            try:
                await self._send({
                    "ticks_history": SYMBOL,
                    "granularity":   gran,
                    "count":         200,
                    "style":         "candles",
                    "end":           "latest",
                })
                resp = await self._recv_type("candles", timeout=15)
                if resp and "candles" in resp:
                    engine.load_history(gran, resp["candles"])
                else:
                    warn(f"No candle history for {gran}s")
            except Exception as e:
                warn(f"History load {gran}s failed: {e}")

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            warn(f"Balance fetch: {exc}")
        return None

    async def place_trade(self, direction: str, stake: float,
                          duration_min: int) -> tuple:
        """
        Returns (contract_id, expiry_time) where expiry_time is a Unix
        timestamp from Deriv's buy response (date_expiry field).
        Returns (None, None) on any failure.
        """
        contract_type = "CALL" if direction == "BUY_RISE" else "PUT"
        duration_sec  = duration_min * 60

        await self._send({
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": contract_type,
            "currency":      "USD",
            "duration":      duration_sec,
            "duration_unit": "s",
            "symbol":        SYMBOL,
        })
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            msg = (proposal or {}).get("error", {}).get("message", "timeout")
            err(f"Proposal error: {msg}")
            return None, None

        prop   = proposal.get("proposal", {})
        pid    = prop.get("id")
        ask    = float(prop.get("ask_price", stake))
        payout = float(prop.get("payout", 0))
        roi    = (payout - ask) / ask * 100 if ask > 0 else 0
        info(f"Proposal OK  |  {contract_type}  ask=${ask:.2f}  "
             f"payout=${payout:.2f}  ROI={roi:.1f}%")

        if not pid:
            err("No proposal ID")
            return None, None

        buy_ts      = time.time()
        contract_id = None
        expiry_time = None           # ← will be populated from buy response
        await self._send({"buy": pid, "price": ask})

        for attempt in range(BUY_RETRIES):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                warn(f"Buy no response attempt {attempt + 1}")
                continue
            if "error" in resp:
                err(f"Buy error: {resp['error'].get('message', '')}")
                return None, None
            buy_data    = resp.get("buy", {})
            contract_id = buy_data.get("contract_id")
            expiry_time = buy_data.get("date_expiry")  # ← Unix timestamp
            if contract_id:
                break

        if not contract_id:
            # Orphan recovery
            warn("No contract_id — orphan recovery")
            for _ in range(4):
                await asyncio.sleep(3)
                await self._send({"profit_table": 1, "description": 1,
                                  "sort": "DESC", "limit": 5})
                resp = await self._recv_type("profit_table", timeout=10)
                if resp and "profit_table" in resp:
                    for tx in resp["profit_table"].get("transactions", []):
                        if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                                float(tx.get("purchase_time", 0)) >= buy_ts - 10):
                            contract_id = tx.get("contract_id")
                            info(f"Orphan recovered → {contract_id}")
                            break
                if contract_id:
                    break
            if not contract_id:
                err("Orphan recovery failed")
                return None, None

        try:
            await self._send({"proposal_open_contract": 1,
                              "contract_id": contract_id,
                              "subscribe":   1})
        except Exception:
            pass

        tlog(f"Placed  |  contract={contract_id}  |  {contract_type}  "
             f"${ask:.2f}  {duration_min}min  expiry_ts={expiry_time}")
        return contract_id, expiry_time

    async def poll_contract(self, contract_id: int) -> Optional[dict]:
        try:
            await self._send({"proposal_open_contract": 1,
                              "contract_id": contract_id})
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            warn(f"Poll: {exc}")
        return None

    @staticmethod
    def is_settled(data: dict) -> bool:
        if data.get("is_settled") or data.get("is_sold"):
            return True
        return data.get("status", "").lower() in ("sold", "won", "lost")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    bot_ref = None

    def do_GET(self):
        body = b"OK"
        if self.path == "/status" and self.bot_ref:
            b   = self.bot_ref
            r   = b.risk
            tot = r.wins + r.losses
            body = json.dumps({
                "status":       "running",
                "symbol":       SYMBOL,
                "regime":       b.current_regime,
                "candles_1m":   len(b.engine.candles[60]),
                "ready":        b.engine.ready(),
                "locked":       b.waiting_for_result,
                "contract_id":  (b.current_trade or {}).get("id"),
                "expiry_time":  (b.current_trade or {}).get("expiry_time"),
                "lock_age_s":   round(time.monotonic() - b.lock_since, 1)
                                if b.lock_since else None,
                "trades":       tot,
                "wins":         r.wins,
                "losses":       r.losses,
                "wr":           round(r.wins/tot, 4) if tot else 0,
                "pnl":          round(r.session_pnl, 4),
                "marti_step":   r.marti_step,
                "balance":      r.balance,
                "last_eval":    b.last_decision_time,
            }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_health_server(bot):
    HealthHandler.bot_ref = bot
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    info(f"Health server on :{PORT}  (GET / or /status)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────────────────────────────────────

class R100EnsembleBot:
    def __init__(self):
        self.client   = DerivClient()
        self.engine   = CandleEngine()
        self.features = FeatureExtractor()
        self.calibrator = FeatureCalibrator()           # live adaptive thresholds
        self.perf       = StrategyPerformanceTracker()  # adaptive ensemble weights
        self.regime   = RegimeDetector()
        self.guardian = VolatilityGuardian()
        self.ensemble = EnsembleCombiner()
        self.risk     = RiskManager()

        self.waiting_for_result:  bool            = False
        self._evaluating:         bool            = False
        self.current_trade:       Optional[dict]  = None
        self.lock_since:          Optional[float] = None
        self._stop:               bool            = False
        self._bal_before:         Optional[float] = None

        self.last_decision_time:  float           = 0.0
        self.current_regime:      str             = "CALM"
        self.tick_count:          int             = 0
        self.signal_count:        int             = 0
        self._last_1m_candle_count: int           = 0   # tracks new candles for calibrator

    def _unlock(self, reason: str = "manual"):
        if self.waiting_for_result:
            cid = (self.current_trade or {}).get("id", "?")
            info(f"Unlock: contract={cid}  reason={reason}")
        self.waiting_for_result = False
        self.current_trade      = None
        self.lock_since         = None
        self._evaluating        = False

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        if time.monotonic() - self.lock_since >= LOCK_TIMEOUT:
            warn("Lock timeout — auto-unlocking")
            self._unlock("timeout")

    # ── Tick handler ──────────────────────────────────────────────────────────

    async def on_tick(self, tick: dict):
        price = float(tick.get("quote", 0))
        epoch = int(tick.get("epoch", time.time()))
        if not price:
            return

        self.tick_count += 1
        self.engine.ingest(epoch, price)
        self._check_lock_timeout()

        # Feed FeatureCalibrator once per completed 1m candle.
        # We detect a new candle by a change in the candle count.
        n1m = len(self.engine.candles[60])
        if n1m > self._last_1m_candle_count and n1m >= 2:
            self._last_1m_candle_count = n1m
            feat_for_cal = self.features.extract(self.engine)
            if feat_for_cal is not None:
                self.calibrator.record(feat_for_cal)

        if self.tick_count % 20 == 0:
            warmup_left = max(0, WARMUP_CANDLES - n1m)
            cal_status  = ("calibrated" if self.calibrator.calibrated
                           else f"pre-cal({n1m}/{WARMUP_CANDLES})")
            status = ("LOCKED" if self.waiting_for_result else
                      f"WARMUP({warmup_left})" if warmup_left > 0 else
                      f"READY  regime={self.current_regime}  cal={cal_status}")
            info(f"tick={self.tick_count}  price={price:.4f}  "
                 f"1m_candles={n1m}  {status}")

        if self.waiting_for_result or self._evaluating:
            return
        if not self.engine.ready():
            return
        if time.time() - self.last_decision_time < DECISION_COOLDOWN:
            return
        if not self.risk.can_trade():
            self._stop = True
            return

        self._evaluating = True
        try:
            await self._evaluate(price)
        finally:
            self._evaluating = False

    # ── Market evaluation ─────────────────────────────────────────────────────

    async def _evaluate(self, price: float):
        if self.waiting_for_result:
            return

        feat = self.features.extract(self.engine)
        if feat is None:
            return

        # Require calibration before trading — thresholds must be data-driven
        if not self.calibrator.calibrated:
            return

        regime = self.regime.detect(feat, self.calibrator)
        self.current_regime = regime

        # Block CHAOS entirely
        if regime == "CHAOS":
            return

        # Volatility guardian — uses calibrated thresholds
        allow, guard_reason = self.guardian.check(feat, self.calibrator)
        if not allow:
            return

        # Ensemble signal — calibrated strategies + adaptive weights
        result = self.ensemble.evaluate(feat, regime, self.calibrator, self.perf)

        if not result["trade"]:
            return

        direction  = result["direction"]
        confidence = result["confidence"]
        duration   = EXPIRY_MAP.get(regime, 3)

        self.signal_count += 1
        # Record signal in perf tracker BEFORE placing trade
        self.perf.record_signal(result)

        w = result.get("weights", ["-", "-", "-"])
        info("=" * 60)
        info(f"SIGNAL #{self.signal_count}  |  tick={self.tick_count}  "
             f"|  regime={regime}")
        info(f"  Direction : {direction}  |  Confidence: {confidence:.4f}  "
             f"|  Score: {result['score']:.4f}")
        info(f"  Expiry    : {duration}min")
        info(f"  Weights   : trend={w[0]}  mean_rev={w[1]}  breakout={w[2]}  "
             f"(blend={PERF_BLEND})")
        info(f"  Strategies: "
             f"trend={result['strategies']['trend']['signal']}({result['strategies']['trend']['conf']:.2f})  "
             f"mr={result['strategies']['mean_rev']['signal']}({result['strategies']['mean_rev']['conf']:.2f})  "
             f"br={result['strategies']['breakout']['signal']}({result['strategies']['breakout']['conf']:.2f})")
        info(f"  Agree     : {result['agree_count']}/3 strategies")
        info(f"  PerfTrack : {self.perf.status(regime)}")
        info(f"  Calibrator: {self.calibrator.summary()}")
        info("=" * 60)

        stake = self.risk.get_stake(confidence)
        info(f"Placing trade  stake=${stake:.2f}  marti_step={self.risk.marti_step}")

        self._bal_before = await self.client.fetch_balance()
        if self._bal_before:
            info(f"Pre-trade balance: ${self._bal_before:.2f}")

        contract_id, expiry_time = await self.client.place_trade(
            direction, stake, duration
        )

        if contract_id:
            self.current_trade = {
                "id":          contract_id,
                "direction":   direction,
                "stake":       stake,
                "duration":    duration,
                "confidence":  confidence,
                "regime":      regime,
                "expiry_time": expiry_time,   # Unix ts from Deriv buy response
            }
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            self.last_decision_time = time.time()

            # Launch active expiry poller so the bot never relies solely
            # on the passive WS push for settlement detection.
            asyncio.create_task(
                self._expiry_poller(contract_id, expiry_time, duration),
                name=f"poller_{contract_id}"
            )

            jlog({
                "type":        "trade",
                "cid":         contract_id,
                "direction":   direction,
                "stake":       stake,
                "duration":    duration,
                "confidence":  round(confidence, 4),
                "regime":      regime,
                "score":       result["score"],
                "agree":       result["agree_count"],
                "weights":     result.get("weights"),
                "cal_chaos":   round(self.calibrator.chaos_thresh, 5),
                "cal_trend":   round(self.calibrator.trending_thresh, 5),
                "cal_calm":    round(self.calibrator.calm_thresh, 5),
                "recals":      len(self.calibrator._history),
                "expiry_time": expiry_time,
                "ts":          _ts(),
            })
        else:
            self._bal_before = None
            warn("Trade placement failed — ready for next signal")
            self.last_decision_time = time.time()

    # ── Expiry poller ─────────────────────────────────────────────────────────

    async def _expiry_poller(self, contract_id: int,
                              expiry_time: Optional[int],
                              duration_min: int):
        """
        Active settlement guard. Fires after the contract expires and
        polls Deriv directly if the passive WS push hasn't arrived.

        Root cause this fixes:
          The main loop receives messages sequentially. With R_100
          ticking ~1/sec, 120-300 tick messages accumulate in the inbox
          during a 2-5min contract. The settlement proposal_open_contract
          message can be buried behind all of them. Without this poller
          the bot waited the full LOCK_TIMEOUT (350s) before unlocking,
          which is what caused the forced unlocks.

        Timeline:
          T+0              trade placed, WS subscription active
          T+duration+5s    poller wakes, checks if still locked
          T+duration+5-35s polls every 5s up to 6 times
          T+duration+35s   force-unlock if no result (LOCK_TIMEOUT
                           at 350s is a further fallback)
        """
        if expiry_time:
            wait = max(5.0, (expiry_time - time.time()) + 5)
        else:
            # Fallback: use stated duration + 10s buffer
            wait = duration_min * 60 + 10

        info(f"Expiry poller started: contract={contract_id} "
             f"sleeping {wait:.1f}s")
        await asyncio.sleep(wait)

        # If WS push already settled this — nothing to do
        if not self.waiting_for_result:
            info(f"Expiry poller: {contract_id} already settled via WS ✅")
            return
        if not self.current_trade or self.current_trade.get("id") != contract_id:
            return

        warn(f"Expiry poller: {contract_id} still locked after expiry "
             f"— polling Deriv directly")

        # Poll up to 6 × 5s = 30s
        for attempt in range(1, 7):
            try:
                data = await self.client.poll_contract(contract_id)
                if data:
                    if self.client.is_settled(data):
                        info(f"Expiry poller: settled on attempt {attempt} ✅")
                        ok = await self.handle_settlement(data)
                        if not ok:
                            self._stop = True
                        return
                    else:
                        info(f"Expiry poller: attempt {attempt} — "
                             f"not yet settled (status={data.get('status','?')})")
                else:
                    warn(f"Expiry poller: attempt {attempt} — no data returned")
            except Exception as exc:
                warn(f"Expiry poller poll error (attempt {attempt}): {exc}")

            await asyncio.sleep(5)

        # All polls exhausted — force-unlock so trading can resume
        if (self.waiting_for_result and
                self.current_trade and
                self.current_trade.get("id") == contract_id):
            warn(f"Expiry poller: exhausted all attempts for {contract_id} "
                 f"— force-unlocking")
            self._unlock("expiry_poller_exhausted")

    # ── Settlement ────────────────────────────────────────────────────────────

    async def handle_settlement(self, data: dict) -> bool:
        cid = data.get("contract_id")
        if not self.current_trade or cid != self.current_trade["id"]:
            return True
        if not self.client.is_settled(data):
            return True

        profit = float(data.get("profit", 0))
        status = data.get("status", "unknown")
        stake  = self.current_trade["stake"]

        bal_after = await self.client.fetch_balance()
        if bal_after is not None and self._bal_before is not None:
            actual = round(bal_after - self._bal_before, 4)
        else:
            actual = profit

        tlog(f"SETTLED  |  contract={cid}  |  "
             f"status={status}  |  profit=${actual:+.4f}")

        won = actual > 0
        if won:
            self.risk.record_win(actual)
        else:
            self.risk.record_loss(stake)

        # Feed outcome into performance tracker for adaptive weight learning
        trade_regime = (self.current_trade or {}).get("regime", self.current_regime)
        self.perf.record_outcome(won, trade_regime)

        if bal_after:
            self.risk.balance = bal_after

        jlog({
            "type":    "result",
            "cid":     cid,
            "status":  status,
            "profit":  actual,
            "pnl":     round(self.risk.session_pnl, 4),
            "wins":    self.risk.wins,
            "losses":  self.risk.losses,
            "ts":      _ts(),
        })

        self._bal_before = None
        self._unlock("settlement")
        info("Ready for next signal")
        return self.risk.can_trade()

    # ── Reconnect ─────────────────────────────────────────────────────────────

    async def _reconnect(self) -> bool:
        delay   = RECONNECT_MIN
        attempt = 0
        while not self._stop:
            attempt += 1
            warn(f"Reconnect attempt {attempt} in {delay}s ...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)
            await self.client.close()
            self.client = DerivClient()
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                await self.client.subscribe_balance()
                # Re-attach open contract
                if self.waiting_for_result and self.current_trade:
                    cid  = self.current_trade["id"]
                    info(f"Re-attaching contract {cid}")
                    data = await self.client.poll_contract(cid)
                    if data:
                        await self.handle_settlement(data)
                    if self.waiting_for_result:
                        await self.client._send({
                            "proposal_open_contract": 1,
                            "contract_id": cid,
                            "subscribe":   1,
                        })
                info("Reconnect OK")
                return True
            except Exception as exc:
                err(f"Reconnect error: {exc}")
        return False

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self):
        info("=" * 64)
        info(f"R_100 ENSEMBLE BOT  v2  —  {SYMBOL}  —  Railway")
        info(f"Strategies : Trend + Mean-Reversion + Breakout")
        info(f"Regimes    : TRENDING/RANGING/CALM (CHAOS blocked)")
        info(f"Calibrator : candle-based  recal={int(RECAL_CANDLES)} candles  "
             f"window={int(RECAL_WINDOW)}  warmup={WARMUP_CANDLES}")
        info(f"PerfBlend  : {PERF_BLEND}  reset_every={int(WEIGHT_RESET_TRADES)} trades  "
             f"min_data={int(MIN_STRAT_TRADES)} trades/strategy")
        info(f"Gate       : min_conf={MIN_CONFIDENCE}  cooldown={DECISION_COOLDOWN}s")
        info(f"Risk       : stake=${BASE_STAKE}  marti={MARTI_MULT}×/{MAX_MARTI_STEPS}  "
             f"target=${TARGET_PROFIT}  stop=-${STOP_LOSS}")
        info("=" * 64)

        if API_TOKEN in ("REPLACE_WITH_YOUR_TOKEN", ""):
            err("Set DERIV_API_TOKEN before running")
            return

        if not await self.client.connect():
            return

        # Load historical candles for warmup
        info("Loading historical candles ...")
        await self.client.load_history(self.engine)

        if not await self.client.subscribe_ticks():
            return
        await self.client.subscribe_balance()

        info(f"Bot live  |  warmup={WARMUP_CANDLES} 1m-candles  "
             f"(have {len(self.engine.candles[60])} so far)")

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    warn("WS disconnected — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        warn("Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    await self.on_tick(response["tick"])

                if "balance" in response:
                    bal = response["balance"].get("balance")
                    if bal is not None:
                        self.risk.balance = float(bal)

                for key in ("proposal_open_contract", "buy"):
                    if key in response:
                        ok = await self.handle_settlement(response[key])
                        if not ok:
                            self._stop = True

                if "transaction" in response:
                    tx = response["transaction"]
                    if "contract_id" in tx:
                        ok = await self.handle_settlement({
                            "contract_id": tx.get("contract_id"),
                            "profit":      tx.get("profit", 0),
                            "status":      tx.get("action", "sold"),
                            "is_settled":  True,
                        })
                        if not ok:
                            self._stop = True

        except KeyboardInterrupt:
            info("Stopped by user")
        except Exception as e:
            err(f"Fatal: {e}")
            traceback.print_exc(file=sys.stdout)
        finally:
            self._print_final()
            await self.client.close()
            info("Bot exited.")

    def _print_final(self):
        r     = self.risk
        total = r.wins + r.losses
        wr    = r.wins / total * 100 if total else 0
        info("=" * 64)
        info("FINAL SESSION STATS")
        info(f"Live ticks    : {self.tick_count}")
        info(f"Signals fired : {self.signal_count}")
        info(f"Trades        : {total}  (W:{r.wins}  L:{r.losses})")
        info(f"Win rate      : {wr:.1f}%  (breakeven 51.3%)")
        info(f"Net P&L       : ${r.session_pnl:.4f}")
        info(f"Balance       : ${r.balance:.2f}")
        info(f"Calibrator    : {self.calibrator.summary()}")
        for reg in StrategyPerformanceTracker.REGIMES:
            ps = self.perf.status(reg)
            w  = self.perf.adjusted_weights(reg)
            info(f"Perf [{reg:8s}]: {ps}  →  weights={[round(x,3) for x in w]}")
        info("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    bot = R100EnsembleBot()
    start_health_server(bot)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
