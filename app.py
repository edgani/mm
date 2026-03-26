"""
V4 Core Lane Skeleton
Institutional-leaning long-only research engine for:
- Tactical module (10 / 20 bars)
- Runner module (40 / 60 / 120 bars)

Focus:
- US trend leaders + gold
- setup -> confirmation -> hold/failure
- adaptive sizing
- walk-forward backtest scaffold

This is a research skeleton, intended to be extended and calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
@dataclass
class UniverseConfig:
    symbols: List[str]


@dataclass
class TacticalConfig:
    train_bars: int = 180
    step_bars: int = 10
    forward_bars: Tuple[int, ...] = (10, 20)
    min_confidence: float = 60.0
    min_rank: float = 70.0
    max_atr_percentile: float = 0.85
    max_distance_from_avg_atr: float = 2.5


@dataclass
class RunnerConfig:
    train_bars: int = 220
    step_bars: int = 20
    forward_bars: Tuple[int, ...] = (40, 60, 120)
    min_confidence: float = 65.0
    min_rank: float = 80.0
    max_atr_percentile: float = 0.80
    max_distance_from_base_atr: float = 5.0


@dataclass
class V4Config:
    tactical: TacticalConfig = TacticalConfig()
    runner: RunnerConfig = RunnerConfig()
    base_lookback: int = 150
    base_min_len: int = 20
    base_max_len: int = 70
    atr_len: int = 14
    vol_window: int = 20
    percentile_window: int = 120
    weekly_fast_ma: int = 20
    weekly_slow_ma: int = 40


# =========================================================
# Helpers
# =========================================================
def safe_volume(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, data=0.0)
    return pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _rank(x):
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])
    return series.rolling(window, min_periods=max(10, window // 3)).apply(_rank, raw=False)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length, min_periods=length).mean()


def weekly_trend_state(df: pd.DataFrame, fast_ma: int = 20, slow_ma: int = 40) -> pd.Series:
    weekly = df[["Close"]].resample("W-FRI").last().dropna()
    weekly["wma_fast"] = weekly["Close"].rolling(fast_ma, min_periods=fast_ma).mean()
    weekly["wma_slow"] = weekly["Close"].rolling(slow_ma, min_periods=slow_ma).mean()

    state = pd.Series(index=weekly.index, dtype="object")
    bull = (weekly["Close"] > weekly["wma_fast"]) & (weekly["wma_fast"] > weekly["wma_slow"])
    bear = (weekly["Close"] < weekly["wma_fast"]) & (weekly["wma_fast"] < weekly["wma_slow"])
    state.loc[bull] = "bullish"
    state.loc[bear] = "bearish"
    state = state.fillna("neutral")

    # Align back to daily
    aligned = state.reindex(df.index, method="ffill")
    return aligned.fillna("neutral")


def normalize_score(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, x)))


# =========================================================
# Feature Engine
# =========================================================
class FeatureEngine:
    def __init__(self, cfg: V4Config):
        self.cfg = cfg

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["ATR"] = compute_atr(d, self.cfg.atr_len)
        d["ATR_pct"] = d["ATR"] / d["Close"].replace(0, np.nan)
        d["ret_20"] = d["Close"] / d["Close"].shift(20) - 1.0
        d["ret_60"] = d["Close"] / d["Close"].shift(60) - 1.0
        d["vol_20"] = d["Close"].pct_change().rolling(self.cfg.vol_window).std()
        d["ATR_pct_pctile"] = rolling_percentile(d["ATR_pct"], self.cfg.percentile_window)
        d["vol_20_pctile"] = rolling_percentile(d["vol_20"], self.cfg.percentile_window)
        d["ma_20"] = d["Close"].rolling(20).mean()
        d["ma_50"] = d["Close"].rolling(50).mean()
        d["ma_100"] = d["Close"].rolling(100).mean()
        d["ma_200"] = d["Close"].rolling(200).mean()
        d["weekly_state"] = weekly_trend_state(d, self.cfg.weekly_fast_ma, self.cfg.weekly_slow_ma)
        return d

    def detect_base(self, df: pd.DataFrame, end_loc: int) -> Optional[Dict]:
        lookback = self.cfg.base_lookback
        min_len = self.cfg.base_min_len
        max_len = self.cfg.base_max_len

        start_loc = max(0, end_loc - lookback + 1)
        d = df.iloc[start_loc:end_loc + 1].copy()
        if len(d) < min_len + 8:
            return None

        best = None
        best_score = -1e9

        for win in range(min_len, min(max_len, len(d)) + 1):
            seg = d.iloc[-win:]
            before = d.iloc[:-win]
            if len(before) < 8:
                continue

            seg_high = float(seg["High"].max())
            seg_low = float(seg["Low"].min())
            seg_range = max(seg_high - seg_low, 1e-9)
            seg_mean = max(float(seg["Close"].mean()), 1e-9)

            pre = before.tail(min(30, len(before)))
            pre_return = float(pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1.0) if float(pre["Close"].iloc[0]) != 0 else 0.0

            compression = 1.0 - min(seg_range / seg_mean, 1.0)
            duration_score = min(len(seg) / max_len, 1.0)

            low_band = seg_low + 0.25 * seg_range
            high_band = seg_high - 0.25 * seg_range
            low_tests = int((seg["Low"] <= low_band).sum())
            high_tests = int((seg["High"] >= high_band).sum())
            repeated_tests = min((low_tests + high_tests) / max(len(seg) * 0.30, 1), 1.0)

            flatness_penalty = float(seg["Close"].pct_change().std() or 0.0)
            pretrend_score = 1.0 - min(abs(pre_return) * 3.0, 1.0)

            setup_score = (
                100.0 * (
                    0.32 * compression +
                    0.18 * pretrend_score +
                    0.18 * duration_score +
                    0.22 * repeated_tests
                )
                - 400.0 * flatness_penalty
            )
            setup_score = normalize_score(setup_score)

            if setup_score > best_score:
                best_score = setup_score
                best = {
                    "start_idx": seg.index[0],
                    "end_idx": seg.index[-1],
                    "base_high": seg_high,
                    "base_low": seg_low,
                    "base_mid": (seg_high + seg_low) / 2.0,
                    "base_range": seg_range,
                    "compression": compression,
                    "low_tests": low_tests,
                    "high_tests": high_tests,
                    "setup_score": setup_score,
                }

        return best

    def estimate_avg_zone(self, seg: pd.DataFrame) -> Dict:
        high = float(seg["High"].max())
        low = float(seg["Low"].min())
        seg_range = max(high - low, 1e-9)

        typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
        vol = safe_volume(seg)

        midpoint = (high + low) / 2.0
        tp_center = float(typical.mean())
        vwap_center = float((typical * vol).sum() / vol.sum()) if float(vol.sum()) > 0 else tp_center

        bins = np.linspace(low, high, 25)
        hist, edges = np.histogram(seg["Close"].clip(low, high), bins=bins)
        ix = int(np.argmax(hist))
        defend_low = float(edges[max(ix - 1, 0)])
        defend_high = float(edges[min(ix + 2, len(edges) - 1)])
        defended_center = (defend_low + defend_high) / 2.0

        avg_core = float(np.mean([midpoint, tp_center, vwap_center, defended_center]))
        avg_half = 0.12 * seg_range

        return {
            "avg_lower": avg_core - avg_half,
            "avg_core": avg_core,
            "avg_upper": avg_core + avg_half,
            "defend_low": defend_low,
            "defend_high": defend_high,
        }

    def anchored_vwap(self, df: pd.DataFrame, start_idx) -> pd.Series:
        seg = df.loc[start_idx:].copy()
        typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
        vol = safe_volume(seg).replace(0, np.nan)
        avwap = (typical * vol).fillna(0).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)

        out = pd.Series(index=df.index, dtype=float)
        out.loc[seg.index] = avwap
        return out


# =========================================================
# Signal Engine
# =========================================================
class SignalEngine:
    def __init__(self, cfg: V4Config):
        self.cfg = cfg

    def setup_score(self, base: Dict) -> float:
        return normalize_score(base["setup_score"])

    def confirmation_score(self, row: pd.Series, base: Dict, zone: Dict, avwap_value: float) -> float:
        breakout_confirm = 1.0 if row["Close"] > base["base_high"] else 0.0
        above_avg_core = 1.0 if row["Close"] > zone["avg_core"] else 0.0
        above_avwap = 1.0 if pd.notna(avwap_value) and row["Close"] > avwap_value else 0.0
        breakout_distance_quality = 1.0 if row["Close"] <= zone["avg_upper"] else 0.6

        score = 100.0 * (
            0.35 * breakout_confirm +
            0.25 * above_avg_core +
            0.25 * above_avwap +
            0.15 * breakout_distance_quality
        )
        return normalize_score(score)

    def hold_score(self, row: pd.Series, zone: Dict, avwap_value: float) -> float:
        above_avg_lower = 1.0 if row["Close"] >= zone["avg_lower"] else 0.0
        above_avwap = 1.0 if pd.notna(avwap_value) and row["Close"] >= avwap_value else 0.0
        reclaim_quality = 1.0 if row["Close"] >= zone["avg_core"] else 0.5
        trend_integrity = 1.0 if row["Close"] >= row.get("ma_50", row["Close"]) else 0.5

        score = 100.0 * (
            0.30 * above_avg_lower +
            0.30 * above_avwap +
            0.20 * reclaim_quality +
            0.20 * trend_integrity
        )
        return normalize_score(score)

    def regime_score(self, row: pd.Series, zone: Dict, base: Dict) -> float:
        atr_ok = 1.0 if pd.notna(row["ATR_pct_pctile"]) and row["ATR_pct_pctile"] <= 0.85 else 0.3
        vol_ok = 1.0 if pd.notna(row["vol_20_pctile"]) and row["vol_20_pctile"] <= 0.85 else 0.4
        weekly_trend_ok = 1.0 if row["weekly_state"] == "bullish" else (0.5 if row["weekly_state"] == "neutral" else 0.0)

        atr = row.get("ATR", np.nan)
        distance_from_avg_atr = 0.0
        if pd.notna(atr) and atr > 0:
            distance_from_avg_atr = max((row["Close"] - zone["avg_core"]) / atr, 0.0)

        extension_penalty = min(distance_from_avg_atr / 4.0, 1.0)
        score = 100.0 * (
            0.35 * atr_ok +
            0.25 * vol_ok +
            0.25 * weekly_trend_ok +
            0.15 * (1.0 - extension_penalty)
        )
        return normalize_score(score)

    def final_rank(self, setup_score: float, confirmation_score: float, hold_score: float, regime_score: float) -> float:
        rank = (
            0.30 * setup_score +
            0.30 * confirmation_score +
            0.25 * hold_score +
            0.15 * regime_score
        )
        return normalize_score(rank)

    def tier(self, rank: float) -> str:
        if rank >= 80:
            return "A+"
        if rank >= 70:
            return "A"
        if rank >= 60:
            return "B"
        return "C"

    def tactical_long_signal(self, row: pd.Series, base: Dict, zone: Dict, avwap_value: float, confidence: float, final_rank: float) -> bool:
        atr = row.get("ATR", np.nan)
        distance_from_avg_atr = np.inf
        if pd.notna(atr) and atr > 0:
            distance_from_avg_atr = (row["Close"] - zone["avg_core"]) / atr

        if row["Close"] <= base["base_high"]:
            return False
        if row["Close"] <= zone["avg_core"]:
            return False
        if pd.isna(avwap_value) or row["Close"] <= avwap_value:
            return False
        if confidence < self.cfg.tactical.min_confidence:
            return False
        if pd.notna(row["ATR_pct_pctile"]) and row["ATR_pct_pctile"] > self.cfg.tactical.max_atr_percentile:
            return False
        if distance_from_avg_atr > self.cfg.tactical.max_distance_from_avg_atr:
            return False
        if row["weekly_state"] == "bearish":
            return False
        if final_rank < self.cfg.tactical.min_rank:
            return False
        return True

    def runner_long_signal(self, row: pd.Series, base: Dict, zone: Dict, avwap_value: float, confidence: float, final_rank: float) -> bool:
        atr = row.get("ATR", np.nan)
        distance_from_base_atr = np.inf
        if pd.notna(atr) and atr > 0:
            distance_from_base_atr = (row["Close"] - base["base_high"]) / atr

        if not self.tactical_long_signal(row, base, zone, avwap_value, confidence, final_rank):
            return False
        if row["weekly_state"] != "bullish":
            return False
        if distance_from_base_atr > self.cfg.runner.max_distance_from_base_atr:
            return False
        if confidence < self.cfg.runner.min_confidence:
            return False
        if final_rank < self.cfg.runner.min_rank:
            return False
        return True

    def tactical_exit(self, row: pd.Series, zone: Dict, avwap_value: float) -> bool:
        if row["Close"] < zone["avg_lower"]:
            return True
        if pd.notna(avwap_value) and row["Close"] < avwap_value:
            return True
        return False

    def runner_exit(self, row: pd.Series, zone: Dict, avwap_value: float) -> bool:
        if row["weekly_state"] == "bearish":
            return True
        if row["Close"] < zone["avg_lower"] and pd.notna(avwap_value) and row["Close"] < avwap_value:
            return True
        return False

    def confidence(self, setup_score: float, confirmation_score: float, hold_score: float, regime_score: float) -> float:
        conf = 0.25 * setup_score + 0.35 * confirmation_score + 0.25 * hold_score + 0.15 * regime_score
        return normalize_score(conf)


# =========================================================
# Risk Engine
# =========================================================
class RiskEngine:
    def size_from_tier(self, tier: str, atr_pct_pctile: float, vol_pctile: float) -> float:
        if tier == "A+":
            size = 1.0
        elif tier == "A":
            size = 0.75
        elif tier == "B":
            size = 0.50
        else:
            size = 0.0

        if pd.notna(atr_pct_pctile) and atr_pct_pctile > 0.85:
            size *= 0.5
        if pd.notna(vol_pctile) and vol_pctile > 0.85:
            size *= 0.5

        return float(size)

    def no_trade_filter(self, confidence: float, final_rank: float, weekly_state: str) -> bool:
        if confidence < 60:
            return True
        if final_rank < 70:
            return True
        if weekly_state == "bearish":
            return True
        return False


# =========================================================
# Backtest Engine
# =========================================================
class BacktestEngine:
    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.features = FeatureEngine(cfg)
        self.signals = SignalEngine(cfg)
        self.risk = RiskEngine()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.features.enrich(df)

    def _compute_snapshot(self, d: pd.DataFrame, end_loc: int) -> Optional[Dict]:
        base = self.features.detect_base(d, end_loc)
        if not base:
            return None

        seg = d.loc[base["start_idx"]:base["end_idx"]].copy()
        zone = self.features.estimate_avg_zone(seg)
        d = d.copy()
        d["AVWAP"] = self.features.anchored_vwap(d, base["start_idx"])

        row = d.iloc[end_loc]
        avwap_value = d["AVWAP"].iloc[end_loc]

        setup_score = self.signals.setup_score(base)
        confirmation_score = self.signals.confirmation_score(row, base, zone, avwap_value)
        hold_score = self.signals.hold_score(row, zone, avwap_value)
        regime_score = self.signals.regime_score(row, zone, base)
        confidence = self.signals.confidence(setup_score, confirmation_score, hold_score, regime_score)
        final_rank = self.signals.final_rank(setup_score, confirmation_score, hold_score, regime_score)
        tier = self.signals.tier(final_rank)
        size = self.risk.size_from_tier(tier, row.get("ATR_pct_pctile", np.nan), row.get("vol_20_pctile", np.nan))

        return {
            "base": base,
            "zone": zone,
            "row": row,
            "avwap": avwap_value,
            "setup_score": setup_score,
            "confirmation_score": confirmation_score,
            "hold_score": hold_score,
            "regime_score": regime_score,
            "confidence": confidence,
            "final_rank": final_rank,
            "tier": tier,
            "size": size,
            "data": d,
        }

    def run_tactical_walk_forward(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = self.prepare(df)
        rows = []
        max_h = max(self.cfg.tactical.forward_bars)

        for end_loc in range(self.cfg.tactical.train_bars, len(d) - max_h, self.cfg.tactical.step_bars):
            snap = self._compute_snapshot(d.iloc[: end_loc + 1].copy(), end_loc)
            if not snap:
                continue

            row = snap["row"]
            signal = self.signals.tactical_long_signal(
                row=row,
                base=snap["base"],
                zone=snap["zone"],
                avwap_value=snap["avwap"],
                confidence=snap["confidence"],
                final_rank=snap["final_rank"],
            )

            if not signal or self.risk.no_trade_filter(snap["confidence"], snap["final_rank"], row["weekly_state"]):
                trade = "FLAT"
            else:
                trade = "LONG"

            entry_close = float(d["Close"].iloc[end_loc])

            out = {
                "symbol": symbol,
                "module": "TACTICAL",
                "signal_date": d.index[end_loc],
                "signal": trade,
                "confidence": snap["confidence"],
                "final_rank": snap["final_rank"],
                "tier": snap["tier"],
                "size": snap["size"],
                "entry_close": entry_close,
            }

            for h in self.cfg.tactical.forward_bars:
                exit_close = float(d["Close"].iloc[end_loc + h])
                raw_ret = exit_close / entry_close - 1.0
                strat_ret = raw_ret * snap["size"] if trade == "LONG" else 0.0
                out[f"ret_{h}"] = raw_ret
                out[f"strat_ret_{h}"] = strat_ret

            rows.append(out)

        return pd.DataFrame(rows)

    def run_runner_walk_forward(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = self.prepare(df)
        rows = []
        max_h = max(self.cfg.runner.forward_bars)

        for end_loc in range(self.cfg.runner.train_bars, len(d) - max_h, self.cfg.runner.step_bars):
            snap = self._compute_snapshot(d.iloc[: end_loc + 1].copy(), end_loc)
            if not snap:
                continue

            row = snap["row"]
            signal = self.signals.runner_long_signal(
                row=row,
                base=snap["base"],
                zone=snap["zone"],
                avwap_value=snap["avwap"],
                confidence=snap["confidence"],
                final_rank=snap["final_rank"],
            )

            if not signal or self.risk.no_trade_filter(snap["confidence"], snap["final_rank"], row["weekly_state"]):
                trade = "FLAT"
            else:
                trade = "LONG"

            entry_close = float(d["Close"].iloc[end_loc])

            out = {
                "symbol": symbol,
                "module": "RUNNER",
                "signal_date": d.index[end_loc],
                "signal": trade,
                "confidence": snap["confidence"],
                "final_rank": snap["final_rank"],
                "tier": snap["tier"],
                "size": snap["size"],
                "entry_close": entry_close,
            }

            for h in self.cfg.runner.forward_bars:
                exit_close = float(d["Close"].iloc[end_loc + h])
                raw_ret = exit_close / entry_close - 1.0
                strat_ret = raw_ret * snap["size"] if trade == "LONG" else 0.0
                out[f"ret_{h}"] = raw_ret
                out[f"strat_ret_{h}"] = strat_ret

            rows.append(out)

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(trades: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        rows = []
        active = trades[trades["signal"] == "LONG"].copy()

        for h in horizons:
            col = f"strat_ret_{h}"
            if col not in active.columns or active.empty:
                continue

            eq = (1.0 + active[col].fillna(0.0)).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1.0

            rows.append({
                "horizon": h,
                "signals": int(len(active)),
                "avg_return": float(active[col].mean()),
                "median_return": float(active[col].median()),
                "hit_rate": float((active[col] > 0).mean()),
                "cum_return": float(eq.iloc[-1] - 1.0) if len(eq) else np.nan,
                "max_drawdown": float(dd.min()) if len(dd) else np.nan,
            })

        return pd.DataFrame(rows)


# =========================================================
# I/O scaffold
# =========================================================
def load_csv_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {cols["date"]: "Date", cols["open"]: "Open", cols["high"]: "High", cols["low"]: "Low", cols["close"]: "Close"}
    if "volume" in cols:
        rename_map[cols["volume"]] = "Volume"

    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def run_single_file(csv_path: str, symbol: str, outdir: str = "v4_results") -> Dict[str, str]:
    cfg = V4Config()
    bt = BacktestEngine(cfg)
    df = load_csv_ohlcv(csv_path)

    tactical_trades = bt.run_tactical_walk_forward(df, symbol)
    runner_trades = bt.run_runner_walk_forward(df, symbol)

    tactical_summary = bt.summarize(tactical_trades, cfg.tactical.forward_bars)
    runner_summary = bt.summarize(runner_trades, cfg.runner.forward_bars)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    tactical_trades.to_csv(out / f"{symbol}_tactical_trades.csv", index=False)
    runner_trades.to_csv(out / f"{symbol}_runner_trades.csv", index=False)
    tactical_summary.to_csv(out / f"{symbol}_tactical_summary.csv", index=False)
    runner_summary.to_csv(out / f"{symbol}_runner_summary.csv", index=False)

    return {
        "tactical_trades": str(out / f"{symbol}_tactical_trades.csv"),
        "runner_trades": str(out / f"{symbol}_runner_trades.csv"),
        "tactical_summary": str(out / f"{symbol}_tactical_summary.csv"),
        "runner_summary": str(out / f"{symbol}_runner_summary.csv"),
    }


if __name__ == "__main__":
    print("V4 core lane skeleton ready. Use run_single_file(csv_path, symbol).")
