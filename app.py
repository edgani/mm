import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

st.set_page_config(page_title="V4 Core Lane App", layout="wide")

# =========================================================
# Config
# =========================================================
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
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    base_lookback: int = 150
    base_min_len: int = 20
    base_max_len: int = 70
    atr_len: int = 14
    vol_window: int = 20
    percentile_window: int = 120
    weekly_fast_ma: int = 20
    weekly_slow_ma: int = 40


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
    return state.reindex(df.index, method="ffill").fillna("neutral")


def normalize_score(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, x)))


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
        return normalize_score(100.0 * (0.35 * breakout_confirm + 0.25 * above_avg_core + 0.25 * above_avwap + 0.15 * breakout_distance_quality))

    def hold_score(self, row: pd.Series, zone: Dict, avwap_value: float) -> float:
        above_avg_lower = 1.0 if row["Close"] >= zone["avg_lower"] else 0.0
        above_avwap = 1.0 if pd.notna(avwap_value) and row["Close"] >= avwap_value else 0.0
        reclaim_quality = 1.0 if row["Close"] >= zone["avg_core"] else 0.5
        trend_integrity = 1.0 if row["Close"] >= row.get("ma_50", row["Close"]) else 0.5
        return normalize_score(100.0 * (0.30 * above_avg_lower + 0.30 * above_avwap + 0.20 * reclaim_quality + 0.20 * trend_integrity))

    def regime_score(self, row: pd.Series, zone: Dict) -> float:
        atr_ok = 1.0 if pd.notna(row["ATR_pct_pctile"]) and row["ATR_pct_pctile"] <= 0.85 else 0.3
        vol_ok = 1.0 if pd.notna(row["vol_20_pctile"]) and row["vol_20_pctile"] <= 0.85 else 0.4
        weekly_trend_ok = 1.0 if row["weekly_state"] == "bullish" else (0.5 if row["weekly_state"] == "neutral" else 0.0)
        atr = row.get("ATR", np.nan)
        distance_from_avg_atr = 0.0
        if pd.notna(atr) and atr > 0:
            distance_from_avg_atr = max((row["Close"] - zone["avg_core"]) / atr, 0.0)
        extension_penalty = min(distance_from_avg_atr / 4.0, 1.0)
        return normalize_score(100.0 * (0.35 * atr_ok + 0.25 * vol_ok + 0.25 * weekly_trend_ok + 0.15 * (1.0 - extension_penalty)))

    def final_rank(self, setup_score: float, confirmation_score: float, hold_score: float, regime_score: float) -> float:
        return normalize_score(0.30 * setup_score + 0.30 * confirmation_score + 0.25 * hold_score + 0.15 * regime_score)

    def tier(self, rank: float) -> str:
        if rank >= 80:
            return "A+"
        if rank >= 70:
            return "A"
        if rank >= 60:
            return "B"
        return "C"


def load_csv_ohlcv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {
        cols["date"]: "Date",
        cols["open"]: "Open",
        cols["high"]: "High",
        cols["low"]: "Low",
        cols["close"]: "Close",
    }
    if "volume" in cols:
        rename_map[cols["volume"]] = "Volume"

    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def analyze_latest(df: pd.DataFrame, cfg: V4Config):
    feat = FeatureEngine(cfg)
    sig = SignalEngine(cfg)

    d = feat.enrich(df)
    end_loc = len(d) - 1
    base = feat.detect_base(d, end_loc)
    if not base:
        return None

    seg = d.loc[base["start_idx"]:base["end_idx"]].copy()
    zone = feat.estimate_avg_zone(seg)
    d["AVWAP"] = feat.anchored_vwap(d, base["start_idx"])
    row = d.iloc[end_loc]
    avwap_value = d["AVWAP"].iloc[end_loc]

    setup_score = sig.setup_score(base)
    confirmation_score = sig.confirmation_score(row, base, zone, avwap_value)
    hold_score = sig.hold_score(row, zone, avwap_value)
    regime_score = sig.regime_score(row, zone)
    confidence = normalize_score(0.25 * setup_score + 0.35 * confirmation_score + 0.25 * hold_score + 0.15 * regime_score)
    final_rank = sig.final_rank(setup_score, confirmation_score, hold_score, regime_score)
    tier = sig.tier(final_rank)

    tactical_ok = (
        row["Close"] > base["base_high"]
        and row["Close"] > zone["avg_core"]
        and pd.notna(avwap_value)
        and row["Close"] > avwap_value
        and confidence >= cfg.tactical.min_confidence
        and final_rank >= cfg.tactical.min_rank
    )

    runner_ok = (
        tactical_ok
        and row["weekly_state"] == "bullish"
        and confidence >= cfg.runner.min_confidence
        and final_rank >= cfg.runner.min_rank
    )

    return {
        "data": d,
        "base": base,
        "zone": zone,
        "avwap_value": avwap_value,
        "setup_score": setup_score,
        "confirmation_score": confirmation_score,
        "hold_score": hold_score,
        "regime_score": regime_score,
        "confidence": confidence,
        "final_rank": final_rank,
        "tier": tier,
        "tactical_signal": "LONG" if tactical_ok else "FLAT",
        "runner_signal": "LONG" if runner_ok else "FLAT",
    }


st.title("V4 Core Lane App")
st.caption("Sekarang ada isi. Upload CSV OHLCV lalu app bakal baca structure V4 buat latest bar.")

with st.sidebar:
    st.header("Config")
    symbol = st.text_input("Symbol label", value="AAPL")
    uploaded = st.file_uploader("Upload CSV OHLCV", type=["csv"])
    st.markdown("CSV minimal:")
    st.code("Date,Open,High,Low,Close,Volume", language=None)

cfg = V4Config()

if uploaded is None:
    st.info("Upload CSV dulu. File yang kemarin kosong karena yang dideploy itu module skeleton, bukan UI Streamlit.")
else:
    try:
        df = load_csv_ohlcv(uploaded)
        if len(df) < 250:
            st.warning(f"Data cuma {len(df)} bar. Lebih bagus >= 250 bar.")
        res = analyze_latest(df, cfg)
        if res is None:
            st.error("Belum ketemu base/range yang cukup jelas di latest window.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tactical", res["tactical_signal"])
            c2.metric("Runner", res["runner_signal"])
            c3.metric("Confidence", f'{res["confidence"]:.1f}')
            c4.metric("Final Rank", f'{res["final_rank"]:.1f}')

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Tier", res["tier"])
            c6.metric("Base High", f'{res["base"]["base_high"]:.4f}')
            c7.metric("Avg Core", f'{res["zone"]["avg_core"]:.4f}')
            c8.metric("AVWAP", f'{res["avwap_value"]:.4f}' if pd.notna(res["avwap_value"]) else "NA")

            st.subheader(f"{symbol} Close vs AVWAP")
            chart_df = res["data"][["Close", "AVWAP"]].copy()
            st.line_chart(chart_df)

            st.subheader("Scores")
            score_df = pd.DataFrame(
                [
                    ["Setup Score", round(res["setup_score"], 2)],
                    ["Confirmation Score", round(res["confirmation_score"], 2)],
                    ["Hold Score", round(res["hold_score"], 2)],
                    ["Regime Score", round(res["regime_score"], 2)],
                    ["Confidence", round(res["confidence"], 2)],
                    ["Final Rank", round(res["final_rank"], 2)],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.subheader("Base / Zone Info")
            base_df = pd.DataFrame(
                [
                    ["Base Start", str(res["base"]["start_idx"])],
                    ["Base End", str(res["base"]["end_idx"])],
                    ["Base Low", round(res["base"]["base_low"], 4)],
                    ["Base High", round(res["base"]["base_high"], 4)],
                    ["Compression", round(res["base"]["compression"], 4)],
                    ["Low Tests", int(res["base"]["low_tests"])],
                    ["High Tests", int(res["base"]["high_tests"])],
                    ["Avg Lower", round(res["zone"]["avg_lower"], 4)],
                    ["Avg Core", round(res["zone"]["avg_core"], 4)],
                    ["Avg Upper", round(res["zone"]["avg_upper"], 4)],
                    ["Defend Low", round(res["zone"]["defend_low"], 4)],
                    ["Defend High", round(res["zone"]["defend_high"], 4)],
                ],
                columns=["Field", "Value"],
            )
            st.dataframe(base_df, use_container_width=True, hide_index=True)

            st.subheader("Interpretation")
            notes = []
            if res["tactical_signal"] == "LONG":
                notes.append("- Tactical long valid: latest close lolos setup + confirmation minimum.")
            else:
                notes.append("- Tactical long belum valid: masih kurang di breakout / avg core / AVWAP / score.")
            if res["runner_signal"] == "LONG":
                notes.append("- Runner long valid: structure cukup kuat untuk horizon panjang.")
            else:
                notes.append("- Runner long belum valid: HTF / confidence / final rank belum cukup.")
            if res["data"]["weekly_state"].iloc[-1] == "bullish":
                notes.append("- Weekly state bullish.")
            else:
                notes.append(f"- Weekly state sekarang {res['data']['weekly_state'].iloc[-1]}.")
            st.markdown("\n".join(notes))

            with st.expander("Raw latest rows"):
                st.dataframe(res["data"].tail(30), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
