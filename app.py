import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Observational Market Structure Engine V4", layout="wide")

TIMEFRAME_MAP = {
    "1D": {"period": "3y", "interval": "1d"},
    "1W": {"period": "8y", "interval": "1wk"},
    "4H": {"period": "730d", "interval": "1h"},  # proxied from 1H
    "1H": {"period": "730d", "interval": "1h"},
}

DEFAULT_WATCHLIST = """AAPL,NVDA,MSFT,AMZN,TSLA,BBCA.JK,TLKM.JK,BMRI.JK,^JKSE,GC=F,CL=F,EURUSD=X,USDJPY=X,BTC-USD,ETH-USD"""


# ============================================================
# HELPERS
# ============================================================
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    aliases = {
        "XAUUSD": "XAUUSD=X",
        "XAGUSD": "XAGUSD=X",
        "WTI": "CL=F",
        "BRENT": "BZ=F",
        "DXY": "DX-Y.NYB",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "IHSG": "^JKSE",
    }
    return aliases.get(s, s)


def classify_asset(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".JK") or s == "^JKSE":
        return "IHSG"
    if s.endswith("=X"):
        return "Forex/Metals"
    if s.endswith("-USD"):
        return "Crypto"
    if s.endswith("=F"):
        return "Futures/Commodities"
    return "US Stocks/Other"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol} ({period}, {interval})")
    df = _flatten_columns(df).copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna()
    if interval == "1h" and len(df) > 10:
        # Proxy 4H from 1H later by resample, keep original here.
        pass
    return df


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    df = df_1h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    out = pd.DataFrame()
    out["Open"] = df["Open"].resample("4H").first()
    out["High"] = df["High"].resample("4H").max()
    out["Low"] = df["Low"].resample("4H").min()
    out["Close"] = df["Close"].resample("4H").last()
    out["Volume"] = df["Volume"].resample("4H").sum() if "Volume" in df.columns else 0
    return out.dropna()


def rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def bounded(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def safe_mean(vals: List[float], default: float = np.nan) -> float:
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else default


@dataclass
class AnalysisResult:
    symbol: str
    asset_class: str
    timeframe: str
    last_close: float
    base_low: float
    base_high: float
    core_low: float
    core_high: float
    avg_lower: float
    avg_core: float
    avg_upper: float
    defended_level: float
    invalidation: float
    avwap: float
    accumulation_score: float
    distribution_score: float
    holding_score: float
    release_risk: float
    confidence: float
    base_quality: float
    reaccumulation_score: float
    redistribution_score: float
    zone_alert: str
    structure_state: str
    notes: List[str]
    df: pd.DataFrame
    profile: pd.DataFrame


# ============================================================
# CORE ENGINE
# ============================================================
def detect_base_window(df: pd.DataFrame, lookback: int = 180) -> Tuple[int, int, Dict[str, float]]:
    data = df.tail(max(lookback, 80)).copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    vol = data["Volume"] if "Volume" in data.columns else pd.Series(0, index=data.index)

    n = len(data)
    if n < 60:
        return 0, n - 1, {"base_quality": 30.0}

    best = None
    best_score = -1e9
    ranges = []

    for win in range(25, min(120, n - 5), 5):
        seg = data.iloc[-win:]
        seg_low = seg["Low"].min()
        seg_high = seg["High"].max()
        width = (seg_high - seg_low) / max(seg["Close"].iloc[-1], 1e-9)
        if width <= 0:
            continue

        pre = data.iloc[:-win]
        if len(pre) < 20:
            continue
        pre_win = min(len(pre), max(20, win // 2))
        pre_seg = pre.iloc[-pre_win:]
        pre_drop = (pre_seg["Close"].iloc[-1] - pre_seg["Close"].iloc[0]) / max(pre_seg["Close"].iloc[0], 1e-9)
        downside_context = -pre_drop

        touch_low = ((seg["Low"] <= seg_low * 1.01)).sum()
        touch_high = ((seg["High"] >= seg_high * 0.99)).sum()
        close_pos = (seg["Close"] - seg_low) / max(seg_high - seg_low, 1e-9)
        center_bias = 1 - abs(close_pos.mean() - 0.5) * 2
        rejection_low = (((seg["Close"] - seg["Low"]) / (seg["High"] - seg["Low"] + 1e-9)) > 0.6).mean()
        contraction = 1 - min(1.0, (seg["High"] - seg["Low"]).rolling(10).mean().iloc[-1] / max((pre_seg["High"] - pre_seg["Low"]).rolling(10).mean().iloc[-1], 1e-9))
        vol_support = float((vol.loc[seg.index] > vol.rolling(30).mean().loc[seg.index].fillna(0)).mean())

        score = (
            30 * min(max(downside_context, 0), 0.25) / 0.25
            + 15 * min(touch_low / 4, 1)
            + 8 * min(touch_high / 4, 1)
            + 15 * center_bias
            + 15 * rejection_low
            + 10 * max(contraction, 0)
            + 7 * vol_support
        ) - 20 * min(width / 0.25, 1)

        ranges.append((win, score))
        if score > best_score:
            best_score = score
            best = {
                "win": win,
                "seg_low": seg_low,
                "seg_high": seg_high,
                "base_quality": bounded(score),
            }

    if not best:
        return 0, n - 1, {"base_quality": 25.0}

    start = len(df) - best["win"]
    end = len(df) - 1
    return start, end, best



def build_price_profile(seg: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    seg_low = float(seg["Low"].min())
    seg_high = float(seg["High"].max())
    if seg_high <= seg_low:
        return pd.DataFrame({"price": [seg_low], "weight": [1.0]})
    edges = np.linspace(seg_low, seg_high, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    weights = np.zeros(bins)

    vol_series = seg["Volume"] if "Volume" in seg.columns else pd.Series(1.0, index=seg.index)
    vol_fallback = max(float(vol_series.replace(0, np.nan).median() or 1.0), 1.0)

    for _, row in seg.iterrows():
        lo, hi, close = float(row["Low"]), float(row["High"]), float(row["Close"])
        vol = float(row.get("Volume", vol_fallback))
        bar_low = min(lo, hi)
        bar_high = max(lo, hi)
        span = max(bar_high - bar_low, 1e-9)
        overlap = np.maximum(0, np.minimum(edges[1:], bar_high) - np.maximum(edges[:-1], bar_low))
        frac = overlap / span
        pos_bonus = 0.2 + 0.8 * np.exp(-((mids - close) ** 2) / max((span * 0.6) ** 2, 1e-9))
        weights += vol * frac * pos_bonus

    profile = pd.DataFrame({"price": mids, "weight": weights})
    profile["weight"] = profile["weight"] / max(profile["weight"].sum(), 1e-9)
    return profile.sort_values("price").reset_index(drop=True)



def anchored_vwap_from_index(df: pd.DataFrame, start_idx: int) -> pd.Series:
    sub = df.iloc[start_idx:].copy()
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3.0
    vol = sub["Volume"].fillna(0).replace(0, 1)
    cum_pv = (typical * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    avwap = cum_pv / cum_v
    out = pd.Series(index=df.index, dtype=float)
    out.loc[sub.index] = avwap
    return out



def score_structure(df: pd.DataFrame, timeframe: str, symbol: str) -> AnalysisResult:
    base_start, base_end, meta = detect_base_window(df)
    base = df.iloc[base_start: base_end + 1].copy()
    profile = build_price_profile(base)

    base_low = float(base["Low"].min())
    base_high = float(base["High"].max())
    last_close = float(df["Close"].iloc[-1])
    base_width = max(base_high - base_low, 1e-9)

    cumw = profile["weight"].cumsum()
    core = profile[(cumw >= 0.2) & (cumw <= 0.8)]
    if core.empty:
        core = profile.iloc[max(0, len(profile)//3): min(len(profile), 2*len(profile)//3 + 1)]
    core_low = float(core["price"].min())
    core_high = float(core["price"].max())

    time_center = float(base["Close"].median())
    vol_center = float((profile["price"] * profile["weight"]).sum())
    base_mid = (base_low + base_high) / 2

    avwap_series = anchored_vwap_from_index(df, base_start)
    avwap = float(avwap_series.dropna().iloc[-1]) if not avwap_series.dropna().empty else last_close

    # Most defended level: cluster around lows where subsequent bounce is strongest.
    lows = base["Low"].rolling(3).min().dropna()
    defenses = []
    for idx in range(3, len(base) - 5):
        lvl = float(base["Low"].iloc[idx])
        fwd = base["Close"].iloc[idx + 1: idx + 6]
        bounce = float((fwd.max() - lvl) / max(lvl, 1e-9))
        defenses.append((lvl, bounce))
    defended_level = float(sorted(defenses, key=lambda x: x[1], reverse=True)[0][0]) if defenses else core_low

    avg_core = safe_mean([base_mid, time_center, vol_center, defended_level, avwap], default=base_mid)
    avg_lower = safe_mean([core_low, defended_level, min(avwap, avg_core)], default=core_low)
    avg_upper = safe_mean([core_high, max(avwap, avg_core), base_mid], default=core_high)
    if avg_lower > avg_upper:
        avg_lower, avg_upper = avg_upper, avg_lower

    # Structural features
    recent = df.tail(30).copy()
    if len(recent) < 10:
        recent = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0) if "Volume" in df.columns else pd.Series(0, index=df.index)

    touch_base_low_recent = float((recent["Low"] <= base_low * 1.01).mean())
    touch_base_high_recent = float((recent["High"] >= base_high * 0.99).mean())
    above_base_mid = float((recent["Close"] > base_mid).mean())
    above_avg = float((recent["Close"] > avg_core).mean())
    above_avwap = float((recent["Close"] > avwap).mean())

    pullback_depth = float((recent["Close"].max() - recent["Close"].min()) / max(base_width, 1e-9))
    downside_reject_recent = float((((recent["Close"] - recent["Low"]) / (recent["High"] - recent["Low"] + 1e-9)) > 0.6).mean())
    upside_reject_recent = float((((recent["High"] - recent["Close"]) / (recent["High"] - recent["Low"] + 1e-9)) > 0.6).mean())

    breakout_progress = float((last_close - base_high) / max(base_width, 1e-9))
    breakdown_progress = float((base_low - last_close) / max(base_width, 1e-9))

    # Directional efficiency
    ret20 = close.pct_change(20)
    ret5 = close.pct_change(5)
    vol_z = rolling_zscore(vol.replace(0, np.nan).fillna(method="ffill").fillna(0), 50).iloc[-1]
    dist_to_avg = (last_close - avg_core) / max(base_width, 1e-9)
    dist_to_defended = (last_close - defended_level) / max(base_width, 1e-9)

    # Reaccumulation / redistribution logic after initial breakout
    post_base = df.iloc[base_end + 1:].copy()
    reaccumulation_score = 0.0
    redistribution_score = 0.0
    if len(post_base) >= 15:
        pb_high = float(post_base["High"].max())
        pb_low = float(post_base["Low"].min())
        pb_width = max(pb_high - pb_low, 1e-9)
        post_above_base = float((post_base["Close"] > base_high).mean())
        post_below_base = float((post_base["Close"] < base_low).mean())
        shallow_pullbacks = float(((post_base["Low"] > avg_lower * 0.98)).mean())
        failure_breaks_down = float(((post_base["Low"] < base_high) & (post_base["Close"] > base_high)).mean()) if base_high else 0
        failure_breaks_up = float(((post_base["High"] > base_low) & (post_base["Close"] < base_low)).mean()) if base_low else 0
        reaccumulation_score = bounded(
            35 * max(post_above_base, 0)
            + 25 * max(shallow_pullbacks, 0)
            + 25 * max(failure_breaks_down, 0)
            + 15 * max((last_close - max(base_high, avg_core)) / max(pb_width, 1e-9), 0)
        )
        redistribution_score = bounded(
            35 * max(post_below_base, 0)
            + 25 * max((1 - shallow_pullbacks), 0)
            + 25 * max(failure_breaks_up, 0)
            + 15 * max((min(base_low, avg_core) - last_close) / max(pb_width, 1e-9), 0)
        )

    accumulation_score = bounded(
        18 * meta.get("base_quality", 30) / 100
        + 12 * touch_base_low_recent
        + 16 * downside_reject_recent
        + 10 * above_base_mid
        + 12 * above_avg
        + 10 * above_avwap
        + 12 * max(dist_to_avg, -0.5) + 6
        + 10 * max(breakout_progress, 0)
        + 8 * (reaccumulation_score / 100)
    )

    distribution_score = bounded(
        15 * meta.get("base_quality", 30) / 100
        + 15 * touch_base_high_recent
        + 18 * upside_reject_recent
        + 14 * (1 - above_base_mid)
        + 14 * (1 - above_avg)
        + 10 * max(-dist_to_avg, -0.5) + 6
        + 10 * max(breakdown_progress, 0)
        + 8 * (redistribution_score / 100)
    )

    holding_score = bounded(
        22 * above_avg
        + 18 * above_avwap
        + 16 * max(dist_to_defended, -0.5) + 8
        + 16 * max(breakout_progress, 0)
        + 16 * (1 - min(max(pullback_depth / 3.0, 0), 1))
        + 12 * (reaccumulation_score / 100)
    )

    release_risk = bounded(
        22 * upside_reject_recent
        + 18 * (1 - above_avg)
        + 16 * max(breakdown_progress, 0)
        + 14 * min(max(pullback_depth / 3.0, 0), 1)
        + 15 * (redistribution_score / 100)
        + 15 * max(-dist_to_defended, 0)
    )

    base_quality = float(meta.get("base_quality", 30.0))
    confidence = bounded(
        0.35 * base_quality
        + 0.2 * max(accumulation_score, distribution_score)
        + 0.25 * abs(accumulation_score - distribution_score)
        + 0.2 * min(abs(float(vol_z) if pd.notna(vol_z) else 0), 2.5) * 20 / 2.5
    )

    invalidation = float(min(defended_level, avg_lower) * 0.985) if holding_score >= release_risk else float(max(defended_level, avg_upper) * 1.015)

    # Zone alert
    zone_alert = "Neutral"
    if abs(last_close - avg_core) / max(base_width, 1e-9) <= 0.20:
        zone_alert = "Near Avg Zone"
    if abs(last_close - defended_level) / max(base_width, 1e-9) <= 0.15:
        zone_alert = "Near Defended Level"
    if last_close > avg_upper and holding_score > 60:
        zone_alert = "Above Avg Zone / Structure Healthy"
    if last_close < avg_lower and release_risk > 60:
        zone_alert = "Below Avg Zone / Structure Weak"

    # Structure state
    if accumulation_score >= 60 and holding_score >= release_risk:
        structure_state = "Accumulation / Holding Bias"
    elif distribution_score >= 60 and release_risk > holding_score:
        structure_state = "Distribution / Release Bias"
    elif reaccumulation_score >= 60:
        structure_state = "Re-Accumulation Bias"
    elif redistribution_score >= 60:
        structure_state = "Re-Distribution Bias"
    else:
        structure_state = "Neutral / Mixed"

    notes = []
    if reaccumulation_score >= 60:
        notes.append("Post-base action still holding above key structure; re-accumulation probability elevated.")
    if redistribution_score >= 60:
        notes.append("Post-base action is losing key structure; re-distribution probability elevated.")
    if last_close > avwap:
        notes.append("Price is above base-anchored VWAP.")
    else:
        notes.append("Price is below base-anchored VWAP.")
    if last_close > defended_level:
        notes.append("Market still trades above the strongest defended area.")
    else:
        notes.append("Market is trading at or below the strongest defended area.")

    return AnalysisResult(
        symbol=symbol,
        asset_class=classify_asset(symbol),
        timeframe=timeframe,
        last_close=last_close,
        base_low=base_low,
        base_high=base_high,
        core_low=core_low,
        core_high=core_high,
        avg_lower=avg_lower,
        avg_core=avg_core,
        avg_upper=avg_upper,
        defended_level=defended_level,
        invalidation=invalidation,
        avwap=avwap,
        accumulation_score=accumulation_score,
        distribution_score=distribution_score,
        holding_score=holding_score,
        release_risk=release_risk,
        confidence=confidence,
        base_quality=base_quality,
        reaccumulation_score=reaccumulation_score,
        redistribution_score=redistribution_score,
        zone_alert=zone_alert,
        structure_state=structure_state,
        notes=notes,
        df=df,
        profile=profile,
    )



def analyze_symbol(symbol: str, timeframe: str) -> AnalysisResult:
    s = normalize_symbol(symbol)
    tf = TIMEFRAME_MAP[timeframe]
    raw = fetch_ohlcv(s, tf["period"], tf["interval"])
    if timeframe == "4H":
        raw = resample_4h(raw)
    return score_structure(raw, timeframe, s)



def consensus_view(results: List[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Timeframe": r.timeframe,
                "State": r.structure_state,
                "Accumulation": round(r.accumulation_score, 1),
                "Distribution": round(r.distribution_score, 1),
                "Holding": round(r.holding_score, 1),
                "ReleaseRisk": round(r.release_risk, 1),
                "ReAccum": round(r.reaccumulation_score, 1),
                "ReDistrib": round(r.redistribution_score, 1),
                "Confidence": round(r.confidence, 1),
                "ZoneAlert": r.zone_alert,
            }
        )
    return pd.DataFrame(rows)



def ranking_score(r: AnalysisResult) -> float:
    return float(
        0.32 * r.accumulation_score
        + 0.26 * r.holding_score
        + 0.18 * r.reaccumulation_score
        + 0.14 * r.confidence
        - 0.10 * r.release_risk
    )



def ranking_score_short(r: AnalysisResult) -> float:
    return float(
        0.32 * r.distribution_score
        + 0.26 * r.release_risk
        + 0.18 * r.redistribution_score
        + 0.14 * r.confidence
        - 0.10 * r.holding_score
    )



def walk_forward_backtest(symbol: str, timeframe: str, lookahead: int = 10) -> pd.DataFrame:
    s = normalize_symbol(symbol)
    tf = TIMEFRAME_MAP[timeframe]
    raw = fetch_ohlcv(s, tf["period"], tf["interval"])
    if timeframe == "4H":
        raw = resample_4h(raw)
    raw = raw.dropna().copy()

    min_bars = 140
    if len(raw) < min_bars + lookahead + 20:
        return pd.DataFrame()

    rows = []
    step = max(5, lookahead)
    for end in range(min_bars, len(raw) - lookahead, step):
        sub = raw.iloc[:end].copy()
        try:
            res = score_structure(sub, timeframe, s)
        except Exception:
            continue
        fwd = raw["Close"].iloc[end + lookahead - 1] / raw["Close"].iloc[end - 1] - 1
        bull_signal = ranking_score(res)
        bear_signal = ranking_score_short(res)
        rows.append(
            {
                "Date": raw.index[end - 1],
                "ForwardReturn": fwd,
                "BullScore": bull_signal,
                "BearScore": bear_signal,
                "Acc": res.accumulation_score,
                "Hold": res.holding_score,
                "Release": res.release_risk,
                "Conf": res.confidence,
            }
        )
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["BullBucket"] = pd.qcut(bt["BullScore"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    bt["BearBucket"] = pd.qcut(bt["BearScore"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return bt


# ============================================================
# VISUALS
# ============================================================
def make_chart(r: AnalysisResult) -> go.Figure:
    df = r.df.tail(220).copy()
    avwap_series = anchored_vwap_from_index(r.df, max(0, len(r.df) - len(df)))
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for y, name, color in [
        (r.base_low, "Base Low", "rgba(255,120,120,0.6)"),
        (r.base_high, "Base High", "rgba(120,255,120,0.6)"),
        (r.avg_lower, "Avg Lower", "rgba(255,255,120,0.5)"),
        (r.avg_core, "Avg Core", "rgba(255,200,0,0.9)"),
        (r.avg_upper, "Avg Upper", "rgba(255,255,120,0.5)"),
        (r.defended_level, "Defended", "rgba(0,180,255,0.9)"),
        (r.avwap, "Base AVWAP", "rgba(255,0,255,0.8)"),
    ]:
        fig.add_hline(y=y, line_width=1.2, line_dash="dot", line_color=color, annotation_text=name)

    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================================================
# UI
# ============================================================
st.title("Observational Market Structure Engine V4")
st.caption("Pure chart-observation engine: accumulation, distribution, estimated average zone, defended level, AVWAP, re-accumulation / re-distribution, scanner, and walk-forward sanity test.")

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Single Symbol", value="AAPL")
    timeframe = st.selectbox("Timeframe", ["1W", "1D", "4H", "1H"], index=1)
    run_single = st.button("Run Single Analysis", type="primary")
    st.divider()
    watchlist_text = st.text_area("Scanner Watchlist (comma separated)", value=DEFAULT_WATCHLIST, height=120)
    scan_timeframe = st.selectbox("Scanner Timeframe", ["1W", "1D", "4H", "1H"], index=1)
    run_scan = st.button("Run Scanner")
    st.divider()
    bt_symbol = st.text_input("Backtest Symbol", value="AAPL")
    bt_timeframe = st.selectbox("Backtest Timeframe", ["1W", "1D", "4H", "1H"], index=1)
    lookahead = st.slider("Forward Lookahead Bars", 3, 30, 10)
    run_bt = st.button("Run Walk-Forward Test")


def render_result(r: AnalysisResult):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("State", r.structure_state)
    c2.metric("Last Close", f"{r.last_close:,.4f}")
    c3.metric("Zone Alert", r.zone_alert)
    c4.metric("Asset Class", r.asset_class)
    c5.metric("Confidence", f"{r.confidence:.1f}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accumulation", f"{r.accumulation_score:.1f}")
    m2.metric("Distribution", f"{r.distribution_score:.1f}")
    m3.metric("Holding", f"{r.holding_score:.1f}")
    m4.metric("Release Risk", f"{r.release_risk:.1f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Re-Accum", f"{r.reaccumulation_score:.1f}")
    m6.metric("Re-Distrib", f"{r.redistribution_score:.1f}")
    m7.metric("Base Quality", f"{r.base_quality:.1f}")
    m8.metric("Base AVWAP", f"{r.avwap:,.4f}")

    t1, t2 = st.columns([2, 1])
    with t1:
        st.plotly_chart(make_chart(r), use_container_width=True)
    with t2:
        table = pd.DataFrame(
            {
                "Metric": [
                    "Base Low", "Base High", "Core Low", "Core High",
                    "Avg Lower", "Avg Core", "Avg Upper", "Defended Level", "Invalidation"
                ],
                "Value": [
                    r.base_low, r.base_high, r.core_low, r.core_high,
                    r.avg_lower, r.avg_core, r.avg_upper, r.defended_level, r.invalidation,
                ]
            }
        )
        st.dataframe(table.style.format({"Value": "{:,.4f}"}), use_container_width=True, hide_index=True)
        st.markdown("**Logic Notes**")
        for note in r.notes:
            st.write(f"- {note}")

    st.markdown("**Price Profile Inside Base**")
    prof = r.profile.copy()
    prof["weight_pct"] = prof["weight"] * 100
    st.bar_chart(prof.set_index("price")["weight_pct"], use_container_width=True)


if run_single:
    try:
        tfs = ["1W", "1D", "4H"]
        results = [analyze_symbol(symbol, tf) for tf in tfs]
        primary = next((r for r in results if r.timeframe == timeframe), results[1])
        st.subheader(f"Single Instrument Analysis — {primary.symbol}")
        render_result(primary)
        st.markdown("**Multi-Timeframe Consensus**")
        st.dataframe(consensus_view(results), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

if run_scan:
    syms = [normalize_symbol(s) for s in watchlist_text.split(",") if s.strip()]
    rows = []
    progress = st.progress(0)
    status = st.empty()
    for i, s in enumerate(syms, start=1):
        status.write(f"Scanning {s} ({i}/{len(syms)})")
        try:
            r = analyze_symbol(s, scan_timeframe)
            rows.append(
                {
                    "Symbol": r.symbol,
                    "AssetClass": r.asset_class,
                    "State": r.structure_state,
                    "LastClose": r.last_close,
                    "Accumulation": round(r.accumulation_score, 1),
                    "Distribution": round(r.distribution_score, 1),
                    "Holding": round(r.holding_score, 1),
                    "ReleaseRisk": round(r.release_risk, 1),
                    "ReAccum": round(r.reaccumulation_score, 1),
                    "ReDistrib": round(r.redistribution_score, 1),
                    "Confidence": round(r.confidence, 1),
                    "BullRank": round(ranking_score(r), 2),
                    "BearRank": round(ranking_score_short(r), 2),
                    "AvgCore": r.avg_core,
                    "Defended": r.defended_level,
                    "ZoneAlert": r.zone_alert,
                }
            )
        except Exception as e:
            rows.append({"Symbol": s, "Error": str(e)})
        progress.progress(i / len(syms))
    status.empty()
    scan_df = pd.DataFrame(rows)
    if not scan_df.empty:
        st.subheader("Scanner Results")
        if "BullRank" in scan_df.columns:
            bull = scan_df.dropna(subset=["BullRank"]).sort_values("BullRank", ascending=False)
            bear = scan_df.dropna(subset=["BearRank"]).sort_values("BearRank", ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top Bullish Structure Candidates**")
                st.dataframe(bull.head(20), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Top Bearish / Release Candidates**")
                st.dataframe(bear.head(20), use_container_width=True, hide_index=True)
            csv = scan_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Scanner CSV", data=csv, file_name="scanner_results_v4.csv", mime="text/csv")
        else:
            st.dataframe(scan_df, use_container_width=True, hide_index=True)

if run_bt:
    try:
        bt = walk_forward_backtest(bt_symbol, bt_timeframe, lookahead=lookahead)
        if bt.empty:
            st.warning("Not enough data for backtest.")
        else:
            st.subheader(f"Walk-Forward Sanity Test — {normalize_symbol(bt_symbol)}")
            summary = pd.DataFrame(
                {
                    "BullBucketMeanForwardReturn": bt.groupby("BullBucket", observed=False)["ForwardReturn"].mean(),
                    "BearBucketMeanForwardReturn": bt.groupby("BearBucket", observed=False)["ForwardReturn"].mean(),
                }
            )
            st.dataframe(summary.style.format("{:.2%}"), use_container_width=True)
            st.line_chart(bt.set_index("Date")[["BullScore", "BearScore"]], use_container_width=True)
            st.line_chart(bt.set_index("Date")[["ForwardReturn"]], use_container_width=True)
            st.dataframe(bt.tail(25), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.markdown(
    "**Important**: ini inferensi observasional dari OHLCV + struktur. Ini bukan pembaca inventory asli pelaku besar. "
    "Untuk forex/crypto/futures, kualitas volume dan latensi data publik bisa beda dari data exchange-grade."
)
