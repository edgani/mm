import importlib.util
import streamlit as st

st.set_page_config(page_title="Observational Accumulation Engine", layout="wide")

# Graceful dependency check so Streamlit Cloud shows a useful message
_missing = [pkg for pkg in ["numpy", "pandas", "yfinance"] if importlib.util.find_spec(pkg) is None]
if _missing:
    st.error(
        "Missing Python packages: " + ", ".join(_missing) + "\n\n"
        "Put a file named requirements.txt in the SAME folder as app.py with:\n"
        "streamlit>=1.36.0\n"
        "pandas>=2.2.0\n"
        "numpy>=1.26.0\n"
        "yfinance>=0.2.54\n\n"
        "Then redeploy / reboot the app."
    )
    st.stop()

import numpy as np
import pandas as pd
import yfinance as yf


@st.cache_data(show_spinner=False, ttl=900)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol.strip().upper(),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy().dropna(subset=["Close"])
    return df


def safe_volume(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, data=0.0)
    return pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)


def find_base_window(df: pd.DataFrame, lookback: int = 120, min_window: int = 18, max_window: int = 60):
    d = df.tail(lookback).copy()
    if len(d) < min_window + 5:
        return None

    best = None
    best_score = -1e9

    for end_idx in range(min_window, len(d) + 1):
        for win in range(min_window, min(max_window, end_idx) + 1):
            seg = d.iloc[end_idx - win:end_idx]
            before = d.iloc[: end_idx - win]
            if len(seg) < min_window or len(before) < 8:
                continue

            seg_high = float(seg["High"].max())
            seg_low = float(seg["Low"].min())
            rng = max(seg_high - seg_low, 1e-9)

            pre = before.tail(min(30, len(before)))
            pre_return = (
                float(pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1.0)
                if float(pre["Close"].iloc[0]) != 0
                else 0.0
            )

            width_score = 1.0 - min(rng / max(float(seg["Close"].mean()), 1e-9), 1.0)
            flat_score = 1.0 - min(abs(pre_return) * 3.0, 1.0)
            dur_score = min(len(seg) / max_window, 1.0)

            lower_band = seg_low + 0.25 * rng
            upper_band = seg_high - 0.25 * rng
            tests = int((seg["Low"] <= lower_band).sum()) + int((seg["High"] >= upper_band).sum())
            test_score = min(tests / max(len(seg) * 0.30, 1), 1.0)

            score = 0.35 * width_score + 0.20 * flat_score + 0.20 * dur_score + 0.25 * test_score

            if score > best_score:
                best_score = score
                best = {
                    "start": seg.index[0],
                    "end": seg.index[-1],
                    "base_low": seg_low,
                    "base_high": seg_high,
                    "base_len": len(seg),
                    "score": float(score),
                }

    return best


def estimate_zones(df: pd.DataFrame, base: dict):
    seg = df.loc[base["start"]:base["end"]].copy()
    base_low = float(base["base_low"])
    base_high = float(base["base_high"])
    base_range = max(base_high - base_low, 1e-9)

    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
    vol = safe_volume(seg)
    vwap_center = float((typical * vol).sum() / vol.sum()) if float(vol.sum()) > 0 else float(typical.mean())
    tp_center = float(typical.mean())
    midpoint = (base_high + base_low) / 2.0

    bins = np.linspace(base_low, base_high, 21)
    hist, edges = np.histogram(seg["Close"].clip(base_low, base_high), bins=bins)
    ix = int(np.argmax(hist))
    defend_low = float(edges[max(ix - 1, 0)])
    defend_high = float(edges[min(ix + 2, len(edges) - 1)])
    defended_center = (defend_low + defend_high) / 2.0

    avg_core = float(np.mean([midpoint, vwap_center, tp_center, defended_center]))
    avg_half = 0.12 * base_range

    return {
        "defend_low": defend_low,
        "defend_high": defend_high,
        "avg_core": avg_core,
        "avg_lower": avg_core - avg_half,
        "avg_upper": avg_core + avg_half,
    }


def anchored_vwap_from_base(df: pd.DataFrame, base: dict) -> pd.Series:
    seg = df.loc[base["start"]:].copy()
    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
    vol = safe_volume(seg).replace(0, np.nan)
    avwap = (typical * vol).fillna(0).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
    out = pd.Series(index=df.index, dtype=float)
    out.loc[seg.index] = avwap
    return out


def compute_scores(df: pd.DataFrame, base: dict, zones: dict):
    post = df.loc[base["end"]:].copy()
    base_seg = df.loc[base["start"]:base["end"]].copy()

    if len(post) < 5:
        return {
            "accumulation": 50.0,
            "distribution": 50.0,
            "holding": 50.0,
            "release_risk": 50.0,
            "confidence": 35.0,
        }

    base_low = float(base["base_low"])
    base_high = float(base["base_high"])
    base_range = max(base_high - base_low, 1e-9)
    last_close = float(df["Close"].iloc[-1])

    pre = df.loc[:base["start"]].tail(25)
    pre_return = (
        float(pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1.0)
        if len(pre) >= 5 and float(pre["Close"].iloc[0]) != 0
        else 0.0
    )

    post_high = float(post["High"].max())
    post_low = float(post["Low"].min())
    markup = (post_high - base_high) / base_range
    markdown = (base_low - post_low) / base_range
    defended = 1.0 if last_close >= zones["avg_lower"] else 0.0
    above_avwap = 1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close >= float(df["AVWAP"].iloc[-1]) else 0.0

    lower_rej = ((base_seg["Close"] - base_seg["Low"]) / (base_seg["High"] - base_seg["Low"]).replace(0, np.nan)).fillna(0).mean()
    upper_rej = ((base_seg["High"] - base_seg["Close"]) / (base_seg["High"] - base_seg["Low"]).replace(0, np.nan)).fillna(0).mean()

    accumulation = 25 + max(0.0, -pre_return) * 140 + max(0.0, markup) * 12 + defended * 12 + above_avwap * 8 + float(lower_rej) * 18 - max(0.0, markdown) * 15
    distribution = 25 + max(0.0, pre_return) * 140 + max(0.0, markdown) * 14 + (1.0 if last_close < zones["avg_lower"] else 0.0) * 12 + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 8 + float(upper_rej) * 18 - max(0.0, markup) * 12
    holding = 20 + max(0.0, (last_close - base_high) / base_range) * 20 + defended * 18 + above_avwap * 15
    release_risk = 20 + max(0.0, (base_low - last_close) / base_range) * 22 + (1.0 if last_close < zones["avg_lower"] else 0.0) * 15 + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 10 - defended * 10
    confidence = 35 + min(base["base_len"] / 60.0, 1.0) * 20 + min(base["score"], 1.0) * 20 + (15 if (safe_volume(df).tail(60) > 0).mean() > 0.6 else 6)

    def clamp(x):
        return float(max(0.0, min(100.0, x)))

    return {
        "accumulation": clamp(accumulation),
        "distribution": clamp(distribution),
        "holding": clamp(holding),
        "release_risk": clamp(release_risk),
        "confidence": clamp(confidence),
        "invalidation": float(min(zones["avg_lower"], zones["defend_low"], base_low + 0.10 * base_range)),
    }


def classify(res: dict) -> str:
    if res["accumulation"] >= 60 and res["holding"] >= 55:
        return "Accumulation Bias"
    if res["distribution"] >= 60 and res["release_risk"] >= 55:
        return "Distribution Bias"
    return "Neutral / Mixed"


def analyze_symbol(symbol: str, period: str, interval: str):
    df = fetch_data(symbol, period, interval)
    if df.empty or len(df) < 40:
        return None, "Data kosong atau terlalu sedikit."
    base = find_base_window(df)
    if not base:
        return None, "Belum nemu base/range yang cukup jelas."
    zones = estimate_zones(df, base)
    df["AVWAP"] = anchored_vwap_from_base(df, base)
    scores = compute_scores(df, base, zones)
    res = {**base, **zones, **scores, "last_close": float(df["Close"].iloc[-1]), "df": df}
    return res, None


WATCHLISTS = {
    "US Mega Caps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
    "IHSG Large Caps": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "ICBP.JK"],
    "Commodities & Futures": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDIDR=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "Mixed Default": ["AAPL", "NVDA", "BBCA.JK", "^JKSE", "GC=F", "EURUSD=X", "BTC-USD"],
}


st.title("Observational Accumulation / Distribution Engine")
st.caption("Versi final tanpa Plotly. Fokus ke OHLCV + struktur.")

with st.sidebar:
    mode = st.radio("Mode", ["Single Instrument", "Watchlist Scanner"])
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)
    st.code("AAPL\nBBCA.JK\n^JKSE\nGC=F\nCL=F\nEURUSD=X\nBTC-USD", language=None)

if mode == "Single Instrument":
    symbol = st.text_input("Symbol", value="AAPL").strip().upper()
    if st.button("Analyze", use_container_width=True):
        res, err = analyze_symbol(symbol, period, interval)
        if err:
            st.error(err)
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("State", classify(res))
            c2.metric("Accumulation", f'{res["accumulation"]:.1f}')
            c3.metric("Distribution", f'{res["distribution"]:.1f}')
            c4.metric("Holding", f'{res["holding"]:.1f}')
            c5.metric("Release Risk", f'{res["release_risk"]:.1f}')

            c6, c7, c8, c9, c10 = st.columns(5)
            c6.metric("Avg Core", f'{res["avg_core"]:.4f}')
            c7.metric("Avg Lower", f'{res["avg_lower"]:.4f}')
            c8.metric("Avg Upper", f'{res["avg_upper"]:.4f}')
            c9.metric("Defended Zone", f'{res["defend_low"]:.4f} - {res["defend_high"]:.4f}')
            c10.metric("Invalidation", f'{res["invalidation"]:.4f}')

            st.subheader("Close vs AVWAP")
            st.line_chart(res["df"][["Close", "AVWAP"]])

            info = pd.DataFrame([
                ["Base Start", str(res["start"])],
                ["Base End", str(res["end"])],
                ["Base Low", round(res["base_low"], 4)],
                ["Base High", round(res["base_high"], 4)],
                ["Base Length", int(res["base_len"])],
                ["Confidence", round(res["confidence"], 1)],
            ], columns=["Field", "Value"])
            st.dataframe(info, use_container_width=True, hide_index=True)

else:
    preset = st.selectbox("Preset", list(WATCHLISTS.keys()))
    custom = st.text_area("Custom symbols (pisahkan koma)", value=", ".join(WATCHLISTS[preset]), height=100)
    if st.button("Run Scanner", use_container_width=True):
        symbols = [x.strip().upper() for x in custom.split(",") if x.strip()]
        rows = []
        prog = st.progress(0.0)
        for i, sym in enumerate(symbols):
            res, err = analyze_symbol(sym, period, interval)
            if res:
                rows.append({
                    "Symbol": sym,
                    "State": classify(res),
                    "Last Close": round(res["last_close"], 4),
                    "Avg Core": round(res["avg_core"], 4),
                    "Accumulation": round(res["accumulation"], 1),
                    "Distribution": round(res["distribution"], 1),
                    "Holding": round(res["holding"], 1),
                    "Release Risk": round(res["release_risk"], 1),
                    "Confidence": round(res["confidence"], 1),
                })
            prog.progress((i + 1) / max(len(symbols), 1))
        prog.empty()

        out = pd.DataFrame(rows)
        if out.empty:
            st.warning("Belum ada hasil scan.")
        else:
            out["Bullish Edge"] = out["Accumulation"] * 0.40 + out["Holding"] * 0.35 + out["Confidence"] * 0.25 - out["Release Risk"] * 0.15
            out["Bearish Edge"] = out["Distribution"] * 0.40 + out["Release Risk"] * 0.35 + out["Confidence"] * 0.25 - out["Holding"] * 0.15
            out = out.sort_values("Bullish Edge", ascending=False).reset_index(drop=True)
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"), "scanner_results.csv", "text/csv")
