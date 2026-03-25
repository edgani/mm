import importlib.util
import io
import re
from typing import List, Tuple

import streamlit as st

st.set_page_config(page_title="Universal Market Engine", layout="wide")

_missing = [pkg for pkg in ["numpy", "pandas", "yfinance", "requests"] if importlib.util.find_spec(pkg) is None]
if _missing:
    st.error(
        "Missing Python packages: " + ", ".join(_missing) + "\n\n"
        "requirements.txt must contain:\n"
        "streamlit>=1.36.0\n"
        "pandas>=2.2.0\n"
        "numpy>=1.26.0\n"
        "yfinance>=0.2.54\n"
        "requests>=2.31.0"
    )
    st.stop()

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# -----------------------------
# Dynamic universe loaders
# -----------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_us_universe() -> pd.DataFrame:
    urls = {
        "nasdaq": "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    }

    frames = []
    for name, url in urls.items():
        try:
            txt = requests.get(url, timeout=20).text
            df = pd.read_csv(io.StringIO(txt), sep="|")
            df = df.iloc[:-1].copy()  # drop file creation time footer
            if name == "nasdaq":
                if "Symbol" in df.columns:
                    tmp = df.rename(columns={"Symbol": "Ticker", "Security Name": "Name"})
                    tmp["Exchange"] = "NASDAQ"
                    frames.append(tmp[["Ticker", "Name", "Exchange"]])
            else:
                # otherlisted format often uses ACT Symbol
                ticker_col = "ACT Symbol" if "ACT Symbol" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
                name_col = "Security Name" if "Security Name" in df.columns else None
                exch_col = "Exchange" if "Exchange" in df.columns else None
                if ticker_col and name_col:
                    tmp = df.rename(columns={ticker_col: "Ticker", name_col: "Name"})
                    if exch_col:
                        tmp["Exchange"] = df[exch_col]
                    else:
                        tmp["Exchange"] = "OTHER"
                    frames.append(tmp[["Ticker", "Name", "Exchange"]])
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["Ticker", "Name", "Exchange"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out = out[out["Ticker"].str.match(r"^[A-Z\.\-]+$|^[A-Z]{1,5}$", na=False)]
    return out


@st.cache_data(show_spinner=False, ttl=86400)
def load_idx_universe() -> pd.DataFrame:
    urls = [
        "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
        "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham",
        "https://www.idx.co.id/en/listed-companies/company-profiles",
    ]

    candidates = []
    pattern = re.compile(r"^[A-Z]{4,5}$")

    for url in urls:
        try:
            tables = pd.read_html(url)
        except Exception:
            tables = []

        for tbl in tables:
            cols = [str(c).strip() for c in tbl.columns]
            tbl.columns = cols
            possible_cols = [c for c in cols if c.lower() in ["code", "ticker", "symbol", "kode", "kode saham"]]
            if possible_cols:
                c = possible_cols[0]
                vals = tbl[c].astype(str).str.upper().str.strip()
                vals = vals[vals.str.match(pattern, na=False)]
                if len(vals) > 10:
                    tmp = pd.DataFrame({"Ticker": vals.unique()})
                    candidates.append(tmp)

        # fallback: search all string cells
        for tbl in tables:
            vals = pd.Series(tbl.astype(str).values.ravel()).str.upper().str.strip()
            vals = vals[vals.str.match(pattern, na=False)]
            if len(vals) > 20:
                tmp = pd.DataFrame({"Ticker": vals.unique()})
                candidates.append(tmp)

    if not candidates:
        return pd.DataFrame(columns=["Ticker", "YahooTicker"])

    out = pd.concat(candidates, ignore_index=True).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    out["YahooTicker"] = out["Ticker"] + ".JK"
    return out


# -----------------------------
# Symbol helpers
# -----------------------------
def normalize_symbol(sym: str) -> str:
    return sym.strip().upper()


def auto_resolve_symbol(sym: str, market_hint: str = "Auto") -> List[str]:
    s = normalize_symbol(sym)
    tries = []

    if market_hint == "IHSG":
        tries = [s if s.endswith(".JK") else s + ".JK", s]
    elif market_hint == "US":
        tries = [s]
    elif market_hint == "Forex":
        tries = [s if s.endswith("=X") else s + "=X", s]
    elif market_hint == "Crypto":
        tries = [s if "-USD" in s else s + "-USD", s]
    elif market_hint == "Futures":
        tries = [s if s.endswith("=F") else s + "=F", s]
    else:
        tries = [s]
        if not s.endswith(".JK"):
            tries.append(s + ".JK")
        if not s.endswith("=X"):
            tries.append(s + "=X")
        if not s.endswith("=F"):
            tries.append(s + "=F")
        if "-USD" not in s:
            tries.append(s + "-USD")

    seen = []
    for x in tries:
        if x not in seen:
            seen.append(x)
    return seen


# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False, ttl=900)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)

    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["Close"])
    return df


def fetch_with_resolution(raw_symbol: str, period: str, interval: str, market_hint: str) -> Tuple[str, pd.DataFrame]:
    for candidate in auto_resolve_symbol(raw_symbol, market_hint):
        df = fetch_data(candidate, period, interval)
        if not df.empty and len(df) >= 30:
            return candidate, df
    return "", pd.DataFrame()


def safe_volume(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, data=0.0)
    return pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)


# -----------------------------
# Engine
# -----------------------------
def find_base_window(df: pd.DataFrame, lookback: int = 150, min_window: int = 18, max_window: int = 70):
    d = df.tail(lookback).copy()
    if len(d) < min_window + 8:
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
            seg_mean = max(float(seg["Close"].mean()), 1e-9)

            pre = before.tail(min(30, len(before)))
            pre_return = float(pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1.0) if float(pre["Close"].iloc[0]) != 0 else 0.0

            width_score = 1.0 - min(rng / seg_mean, 1.0)
            flat_score = 1.0 - min(abs(pre_return) * 3.0, 1.0)
            dur_score = min(len(seg) / max_window, 1.0)

            lower_band = seg_low + 0.25 * rng
            upper_band = seg_high - 0.25 * rng
            low_tests = int((seg["Low"] <= lower_band).sum())
            high_tests = int((seg["High"] >= upper_band).sum())
            test_score = min((low_tests + high_tests) / max(len(seg) * 0.30, 1), 1.0)

            close_std = float(seg["Close"].pct_change().std() or 0.0)
            score = 0.32 * width_score + 0.18 * flat_score + 0.18 * dur_score + 0.22 * test_score - 4.0 * close_std

            if score > best_score:
                best_score = score
                best = {
                    "start": seg.index[0],
                    "end": seg.index[-1],
                    "base_low": seg_low,
                    "base_high": seg_high,
                    "base_len": len(seg),
                    "score": float(score),
                    "low_tests": low_tests,
                    "high_tests": high_tests,
                }
    return best


def estimate_zones(df: pd.DataFrame, base: dict):
    seg = df.loc[base["start"]:base["end"]].copy()
    base_low = float(base["base_low"])
    base_high = float(base["base_high"])
    base_range = max(base_high - base_low, 1e-9)

    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
    vol = safe_volume(seg)
    midpoint = (base_high + base_low) / 2.0
    tp_center = float(typical.mean())
    vwap_center = float((typical * vol).sum() / vol.sum()) if float(vol.sum()) > 0 else tp_center

    bins = np.linspace(base_low, base_high, 25)
    hist, edges = np.histogram(seg["Close"].clip(base_low, base_high), bins=bins)
    ix = int(np.argmax(hist))
    defend_low = float(edges[max(ix - 1, 0)])
    defend_high = float(edges[min(ix + 2, len(edges) - 1)])
    defended_center = (defend_low + defend_high) / 2.0

    avg_core = float(np.mean([midpoint, tp_center, vwap_center, defended_center]))
    avg_half = 0.12 * base_range

    return {
        "midpoint": midpoint,
        "tp_center": tp_center,
        "vwap_center": vwap_center,
        "defend_low": defend_low,
        "defend_high": defend_high,
        "defended_center": defended_center,
        "avg_lower": avg_core - avg_half,
        "avg_core": avg_core,
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
    base_seg = df.loc[base["start"]:base["end"]].copy()
    post = df.loc[base["end"]:].copy()

    if len(post) < 5:
        return {
            "accumulation": 50.0,
            "distribution": 50.0,
            "holding": 50.0,
            "release_risk": 50.0,
            "confidence": 35.0,
            "state": "Neutral / Mixed",
        }

    base_low = float(base["base_low"])
    base_high = float(base["base_high"])
    base_range = max(base_high - base_low, 1e-9)
    last_close = float(df["Close"].iloc[-1])

    pre = df.loc[:base["start"]].tail(25)
    pre_return = float(pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1.0) if len(pre) >= 5 and float(pre["Close"].iloc[0]) != 0 else 0.0

    post_high = float(post["High"].max())
    post_low = float(post["Low"].min())
    markup = (post_high - base_high) / base_range
    markdown = (base_low - post_low) / base_range
    above_base = (last_close - base_high) / base_range
    below_base = (base_low - last_close) / base_range

    defended = 1.0 if last_close >= zones["avg_lower"] else 0.0
    above_avwap = 1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close >= float(df["AVWAP"].iloc[-1]) else 0.0

    body_range = (base_seg["High"] - base_seg["Low"]).replace(0, np.nan)
    lower_rej = ((base_seg["Close"] - base_seg["Low"]) / body_range).fillna(0).mean()
    upper_rej = ((base_seg["High"] - base_seg["Close"]) / body_range).fillna(0).mean()

    vol_valid = 1.0 if (safe_volume(df).tail(60) > 0).mean() > 0.60 else 0.4

    accumulation = 25 + max(0.0, -pre_return) * 140 + max(0.0, markup) * 12 + defended * 12 + above_avwap * 8 + float(lower_rej) * 18 - max(0.0, markdown) * 15
    distribution = 25 + max(0.0, pre_return) * 140 + max(0.0, markdown) * 14 + (1.0 if last_close < zones["avg_lower"] else 0.0) * 12 + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 8 + float(upper_rej) * 18 - max(0.0, markup) * 12
    holding = 20 + max(0.0, above_base) * 20 + defended * 18 + above_avwap * 15 - max(0.0, below_base) * 20
    release_risk = 20 + max(0.0, below_base) * 22 + (1.0 if last_close < zones["avg_lower"] else 0.0) * 15 + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 10 + max(0.0, markdown) * 10 - defended * 10
    confidence = 35 + min(base["base_len"] / 70.0, 1.0) * 20 + min(base["score"], 1.0) * 20 + vol_valid * 15

    def clamp(x):
        return float(max(0.0, min(100.0, x)))

    accumulation = clamp(accumulation)
    distribution = clamp(distribution)
    holding = clamp(holding)
    release_risk = clamp(release_risk)
    confidence = clamp(confidence)
    invalidation = float(min(zones["avg_lower"], zones["defend_low"], base_low + 0.10 * base_range))

    state = "Neutral / Mixed"
    if accumulation >= 60 and holding >= 55:
        state = "Accumulation Bias"
    elif distribution >= 60 and release_risk >= 55:
        state = "Distribution Bias"

    return {
        "accumulation": accumulation,
        "distribution": distribution,
        "holding": holding,
        "release_risk": release_risk,
        "confidence": confidence,
        "invalidation": invalidation,
        "state": state,
        "markup": float(markup),
        "markdown": float(markdown),
        "pre_return": float(pre_return),
        "above_base": float(above_base),
        "below_base": float(below_base),
    }


def interpret_result(res: dict) -> dict:
    state = res["state"]
    lead_score = max(res["accumulation"], res["distribution"], res["holding"], res["release_risk"])
    strength = "weak"
    if lead_score >= 75 and res["confidence"] >= 60:
        strength = "strong"
    elif lead_score >= 62 and res["confidence"] >= 50:
        strength = "moderate"

    reasons, risks = [], []
    action = ""

    if state == "Accumulation Bias":
        if res["pre_return"] < 0:
            reasons.append("range muncul setelah tekanan turun sebelumnya")
        if res["markup"] > 0:
            reasons.append("harga sudah bisa keluar dari atas base")
        if res["holding"] >= 60:
            reasons.append("hasil markup masih relatif dibela")
        if res["last_close"] >= res["avg_lower"] and res["last_close"] <= res["avg_upper"]:
            reasons.append("harga sedang berada di area estimasi average")
        risks.append("kalau close accepted di bawah invalidation, bias bullish melemah")
        if res["release_risk"] >= 55:
            risks.append("ada tanda pertahanan area penting mulai berkurang")
        action = "cek apakah pullback masih ditahan di avg zone / defended zone"

    elif state == "Distribution Bias":
        if res["pre_return"] > 0:
            reasons.append("range muncul setelah kenaikan sebelumnya")
        if res["markdown"] > 0:
            reasons.append("harga sudah kehilangan bagian bawah struktur")
        if res["release_risk"] >= 60:
            reasons.append("area penting tidak lagi dibela dengan baik")
        if res["last_close"] < res["avg_lower"]:
            reasons.append("harga berada di bawah average zone")
        risks.append("kalau harga cepat reclaim avg zone dan bertahan di atasnya, bias bearish melemah")
        action = "cek apakah bounce gagal reclaim avg zone / AVWAP"
    else:
        reasons.append("skor akumulasi dan distribusi belum cukup dominan")
        reasons.append("struktur masih campur atau range belum resolve")
        risks.append("breakout atau breakdown berikutnya akan lebih menentukan arah")
        action = "tunggu struktur lebih jelas, jangan maksa arah"

    return {
        "summary": f"{state} ({strength})",
        "reasons": reasons,
        "risks": risks,
        "action": action
    }


def analyze_symbol(raw_symbol: str, period: str, interval: str, market_hint: str):
    resolved_symbol, df = fetch_with_resolution(raw_symbol, period, interval, market_hint)
    if df.empty or len(df) < 30:
        return None, "Data kosong / terlalu sedikit / simbol belum ter-resolve di Yahoo."

    base = find_base_window(df)
    if not base:
        return None, "Belum ketemu base/range yang cukup jelas."

    zones = estimate_zones(df, base)
    df = df.copy()
    df["AVWAP"] = anchored_vwap_from_base(df, base)
    scores = compute_scores(df, base, zones)

    result = {**base, **zones, **scores}
    result["raw_symbol"] = normalize_symbol(raw_symbol)
    result["symbol"] = resolved_symbol
    result["last_close"] = float(df["Close"].iloc[-1])
    result["df"] = df
    result["interpretation"] = interpret_result(result)
    return result, None


def scanner_table(symbols, period, interval, market_hint):
    rows = []
    progress = st.progress(0.0)

    for i, sym in enumerate(symbols):
        res, err = analyze_symbol(sym, period, interval, market_hint)
        if res is not None:
            rows.append({
                "Input": normalize_symbol(sym),
                "Resolved": res["symbol"],
                "State": res["state"],
                "Bias": "ACCUM" if res["state"] == "Accumulation Bias" else ("DIST" if res["state"] == "Distribution Bias" else "MIXED"),
                "Interpretation": res["interpretation"]["summary"],
                "Last Close": round(res["last_close"], 4),
                "Avg Core": round(res["avg_core"], 4),
                "Accumulation": round(res["accumulation"], 1),
                "Distribution": round(res["distribution"], 1),
                "Holding": round(res["holding"], 1),
                "Release Risk": round(res["release_risk"], 1),
                "Confidence": round(res["confidence"], 1),
            })
        progress.progress((i + 1) / max(len(symbols), 1))

    progress.empty()
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Bullish Edge"] = out["Accumulation"] * 0.40 + out["Holding"] * 0.35 + out["Confidence"] * 0.25 - out["Release Risk"] * 0.15
    out["Bearish Edge"] = out["Distribution"] * 0.40 + out["Release Risk"] * 0.35 + out["Confidence"] * 0.25 - out["Holding"] * 0.15
    out = out.sort_values(["Bullish Edge", "Confidence"], ascending=[False, False]).reset_index(drop=True)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("Universal Observational Accumulation / Distribution Engine")
st.caption("Dynamic universe loader untuk US + IHSG, symbol auto-resolution, interpretasi lebih manusiawi, scanner jelas ACCUM / DIST / MIXED.")

with st.sidebar:
    mode = st.radio("Mode", ["Single Instrument", "Watchlist Scanner", "Universe Browser"])
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)
    market_hint = st.selectbox("Market Hint", ["Auto", "US", "IHSG", "Forex", "Futures", "Crypto"], index=0)
    st.markdown("### Input examples")
    st.code("AAPL\nHUMI.JK or HUMI\nEURUSD or EURUSD=X\nGC or GC=F\nBTC or BTC-USD", language=None)

if mode == "Single Instrument":
    symbol = st.text_input("Symbol", value="AAPL").strip().upper()

    if st.button("Analyze", use_container_width=True):
        res, err = analyze_symbol(symbol, period, interval, market_hint)
        if err:
            st.error(err)
        else:
            st.success(f"Resolved symbol: {res['symbol']}")
            a,b,c,d,e = st.columns(5)
            a.metric("State", res["state"])
            b.metric("Accumulation", f'{res["accumulation"]:.1f}')
            c.metric("Distribution", f'{res["distribution"]:.1f}')
            d.metric("Holding", f'{res["holding"]:.1f}')
            e.metric("Release Risk", f'{res["release_risk"]:.1f}')

            f,g,h,i,j = st.columns(5)
            f.metric("Avg Core", f'{res["avg_core"]:.4f}')
            g.metric("Avg Lower", f'{res["avg_lower"]:.4f}')
            h.metric("Avg Upper", f'{res["avg_upper"]:.4f}')
            i.metric("Defended Zone", f'{res["defend_low"]:.4f} - {res["defend_high"]:.4f}')
            j.metric("Invalidation", f'{res["invalidation"]:.4f}')

            st.subheader("Interpretation")
            st.write(res["interpretation"]["summary"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Why this read**")
                for x in res["interpretation"]["reasons"]:
                    st.write("- " + x)
            with c2:
                st.markdown("**Risk / invalidation**")
                for x in res["interpretation"]["risks"]:
                    st.write("- " + x)
                st.markdown("**Practical focus**")
                st.write("- " + res["interpretation"]["action"])

            st.subheader("Close vs AVWAP")
            st.line_chart(res["df"][["Close", "AVWAP"]])

elif mode == "Watchlist Scanner":
    presets = ["US Mega + Liquid", "Forex Majors", "Futures & Commodities", "Crypto Majors", "Mixed Global", "IHSG Wide (loaded below if available)"]
    preset = st.selectbox("Preset", presets)

    if preset == "US Mega + Liquid":
        default_symbols = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX, AVGO, PLTR, JPM, XOM, LLY, COST, UNH, SPY, QQQ, IWM, DIA"
    elif preset == "Forex Majors":
        default_symbols = "EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, NZDUSD=X, USDCAD=X, USDCHF=X, EURJPY=X, EURGBP=X, USDIDR=X"
    elif preset == "Futures & Commodities":
        default_symbols = "GC=F, SI=F, CL=F, BZ=F, NG=F, HG=F, ZC=F, ZS=F, ZW=F, LE=F, HE=F, PL=F, PA=F"
    elif preset == "Crypto Majors":
        default_symbols = "BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, DOGE-USD, AVAX-USD, LINK-USD, DOT-USD"
    elif preset == "Mixed Global":
        default_symbols = "AAPL, NVDA, SPY, ^JKSE, BBCA.JK, TLKM.JK, EURUSD=X, USDJPY=X, GC=F, CL=F, BTC-USD, ETH-USD"
    else:
        idx_df = load_idx_universe()
        default_symbols = ", ".join(idx_df["Ticker"].head(200).tolist()) if not idx_df.empty else "^JKSE, BBCA, BBRI, BMRI, TLKM"

    custom = st.text_area("Symbols (comma separated)", value=default_symbols, height=220)

    if st.button("Run Scanner", use_container_width=True):
        symbols = [x.strip().upper() for x in custom.split(",") if x.strip()]
        out = scanner_table(symbols, period, interval, market_hint)

        if out.empty:
            st.warning("No scanner result. Symbols may not resolve in Yahoo or data is too short.")
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")

else:
    tab1, tab2 = st.tabs(["US Universe", "IHSG Universe"])

    with tab1:
        us = load_us_universe()
        st.write(f"Loaded US symbols: {len(us)}")
        q = st.text_input("Search US ticker or name")
        view = us.copy()
        if q:
            q = q.upper()
            view = view[view["Ticker"].astype(str).str.contains(q, na=False) | view["Name"].astype(str).str.upper().str.contains(q, na=False)]
        st.dataframe(view.head(1000), use_container_width=True, hide_index=True)

    with tab2:
        idx = load_idx_universe()
        st.write(f"Loaded IHSG/IDX symbols: {len(idx)}")
        q2 = st.text_input("Search IHSG ticker")
        view2 = idx.copy()
        if q2:
            q2 = q2.upper()
            view2 = view2[view2["Ticker"].astype(str).str.contains(q2, na=False)]
        st.dataframe(view2.head(1000), use_container_width=True, hide_index=True)
