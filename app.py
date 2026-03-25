import importlib.util
import streamlit as st

st.set_page_config(page_title="Universal Observational Market Engine", layout="wide")

_missing = [pkg for pkg in ["numpy", "pandas", "yfinance"] if importlib.util.find_spec(pkg) is None]
if _missing:
    st.error(
        "Missing Python packages: " + ", ".join(_missing) + "\n\n"
        "Put requirements.txt in the SAME folder as app.py with:\n"
        "streamlit>=1.36.0\n"
        "pandas>=2.2.0\n"
        "numpy>=1.26.0\n"
        "yfinance>=0.2.54"
    )
    st.stop()

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Symbol universe presets
# -----------------------------
IHSG_WIDE = [
    "^JKSE",
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK","ICBP.JK","INDF.JK","CPIN.JK","SMGR.JK","UNTR.JK",
    "ADRO.JK","PTBA.JK","ANTM.JK","MDKA.JK","TINS.JK","HRUM.JK","ITMG.JK","INDY.JK","PGEO.JK","MEDC.JK","AKRA.JK",
    "AMMN.JK","BYAN.JK","TPIA.JK","BRPT.JK","ESSA.JK","SIDO.JK","KLBF.JK","MIKA.JK","HEAL.JK","SILO.JK","ACES.JK",
    "MAPI.JK","MAPA.JK","ERAA.JK","EXCL.JK","ISAT.JK","MTEL.JK","GOTO.JK","BUKA.JK","BRIS.JK","ARTO.JK","BTPS.JK",
    "BNGA.JK","NISP.JK","JPFA.JK","MAIN.JK","MYOR.JK","ULTJ.JK","INKP.JK","TKIM.JK","ICBP.JK","INKP.JK","TKIM.JK",
    "PANI.JK","BREN.JK","NCKL.JK","MBMA.JK","AMRT.JK","RALS.JK","SCMA.JK","MNCN.JK","ADMR.JK","DOID.JK","DMAS.JK",
    "PWON.JK","CTRA.JK","BSDE.JK","SMRA.JK","PWON.JK","WIKA.JK","PTPP.JK","JSMR.JK","WSKT.JK"
]

MARKET_PRESETS = {
    "US Mega + Liquid": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO","PLTR","JPM","XOM","LLY","COST","UNH",
        "SPY","QQQ","IWM","DIA"
    ],
    "IHSG Wide": IHSG_WIDE,
    "Forex Majors": [
        "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","USDCAD=X","USDCHF=X","EURJPY=X","EURGBP=X","USDIDR=X"
    ],
    "Futures & Commodities": [
        "GC=F","SI=F","CL=F","BZ=F","NG=F","HG=F","ZC=F","ZS=F","ZW=F","LE=F","HE=F","PL=F","PA=F"
    ],
    "Crypto Majors": [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","DOT-USD"
    ],
    "Mixed Global": [
        "AAPL","NVDA","SPY","^JKSE","BBCA.JK","TLKM.JK","EURUSD=X","USDJPY=X","GC=F","CL=F","BTC-USD","ETH-USD"
    ],
}


# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False, ttl=900)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    symbol = symbol.strip().upper()

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

    accumulation = (
        25
        + max(0.0, -pre_return) * 140
        + max(0.0, markup) * 12
        + defended * 12
        + above_avwap * 8
        + float(lower_rej) * 18
        - max(0.0, markdown) * 15
    )

    distribution = (
        25
        + max(0.0, pre_return) * 140
        + max(0.0, markdown) * 14
        + (1.0 if last_close < zones["avg_lower"] else 0.0) * 12
        + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 8
        + float(upper_rej) * 18
        - max(0.0, markup) * 12
    )

    holding = (
        20
        + max(0.0, above_base) * 20
        + defended * 18
        + above_avwap * 15
        - max(0.0, below_base) * 20
    )

    release_risk = (
        20
        + max(0.0, below_base) * 22
        + (1.0 if last_close < zones["avg_lower"] else 0.0) * 15
        + (1.0 if not np.isnan(df["AVWAP"].iloc[-1]) and last_close < float(df["AVWAP"].iloc[-1]) else 0.0) * 10
        + max(0.0, markdown) * 10
        - defended * 10
    )

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
    strength = "weak"
    lead_score = max(res["accumulation"], res["distribution"], res["holding"], res["release_risk"])

    if lead_score >= 75 and res["confidence"] >= 60:
        strength = "strong"
    elif lead_score >= 62 and res["confidence"] >= 50:
        strength = "moderate"

    reasons = []
    risks = []
    action = ""

    if state == "Accumulation Bias":
        if res["pre_return"] < 0:
            reasons.append("base terbentuk setelah tekanan turun sebelumnya")
        if res["markup"] > 0:
            reasons.append("harga sudah mampu keluar dari atas base")
        if res["holding"] >= 60:
            reasons.append("hasil markup masih relatif dibela")
        if res["last_close"] >= res["avg_lower"] and res["last_close"] <= res["avg_upper"]:
            reasons.append("harga sedang berada di area estimasi average")
        risks.append("kalau close accepted di bawah invalidation, hipotesis bullish melemah")
        if res["release_risk"] >= 55:
            risks.append("ada tanda pertahanan area penting mulai berkurang")
        action = "fokus cari apakah pullback masih ditahan di avg zone / defended zone"

    elif state == "Distribution Bias":
        if res["pre_return"] > 0:
            reasons.append("base/range muncul setelah kenaikan sebelumnya")
        if res["markdown"] > 0:
            reasons.append("harga sudah kehilangan bagian bawah struktur")
        if res["release_risk"] >= 60:
            reasons.append("area penting tidak lagi dibela dengan baik")
        if res["last_close"] < res["avg_lower"]:
            reasons.append("harga berada di bawah average zone")
        risks.append("kalau harga cepat reclaim avg zone dan bertahan di atasnya, bias bearish melemah")
        action = "fokus lihat apakah bounce gagal reclaim avg zone / AVWAP"

    else:
        reasons.append("skor akumulasi dan distribusi belum cukup dominan")
        reasons.append("struktur masih campur atau range belum resolve")
        risks.append("breakout atau breakdown berikutnya akan lebih menentukan arah")
        action = "jangan maksa baca arah; tunggu struktur lebih jelas"

    summary = f"{state} ({strength})"
    return {"summary": summary, "reasons": reasons, "risks": risks, "action": action}


def analyze_symbol(symbol: str, period: str, interval: str):
    df = fetch_data(symbol, period, interval)
    if df.empty or len(df) < 30:
        return None, "Data kosong / terlalu sedikit / Yahoo tidak balikin data buat simbol ini."

    base = find_base_window(df)
    if not base:
        return None, "Belum ketemu base/range yang cukup jelas."

    zones = estimate_zones(df, base)
    df = df.copy()
    df["AVWAP"] = anchored_vwap_from_base(df, base)
    scores = compute_scores(df, base, zones)

    result = {**base, **zones, **scores}
    result["last_close"] = float(df["Close"].iloc[-1])
    result["df"] = df
    result["interpretation"] = interpret_result(result)
    return result, None


def scanner_table(symbols, period, interval):
    rows = []
    progress = st.progress(0.0)

    for i, sym in enumerate(symbols):
        sym = sym.strip().upper()
        res, err = analyze_symbol(sym, period, interval)
        if res is not None:
            rows.append({
                "Symbol": sym,
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
st.caption("Interpretasi lebih manusiawi. Coverage lintas US stocks, IHSG, forex, futures/commodities, crypto. Scanner kasih bias ACCUM / DIST / MIXED dengan ringkasan.")

with st.sidebar:
    mode = st.radio("Mode", ["Single Instrument", "Watchlist Scanner"])
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)
    st.markdown("### Contoh simbol")
    st.code("AAPL\nBBCA.JK\n^JKSE\nEURUSD=X\nGC=F\nCL=F\nBTC-USD", language=None)

if mode == "Single Instrument":
    symbol = st.text_input("Symbol", value="AAPL").strip().upper()

    if st.button("Analyze", use_container_width=True):
        res, err = analyze_symbol(symbol, period, interval)
        if err:
            st.error(err)
        else:
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

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Kenapa engine baca begini**")
                for x in res["interpretation"]["reasons"]:
                    st.write("- " + x)
            with col2:
                st.markdown("**Risk / invalidation**")
                for x in res["interpretation"]["risks"]:
                    st.write("- " + x)
                st.markdown("**Practical focus**")
                st.write("- " + res["interpretation"]["action"])

            st.subheader("Close vs AVWAP")
            st.line_chart(res["df"][["Close", "AVWAP"]])

            info = pd.DataFrame([
                ["Base Start", str(res["start"])],
                ["Base End", str(res["end"])],
                ["Base Low", round(res["base_low"], 4)],
                ["Base High", round(res["base_high"], 4)],
                ["Base Length", int(res["base_len"])],
                ["Low Tests", int(res["low_tests"])],
                ["High Tests", int(res["high_tests"])],
                ["Confidence", round(res["confidence"], 1)],
            ], columns=["Field", "Value"])
            st.dataframe(info, use_container_width=True, hide_index=True)

else:
    preset = st.selectbox("Preset", list(MARKET_PRESETS.keys()))
    default_symbols = ", ".join(MARKET_PRESETS[preset])
    custom = st.text_area(
        "Symbols (pisahkan koma). Bisa campur US / IHSG / forex / futures / crypto.",
        value=default_symbols,
        height=220
    )

    if st.button("Run Scanner", use_container_width=True):
        symbols = [x.strip().upper() for x in custom.split(",") if x.strip()]
        out = scanner_table(symbols, period, interval)

        if out.empty:
            st.warning("Belum ada hasil scan. Bisa jadi simbol tidak valid di Yahoo atau data terlalu sedikit.")
        else:
            st.subheader("Scanner Result")
            st.dataframe(out, use_container_width=True, hide_index=True)

            st.subheader("Bias Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accumulation Bias", int((out["Bias"] == "ACCUM").sum()))
            c2.metric("Distribution Bias", int((out["Bias"] == "DIST").sum()))
            c3.metric("Mixed", int((out["Bias"] == "MIXED").sum()))

            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")
