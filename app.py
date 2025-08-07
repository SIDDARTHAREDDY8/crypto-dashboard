import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from email.mime.text import MIMEText
import smtplib, ssl
import numpy as np

# ---------- Helpers ----------
def fmt_usd(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------- Page Config ----------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.markdown("""
    <style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stDataFrame, .stMetric { background-color: #1E222A; }
    </style>
""", unsafe_allow_html=True)
st.title("💹 Real-Time Crypto Market Dashboard (CoinGecko) — Candles + Indicators + Compare")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh", value=False)
    auto_refresh_sec = st.number_input("Interval (sec)", 10, 300, 60)

    st.divider()
    st.subheader("📈 Indicators (on Close)")
    show_sma = st.checkbox("Show SMA", value=True)
    sma_short = st.number_input("SMA Short", 5, 200, 20)
    sma_long  = st.number_input("SMA Long", 5, 400, 50)
    show_ema = st.checkbox("Show EMA", value=False)
    ema_win  = st.number_input("EMA Window", 5, 400, 20)
    show_bb  = st.checkbox("Show Bollinger Bands", value=True)
    bb_win   = st.number_input("BB Window", 5, 200, 20)
    bb_k     = st.number_input("BB K (σ)", 1.0, 4.0, 2.0, step=0.5)

    st.divider()
    st.subheader("🔔 Price Alert")
    alert_enabled = st.checkbox("Enable alert check", value=False)
    threshold_direction = st.selectbox("Trigger when price is…", ["Above", "Below"])
    threshold_price = st.number_input("Threshold (USD)", min_value=0.0, value=100000.0, step=100.0)

    email_enabled = st.checkbox("Send email (requires secrets)", value=False)
    alert_email_to = st.text_input("Recipient email", value="")
    st.caption("Add secrets in .streamlit/secrets.toml – see earlier steps.")

# ---------- Data Fetchers ----------
@st.cache_data(ttl=60)
def load_market_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"market_cap_desc","per_page":50,"page":1,"sparkline":"false"}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as e:
        st.error(f"❌ Failed to load market data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_ohlc(coin_id: str, days: int):
    """
    CoinGecko OHLC: /coins/{id}/ohlc?vs_currency=usd&days=1|7|14|30|90|180|365|max
    Returns list of [time, open, high, low, close]
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}
    try:
        time.sleep(0.7)  # be polite
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ Error fetching OHLC data: {e}")
        return None

# ---------- Indicators ----------
def add_indicators_on_close(df: pd.DataFrame,
                            sma_short=20, sma_long=50,
                            ema_win=20,
                            bb_win=20, bb_k=2.0) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    if show_sma:
        out[f"SMA_{sma_short}"] = close.rolling(int(sma_short)).mean()
        out[f"SMA_{sma_long}"]  = close.rolling(int(sma_long)).mean()
    if show_ema:
        out[f"EMA_{ema_win}"]   = close.ewm(span=int(ema_win), adjust=False).mean()
    if show_bb:
        ma  = close.rolling(int(bb_win)).mean()
        std = close.rolling(int(bb_win)).std()
        out["BB_Mid"]   = ma
        out["BB_Upper"] = ma + float(bb_k) * std
        out["BB_Lower"] = ma - float(bb_k) * std
    return out

# ---------- Email Helper ----------
def send_email(subject: str, body: str, to_addr: str) -> bool:
    try:
        creds = st.secrets["email"]
        user = creds["user"]
        pw = creds["password"]          # App password for Gmail
        smtp_server = creds.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(creds.get("smtp_port", 465))
    except Exception:
        st.warning("⚠️ Email secrets missing in .streamlit/secrets.toml")
        return False
    if not to_addr:
        st.warning("⚠️ No recipient email provided.")
        return False
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    try:
        context = ssl.create_default_context()
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=20) as s:
                s.login(user, pw)
                s.sendmail(user, [to_addr], msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as s:
                s.starttls(context=context)
                s.login(user, pw)
                s.sendmail(user, [to_addr], msg.as_string())
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

# ---------- Main ----------
df = load_market_data()
if df.empty:
    st.warning("❌ Market data could not be loaded.")
    st.stop()

coin_names = df["name"].tolist()
selected_coin = st.selectbox("🔽 Select a Coin", coin_names, index=0)
selected_id = df[df["name"] == selected_coin]["id"].values[0]
row = df[df["id"] == selected_id].iloc[0]
current_price = float(row["current_price"])

# KPIs
st.subheader("📊 Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Top Coin", df.iloc[0]["name"])
try:
    c2.metric("BTC Price", fmt_usd(df[df['id']=='bitcoin'].iloc[0]['current_price']))
    c3.metric("ETH Price", fmt_usd(df[df['id']=='ethereum'].iloc[0]['current_price']))
except Exception:
    st.warning("⚠️ BTC/ETH data temporarily unavailable.")

# Market table
st.subheader("📋 Market Overview (Top 50)")
st.dataframe(
    df[["name","symbol","current_price","market_cap","price_change_percentage_24h","total_volume"]],
    use_container_width=True
)

# Scatter overview
st.subheader("📉 Price vs 24h Volume")
fig_over = px.scatter(
    df, x="current_price", y="total_volume", size="market_cap", color="name",
    hover_name="name", log_y=True, title="Price vs Volume (Log Scale)"
)
st.plotly_chart(fig_over, use_container_width=True)

# ---- Quick Range Buttons (1D / 7D / 30D) ----
st.subheader("🕒 Range")
quick_range = st.radio("Select range", options=["1D","7D","30D"], index=1, horizontal=True)
range_map = {"1D":1, "7D":7, "30D":30}
ohlc_days = range_map[quick_range]

# Candlestick + Indicators
st.subheader(f"🕯️ Candlestick (OHLC) + Indicators — {selected_coin} ({quick_range})")
ohlc = get_ohlc(selected_id, ohlc_days)
if ohlc is None or ohlc.empty:
    st.warning("⚠️ OHLC data not available for this coin/range.")
    st.stop()

chart_df = add_indicators_on_close(
    ohlc, sma_short=sma_short, sma_long=sma_long, ema_win=ema_win, bb_win=bb_win, bb_k=bb_k
)

# Candlestick chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df["timestamp"],
    open=chart_df["open"],
    high=chart_df["high"],
    low=chart_df["low"],
    close=chart_df["close"],
    name="OHLC"
))

# Overlays on Close
if show_sma:
    if f"SMA_{sma_short}" in chart_df:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df[f"SMA_{sma_short}"],
                                 name=f"SMA {sma_short}", mode="lines"))
    if f"SMA_{sma_long}" in chart_df:
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df[f"SMA_{sma_long}"],
                                 name=f"SMA {sma_long}", mode="lines"))
if show_ema and f"EMA_{ema_win}" in chart_df:
    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df[f"EMA_{ema_win}"],
                             name=f"EMA {ema_win}", mode="lines"))
if show_bb and {"BB_Upper","BB_Lower"}.issubset(chart_df.columns):
    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["BB_Upper"],
                             name="BB Upper", mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["BB_Lower"],
                             name="BB Lower", mode="lines", line=dict(width=1), fill="tonexty"))

fig.update_layout(
    title=f"{selected_coin} — Candlestick with SMA/EMA/Bollinger",
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ---- CSV Download (OHLC + indicators) ----
csv_filename = f"{selected_id}_{ohlc_days}d_ohlc_indicators.csv"
csv_buf = chart_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV (OHLC + indicators)",
    data=csv_buf,
    file_name=csv_filename,
    mime="text/csv",
)

# ---- Compare Two Coins (Normalized to 100) ----
# ---- Compare Two Coins (Normalized to 100) ----
st.subheader("🆚 Compare Two Coins — Normalized (Index = 100 at start)")

# Pick coins
top_row = st.columns([1, 1, 1, 1])
with top_row[0]:
    coinA = st.selectbox("Coin A", coin_names,
                         index=coin_names.index(selected_coin) if selected_coin in coin_names else 0,
                         key="cmp_coinA")
with top_row[1]:
    # Preselect a different coin than A if possible
    default_b = 1 if len(coin_names) > 1 and coin_names[0] == coinA else 0
    if default_b < len(coin_names) and coin_names[default_b] == coinA and len(coin_names) > 2:
        default_b = 2
    coinB = st.selectbox("Coin B", coin_names, index=default_b, key="cmp_coinB")

# Color pickers
with top_row[2]:
    colorA = st.color_picker("Color A", "#1f77b4")  # Plotly blue
with top_row[3]:
    colorB = st.color_picker("Color B", "#ef553b")  # Plotly red

if coinA == coinB:
    st.warning("⚠️ Please choose two different coins for comparison.")
else:
    coinA_id = df[df["name"] == coinA]["id"].values[0]
    coinB_id = df[df["name"] == coinB]["id"].values[0]

    ohlcA = get_ohlc(coinA_id, ohlc_days)
    ohlcB = get_ohlc(coinB_id, ohlc_days)

    if ohlcA is None or ohlcA.empty or ohlcB is None or ohlcB.empty:
        st.warning("⚠️ OHLC data not available for one or both selected coins/range.")
    else:
        # Merge on common timestamps
        comp = (
            ohlcA[["timestamp", "close"]].rename(columns={"close": f"{coinA}_close"})
            .merge(
                ohlcB[["timestamp", "close"]].rename(columns={"close": f"{coinB}_close"}),
                on="timestamp",
                how="inner",
            )
            .dropna()
            .reset_index(drop=True)
        )

        # Normalize to 100 at the first common timestamp
        comp[f"{coinA}_idx"] = comp[f"{coinA}_close"] / comp[f"{coinA}_close"].iloc[0] * 100.0
        comp[f"{coinB}_idx"] = comp[f"{coinB}_close"] / comp[f"{coinB}_close"].iloc[0] * 100.0

        # Compute returns/volatility/drawdown
        comp[f"{coinA}_ret"] = comp[f"{coinA}_close"].pct_change()
        comp[f"{coinB}_ret"] = comp[f"{coinB}_close"].pct_change()

        def max_drawdown(series: pd.Series) -> float:
            cummax = series.cummax()
            dd = (series / cummax) - 1.0
            return dd.min()

        stats = pd.DataFrame({
            "Metric": ["Period Return", "Stdev of Returns", "Max Drawdown"],
            coinA: [
                (comp[f"{coinA}_close"].iloc[-1] / comp[f"{coinA}_close"].iloc[0] - 1.0),
                comp[f"{coinA}_ret"].std(),
                max_drawdown(comp[f"{coinA}_close"])
            ],
            coinB: [
                (comp[f"{coinB}_close"].iloc[-1] / comp[f"{coinB}_close"].iloc[0] - 1.0),
                comp[f"{coinB}_ret"].std(),
                max_drawdown(comp[f"{coinB}_close"])
            ]
        })

        # Nicely formatted table
        pretty = stats.copy()
        for col in [coinA, coinB]:
            pretty[col] = np.where(
                pretty["Metric"].isin(["Period Return", "Stdev of Returns", "Max Drawdown"]),
                (pretty[col] * 100).map(lambda x: f"{x:,.2f}%"),
                pretty[col]
            )
        st.subheader(f"📊 Returns & Volatility ({quick_range})")
        st.dataframe(pretty, use_container_width=True)

        # Plot normalized chart with custom colors
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=comp["timestamp"], y=comp[f"{coinA}_idx"],
            name=f"{coinA} (idx)",
            mode="lines",
            line=dict(color=colorA, width=2)
        ))
        fig_cmp.add_trace(go.Scatter(
            x=comp["timestamp"], y=comp[f"{coinB}_idx"],
            name=f"{coinB} (idx)",
            mode="lines",
            line=dict(color=colorB, width=2)
        ))
        fig_cmp.update_layout(
            title=f"{coinA} vs {coinB} — Normalized Performance (Index=100 at start, {quick_range})",
            xaxis_title="Time",
            yaxis_title="Index (100 = start)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ---- CSV Export of comparison data ----
        export_cols = ["timestamp",
                       f"{coinA}_close", f"{coinB}_close",
                       f"{coinA}_idx", f"{coinB}_idx",
                       f"{coinA}_ret", f"{coinB}_ret"]
        comp_csv = comp[export_cols].copy()
        comp_filename = f"compare_{coinA}_{coinB}_{ohlc_days}d.csv"
        st.download_button(
            "⬇️ Download Comparison CSV",
            data=comp_csv.to_csv(index=False).encode("utf-8"),
            file_name=comp_filename,
            mime="text/csv",
        )

# ---- Alerts ----
st.subheader("🔔 Alert Status")
if "last_alert_signature" not in st.session_state:
    st.session_state.last_alert_signature = None

direction_str = "above" if threshold_direction == "Above" else "below"
if alert_enabled:
    crossed = (threshold_direction == "Above" and current_price >= threshold_price) or \
              (threshold_direction == "Below" and current_price <= threshold_price)
    if crossed:
        st.error(f"🔔 ALERT: {selected_coin} is {direction_str} {fmt_usd(threshold_price)} "
                 f"(current: {fmt_usd(current_price)})")
        if email_enabled:
            signature = f"{selected_id}:{direction_str}:{threshold_price}"
            if st.session_state.last_alert_signature != signature:
                subject = f"[Crypto Alert] {selected_coin} is {direction_str.upper()} {fmt_usd(threshold_price)}"
                body = (
                    f"{selected_coin} ({selected_id}) crossed your threshold.\n\n"
                    f"Direction: {direction_str}\n"
                    f"Threshold: {fmt_usd(threshold_price)}\n"
                    f"Current:   {fmt_usd(current_price)}\n"
                )
                if send_email(subject, body, alert_email_to):
                    st.success(f"📧 Email sent to {alert_email_to}")
                    st.session_state.last_alert_signature = signature
            else:
                st.caption("Email already sent for this condition in this session.")
    else:
        st.info(f"No alert: {selected_coin} is {direction_str} {fmt_usd(threshold_price)}. "
                f"Current: {fmt_usd(current_price)}")
else:
    st.caption("Enable alert check in the sidebar to monitor and email on threshold crossing.")

# ---- Auto-refresh (last) ----
if auto_refresh:
    time.sleep(float(auto_refresh_sec))
    st.rerun()

