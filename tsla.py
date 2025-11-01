"""
Streamlit App: 即時 5分鐘K線 + 動態支撐/阻力線
已修復：
1. ValueError: x must be a 1-D array
2. The truth value of a Series is ambiguous
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ========================================
# 頁面設定
# ========================================
st.set_page_config(page_title="動態支撐阻力線", layout="wide", initial_sidebar_state="expanded")
st.title("即時 5分鐘K線 + 動態支撐/阻力線")
st.markdown("---")

# ========================================
# 側邊欄
# ========================================
with st.sidebar:
    st.header("設定參數")
    ticker = st.text_input("股票代碼", value="TSLA")
    lookback_days = st.slider("回看天數", 1, 30, 7)
    interval = "5m"

    st.markdown("### Swing 偵測")
    swing_distance = st.slider("最小間距", 1, 15, 5)
    prominence_factor = st.slider("Prominence 係數", 0.1, 1.0, 0.3, step=0.05)
    num_swings = st.slider("擬合點數", 2, 8, 4)

    st.markdown("### Donchian")
    donchian_period = st.slider("週期", 10, 50, 20)

    refresh_interval = st.slider("刷新秒數", 30, 300, 60, step=30)
    debug = st.checkbox("除錯模式", value=False)

# ========================================
# 快取資料
# ========================================
@st.cache_data(ttl=55, show_spinner=False)
def fetch_data(ticker, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days + 2)
    try:
        df = yf.download(ticker.upper(), start=start, end=end,
                         interval=interval, progress=False, prepost=True, auto_adjust=True)
        if df.empty:
            return None, "無資料"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()
        return df, None
    except Exception as e:
        return None, f"下載錯誤：{e}"

# ========================================
# ATR
# ========================================
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# ========================================
# 修正版 detect_swings（雙重保險）
# ========================================
def detect_swings(df, distance, factor):
    prices = df['Close'].values.astype(np.float64)
    if len(prices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    atr = calculate_atr(df)
    min_prom = df['Close'].std() * 0.05
    prom_array = np.maximum(atr * factor, min_prom).values  # 強制 .values

    # 保險：長度對齊
    if len(prom_array) != len(prices):
        prom_array = np.full(len(prices), prom_array.mean())

    try:
        peaks_idx, _ = find_peaks(prices, distance=distance, prominence=prom_array)
        troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=prom_array)
    except Exception as e:
        if debug:
            st.warning(f"find_peaks 失敗：{e}")
        const_prom = max(min_prom, atr.mean() * factor)
        peaks_idx, _ = find_peaks(prices, distance=distance, prominence=const_prom)
        troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=const_prom)

    return np.array(peaks_idx, dtype=int), np.array(troughs_idx, dtype=int)

# ========================================
# 安全擬合
# ========================================
def fit_line(df, idx_list):
    if not idx_list or len(idx_list) < 2:
        return None
    try:
        idx = np.array(idx_list, dtype=int)
        y = df['Close'].iloc[idx].values
        if np.any(np.isnan(y)):
            return None
        slope, intercept = np.polyfit(idx, y, 1)
        return slope * np.arange(len(df)) + intercept
    except:
        return None

# ========================================
# Donchian
# ========================================
def donchian_channel(df, period):
    high = df['High'].rolling(window=period, min_periods=1).max()
    low = df['Low'].rolling(window=period, min_periods=1).min()
    return high, low

# ========================================
# 繪圖
# ========================================
def plot_chart(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=ticker.upper()))

    if len(peaks_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[peaks_idx], y=df['High'].iloc[peaks_idx],
                                 mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'),
                                 name='Swing High'))
    if len(troughs_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[troughs_idx], y=df['Low'].iloc[troughs_idx],
                                 mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'),
                                 name='Swing Low'))

    if res_line is not None:
        fig.add_trace(go.Scatter(x=df.index, y=res_line, mode='lines',
                                 line=dict(color='red', width=2, dash='dot'), name='阻力線'))
    if sup_line is not None:
        fig.add_trace(go.Scatter(x=df.index, y=sup_line, mode='lines',
                                 line=dict(color='green', width=2, dash='dot'), name='支撐線'))

    fig.add_trace(go.Scatter(x=df.index, y=dc_high, mode='lines', line=dict(color='gray', dash='dash'), name='Donchian High'))
    fig.add_trace(go.Scatter(x=df.index, y=dc_low, mode='lines', line=dict(color='gray', dash='dash'), name='Donchian Low',
                             fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))

    fig.update_layout(title=f"{ticker.upper()} 5分鐘K線", xaxis_title="時間", yaxis_title="價格",
                      xaxis_rangeslider_visible=False, template="plotly_white",
                      hovermode="x unified", height=700)
    return fig

# ========================================
# 主迴圈
# ========================================
placeholder = st.empty()

while True:
    with placeholder.container():
        df, error = fetch_data(ticker, lookback_days)

        if df is None or df.empty:
            st.error(f"無法取得資料：{error or '資料為空'}")
            time.sleep(refresh_interval)
            continue

        peaks_idx, troughs_idx = detect_swings(df, swing_distance, prominence_factor)

        chosen_peaks = peaks_idx[-num_swings:].tolist() if len(peaks_idx) >= num_swings else peaks_idx.tolist()
        chosen_troughs = troughs_idx[-num_swings:].tolist() if len(troughs_idx) >= num_swings else troughs_idx.tolist()

        res_line = fit_line(df, chosen_peaks)
        sup_line = fit_line(df, chosen_troughs)
        dc_high, dc_low = donchian_channel(df, donchian_period)

        fig = plot_chart(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("最新價", f"${df['Close'].iloc[-1]:.2f}")
        with col2: st.metric("High 點", len(peaks_idx))
        with col3: st.metric("Low 點", len(troughs_idx))

        if debug:
            with st.expander("除錯"):
                st.write("ATR shape:", calculate_atr(df).shape)
                st.write("prom_array:", prom_array[:5] if 'prom_array' in locals() else "N/A")

        st.caption(f"每 {refresh_interval}s 更新 | {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(refresh_interval)
    st.rerun()
