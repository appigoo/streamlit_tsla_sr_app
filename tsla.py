"""
Streamlit App: 即時 5分鐘K線 + 動態支撐/阻力線
執行方式：
    pip install streamlit yfinance pandas numpy scipy plotly
    streamlit run app.py
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
# Streamlit 頁面設定
# ========================================
st.set_page_config(
    page_title="動態支撐阻力線",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("即時 5分鐘K線 + 動態支撐/阻力線")
st.markdown("---")

# ========================================
# 側邊欄：參數設定
# ========================================
with st.sidebar:
    st.header("設定參數")
    
    ticker = st.text_input("股票代碼", value="TSLA", help="例如：AAPL, NVDA, 0050.TW")
    lookback_days = st.slider("回看天數", 1, 30, 7)
    interval = st.selectbox("K線週期", ["5m"], disabled=True)  # 目前僅支援 5m
    
    st.markdown("### Swing 偵測")
    swing_distance = st.slider("最小間距 (bars)", 1, 15, 5)
    prominence_factor = st.slider("Prominence 係數 (ATR ×)", 0.1, 1.0, 0.3, step=0.05)
    num_swings = st.slider("擬合線使用點數", 2, 8, 4)
    
    st.markdown("### Donchian Channel")
    donchian_period = st.slider("Donchian 週期", 10, 50, 20)

    refresh_interval = st.slider("自動刷新 (秒)", 30, 300, 60)

# ========================================
# 快取資料（避免每次互動都重下載）
# ========================================
@st.cache_data(ttl=60, show_spinner=False)  # 快取 60 秒
def fetch_data(ticker, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days + 1)  # 多取一天確保足夠資料
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="5m",
            progress=False,
            prepost=True,
            auto_adjust=True
        )
        if df.empty:
            return None, "無資料"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()
        return df, None
    except Exception as e:
        return None, f"下載失敗：{str(e)}"

# ========================================
# 計算 ATR（用於 prominence）
# ========================================
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# ========================================
# 偵測 Swing High / Low
# ========================================
def detect_swings(df, distance, factor):
    prices = df['Close'].values
    atr = calculate_atr(df)
    prominence = atr * factor
    min_prom = df['Close'].std() * 0.05
    prominence = np.maximum(prominence, min_prom)

    peaks_idx, _ = find_peaks(prices, distance=distance, prominence=prominence)
    troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=prominence)
    return peaks_idx, troughs_idx

# ========================================
# 擬合直線（使用相對序數）
# ========================================
def fit_line(df, idx_list):
    if len(idx_list) < 2:
        return None
    x = np.array(idx_list)
    y = df['Close'].values[idx_list]
    slope, intercept = np.polyfit(x, y, 1)
    x_all = np.arange(len(df))
    return slope * x_all + intercept

# ========================================
# Donchian Channel
# ========================================
def donchian_channel(df, period):
    high = df['High'].rolling(window=period, min_periods=1).max()
    low = df['Low'].rolling(window=period, min_periods=1).min()
    return high, low

# ========================================
# 主圖表繪製
# ========================================
def plot_chart(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low):
    fig = go.Figure()

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=ticker
    ))

    # Swing 點
    if len(peaks_idx) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[peaks_idx], y=df['High'].iloc[peaks_idx],
            mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'),
            name='Swing High'
        ))
    if len(troughs_idx) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[troughs_idx], y=df['Low'].iloc[troughs_idx],
            mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'),
            name='Swing Low'
        ))

    # 擬合線
    if res_line is not None:
        fig.add_trace(go.Scatter(x=df.index, y=res_line, mode='lines',
                                 line=dict(color='red', width=2, dash='dot'),
                                 name='動態阻力'))
    if sup_line is not None:
        fig.add_trace(go.Scatter(x=df.index, y=sup_line, mode='lines',
                                 line=dict(color='green', width=2, dash='dot'),
                                 name='動態支撐'))

    # Donchian
    fig.add_trace(go.Scatter(x=df.index, y=dc_high, mode='lines',
                             line=dict(color='gray', dash='dash'), name=f'Donchian High'))
    fig.add_trace(go.Scatter(x=df.index, y=dc_low, mode='lines',
                             line=dict(color='gray', dash='dash'), name=f'Donchian Low',
                             fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))

    fig.update_layout(
        title=f"{ticker.upper()} 5分鐘K線 + 動態支撐/阻力",
        xaxis_title="時間",
        yaxis_title="價格",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
        height=700
    )
    return fig

# ========================================
# 主邏輯
# ========================================
placeholder = st.empty()

while True:
    with placeholder.container():
        # 取得資料
        df, error = fetch_data(ticker.upper(), lookback_days)
        
        if error:
            st.error(f"無法取得資料：{error}")
            st.info("請檢查：\n"
                    "- 股票代碼是否正確（支援 .TW, .SS 等）\n"
                    "- 是否在交易時段\n"
                    "- 網路連線")
            time.sleep(refresh_interval)
            continue

        if df.empty:
            st.warning("資料為空")
            time.sleep(refresh_interval)
            continue

        # 偵測 swing
        peaks_idx, troughs_idx = detect_swings(df, swing_distance, prominence_factor)
        chosen_peaks = peaks_idx[-num_swings:] if len(peaks_idx) >= num_swings else peaks_idx
        chosen_troughs = troughs_idx[-num_swings:] if len(troughs_idx) >= num_swings else troughs_idx

        # 擬合線
        res_line = fit_line(df, chosen_peaks)
        sup_line = fit_line(df, chosen_troughs)

        # Donchian
        dc_high, dc_low = donchian_channel(df, donchian_period)

        # 繪圖
        fig = plot_chart(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low)
        st.plotly_chart(fig, use_container_width=True)

        # 資訊欄
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新價格", f"${df['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Swing High 數", len(peaks_idx))
        with col3:
            st.metric("Swing Low 數", len(troughs_idx))

        # 自動刷新倒數
        st.caption(f"每 {refresh_interval} 秒自動更新 | 最後更新：{datetime.now().strftime('%H:%M:%S')}")

    # 等待並刷新
    time.sleep(refresh_interval)
    st.rerun()
