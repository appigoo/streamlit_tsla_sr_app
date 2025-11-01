"""
Streamlit App: 即時 5分鐘K線 + 動態支撐/阻力線 (已修復所有錯誤)
執行：
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
# 頁面設定
# ========================================
st.set_page_config(
    page_title="動態支撐阻力線",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("即時 5分鐘K線 + 動態支撐/阻力線")
st.markdown("---")

# ========================================
# 側邊欄參數
# ========================================
with st.sidebar:
    st.header("設定參數")
    ticker = st.text_input("股票代碼", value="TSLA", help="例如：AAPL, NVDA, 0050.TW")
    lookback_days = st.slider("回看天數", 1, 30, 7)
    interval = "5m"

    st.markdown("### Swing 偵測")
    swing_distance = st.slider("最小間距 (bars)", 1, 15, 5)
    prominence_factor = st.slider("Prominence 係數 (ATR ×)", 0.1, 1.0, 0.3, step=0.05)
    num_swings = st.slider("擬合線使用點數", 2, 8, 4)

    st.markdown("### Donchian Channel")
    donchian_period = st.slider("Donchian 週期", 10, 50, 20)

    refresh_interval = st.slider("自動刷新 (秒)", 30, 300, 60, step=30)

    st.markdown("### 除錯資訊")
    debug = st.checkbox("顯示除錯資訊", value=False)

# ========================================
# 快取資料
# ========================================
@st.cache_data(ttl=55, show_spinner=False)
def fetch_data(ticker, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days + 2)
    try:
        df = yf.download(
            ticker.upper(),
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            prepost=True,
            auto_adjust=True
        )
        if df.empty:
            return None, "無資料（可能非交易時段）"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()
        if len(df) == 0:
            return None, "資料為空（已過濾）"
        return df, None
    except Exception as e:
        return None, f"下載失敗：{str(e)}"

# ========================================
# 計算 ATR
# ========================================
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# ========================================
# 偵測 Swing（安全回傳整數陣列）
# ========================================
def detect_swings(df, distance, factor):
    prices = df['Close'].values
    atr = calculate_atr(df)
    prominence = np.maximum(atr * factor, df['Close'].std() * 0.05).values

    peaks_idx, _ = find_peaks(prices, distance=distance, prominence=prominence)
    troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=prominence)

    # 強制轉為 int ndarray
    peaks_idx = np.array(peaks_idx, dtype=int) if len(peaks_idx) > 0 else np.array([], dtype=int)
    troughs_idx = np.array(troughs_idx, dtype=int) if len(troughs_idx) > 0 else np.array([], dtype=int)

    return peaks_idx, troughs_idx

# ========================================
# 安全擬合線（防呆版）
# ========================================
def fit_line(df, idx_list):
    """
    安全擬合：過濾空值、nan、不足點數
    回傳 line_vals (array) 或 None
    """
    if idx_list is None or len(idx_list) == 0:
        return None

    # 轉為 numpy array 並清理
    try:
        idx_array = np.array(idx_list, dtype=float)
        idx_array = idx_array[~np.isnan(idx_array)]
        idx_array = idx_array.astype(int)
    except:
        return None

    if len(idx_array) < 2:
        return None

    try:
        y = df['Close'].iloc[idx_array].values
        if len(y) == 0 or np.any(np.isnan(y)):
            return None
        slope, intercept = np.polyfit(idx_array, y, 1)
        x_all = np.arange(len(df))
        return slope * x_all + intercept
    except Exception as e:
        if debug:
            st.warning(f"擬合失敗：{e}")
        return None

# ========================================
# Donchian Channel
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

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=ticker.upper()
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
                             line=dict(color='gray', dash='dash'), name='Donchian High'))
    fig.add_trace(go.Scatter(x=df.index, y=dc_low, mode='lines',
                             line=dict(color='gray', dash='dash'), name='Donchian Low',
                             fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))

    fig.update_layout(
        title=f"{ticker.upper()} 5分鐘K線 + 動態支撐/阻力",
        xaxis_title="時間",
        yaxis_title="價格",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
        height=700,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

# ========================================
# 主迴圈
# ========================================
placeholder = st.empty()

while True:
    with placeholder.container():
        df, error = fetch_data(ticker, lookback_days)

        if error:
            st.error(f"無法取得資料：{error}")
            st.info("提示：\n"
                    "- 確認股票代碼正確（含 .TW, .SS）\n"
                    "- 交易時段：美股 09:30-16:00 ET\n"
                    "- 網路正常")
            time.sleep(refresh_interval)
            continue

        # 偵測 swing
        peaks_idx, troughs_idx = detect_swings(df, swing_distance, prominence_factor)

        # 選取最近 N 個點
        chosen_peaks = peaks_idx[-num_swings:].tolist() if len(peaks_idx) >= num_swings else peaks_idx.tolist()
        chosen_troughs = troughs_idx[-num_swings:].tolist() if len(troughs_idx) >= num_swings else troughs_idx.tolist()

        # 擬合線（安全）
        res_line = fit_line(df, chosen_peaks)
        sup_line = fit_line(df, chosen_troughs)

        # Donchian
        dc_high, dc_low = donchian_channel(df, donchian_period)

        # 繪圖
        fig = plot_chart(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low)
        st.plotly_chart(fig, use_container_width=True)

        # 資訊欄
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最新價格", f"${df['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Swing High", len(peaks_idx))
        with col3:
            st.metric("Swing Low", len(troughs_idx))
        with col4:
            st.metric("資料筆數", len(df))

        # 除錯資訊
        if debug:
            with st.expander("除錯資訊"):
                st.write("peaks_idx:", peaks_idx.tolist())
                st.write("troughs_idx:", troughs_idx.tolist())
                st.write("chosen_peaks:", chosen_peaks)
                st.write("chosen_troughs:", chosen_troughs)
                st.write("res_line shape:", res_line.shape if res_line is not None else None)

        # 刷新提示
        st.caption(f"每 {refresh_interval} 秒更新 | 最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(refresh_interval)
    st.rerun()
