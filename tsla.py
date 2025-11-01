"""
Streamlit app: TSLA 5-minute Dynamic Support / Resistance
Fixed syntax issue on multiline markdown string.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

@st.cache_data(ttl=30)
def fetch_5m_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     interval="5m", progress=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    if df.index.tz is not None:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    else:
        df.index = pd.to_datetime(df.index)
    return df

def detect_swings(df: pd.DataFrame, col: str = 'Close', distance: int = 3, prominence: float = 0.5):
    prices = df[col].values
    if len(prices) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    peaks_idx, _ = find_peaks(prices, distance=distance, prominence=prominence)
    troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=prominence)
    return peaks_idx, troughs_idx

def get_recent_n_indices(idx_array: np.ndarray, n: int):
    if len(idx_array) == 0:
        return []
    return list(idx_array[-n:]) if len(idx_array) >= n else list(idx_array)

def fit_line_through_points(df: pd.DataFrame, idx_list):
    if not idx_list or len(idx_list) < 2:
        return None
    x_points = np.array([df.index[i].astype('int64') / 1e9 for i in idx_list])
    y_points = df['Close'].values[idx_list]
    slope, intercept = np.polyfit(x_points, y_points, 1)
    x_all = np.array([ts.astype('int64') / 1e9 for ts in df.index])
    line_vals = slope * x_all + intercept
    return slope, intercept, line_vals

def donchian_channel(df: pd.DataFrame, period: int):
    high = df['High'].rolling(window=period, min_periods=1).max()
    low = df['Low'].rolling(window=period, min_periods=1).min()
    mid = (high + low) / 2.0
    return high, low, mid

def build_figure(df: pd.DataFrame, peaks_idx, troughs_idx, res_line, sup_line, donchian_high, donchian_low, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name=f"{ticker} 5m"))
    if len(peaks_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[peaks_idx], y=df['High'].values[peaks_idx],
                                 mode='markers', marker=dict(symbol='triangle-up', size=9),
                                 name='Swing Highs'))
    if len(troughs_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[troughs_idx], y=df['Low'].values[troughs_idx],
                                 mode='markers', marker=dict(symbol='triangle-down', size=9),
                                 name='Swing Lows'))
    if res_line is not None:
        _, _, line_vals = res_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines', name='Fitted Resistance'))
    if sup_line is not None:
        _, _, line_vals = sup_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines', name='Fitted Support'))
    fig.add_trace(go.Scatter(x=df.index, y=donchian_high, mode='lines', name='Donchian High', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=donchian_low, mode='lines', name='Donchian Low', line=dict(dash='dash')))
    fig.update_layout(title=f"{ticker} 5-minute Candlestick with Dynamic S/R",
                      xaxis_rangeslider_visible=False,
                      template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

st.set_page_config(page_title='TSLA 5m Dynamic Support/Resistance', layout='wide')
st.sidebar.title('Parameters')
with st.sidebar.form('params'):
    ticker = st.text_input('Ticker', 'TSLA').upper()
    lookback_days = st.number_input('Lookback days', min_value=1, max_value=30, value=7)
    distance = st.number_input('Swing distance (bars)', min_value=1, max_value=50, value=3)
    prominence = st.number_input('Swing prominence', min_value=0.0, value=0.5, step=0.1)
    num_swings = st.number_input('Num swings', min_value=2, max_value=20, value=4)
    donchian_period = st.number_input('Donchian period', min_value=1, max_value=200, value=20)
    download_html_name = st.text_input('Export HTML filename', 'tsla_5m_sr.html')
    submitted = st.form_submit_button('Apply')

col1, col2 = st.columns([1, 3])
with col1:
    refresh = st.button('Refresh / Fetch')
with col2:
    st.write('按「Refresh / Fetch」重新取得資料並更新圖表。')

if submitted or refresh:
    df = fetch_5m_data(ticker, int(lookback_days))
    if df.empty:
        st.error('未取得資料，請檢查 ticker、網路或交易時段。')
    else:
        peaks_idx, troughs_idx = detect_swings(df, distance=int(distance), prominence=float(prominence))
        chosen_peaks = get_recent_n_indices(peaks_idx, int(num_swings))
        chosen_troughs = get_recent_n_indices(troughs_idx, int(num_swings))
        res_line = fit_line_through_points(df, chosen_peaks)
        sup_line = fit_line_through_points(df, chosen_troughs)
        dc_high, dc_low, _ = donchian_channel(df, int(donchian_period))
        fig = build_figure(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low, ticker)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button('Download HTML', data=fig.to_html(), file_name=download_html_name)
else:
    st.info('設定參數後按 Apply，再按 Refresh / Fetch 取得最新 5 分鐘 K 線與支撐/阻力。')

st.markdown("""---
**Tips:**
- 調高 `prominence` 可減少噪音，但可能漏掉小幅 swing。
- `distance` 為 bar 數，5m * 3 = 15 分鐘。
- `num_swings` 小 → 線更貼近近期價格；大 → 線更平滑。
""")
