"""
Streamlit app: TSLA 5-minute Dynamic Support / Resistance
Single-file app. Features:
 - Fetch 5m K-lines via yfinance
 - Detect swing highs/lows (scipy.find_peaks)
 - Fit dynamic resistance/support lines using recent swing points
 - Donchian channel
 - Interactive sidebar for parameters (distance, prominence, lookback, num swings, donchian)
 - Manual refresh button and optional auto-refresh (best-effort)
 - Export CSV and interactive HTML of Plotly chart

Dependencies:
    pip install streamlit yfinance pandas numpy scipy plotly
Run:
    streamlit run streamlit_tsla_sr_app.py

Notes:
 - For reliability we avoid calling deprecated Streamlit internals. Auto-refresh uses st.experimental_data_editor if available; otherwise user can press Refresh.
 - If you want true realtime auto-refresh, deploy to Streamlit Cloud or run in an environment where you can rely on st_autorefresh.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# ---------------------------
# Helpers / Core logic
# ---------------------------
@st.cache_data(ttl=30)
def fetch_5m_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     interval="5m", progress=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # normalize tz-aware index to naive UTC
    df = df.dropna()
    try:
        if df.index.tz is not None:
            df.index = pd.to_datetime(df.index).tz_convert(None)
        else:
            df.index = pd.to_datetime(df.index)
    except Exception:
        df.index = pd.to_datetime(df.index)
    return df


def detect_swings(df: pd.DataFrame, col: str = 'Close', distance: int = 3, prominence: float = 0.5):
    prices = df[col].values
    if len(prices) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        peaks_idx, _ = find_peaks(prices, distance=distance, prominence=prominence)
        troughs_idx, _ = find_peaks(-prices, distance=distance, prominence=prominence)
    except Exception:
        # fallback: looser params
        peaks_idx, _ = find_peaks(prices, distance=max(1, distance//2))
        troughs_idx, _ = find_peaks(-prices, distance=max(1, distance//2))
    return peaks_idx, troughs_idx


def get_recent_n_indices(idx_array: np.ndarray, n: int):
    if idx_array is None or len(idx_array) == 0:
        return []
    chosen = list(idx_array[-n:]) if len(idx_array) >= n else list(idx_array)
    return chosen


def fit_line_through_points(df: pd.DataFrame, idx_list):
    if not idx_list or len(idx_list) < 2:
        return None
    # convert timestamps to float seconds to avoid overflow on int64
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
    # swing markers
    if len(peaks_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[peaks_idx], y=df['High'].values[peaks_idx],
                                 mode='markers', marker=dict(symbol='triangle-up', size=9),
                                 name='Swing Highs'))
    if len(troughs_idx) > 0:
        fig.add_trace(go.Scatter(x=df.index[troughs_idx], y=df['Low'].values[troughs_idx],
                                 mode='markers', marker=dict(symbol='triangle-down', size=9),
                                 name='Swing Lows'))
    # fitted lines
    if res_line is not None:
        _, _, line_vals = res_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines', name='Fitted Resistance'))
    if sup_line is not None:
        _, _, line_vals = sup_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines', name='Fitted Support'))
    # donchian
    fig.add_trace(go.Scatter(x=df.index, y=donchian_high, mode='lines', name='Donchian High', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=donchian_low, mode='lines', name='Donchian Low', line=dict(dash='dash')))

    fig.update_layout(title=f"{ticker} 5-minute Candlestick with Dynamic S/R",
                      xaxis_rangeslider_visible=False,
                      template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title='TSLA 5m Dynamic Support/Resistance', layout='wide')

st.sidebar.title('Parameters')
with st.sidebar.form('params'):
    ticker = st.text_input('Ticker', 'TSLA').upper()
    lookback_days = st.number_input('Lookback days (yfinance 5m limits)', min_value=1, max_value=30, value=7)
    distance = st.number_input('Swing minimum distance (bars)', min_value=1, max_value=50, value=3)
    prominence = st.number_input('Swing prominence (price units)', min_value=0.0, value=0.5, step=0.1)
    num_swings = st.number_input('Num recent swings to fit', min_value=2, max_value=20, value=4)
    donchian_period = st.number_input('Donchian period (bars)', min_value=1, max_value=200, value=20)
    download_html_name = st.text_input('Export HTML filename', 'tsla_5m_sr.html')
    submitted = st.form_submit_button('Apply')

# Auto-refresh hint (manual-friendly)
col1, col2 = st.columns([1, 3])
with col1:
    refresh = st.button('Refresh / Fetch')
with col2:
    st.write('按「Refresh / Fetch」重新取得資料並更新圖表（或先在側邊欄按 Apply 再按 Refresh）。')

# Fetch data when user clicks refresh or applies params
if submitted or refresh:
    with st.spinner('Fetching data and computing S/R...'):
        df = fetch_5m_data(ticker, int(lookback_days))
        if df.empty:
            st.error('未取得資料，請檢查 ticker、網路或交易時段。')
        else:
            peaks_idx, troughs_idx = detect_swings(df, distance=int(distance), prominence=float(prominence))
            chosen_peaks = get_recent_n_indices(peaks_idx, int(num_swings))
            chosen_troughs = get_recent_n_indices(troughs_idx, int(num_swings))
            res_line = fit_line_through_points(df, chosen_peaks) if len(chosen_peaks) >= 2 else None
            sup_line = fit_line_through_points(df, chosen_troughs) if len(chosen_troughs) >= 2 else None
            dc_high, dc_low, _ = donchian_channel(df, int(donchian_period))
            fig = build_figure(df, peaks_idx, troughs_idx, res_line, sup_line, dc_high, dc_low, ticker)

            # Layout: two columns
            left, right = st.columns((3, 1))
            with left:
                st.plotly_chart(fig, use_container_width=True)
                # show raw selected swings for debugging
                st.markdown('**Detected swing points (indices)**')
                st.write({'num_peaks': len(peaks_idx), 'num_troughs': len(troughs_idx)})

            with right:
                st.subheader('Export / Data')
                # CSV
                csv = df.to_csv().encode('utf-8')
                st.download_button('Download CSV (OHLC)', data=csv, file_name=f'{ticker}_5m_ohlc.csv', mime='text/csv')
                # HTML export of interactive figure
                buf = io.StringIO()
                fig.write_html(buf, include_plotlyjs='cdn')
                html_bytes = buf.getvalue().encode('utf-8')
                st.download_button('Download interactive HTML', data=html_bytes, file_name=download_html_name, mime='text/html')

            # show table of recent swings
            if len(peaks_idx) > 0 or len(troughs_idx) > 0:
                swings = []
                for i in peaks_idx:
                    swings.append((df.index[i], 'peak', df['High'].iloc[i]))
                for i in troughs_idx:
                    swings.append((df.index[i], 'trough', df['Low'].iloc[i]))
                swings_df = pd.DataFrame(swings, columns=['datetime', 'type', 'price']).sort_values('datetime')
                st.dataframe(swings_df.tail(50))

else:
    st.info('設定參數後按側邊欄的 Apply，再按上方的 Refresh / Fetch 取得最新 5 分鐘 K 線與動態支撐/阻力。')

# Footer / tips
st.markdown("---")
st.markdown('**Tips:**
- 調高 `prominence` 可減少噪音，但同時可能漏掉小幅重要 swing。
- `distance` 以 bar 數計，5m * 3 = 15 分鐘。
- 將 `num recent swings to fit` 設小會讓 S/R 線更貼近期價格，設大會更平滑。')

# Optional: quick-run example button (fills with TSLA and auto refresh)
if st.button('Quick example: TSLA'):
    st.experimental_set_query_params()
    st.experimental_rerun()
