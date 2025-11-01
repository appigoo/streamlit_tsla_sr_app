"""
TSLA 5-minute Kline -> 自動計算並畫出動態支撐/阻力線 (Support / Resistance)
Requirements:
    pip install yfinance pandas numpy scipy plotly
Run:
    python tsla_support_resistance.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config（請依需求調整）
# ---------------------------
TICKER = "TSLA"
INTERVAL = "5m"                    # 5-minute K-line
LOOKBACK_DAYS = 7                  # 最近幾天資料
SWING_DISTANCE = 5                 # 最小間距（根絕波動可調大）
ATR_PERIOD = 14                    # 用於計算 prominence 的 ATR 週期
PROMINENCE_FACTOR = 0.3            # prominence = ATR * factor（建議 0.2~0.5）
NUM_SWINGS_FOR_LINE = 4            # 用最近幾個 swing points 擬合
DONCHIAN_PERIOD = 20               # Donchian channel 週期

# ---------------------------
# Helpers
# ---------------------------
def fetch_5m_data(ticker=TICKER, interval=INTERVAL, days=LOOKBACK_DAYS):
    """取得 5 分鐘 K 線，處理時區與空資料"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            prepost=False,
            auto_adjust=True
        )
        if df.empty:
            raise ValueError("yfinance 回傳空資料框")
        # 統一為 UTC naive
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
        df = df.dropna()
        return df
    except Exception as e:
        raise RuntimeError(f"無法取得資料：{e}\n"
                           "請檢查：1) 網路連線 2) 是否在交易時段 3) Ticker 是否正確") from e


def calculate_atr(df, period=ATR_PERIOD):
    """計算 ATR（用於動態 prominence）"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def detect_swings(df, distance=SWING_DISTANCE, prominence_factor=PROMINENCE_FACTOR):
    """偵測 swing high / low，使用動態 prominence"""
    prices = df['Close'].values
    atr = calculate_atr(df)
    prominence = atr * prominence_factor
    # 為了避免 prominence 為 0，設定最小值
    min_prom = df['Close'].std() * 0.05
    prominence = np.maximum(prominence, min_prom)

    # 偵測 peaks（阻力候選）
    peaks_idx, _ = find_peaks(
        prices,
        distance=distance,
        prominence=prominence
    )
    # 偵測 troughs（支撐候選）
    troughs_idx, _ = find_peaks(
        -prices,
        distance=distance,
        prominence=prominence
    )
    return peaks_idx, troughs_idx


def fit_line_through_points(df, idx_list):
    """
    最小平方法擬合直線，使用「相對序數」作為 x，避免時間戳溢位
    回傳 (slope, intercept, line_vals) 或 None
    """
    if len(idx_list) < 2:
        return None
    x_points = np.array(idx_list)                   # 相對序數 0,1,2,...
    y_points = df['Close'].values[idx_list]
    slope, intercept = np.polyfit(x_points, y_points, 1)
    x_all = np.arange(len(df))
    line_vals = slope * x_all + intercept
    return slope, intercept, line_vals


def get_recent_n_indices(idx_array, n=NUM_SWINGS_FOR_LINE):
    """取最近 n 個（不足則全部）"""
    return list(idx_array[-n:]) if len(idx_array) >= n else list(idx_array)


def donchian_channel(df, period=DONCHIAN_PERIOD):
    high = df['High'].rolling(window=period, min_periods=1).max()
    low = df['Low'].rolling(window=period, min_periods=1).min()
    mid = (high + low) / 2
    return high, low, mid


# ---------------------------
# Main processing
# ---------------------------
def build_chart(ticker=TICKER):
    df = fetch_5m_data(ticker)
    if df.empty:
        raise RuntimeError("資料為空，無法繪圖。")

    # 偵測 swing
    peaks_idx, troughs_idx = detect_swings(df)

    # 選取最近的 swing points
    chosen_peak_idx = get_recent_n_indices(peaks_idx)
    chosen_trough_idx = get_recent_n_indices(troughs_idx)

    # 擬合支撐 / 阻力線
    resistance_line = fit_line_through_points(df, chosen_peak_idx)
    support_line = fit_line_through_points(df, chosen_trough_idx)

    # Donchian Channel
    dc_high, dc_low, dc_mid = donchian_channel(df)

    # === 繪圖 ===
    fig = go.Figure()

    # K 線
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=f"{ticker} 5m"
    ))

    # Swing Highs / Lows
    if len(peaks_idx) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[peaks_idx], y=df['High'].values[peaks_idx],
            mode='markers', marker=dict(symbol='triangle-up', size=9, color='red'),
            name='Swing Highs'
        ))
    if len(troughs_idx) > 0:
        fig.add_trace(go.Scatter(
            x=df.index[troughs_idx], y=df['Low'].values[troughs_idx],
            mode='markers', marker=dict(symbol='triangle-down', size=9, color='green'),
            name='Swing Lows'
        ))

    # 擬合線
    if resistance_line is not None:
        _, _, line_vals = resistance_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines',
                                 line=dict(color='red', width=2, dash='dot'),
                                 name='動態阻力線'))
    if support_line is not None:
        _, _, line_vals = support_line
        fig.add_trace(go.Scatter(x=df.index, y=line_vals, mode='lines',
                                 line=dict(color='green', width=2, dash='dot'),
                                 name='動態支撐線'))

    # Donchian Channel（填色）
    fig.add_trace(go.Scatter(x=df.index, y=dc_high, mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name=f'Donchian High ({DONCHIAN_PERIOD})'))
    fig.add_trace(go.Scatter(x=df.index, y=dc_low, mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name=f'Donchian Low ({DONCHIAN_PERIOD})',
                             fill=None))
    fig.add_trace(go.Scatter(x=df.index, y=dc_low, mode='lines',
                             line=dict(color='rgba(0,0,0,0)'),
                             fill='tonexty', fillcolor='rgba(200,200,200,0.2)',
                             showlegend=False))

    # 佈局
    fig.update_layout(
        title=f"{ticker} 5分鐘K線 + 動態支撐/阻力 (最近 {LOOKBACK_DAYS} 天)",
        xaxis_title="時間",
        yaxis_title="價格 (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig, df


# ---------------------------
# Run and show
# ---------------------------
if __name__ == "__main__":
    try:
        fig, df = build_chart(TICKER)
        fig.show()
        html_path = "tsla_5m_support_resistance.html"
        fig.write_html(html_path)
        print(f"圖表已儲存至：{html_path}")
    except Exception as e:
        print(f"程式執行失敗：{e}")
