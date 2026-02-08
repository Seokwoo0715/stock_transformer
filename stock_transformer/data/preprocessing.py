"""
전처리 모듈 — 정규화, 기술지표, 라벨 생성
"""
import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기술지표 추가: RSI, MA, MACD, 거래량 변화율

    Args:
        df: OHLCV DataFrame

    Returns:
        기술지표가 추가된 DataFrame
    """
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    # RSI (14기간)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 이동평균
    df["MA_10"] = close.rolling(window=10, min_periods=1).mean()
    df["MA_30"] = close.rolling(window=30, min_periods=1).mean()

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 거래량 변화율
    df["Volume_change"] = volume.pct_change().fillna(0).clip(-10, 10)

    return df


def create_labels(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    다음 봉 대비 수익률 기반 라벨 생성

    Args:
        df: Close 컬럼이 있는 DataFrame
        threshold: Flat 판단 기준 (±0.1%)

    Returns:
        라벨 Series (0=Down, 1=Flat, 2=Up)
    """
    returns = df["Close"].pct_change().shift(-1)  # 다음 봉 수익률

    labels = pd.Series(1, index=df.index, dtype=int)  # 기본 Flat
    labels[returns > threshold] = 2   # Up
    labels[returns < -threshold] = 0  # Down

    return labels


def normalize_window(data: np.ndarray) -> tuple:
    """
    윈도우 내 Min-Max 정규화

    Args:
        data: shape (seq_len, features)

    Returns:
        (normalized_data, min_vals, max_vals)
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 0 나눗셈 방지

    normalized = (data - min_vals) / range_vals
    return normalized, min_vals, max_vals
