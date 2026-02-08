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

    # RSI (14기간) — 0~100 범위로 정규화
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = (100 - (100 / (1 + rs))) / 100.0  # 0~1로 스케일링

    # 이동평균 — Close 대비 비율로 변환
    df["MA_10"] = close.rolling(window=10, min_periods=1).mean() / close - 1.0
    df["MA_30"] = close.rolling(window=30, min_periods=1).mean() / close - 1.0

    # MACD (12, 26, 9) — Close 대비 비율로 변환
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = (ema12 - ema26) / close
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 거래량 변화율
    df["Volume_change"] = volume.pct_change().fillna(0).clip(-5, 5)

    # Volume을 로그 스케일로 변환 (큰 값 문제 해결)
    df["Volume"] = np.log1p(df["Volume"])

    return df


def create_labels(df: pd.DataFrame, threshold: float = 0.001, is_returns: bool = False) -> pd.Series:
    """
    다음 봉 대비 수익률 기반 라벨 생성

    Args:
        df: Close 컬럼이 있는 DataFrame
        threshold: Flat 판단 기준 (±0.1%)
        is_returns: True면 Close가 이미 수익률 (multi_ticker 모드)

    Returns:
        라벨 Series (0=Down, 1=Flat, 2=Up)
    """
    if is_returns:
        # 이미 수익률 데이터 → 다음 봉의 Close값 자체가 수익률
        returns = df["Close"].shift(-1)
    else:
        returns = df["Close"].pct_change().shift(-1)

    labels = pd.Series(1, index=df.index, dtype=int)  # 기본 Flat
    labels[returns > threshold] = 2   # Up
    labels[returns < -threshold] = 0  # Down

    return labels


def normalize_window(data: np.ndarray) -> tuple:
    """
    윈도우 내 Min-Max 정규화 (NaN/Inf 안전 처리)

    Args:
        data: shape (seq_len, features)

    Returns:
        (normalized_data, min_vals, max_vals)
    """
    # NaN을 0으로, Inf를 큰 값으로 치환
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 0 나눗셈 방지

    normalized = (data - min_vals) / range_vals

    # 혹시 남은 NaN 제거
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized, min_vals, max_vals
