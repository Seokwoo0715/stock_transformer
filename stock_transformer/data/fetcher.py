"""
GOOGL 10분봉 OHLCV 데이터 수집 모듈
yfinance를 사용하여 데이터를 가져옴
"""
import pandas as pd
import yfinance as yf
from pathlib import Path


def fetch_intraday(
    ticker: str = "GOOGL",
    interval: str = "10m",
    period: str = "60d",
    save_path: str = None,
) -> pd.DataFrame:
    """
    야후 파이낸스에서 인트라데이 데이터 수집

    Args:
        ticker: 종목 코드
        interval: 봉 간격 (1m, 2m, 5m, 10m, 15m, 30m, 60m, 90m)
        period: 수집 기간 (최대 60일 for < 1d interval)
        save_path: CSV 저장 경로 (None이면 저장 안 함)

    Returns:
        OHLCV DataFrame
    """
    # yfinance는 10m을 직접 지원하지 않으므로 5m → 10m 리샘플링
    if interval == "10m":
        raw_interval = "5m"
    else:
        raw_interval = interval

    print(f"[Fetcher] {ticker} {interval} 데이터 수집 중 (period={period})...")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=raw_interval)

    if df.empty:
        raise ValueError(f"{ticker}에 대한 데이터를 가져올 수 없습니다.")

    # 불필요한 컬럼 제거
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # 10분봉 리샘플링
    if interval == "10m":
        df = _resample_to_10min(df)

    # 장중 데이터만 유지 (프리/애프터 마켓 제거)
    df = _filter_market_hours(df)

    # 결측치 처리
    df = df.dropna()

    print(f"[Fetcher] 수집 완료: {len(df)}개 봉 ({df.index[0]} ~ {df.index[-1]})")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"[Fetcher] 저장 완료: {path}")

    return df


def _resample_to_10min(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 10분봉 리샘플링"""
    resampled = df.resample("10min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    return resampled.dropna()


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """미국 정규장 시간만 필터 (9:30 ~ 16:00 ET)"""
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    market_mask = (df.index.time >= pd.Timestamp("09:30").time()) & \
                  (df.index.time < pd.Timestamp("16:00").time())
    return df[market_mask]


def load_cached_or_fetch(
    ticker: str = "GOOGL",
    interval: str = "10m",
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """캐시된 데이터가 있으면 로드, 없으면 새로 수집"""
    cache_path = Path(cache_dir) / f"{ticker}_{interval}.csv"

    if cache_path.exists():
        print(f"[Fetcher] 캐시 로드: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = fetch_intraday(ticker, interval, save_path=str(cache_path))
    return df


def fetch_multi_tickers(
    tickers: list,
    interval: str = "10m",
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """
    여러 종목 데이터를 수집하고 합침

    각 종목의 OHLCV를 개별적으로 정규화하기 위해
    가격을 수익률(pct_change)로 변환하여 스케일을 통일.

    Returns:
        합쳐진 DataFrame (수익률 기반)
    """
    all_dfs = []

    for ticker in tickers:
        try:
            df = load_cached_or_fetch(ticker, interval, cache_dir)

            # 가격을 수익률로 변환 (종목 간 스케일 통일)
            price_cols = ["Open", "High", "Low", "Close"]
            for col in price_cols:
                df[col] = df[col].pct_change()

            # Volume은 로그 스케일 후 변화율
            df["Volume"] = df["Volume"].apply(lambda x: max(x, 1))  # 0 방지
            df["Volume"] = df["Volume"].pct_change()

            # 첫 행 제거 (pct_change로 인한 NaN)
            df = df.iloc[1:]

            # Inf/NaN 클리핑
            df = df.clip(-0.5, 0.5)
            df = df.fillna(0)

            all_dfs.append(df)
            print(f"[Fetcher] {ticker}: {len(df)}개 봉 추가")
        except Exception as e:
            print(f"[Fetcher] {ticker} 수집 실패: {e}")

    if not all_dfs:
        raise ValueError("수집된 데이터가 없습니다.")

    combined = pd.concat(all_dfs, axis=0)
    combined = combined.sort_index()

    print(f"[Fetcher] 총 {len(combined)}개 봉 (종목 {len(all_dfs)}개 합산)")
    return combined
