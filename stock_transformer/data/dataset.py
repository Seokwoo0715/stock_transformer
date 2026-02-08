"""
PyTorch Dataset — 슬라이딩 윈도우 기반 시퀀스 데이터셋
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import add_technical_indicators, create_labels, normalize_window


class StockSequenceDataset(Dataset):
    """
    주식 시퀀스 데이터셋

    각 샘플:
        - x_raw: 정규화된 OHLCV (seq_len, 5)     → Raw Path
        - x_full: 정규화된 전체 피처 (seq_len, F)  → Context Path
        - label: 다음 봉 방향 (0, 1, 2)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 96,
        use_technical: bool = True,
        label_threshold: float = 0.001,
        is_returns: bool = False,
    ):
        self.seq_len = seq_len
        self.ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

        # 기술지표 추가
        if use_technical:
            df = add_technical_indicators(df)

        # 라벨 생성
        labels = create_labels(df, threshold=label_threshold, is_returns=is_returns)

        # 사용할 피처 컬럼
        self.feature_cols = [c for c in df.columns if c in
            self.ohlcv_cols + ["RSI", "MA_10", "MA_30", "MACD", "MACD_signal", "Volume_change"]]

        # numpy 변환
        self.data = df[self.feature_cols].values.astype(np.float32)
        self.ohlcv_data = df[self.ohlcv_cols].values.astype(np.float32)
        self.labels = labels.values

        # 유효 인덱스 (마지막 봉은 라벨 없음)
        self.valid_indices = list(range(self.seq_len, len(self.data) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len

        # Raw Path: OHLCV만
        raw_window = self.ohlcv_data[start_idx:end_idx]
        raw_norm, _, _ = normalize_window(raw_window)

        # Context Path: 전체 피처
        full_window = self.data[start_idx:end_idx]
        full_norm, _, _ = normalize_window(full_window)

        label = self.labels[end_idx]

        return {
            "x_raw": torch.tensor(raw_norm, dtype=torch.float32),
            "x_full": torch.tensor(full_norm, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 96,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    label_threshold: float = 0.001,
    use_technical: bool = True,
    is_returns: bool = False,
) -> tuple:
    """
    시계열 순서를 유지한 train/val/test 분할

    Returns:
        (train_loader, val_loader, test_loader)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end - seq_len:val_end]   # seq_len만큼 겹침 허용
    test_df = df.iloc[val_end - seq_len:]

    train_ds = StockSequenceDataset(train_df, seq_len, use_technical=use_technical, label_threshold=label_threshold, is_returns=is_returns)
    val_ds = StockSequenceDataset(val_df, seq_len, use_technical=use_technical, label_threshold=label_threshold, is_returns=is_returns)
    test_ds = StockSequenceDataset(test_df, seq_len, use_technical=use_technical, label_threshold=label_threshold, is_returns=is_returns)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
