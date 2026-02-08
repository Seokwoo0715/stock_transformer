"""
Phase 1 — Causal Transformer Stock Predictor 설정
GOOGL 10분봉 Dual-Path Transformer
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    ticker: str = "GOOGL"
    interval: str = "10m"           # 10분봉
    seq_len: int = 96               # 입력 시퀀스 길이 (48~144 범위)
    features: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio = 1 - train_ratio - val_ratio


@dataclass
class ModelConfig:
    # 입력 차원
    input_dim: int = 5              # OHLCV
    raw_dim: int = 5                # Raw Path 차원 (OHLCV 그대로)

    # Transformer 설정
    d_model: int = 64               # Context Path 임베딩 차원
    n_heads: int = 4                # Multi-Head Attention 헤드 수
    n_layers: int = 3               # Transformer 레이어 수
    d_ff: int = 256                 # Feed-Forward 히든 차원
    dropout: float = 0.1

    # Prediction Head
    head_hidden: int = 128          # FFN 히든 차원
    num_classes: int = 3            # Up / Down / Flat

    # 기술지표 추가 시 input_dim 변경
    use_technical_indicators: bool = True
    technical_features: List[str] = field(default_factory=lambda: [
        "RSI", "MA_10", "MA_30", "MACD", "MACD_signal", "Volume_change"
    ])

    @property
    def total_input_dim(self) -> int:
        """전처리 후 실제 입력 차원"""
        base = self.input_dim
        if self.use_technical_indicators:
            base += len(self.technical_features)
        return base


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15              # Early stopping
    label_threshold: float = 0.001  # 수익률 ±0.1% 이내 → Flat
    device: str = "auto"            # auto, cpu, cuda, mps
    seed: int = 42
    save_dir: str = "checkpoints"
