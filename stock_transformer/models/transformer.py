"""
Dual-Path Causal Transformer — Phase 1 핵심 모델

아키텍처:
    Input (OHLCV) ──┬── Raw Path (OHLCV 보존, skip connection)
                     └── Context Path (Linear → Transformer → 시간 관계 학습)
                              ↓
                     Causal Masked Multi-Head Self-Attention
                              ↓
                     Concat(raw[-1], context[-1])
                              ↓
                     Prediction Head (FFN → class logits)
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """시퀀스 위치 정보를 부여하는 사인/코사인 포지셔널 인코딩"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalTransformerEncoder(nn.Module):
    """
    Causal Masked Transformer Encoder

    미래 시점을 볼 수 없는 인과적(causal) 마스크를 적용.
    각 봉은 자신과 이전 봉들만 참조할 수 있음.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (더 안정적인 학습)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """상삼각 마스크 생성 — 미래 시점 참조 방지"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) — 인과적 self-attention 결과
        """
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        return self.encoder(x, mask=causal_mask)


class DualPathCausalTransformer(nn.Module):
    """
    Dual-Path Causal Transformer 전체 모델

    Path A (Raw):     정규화된 OHLCV → 변형 없이 보존 → concat
    Path B (Context): 전체 피처 → Linear Projection → Positional Encoding
                      → Causal Transformer → concat
    Merge:            concat(raw[-1], context[-1]) → FFN → 예측
    """

    def __init__(
        self,
        input_dim: int = 11,   # 전체 피처 수 (OHLCV + 기술지표)
        raw_dim: int = 5,      # OHLCV 차원
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 128,
        num_classes: int = 3,
        seq_len: int = 96,
    ):
        super().__init__()

        self.raw_dim = raw_dim
        self.d_model = d_model

        # ── Context Path ──
        # Linear Projection: input_dim → d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        self.transformer = CausalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.context_norm = nn.LayerNorm(d_model)

        # ── Prediction Head ──
        # Concat 차원: raw_dim + d_model
        concat_dim = raw_dim + d_model

        self.prediction_head = nn.Sequential(
            nn.Linear(concat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_raw:  (batch, seq_len, 5)    — 정규화된 OHLCV (Raw Path)
            x_full: (batch, seq_len, F)    — 정규화된 전체 피처 (Context Path)

        Returns:
            logits: (batch, num_classes) — 분류 로짓
        """
        # ── Path A: Raw (skip connection) ──
        # 마지막 시점의 OHLCV만 사용
        raw_last = x_raw[:, -1, :]  # (batch, 5)

        # ── Path B: Context (Transformer) ──
        # Linear Projection
        x_emb = self.input_projection(x_full)    # (batch, seq_len, d_model)
        # Positional Encoding
        x_emb = self.pos_encoding(x_emb)
        # Causal Transformer
        context_out = self.transformer(x_emb)     # (batch, seq_len, d_model)
        context_out = self.context_norm(context_out)
        # 마지막 시점 추출
        context_last = context_out[:, -1, :]      # (batch, d_model)

        # ── Merge: Concatenation ──
        h = torch.cat([raw_last, context_last], dim=-1)  # (batch, 5 + d_model)

        # ── Prediction Head ──
        logits = self.prediction_head(h)  # (batch, num_classes)

        return logits
