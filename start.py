"""
Phase 1 — Causal Transformer Stock Predictor
GOOGL 10분봉 Dual-Path Transformer

Usage:
    python start.py
"""
import torch
import random
import numpy as np

from stock_transformer.config import DataConfig, ModelConfig, TrainConfig
from stock_transformer.data.fetcher import load_cached_or_fetch
from stock_transformer.data.dataset import create_dataloaders
from stock_transformer.models.transformer import DualPathCausalTransformer
from stock_transformer.utils.trainer import get_device, train, test_report


def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ── 설정 로드 ──
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)

    print("=" * 60)
    print("  Phase 1 — Dual-Path Causal Transformer")
    print(f"  Ticker: {data_cfg.ticker} | Interval: {data_cfg.interval}")
    print(f"  Seq Length: {data_cfg.seq_len} | Device: {device}")
    print("=" * 60)

    # ── 1. 데이터 수집 ──
    print("\n[Step 1] 데이터 수집")
    df = load_cached_or_fetch(
        ticker=data_cfg.ticker,
        interval=data_cfg.interval,
    )
    print(f"  총 {len(df)}개 봉 로드됨")

    # ── 2. DataLoader 생성 ──
    print("\n[Step 2] DataLoader 생성")
    train_loader, val_loader, test_loader = create_dataloaders(
        df=df,
        seq_len=data_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        label_threshold=train_cfg.label_threshold,
    )

    # ── 3. 모델 생성 ──
    print("\n[Step 3] 모델 생성")
    model = DualPathCausalTransformer(
        input_dim=model_cfg.total_input_dim,
        raw_dim=model_cfg.raw_dim,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        head_hidden=model_cfg.head_hidden,
        num_classes=model_cfg.num_classes,
        seq_len=data_cfg.seq_len,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: DualPathCausalTransformer")
    print(f"  Total Parameters: {total_params:,}")

    # ── 4. 학습 ──
    print("\n[Step 4] 학습 시작")
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        patience=train_cfg.patience,
        device=device,
        save_dir=train_cfg.save_dir,
    )

    # ── 5. 테스트 ──
    print("\n[Step 5] 테스트 평가")
    test_report(model, test_loader, device)

    print("\nDone!")


if __name__ == "__main__":
    main()
