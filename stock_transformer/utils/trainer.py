"""
학습 & 평가 파이프라인
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


def get_device(preference: str = "auto") -> torch.device:
    """디바이스 자동 선택"""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        x_raw = batch["x_raw"].to(device)
        x_full = batch["x_full"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x_raw, x_full)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """검증/테스트 평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        x_raw = batch["x_raw"].to(device)
        x_full = batch["x_full"].to(device)
        labels = batch["label"].to(device)

        logits = model(x_raw, x_full)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy, all_preds, all_labels


def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 곡선
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 곡선
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Plot] 학습 곡선 저장: {save_path}")


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: torch.device = None,
    save_dir: str = "checkpoints",
):
    """
    전체 학습 루프 (Early Stopping 포함)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    print(f"[Trainer] Device: {device}")
    print(f"[Trainer] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    model_saved = False
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 학습 기록
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # NaN 감지 시 경고
        if train_loss != train_loss:  # NaN check
            print(f"[Epoch {epoch:3d}/{epochs}] ⚠ NaN detected! Skipping...")
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} (NaN)")
                break
            continue

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Logging
        print(
            f"[Epoch {epoch:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
            model_saved = True
            print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch}")
                break

    # 학습 곡선 시각화
    if history["train_loss"]:
        plot_training_curves(history, save_path=str(save_path / "training_curves.png"))

    # 최적 모델 로드
    if model_saved:
        model.load_state_dict(torch.load(save_path / "best_model.pt", weights_only=True))
        print(f"\n[Trainer] 학습 완료. Best Val Loss: {best_val_loss:.4f}")
    else:
        print("\n[Trainer] 학습 완료. (모델 저장 없음 — 마지막 상태 사용)")

    return model


def test_report(model, test_loader, device):
    """테스트 결과 리포트"""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    label_names = ["Down", "Flat", "Up"]

    print("\n" + "=" * 50)
    print(f"  TEST RESULTS")
    print(f"  Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=label_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    return test_loss, test_acc
