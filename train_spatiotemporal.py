import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MSVDDataset, collate_video_batch
from model import VideoCaptionModel


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config():
    with (PROJECT_ROOT / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    pad_idx=0,
    gradient_clip_norm=1.0,
    scheduled_sampling_prob=0.0,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        captions = batch["caption_ids"].to(device)

        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _ = model(
                features,
                decoder_input,
                scheduled_sampling_prob=scheduled_sampling_prob if training else 0.0,
            )

            valid_mask = targets.ne(pad_idx)
            token_count = int(valid_mask.sum().item())

            if token_count == 0:
                continue

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = loss_sum / token_count

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected.")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

        total_loss += loss.item() * token_count
        total_tokens += token_count
        total_batches += 1

    if total_tokens == 0:
        raise RuntimeError("No valid caption tokens were processed.")

    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "batches": total_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train enhanced Spatio-Temporal Video Captioning Model on MSVD."
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs (default: 15)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Training batch size (default: 8)."
    )
    parser.add_argument(
        "--captions-per-video",
        type=int,
        default=3,
        help="Number of diverse captions per video per epoch (default: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience (default: 4).",
    )
    args = parser.parse_args()

    config = load_config()
    set_seed(config.get("seed", 42))

    device = torch.device("cpu")
    print("=" * 70)
    print("TRAINING ENHANCED SPATIO-TEMPORAL VIDEO CAPTION MODEL (STAGE 2)")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Captions per video: {args.captions_per_video}")
    print(f"Learning rate: {args.learning_rate} | Patience: {args.patience}")
    print("=" * 70)

    train_dataset = MSVDDataset(
        "train",
        captions_per_video=args.captions_per_video,
    )
    val_dataset = MSVDDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_video_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_video_batch,
    )

    model = VideoCaptionModel(
        vocab_size=len(train_dataset.vocab),
        feature_dim=config["feature_channels"],
        embed_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        encoder_type="spatiotemporal",
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=train_dataset.pad_idx,
        reduction="sum",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Saved to isolated checkpoint: protects all baseline checkpoints
    best_checkpoint = checkpoint_dir / "spatiotemporal_vgg_best.pt"
    history_path = results_dir / "spatiotemporal_vgg_history.json"

    best_val_loss = float("inf")
    history = []
    epochs_without_improvement = 0

    print(f"Training examples per epoch: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Target checkpoint: {best_checkpoint}")
    print(f"Target history: {history_path}\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Scheduled sampling schedule: 0.0 -> max 0.15 gradually
        ss_prob = min(0.15, (epoch - 1) * 0.015)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            pad_idx=train_dataset.pad_idx,
            gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
            scheduled_sampling_prob=ss_prob,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            pad_idx=val_dataset.pad_idx,
            gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
            scheduled_sampling_prob=0.0,
        )

        scheduler.step(val_metrics["loss"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_tokens": train_metrics["tokens"],
            "val_tokens": val_metrics["tokens"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "scheduled_sampling_prob": ss_prob,
            "seconds": round(time.time() - start_time, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"lr={epoch_record['learning_rate']:.6g} | "
            f"ss_prob={ss_prob:.3f} | "
            f"time={epoch_record['seconds']:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        **config,
                        "encoder_type": "spatiotemporal",
                        "captions_per_video": args.captions_per_video,
                        "batch_size": args.batch_size,
                    },
                    "vocab": train_dataset.vocab,
                    "architecture": "spatiotemporal_vgg",
                },
                best_checkpoint,
            )
            print("  [*] Saved new best Spatio-Temporal checkpoint!")

        else:
            epochs_without_improvement += 1
            print(f"  No improvement: {epochs_without_improvement}/{args.patience}")

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 70)
    print("STAGE 2 TRAINING FINISHED")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best Checkpoint: {best_checkpoint}")
    print(f"Training History: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
