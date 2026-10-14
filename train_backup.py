
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
    path = PROJECT_ROOT / "config.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic operations where supported.
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    pad_idx=0,
    gradient_clip_norm=1.0,
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
            logits, _ = model(features, decoder_input)

            # Token-weighted loss; ignore padding tokens.
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
    config = load_config()
    set_seed(config["seed"])

    device = torch.device("cpu")
    print("Device:", device)
    print("Training configuration:", json.dumps(config, indent=2))

    train_dataset = MSVDDataset("train")
    val_dataset = MSVDDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_video_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_video_batch,
    )

    model = VideoCaptionModel(
        vocab_size=len(train_dataset.vocab),
        feature_dim=config["feature_channels"],
        embed_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=train_dataset.pad_idx,
        reduction="sum",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
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

    best_checkpoint = checkpoint_dir / "best_model.pt"
    history_path = results_dir / "training_history.json"

    # ---------------------------------------------------------
    # Resume training from the existing best checkpoint.
    # ---------------------------------------------------------
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Cannot resume: checkpoint not found at {best_checkpoint}"
        )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: training history not found at {history_path}"
        )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Verify that the saved vocabulary and model configuration
    # match the current training setup.
    if checkpoint["vocab"] != train_dataset.vocab:
        raise RuntimeError(
            "Checkpoint vocabulary does not match the training dataset vocabulary."
        )

    for key in (
        "feature_channels",
        "embedding_dim",
        "hidden_dim",
        "dropout",
    ):
        if checkpoint["config"].get(key) != config.get(key):
            raise RuntimeError(
                f"Checkpoint configuration mismatch for {key}: "
                f"{checkpoint['config'].get(key)} != {config.get(key)}"
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = float(checkpoint["best_val_loss"])

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not history:
        raise RuntimeError("Training history is empty.")

    if history[-1]["epoch"] != checkpoint["epoch"]:
        raise RuntimeError(
            "The history's last epoch does not match the checkpoint epoch. "
            "Training was not started to avoid corrupting the results."
        )

    if start_epoch > config["epochs"]:
        print(
            f"Checkpoint is already at epoch {checkpoint['epoch']}, "
            f"and the configured maximum is {config['epochs']}."
        )
        print("Increase config.json's epochs value to continue training.")
        return

    # The checkpoint does not save the early-stopping counter.
    # Reset it because the saved checkpoint is the best model.
    epochs_without_improvement = 0

    print("\nRESUMING TRAINING")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(f"Best validation loss so far: {best_val_loss:.6f}")
    print("Starting epoch:", start_epoch)
    print("Maximum epoch:", config["epochs"])
    print(
        "Early-stopping patience:",
        config["early_stopping_patience"],
    )

    # ---------------------------------------------------------
    # Continue from the next epoch instead of epoch 1.
    # ---------------------------------------------------------
    for epoch in range(start_epoch, config["epochs"] + 1):
        start_time = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            pad_idx=train_dataset.pad_idx,
            gradient_clip_norm=config["gradient_clip_norm"],
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            pad_idx=val_dataset.pad_idx,
            gradient_clip_norm=config["gradient_clip_norm"],
        )

        scheduler.step(val_metrics["loss"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_tokens": train_metrics["tokens"],
            "val_tokens": val_metrics["tokens"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start_time, 2),
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"lr={epoch_record['learning_rate']:.6g} | "
            f"time={epoch_record['seconds']:.2f}s"
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
                    "config": config,
                    "vocab": train_dataset.vocab,
                },
                best_checkpoint,
            )

            print("  Saved best checkpoint:", best_checkpoint)

        else:
            epochs_without_improvement += 1

            print(
                "  No validation improvement:",
                epochs_without_improvement,
                "/",
                config["early_stopping_patience"],
            )

        # Save history after checkpoint/early-stopping bookkeeping.
        # Existing epochs are retained.
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if (
            epochs_without_improvement
            >= config["early_stopping_patience"]
        ):
            print("Early stopping triggered.")
            break

    print("\nTRAINING FINISHED")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", best_checkpoint)
    print("Training history:", history_path)
    print("Total recorded epochs:", len(history))


if __name__ == "__main__":
    main()
