import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MSVDDataset, collate_video_batch
from generate_caption import greedy_decode
from metrics import evaluate_predictions
from model import VideoCaptionModel

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"


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


def evaluate_val_greedy(model, val_records, id_to_word, max_length, device, suppress_unk=True):
    """Run greedy decoding on all validation videos and score with standard COCO metrics."""
    model.eval()
    hyps = {}
    refs = {}

    for record in val_records:
        video_id = record["video_id"]
        feature_path = FEATURE_DIR / f"{video_id}.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing features for {video_id}: {feature_path}")

        features = np.load(feature_path, allow_pickle=False)
        if features.shape != (8, 512, 7, 7):
            raise ValueError(f"Unexpected feature shape for {video_id}: {features.shape}")
        if features.dtype != np.float32:
            raise TypeError(f"Expected float32 features for {video_id}, got {features.dtype}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite feature values for {video_id}")

        caption, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            repetition_penalty=1.0,
            suppress_unk=suppress_unk,
        )
        hyps[video_id] = caption
        refs[video_id] = record["caption"]

    scores = evaluate_predictions(refs, hyps)
    # Restore model.train() after validation decoding
    model.train()
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Train Video Captioning Model on MSVD."
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default=None,
        choices=["temporal", "temporal_bigru", "spatiotemporal", "spatiotemporal_attention"],
        help="Encoder architecture type (overrides config).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run identifier for checkpoints and logs (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config).",
    )
    parser.add_argument(
        "--captions-per-video",
        type=int,
        default=None,
        help="Number of captions per video per epoch (0 = all, overrides config).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (overrides config).",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        choices=["cider", "loss"],
        help="Metric used for model checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'cpu', etc., overrides config).",
    )
    parser.add_argument(
        "--max-train-videos",
        type=int,
        default=None,
        help="Dev flag: limit number of train videos (smoke tests only).",
    )
    parser.add_argument(
        "--max-val-videos",
        type=int,
        default=None,
        help="Dev flag: limit number of val videos (smoke tests only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from an existing last checkpoint (checkpoints/{run_name}_last.pt).",
    )

    args = parser.parse_args()

    # Load authoritative config and resolve overrides
    raw_config = load_config()
    resolved_config = copy.deepcopy(raw_config)

    if args.encoder_type is not None:
        resolved_config["encoder_type"] = args.encoder_type

    if args.run_name is not None:
        resolved_config["run_name"] = args.run_name
    elif args.encoder_type is not None:
        resolved_config["run_name"] = f"{args.encoder_type}_vgg"
    elif "run_name" not in resolved_config:
        resolved_config["run_name"] = f"{resolved_config.get('encoder_type', 'model')}_vgg"
    if args.epochs is not None:
        resolved_config["epochs"] = args.epochs
    if args.batch_size is not None:
        resolved_config["batch_size"] = args.batch_size
    if args.captions_per_video is not None:
        resolved_config["captions_per_video"] = args.captions_per_video
    if args.learning_rate is not None:
        resolved_config["learning_rate"] = args.learning_rate
    if args.patience is not None:
        resolved_config["early_stopping_patience"] = args.patience
    if args.selection_metric is not None:
        resolved_config["selection_metric"] = args.selection_metric
    if args.seed is not None:
        resolved_config["seed"] = args.seed
    if args.device is not None:
        resolved_config["device"] = args.device

    resolved_config["max_train_videos"] = args.max_train_videos
    resolved_config["max_val_videos"] = args.max_val_videos

    run_name = resolved_config["run_name"]
    encoder_type = resolved_config["encoder_type"]
    selection_metric = resolved_config.get("selection_metric", "cider").lower()
    seed = resolved_config.get("seed", 42)

    device_str = resolved_config.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {device_str} requested but CUDA is not available. Falling back to cpu.")
        device_str = "cpu"
    device = torch.device(device_str)

    set_seed(seed)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = checkpoint_dir / f"{run_name}_best.pt"
    last_checkpoint = checkpoint_dir / f"{run_name}_last.pt"
    history_path = results_dir / f"{run_name}_history.json"
    resolved_config_path = results_dir / f"{run_name}_config_resolved.json"

    # Overwrite protection: without --resume, refuse to overwrite existing checkpoints
    if not args.resume:
        existing = []
        if best_checkpoint.is_file():
            existing.append(str(best_checkpoint))
        if last_checkpoint.is_file():
            existing.append(str(last_checkpoint))
        if existing:
            sys.stderr.write(
                f"ERROR: Checkpoint(s) already exist for run '{run_name}':\n"
                + "\n".join(f"  - {p}" for p in existing)
                + f"\nRefusing to overwrite existing checkpoint(s) without --resume.\n"
                + f"To resume this run, pass --resume.\n"
                + f"To start a new run, specify a different --run-name.\n"
            )
            sys.exit(1)

    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    print("=" * 70)
    print(f"TRAINING VIDEO CAPTION MODEL: {run_name}")
    print(f"Encoder: {encoder_type} | Device: {device} | Selection Metric: {selection_metric}")
    print(f"Epochs: {resolved_config['epochs']} | Batch size: {resolved_config['batch_size']}")
    print(f"Captions per video: {resolved_config['captions_per_video']} | LR: {resolved_config['learning_rate']}")
    print(f"Patience: {resolved_config['early_stopping_patience']}")
    print("=" * 70)

    # Initialize datasets
    train_dataset = MSVDDataset(
        "train",
        captions_per_video=resolved_config["captions_per_video"],
        seed=seed,
    )
    if args.max_train_videos is not None:
        train_dataset.records = train_dataset.records[: args.max_train_videos]
        train_dataset.video_ids = [r["video_id"] for r in train_dataset.records]
        train_dataset.resample(epoch=0)

    # Validation loss evaluates ALL reference captions
    val_loss_dataset = MSVDDataset(
        "val",
        captions_per_video=0,
        seed=seed,
    )
    if args.max_val_videos is not None:
        val_loss_dataset.records = val_loss_dataset.records[: args.max_val_videos]
        val_loss_dataset.video_ids = [r["video_id"] for r in val_loss_dataset.records]
        val_loss_dataset.samples = [
            (record, cap)
            for record in val_loss_dataset.records
            for cap in record["caption"]
        ]

    val_records = val_loss_dataset.records
    id_to_word = {idx: word for word, idx in train_dataset.vocab.items()}

    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved_config["batch_size"],
        shuffle=True,
        num_workers=resolved_config.get("num_workers", 0),
        collate_fn=collate_video_batch,
    )

    val_loss_loader = DataLoader(
        val_loss_dataset,
        batch_size=resolved_config["batch_size"],
        shuffle=False,
        num_workers=resolved_config.get("num_workers", 0),
        collate_fn=collate_video_batch,
    )

    model = VideoCaptionModel(
        vocab_size=len(train_dataset.vocab),
        feature_dim=resolved_config["feature_channels"],
        embed_dim=resolved_config["embedding_dim"],
        hidden_dim=resolved_config["hidden_dim"],
        dropout=resolved_config["dropout"],
        encoder_type=encoder_type,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=train_dataset.pad_idx,
        reduction="sum",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config["learning_rate"],
        weight_decay=resolved_config.get("weight_decay", 1e-4),
    )

    sched_cfg = resolved_config.get("scheduler", {})
    scheduler_mode = "max" if selection_metric == "cider" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
    )

    start_epoch = 1
    best_metric_val = -float("inf") if selection_metric == "cider" else float("inf")
    history = []
    epochs_without_improvement = 0

    if args.resume:
        if not last_checkpoint.is_file():
            sys.stderr.write(
                f"ERROR: Cannot resume run '{run_name}': checkpoint not found at '{last_checkpoint}'.\n"
            )
            sys.exit(1)

        print(f"[*] Resuming run '{run_name}' from: {last_checkpoint}")
        ckpt = torch.load(last_checkpoint, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        if "best_metric_val" in ckpt:
            best_metric_val = ckpt["best_metric_val"]
        elif selection_metric == "cider" and "best_val_cider" in ckpt:
            best_metric_val = ckpt["best_val_cider"]
        elif selection_metric == "loss" and "best_val_loss" in ckpt:
            best_metric_val = ckpt["best_val_loss"]
        else:
            best_metric_val = -float("inf") if selection_metric == "cider" else float("inf")

        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        history = ckpt.get("history", [])

        if not history and history_path.is_file():
            try:
                with history_path.open("r", encoding="utf-8") as f:
                    file_history = json.load(f)
                if isinstance(file_history, list):
                    history = file_history
            except Exception:
                pass

        print(f"    Resuming at epoch {start_epoch} (completed epoch {ckpt['epoch']})")
        print(f"    Best {selection_metric}: {best_metric_val:.4f} | Early-stop count: {epochs_without_improvement}/{resolved_config['early_stopping_patience']}\n")

    if start_epoch > resolved_config["epochs"]:
        print(f"Run '{run_name}' already completed all {resolved_config['epochs']} epochs.")
        return

    print(f"Train samples per epoch: {len(train_dataset)}")
    print(f"Val loss references: {len(val_loss_dataset)} (across {len(val_records)} videos)")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Best checkpoint target: {best_checkpoint}")
    print(f"History target: {history_path}\n")

    ss_cfg = resolved_config.get("scheduled_sampling", {})
    max_ss_prob = ss_cfg.get("max_prob", 0.15)
    ss_inc = ss_cfg.get("increment_per_epoch", 0.015)

    for epoch in range(start_epoch, resolved_config["epochs"] + 1):
        start_time = time.time()

        # Resample training captions for current epoch
        train_dataset.resample(epoch)

        ss_prob = min(max_ss_prob, (epoch - 1) * ss_inc)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            pad_idx=train_dataset.pad_idx,
            gradient_clip_norm=resolved_config.get("gradient_clip_norm", 1.0),
            scheduled_sampling_prob=ss_prob,
        )

        # Val loss over ALL reference captions
        val_loss_metrics = run_epoch(
            model=model,
            loader=val_loss_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            pad_idx=val_loss_dataset.pad_idx,
            gradient_clip_norm=resolved_config.get("gradient_clip_norm", 1.0),
            scheduled_sampling_prob=0.0,
        )

        # Val greedy decoding scored with standard COCO metrics
        val_coco_scores = evaluate_val_greedy(
            model=model,
            val_records=val_records,
            id_to_word=id_to_word,
            max_length=resolved_config["max_caption_length"],
            device=device,
            suppress_unk=resolved_config.get("suppress_unk", True),
        )

        val_loss = val_loss_metrics["loss"]
        val_cider = val_coco_scores["CIDEr"]
        val_bleu4 = val_coco_scores["BLEU-4"]
        val_meteor = val_coco_scores["METEOR"]
        val_rouge_l = val_coco_scores["ROUGE-L"]

        current_metric = val_cider if selection_metric == "cider" else val_loss
        scheduler.step(current_metric)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_loss,
            "val_cider": val_cider,
            "val_bleu4": val_bleu4,
            "val_meteor": val_meteor,
            "val_rouge_l": val_rouge_l,
            "learning_rate": current_lr,
            "scheduled_sampling_prob": ss_prob,
            "seconds": round(time.time() - start_time, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{resolved_config['epochs']:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val CIDEr: {val_cider:.4f} | "
            f"BLEU-4: {val_bleu4:.4f} | "
            f"METEOR: {val_meteor:.4f} | "
            f"ROUGE-L: {val_rouge_l:.4f} | "
            f"learning rate: {current_lr:.6g} | "
            f"seconds: {epoch_record['seconds']:.2f}"
        )

        is_better = (
            val_cider > best_metric_val
            if selection_metric == "cider"
            else val_loss < best_metric_val
        )

        if is_better:
            best_metric_val = current_metric
            epochs_without_improvement = 0
            best_save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric_val": best_metric_val,
                "best_val_loss": val_loss,
                "best_val_cider": val_cider,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "val_scores": val_coco_scores,
                "config": resolved_config,
                "vocab": train_dataset.vocab,
                "architecture": encoder_type,
            }
            torch.save(best_save_dict, best_checkpoint)
            print(f"  [*] Saved new best checkpoint based on {selection_metric}: {best_metric_val:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement in {selection_metric}: {epochs_without_improvement}/{resolved_config['early_stopping_patience']}")

        # The _last.pt checkpoint must contain all of these every epoch:
        # model, optimizer, scheduler, epoch, best metric, early-stopping counter and history
        last_save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric_val": best_metric_val,
            "best_val_loss": val_loss,
            "best_val_cider": val_cider,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "val_scores": val_coco_scores,
            "config": resolved_config,
            "vocab": train_dataset.vocab,
            "architecture": encoder_type,
        }
        torch.save(last_save_dict, last_checkpoint)

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= resolved_config["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(f"Best {selection_metric}: {best_metric_val:.4f}")
    print(f"Best Checkpoint: {best_checkpoint}")
    print(f"Last Checkpoint: {last_checkpoint}")
    print(f"Resolved Config: {resolved_config_path}")
    print(f"Training History: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
