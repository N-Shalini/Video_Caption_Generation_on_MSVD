import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import VideoCaptionModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "temporal_attention_continued_best.pt"
ANNOTATIONS_PATH = PROJECT_DIR / "data" / "annotations" / "msvd_test.json"
FEATURE_DIR = PROJECT_DIR / "data" / "features" / "vgg16_spatial"


def load_model(device=None, checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    if device is None:
        dev_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            dev_str = "cpu"
        device = torch.device(dev_str)

    encoder_type = config.get("encoder_type", "temporal")

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=config["feature_channels"],
        embed_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        encoder_type=encoder_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_word = {idx: word for word, idx in vocab.items()}

    return model, config, vocab, id_to_word, checkpoint


@torch.inference_mode()
def greedy_decode(
    model,
    features=None,
    id_to_word=None,
    max_length=30,
    device=None,
    repetition_penalty=1.0,
    suppress_unk=True,
    encoder_out=None,
):
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")

    if device is None:
        device = next(model.parameters()).device

    if encoder_out is None:
        features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        encoder_out = model.encoder(features_t)

    hidden = encoder_out.mean(dim=1)
    cell = torch.zeros_like(hidden)

    # Precompute key projection W1(encoder_out) once per sequence
    precomputed_W1 = model.decoder.attention.W1(encoder_out)

    token_id = 1  # <bos>
    generated_words = []
    generated_token_ids = []
    attention_history = []

    for _ in range(max_length - 1):
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        embedding = model.decoder.embedding(token)

        context, attention_weights = model.decoder.attention(
            encoder_out, hidden, precomputed_W1=precomputed_W1
        )

        lstm_input = torch.cat([embedding, context], dim=1)
        hidden, cell = model.decoder.lstm_cell(
            lstm_input, (hidden, cell)
        )

        logits = model.decoder.output_projection(hidden).squeeze(0)

        if repetition_penalty > 1.0:
            for previous_id in set(generated_token_ids):
                if previous_id not in (0, 1, 2):
                    if logits[previous_id] > 0:
                        logits[previous_id] /= repetition_penalty
                    else:
                        logits[previous_id] *= repetition_penalty

        # Suppress <pad> and <bos>
        logits[0] = -float("inf")
        logits[1] = -float("inf")
        if suppress_unk:
            logits[3] = -float("inf")

        token_id = int(torch.argmax(logits).item())
        attention_history.append(attention_weights.squeeze(0).cpu())

        if token_id == 2:  # <eos>
            break

        if token_id in (0, 1):
            continue

        generated_words.append(id_to_word.get(token_id, "<unk>"))
        generated_token_ids.append(token_id)

    return " ".join(generated_words), attention_history


class BeamHypothesis:
    def __init__(self, tokens, cumulative_log_prob, hidden, cell, last_token_id):
        self.tokens = tokens
        self.cumulative_log_prob = cumulative_log_prob
        self.hidden = hidden
        self.cell = cell
        self.last_token_id = last_token_id


@torch.inference_mode()
def beam_search_decode(
    model,
    features=None,
    id_to_word=None,
    max_length=30,
    device=None,
    beam_width=5,
    length_penalty=0.7,
    repetition_penalty=1.0,
    ngram_block_size=0,
    suppress_unk=True,
    encoder_out=None,
):
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")

    if device is None:
        device = next(model.parameters()).device

    if encoder_out is None:
        features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        encoder_out = model.encoder(features_t)

    init_hidden = encoder_out.mean(dim=1)
    init_cell = torch.zeros_like(init_hidden)

    # Precompute key projection W1(encoder_out) once per sequence
    precomputed_W1 = model.decoder.attention.W1(encoder_out)

    live = [
        BeamHypothesis(
            tokens=[],
            cumulative_log_prob=0.0,
            hidden=init_hidden,
            cell=init_cell,
            last_token_id=1,  # <bos>
        )
    ]
    finished = []

    for _ in range(max_length - 1):
        candidates = []

        for hyp in live:
            token = torch.tensor([hyp.last_token_id], dtype=torch.long, device=device)
            embed = model.decoder.embedding(token)
            context, _ = model.decoder.attention(
                encoder_out, hyp.hidden, precomputed_W1=precomputed_W1
            )
            lstm_input = torch.cat([embed, context], dim=1)
            next_hidden, next_cell = model.decoder.lstm_cell(
                lstm_input, (hyp.hidden, hyp.cell)
            )
            logits = model.decoder.output_projection(next_hidden).squeeze(0)

            # Repetition penalty
            if repetition_penalty > 1.0:
                for prev_id in set(hyp.tokens):
                    if prev_id not in (0, 1, 2):
                        if logits[prev_id] > 0:
                            logits[prev_id] /= repetition_penalty
                        else:
                            logits[prev_id] *= repetition_penalty

            # N-gram blocking
            if ngram_block_size > 0 and len(hyp.tokens) >= ngram_block_size - 1:
                ngram_prefix = tuple(hyp.tokens[-(ngram_block_size - 1):])
                for i in range(len(hyp.tokens) - ngram_block_size + 1):
                    if tuple(hyp.tokens[i : i + ngram_block_size - 1]) == ngram_prefix:
                        blocked_token = hyp.tokens[i + ngram_block_size - 1]
                        logits[blocked_token] = -float("inf")

            # Suppress <pad> and <bos>
            logits[0] = -float("inf")
            logits[1] = -float("inf")
            if suppress_unk:
                logits[3] = -float("inf")

            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width)

            for k in range(beam_width):
                token_id = int(topk_indices[k].item())
                cand_log_prob = hyp.cumulative_log_prob + float(topk_log_probs[k].item())

                if token_id == 2:  # <eos>
                    completed_tokens = hyp.tokens + [token_id]
                    length = len(completed_tokens)
                    norm_score = cand_log_prob / (length ** length_penalty)
                    finished.append((norm_score, completed_tokens))
                else:
                    new_hyp = BeamHypothesis(
                        tokens=hyp.tokens + [token_id],
                        cumulative_log_prob=cand_log_prob,
                        hidden=next_hidden,
                        cell=next_cell,
                        last_token_id=token_id,
                    )
                    candidates.append(new_hyp)

        # Prune live hypotheses by raw cumulative log-prob
        candidates.sort(key=lambda h: h.cumulative_log_prob, reverse=True)
        live = candidates[:beam_width]

        if not live:
            break

        # Early stopping bound: valid only when length_penalty == 0.0
        if length_penalty == 0.0 and finished:
            best_finished_score = max(score for score, _ in finished)
            if live[0].cumulative_log_prob <= best_finished_score:
                break

    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        best_tokens = finished[0][1]
    elif live:
        # Fall back to best live hypothesis with same normalization
        scored_live = []
        for h in live:
            length = max(len(h.tokens), 1)
            score = h.cumulative_log_prob / (length ** length_penalty)
            scored_live.append((score, h.tokens))
        scored_live.sort(key=lambda x: x[0], reverse=True)
        best_tokens = scored_live[0][1]
    else:
        best_tokens = []

    words = [id_to_word.get(t, "<unk>") for t in best_tokens if t not in (0, 1, 2)]
    return " ".join(words), []


# For backwards-compatibility with existing scripts:
generate_caption = greedy_decode


def main():
    parser = argparse.ArgumentParser(
        description="Generate a caption using the best MSVD temporal-attention model."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index in msvd_test.json (default: 0).",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional specific video_id to evaluate (overrides --index).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT.name}).",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        choices=["greedy", "beam"],
        default="greedy",
        help="Decoding method: 'greedy' or 'beam' (default: greedy).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for beam search (default: 5).",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.7,
        help="Length penalty exponent for beam search (default: 0.7).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0; disabled).",
    )
    parser.add_argument(
        "--ngram-block-size",
        type=int,
        default=0,
        help="N-gram repetition blocking size for beam search (default: 0; disabled).",
    )
    parser.add_argument(
        "--suppress-unk",
        action="store_true",
        default=True,
        help="Suppress <unk> during generation (default: True).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', etc.; overrides checkpoint/config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device_arg = None
    if args.device is not None:
        dev_str = args.device
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: {dev_str} requested but CUDA is not available. Falling back to cpu.")
            dev_str = "cpu"
        device_arg = torch.device(dev_str)

    model, config, vocab, id_to_word, checkpoint = load_model(
        device=device_arg, checkpoint_path=args.checkpoint
    )
    device = next(model.parameters()).device

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if args.video_id:
        record = next((r for r in records if r["video_id"] == args.video_id), None)
        if record is None:
            raise ValueError(f"Video ID '{args.video_id}' not found in {ANNOTATIONS_PATH.name}.")
        args.index = records.index(record)
    else:
        if not 0 <= args.index < len(records):
            raise IndexError(f"Index must be between 0 and {len(records) - 1}.")
        record = records[args.index]

    video_id = record["video_id"]
    feature_path = FEATURE_DIR / f"{video_id}.npy"

    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    features = np.load(feature_path, allow_pickle=False)

    expected_shape = (8, 512, 7, 7)
    if features.shape != expected_shape:
        raise ValueError(f"Expected feature shape {expected_shape}, got {features.shape}")

    if features.dtype != np.float32:
        raise TypeError(f"Expected float32 features, got {features.dtype}")

    if not np.isfinite(features).all():
        raise ValueError("Feature array contains non-finite values.")

    model, config, vocab, id_to_word, checkpoint = load_model(
        device, checkpoint_path=args.checkpoint
    )

    if args.decoding == "beam":
        caption, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=config["max_caption_length"],
            device=device,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            ngram_block_size=args.ngram_block_size,
        )
    else:
        caption, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=config["max_caption_length"],
            device=device,
            repetition_penalty=args.repetition_penalty,
        )

    print(f"Checkpoint: {Path(args.checkpoint).name} (Epoch {checkpoint['epoch']}, Best Val Loss {checkpoint['best_val_loss']:.4f})")
    print(f"Test record index: {args.index}")
    print(f"Video ID: {video_id}")
    print(f"Decoding: {args.decoding.upper()}" + (f" (k={args.beam_width})" if args.decoding == 'beam' else ''))
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"Reference captions count: {len(record['caption'])}")
    print(f"\nGenerated caption: {caption}")

    print("\nReference captions (first 5):")
    for reference in record["caption"][:5]:
        print(f"- {reference}")


if __name__ == "__main__":
    main()
