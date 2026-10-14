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


def load_model(device, checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

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
    features,
    id_to_word,
    max_length,
    device,
    repetition_penalty=1.0,
):
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")

    features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
    encoder_out = model.encoder(features_t)

    hidden = encoder_out.mean(dim=1)
    cell = torch.zeros_like(hidden)

    token_id = 1  # <bos>
    generated_words = []
    generated_token_ids = []
    attention_history = []

    for _ in range(max_length - 1):
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        embedding = model.decoder.embedding(token)

        context, attention_weights = model.decoder.attention(
            encoder_out, hidden
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
    def __init__(self, tokens, score, hidden, cell, last_token_id):
        self.tokens = tokens
        self.score = score
        self.hidden = hidden
        self.cell = cell
        self.last_token_id = last_token_id


@torch.inference_mode()
def beam_search_decode(
    model,
    features,
    id_to_word,
    max_length,
    device,
    beam_width=5,
    length_penalty=0.7,
    repetition_penalty=1.1,
    ngram_block_size=3,
):
    features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
    encoder_out = model.encoder(features_t)
    init_hidden = encoder_out.mean(dim=1)
    init_cell = torch.zeros_like(init_hidden)

    beams = [
        BeamHypothesis(
            tokens=[],
            score=0.0,
            hidden=init_hidden,
            cell=init_cell,
            last_token_id=1,
        )
    ]
    completed = []

    for _ in range(max_length - 1):
        candidates = []
        for hyp in beams:
            token = torch.tensor([hyp.last_token_id], dtype=torch.long, device=device)
            embed = model.decoder.embedding(token)
            context, _ = model.decoder.attention(encoder_out, hyp.hidden)
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
                    if tuple(hyp.tokens[i:i + ngram_block_size - 1]) == ngram_prefix:
                        blocked_token = hyp.tokens[i + ngram_block_size - 1]
                        logits[blocked_token] = -float("inf")

            # Suppress <pad> and <bos>
            logits[0] = -float("inf")
            logits[1] = -float("inf")

            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width)

            for k in range(beam_width):
                token_id = int(topk_indices[k].item())
                cand_score = hyp.score + float(topk_log_probs[k].item())

                if token_id == 2:  # <eos>
                    norm_score = cand_score / (max(len(hyp.tokens), 1) ** length_penalty)
                    completed.append((norm_score, hyp.tokens))
                else:
                    new_hyp = BeamHypothesis(
                        tokens=hyp.tokens + [token_id],
                        score=cand_score,
                        hidden=next_hidden,
                        cell=next_cell,
                        last_token_id=token_id,
                    )
                    norm_score = cand_score / (max(len(new_hyp.tokens), 1) ** length_penalty)
                    candidates.append((norm_score, new_hyp))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [c[1] for c in candidates[:beam_width]]

        if len(completed) >= beam_width:
            break

    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        best_tokens = completed[0][1]
    elif beams:
        beams.sort(key=lambda h: h.score / (max(len(h.tokens), 1) ** length_penalty), reverse=True)
        best_tokens = beams[0].tokens
    else:
        best_tokens = []

    words = [id_to_word.get(t, "<unk>") for t in best_tokens]
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
        default=1.2,
        help="Repetition penalty (default: 1.2; 1.0 disables).",
    )
    parser.add_argument(
        "--ngram-block-size",
        type=int,
        default=3,
        help="N-gram repetition blocking size for beam search (default: 3).",
    )

    args = parser.parse_args()
    device = torch.device("cpu")

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
