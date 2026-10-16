import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import load_vocab, tokenize
from generate_caption import compute_eos_blocklist, greedy_decode, beam_search_decode
from model import VideoCaptionModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "msvd_train.json"


def test_eos_blocklist_includes_a_the_and_excludes_common_sentence_ends():
    vocab = load_vocab(PROJECT_ROOT / "data" / "annotations" / "vocab.json")

    # Compute or load blocklist
    blocklist = compute_eos_blocklist(
        train_annotations_path=TRAIN_ANNOTATIONS,
        vocab=vocab,
        min_count=50,
        max_eos_rate=0.002,
    )

    assert isinstance(blocklist, set)
    assert len(blocklist) > 0

    # Requirement 6: the blocklist includes 'a' and 'the'
    assert "a" in vocab
    assert "the" in vocab
    assert vocab["a"] in blocklist, "'a' must be included in EOS blocklist"
    assert vocab["the"] in blocklist, "'the' must be included in EOS blocklist"

    # Requirement 6: and excludes tokens that often end captions
    # Nouns and verbs that naturally conclude descriptions in MSVD
    frequent_ending_words = [
        "bowl",
        "egg",
        "water",
        "food",
        "cat",
        "dog",
        "car",
        "guitar",
        "meat",
        "ball",
    ]
    for word in frequent_ending_words:
        if word in vocab:
            assert vocab[word] not in blocklist, (
                f"Frequent sentence-ending word '{word}' should be excluded from EOS blocklist"
            )

    # Verify cache file was written to data/annotations/eos_blocklist.json
    cache_path = PROJECT_ROOT / "data" / "annotations" / "eos_blocklist.json"
    assert cache_path.is_file(), "Cache file must exist at data/annotations/eos_blocklist.json"

    with cache_path.open("r", encoding="utf-8") as f:
        cache_data = json.load(f)

    assert "TRAIN" in cache_data.get("note", "").upper()
    assert cache_data.get("parameters", {}).get("min_count") == 50
    assert cache_data.get("parameters", {}).get("max_eos_rate") == 0.002
    assert "blocked_words" in cache_data
    assert "a" in cache_data["blocked_words"]
    assert "the" in cache_data["blocked_words"]

    # Verify cache reuse
    reused = compute_eos_blocklist(
        train_annotations_path=TRAIN_ANNOTATIONS,
        vocab=vocab,
        min_count=50,
        max_eos_rate=0.002,
    )
    assert reused == blocklist


def test_tiny_seeded_model_never_emits_eos_after_blocked_token():
    """Verify that whenever generation terminates via <eos>, the token directly preceding <eos>
    is NEVER in the blocklist for both greedy and beam search."""
    torch.manual_seed(999)
    np.random.seed(999)

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "a": 4,
        "person": 5,
        "is": 6,
        "walking": 7,
        "running": 8,
        "cat": 9,
        "the": 10,
        "dog": 11,
        "plays": 12,
        "in": 13,
        "water": 14,
    }
    id_to_word = {i: w for w, i in vocab.items()}

    device = torch.device("cpu")
    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=512,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
        encoder_type="temporal",
    ).to(device)
    model.eval()

    # Block 'a' (4), 'the' (10), 'in' (13)
    blocked_ids = {vocab["a"], vocab["the"], vocab["in"]}
    blocked_words = {"a", "the", "in"}

    num_trials = 25
    max_length = 20

    for i in range(num_trials):
        features = np.random.randn(8, 512, 7, 7).astype(np.float32)

        # 1. Greedy with blocklist
        greedy_caption, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            repetition_penalty=1.0,
            suppress_unk=True,
            block_eos_after=blocked_ids,
        )

        tokens = greedy_caption.split()
        # If caption terminated early (< max_length - 1), it terminated because <eos> was emitted
        if tokens and len(tokens) < max_length - 1:
            last_word = tokens[-1]
            assert last_word not in blocked_words, (
                f"<eos> was emitted right after blocked word '{last_word}' in greedy: '{greedy_caption}'"
            )

        # 2. Beam search with blocklist (k=3 and k=5)
        for bw in [3, 5]:
            beam_caption, _ = beam_search_decode(
                model=model,
                features=features,
                id_to_word=id_to_word,
                max_length=max_length,
                device=device,
                beam_width=bw,
                length_penalty=0.7,
                repetition_penalty=1.0,
                ngram_block_size=0,
                suppress_unk=True,
                block_eos_after=blocked_ids,
            )

            b_tokens = beam_caption.split()
            if b_tokens and len(b_tokens) < max_length - 1:
                b_last_word = b_tokens[-1]
                assert b_last_word not in blocked_words, (
                    f"<eos> was emitted right after blocked word '{b_last_word}' in beam (bw={bw}): '{beam_caption}'"
                )


def test_eos_blocking_enforcement_prevents_premature_eos():
    """Verify that when a model would otherwise emit <eos> right after a blocked token,
    the rule sets logits[2] to -inf and forces the model to emit an alternative token."""
    device = torch.device("cpu")
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "the": 4,
        "cat": 5,
    }
    id_to_word = {i: w for w, i in vocab.items()}

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=512,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
        encoder_type="temporal",
    ).to(device)
    model.eval()

    dummy_encoder_out = torch.zeros(1, 8, 32, device=device)

    # Use forward hook to set logits:
    # <eos> (2) is given highest logit (10.0), and 'cat' (5) is second highest (5.0)
    def hook_fn(module, input, output):
        modified = output.clone()
        modified[:, 2] = 10.0
        modified[:, 5] = 5.0
        return modified

    handle = model.decoder.output_projection.register_forward_hook(hook_fn)

    try:
        # Without blocklist: step 0 from <bos> produces <eos> (2) immediately -> empty caption
        greedy_no_block, _ = greedy_decode(
            model=model,
            encoder_out=dummy_encoder_out,
            id_to_word=id_to_word,
            max_length=5,
            device=device,
            block_eos_after=None,
        )
        assert greedy_no_block == ""

        # With blocklist blocking <eos> after <bos> (1):
        # step 0: <eos> is blocked -> picks 'cat' (5)
        # step 1: from 'cat' (5), not in blocklist -> picks <eos> (2) -> stops
        greedy_block, _ = greedy_decode(
            model=model,
            encoder_out=dummy_encoder_out,
            id_to_word=id_to_word,
            max_length=5,
            device=device,
            block_eos_after={1},
        )
        assert greedy_block == "cat"

        # Beam search with and without blocklist
        beam_no_block, _ = beam_search_decode(
            model=model,
            encoder_out=dummy_encoder_out,
            id_to_word=id_to_word,
            max_length=5,
            device=device,
            beam_width=2,
            length_penalty=0.0,
            block_eos_after=None,
        )
        assert beam_no_block == ""

        beam_block, _ = beam_search_decode(
            model=model,
            encoder_out=dummy_encoder_out,
            id_to_word=id_to_word,
            max_length=5,
            device=device,
            beam_width=2,
            length_penalty=0.0,
            block_eos_after={1},
        )
        assert beam_block == "cat"

    finally:
        handle.remove()


def test_decoding_output_unchanged_when_rule_disabled():
    torch.manual_seed(42)
    np.random.seed(42)

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "a": 4,
        "person": 5,
        "is": 6,
        "walking": 7,
        "running": 8,
        "cat": 9,
        "dog": 10,
        "plays": 11,
        "ball": 12,
        "in": 13,
        "the": 14,
        "water": 15,
    }
    id_to_word = {i: w for w, i in vocab.items()}

    device = torch.device("cpu")
    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=512,
        embed_dim=16,
        hidden_dim=32,
        dropout=0.0,
        encoder_type="temporal",
    ).to(device)
    model.eval()

    for i in range(5):
        features = np.random.randn(8, 512, 7, 7).astype(np.float32)

        # 1. Greedy: None vs default arg
        cap_none, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=15,
            device=device,
            block_eos_after=None,
        )
        cap_default, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=15,
            device=device,
        )
        assert cap_none == cap_default

        # 2. Beam search: None vs default arg
        beam_none, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=15,
            device=device,
            beam_width=3,
            length_penalty=0.7,
            block_eos_after=None,
        )
        beam_default, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=15,
            device=device,
            beam_width=3,
            length_penalty=0.7,
        )
        assert beam_none == beam_default
