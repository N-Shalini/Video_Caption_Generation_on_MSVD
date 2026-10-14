import numpy as np
import pytest
import torch

from model import VideoCaptionModel
from generate_caption import greedy_decode, beam_search_decode


def test_beam_width_1_equals_greedy_and_properties():
    torch.manual_seed(12345)
    np.random.seed(12345)

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
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        encoder_type="temporal",
    ).to(device)
    model.eval()

    num_videos = 20
    max_length = 15

    for i in range(num_videos):
        features = np.random.randn(8, 512, 7, 7).astype(np.float32)

        # 1. Greedy decode
        greedy_caption, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            repetition_penalty=1.0,
            suppress_unk=True,
        )

        # 2. Beam search with beam_width=1
        beam_caption, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            beam_width=1,
            length_penalty=0.0,
            repetition_penalty=1.0,
            ngram_block_size=0,
            suppress_unk=True,
        )

        # Check beam_width=1 equals greedy
        assert beam_caption == greedy_caption, (
            f"Mismatch on video {i}: greedy='{greedy_caption}' vs beam='{beam_caption}'"
        )

        # Check no <pad> or <bos> in output string
        assert "<pad>" not in greedy_caption
        assert "<bos>" not in greedy_caption
        assert "<pad>" not in beam_caption
        assert "<bos>" not in beam_caption

        # Check max length respected
        greedy_tokens = greedy_caption.split()
        beam_tokens = beam_caption.split()
        assert len(greedy_tokens) <= max_length
        assert len(beam_tokens) <= max_length

        # 3. Determinism check: running again produces exact same caption
        greedy_again, _ = greedy_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            repetition_penalty=1.0,
            suppress_unk=True,
        )
        beam_again, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            beam_width=3,
            length_penalty=0.7,
            repetition_penalty=1.0,
            ngram_block_size=0,
            suppress_unk=True,
        )
        beam_again_2, _ = beam_search_decode(
            model=model,
            features=features,
            id_to_word=id_to_word,
            max_length=max_length,
            device=device,
            beam_width=3,
            length_penalty=0.7,
            repetition_penalty=1.0,
            ngram_block_size=0,
            suppress_unk=True,
        )
        assert greedy_caption == greedy_again
        assert beam_again == beam_again_2
