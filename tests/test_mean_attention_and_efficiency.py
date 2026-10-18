import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from model import VideoCaptionModel, CaptionDecoder, reshape_attention_weights
from generate_caption import greedy_decode, beam_search_decode, load_model
from efficiency import count_frozen_vgg16_parameters, get_cpu_model, ProcessMemorySampler
from visualize_attention import (
    plot_frame_attention_heatmap,
    plot_spatiotemporal_spatial_overlay,
    load_manifest,
)


def test_caption_decoder_mean_attention_no_parameters():
    vocab_size = 50
    embed_dim = 64
    hidden_dim = 128
    feature_dim = 128

    # Additive decoder has attention parameters (W1, W2, V)
    decoder_additive = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        attention_type="additive",
    )
    assert decoder_additive.attention is not None
    additive_params = sum(p.numel() for p in decoder_additive.parameters())

    # Mean decoder has NO attention parameters
    decoder_mean = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        attention_type="mean",
    )
    assert decoder_mean.attention is None
    mean_params = sum(p.numel() for p in decoder_mean.parameters())

    # Verify difference matches exactly the AdditiveAttention parameters
    expected_attn_params = sum(p.numel() for p in decoder_additive.attention.parameters())
    assert additive_params - mean_params == expected_attn_params
    assert expected_attn_params > 0


def test_caption_decoder_mean_attention_forward():
    batch_size = 2
    num_positions = 8
    seq_len = 5
    hidden_dim = 64
    vocab_size = 30

    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=hidden_dim,
        feature_dim=hidden_dim,
        dropout=0.0,
        attention_type="mean",
    )

    encoder_out = torch.randn(batch_size, num_positions, hidden_dim)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, attn_weights = decoder(encoder_out, captions)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert attn_weights.shape == (batch_size, seq_len, num_positions)

    # Weights across positions should be uniform (1 / num_positions)
    expected_uniform = torch.full_like(attn_weights, 1.0 / num_positions)
    assert torch.allclose(attn_weights, expected_uniform, atol=1e-6)

    # Gradients flow through mean context
    loss = logits.sum()
    loss.backward()
    assert decoder.lstm_cell.weight_ih.grad is not None


def test_video_caption_model_mean_attention():
    vocab_size = 40
    batch_size = 2
    features = torch.randn(batch_size, 8, 512, 7, 7)
    captions = torch.randint(0, vocab_size, (batch_size, 6))

    model = VideoCaptionModel(
        vocab_size=vocab_size,
        feature_dim=512,
        embed_dim=32,
        hidden_dim=64,
        encoder_type="temporal",
        attention_type="mean",
    )
    assert model.attention_type == "mean"
    assert model.decoder.attention is None

    logits, attn = model(features, captions)
    assert logits.shape == (batch_size, 6, vocab_size)
    assert attn.shape == (batch_size, 6, 8)


def test_greedy_and_beam_decode_mean_attention():
    torch.manual_seed(42)
    np.random.seed(42)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "cat": 4, "jumps": 5}
    id_to_word = {i: w for w, i in vocab.items()}

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=512,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        encoder_type="temporal",
        attention_type="mean",
    )
    model.eval()

    features = np.random.randn(8, 512, 7, 7).astype(np.float32)

    greedy_cap, greedy_attn = greedy_decode(
        model=model,
        features=features,
        id_to_word=id_to_word,
        max_length=10,
        device=torch.device("cpu"),
    )
    assert isinstance(greedy_cap, str)
    assert len(greedy_attn) > 0
    # Every step should have uniform attention across 8 frames
    for step_attn in greedy_attn:
        assert step_attn.shape == (8,)
        assert torch.allclose(step_attn, torch.full((8,), 1.0 / 8), atol=1e-5)

    beam_cap, _ = beam_search_decode(
        model=model,
        features=features,
        id_to_word=id_to_word,
        max_length=10,
        device=torch.device("cpu"),
        beam_width=1,
        length_penalty=0.0,
    )
    # beam_width=1 should equal greedy
    assert beam_cap == greedy_cap


def test_load_model_attention_type_compatibility():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "dog": 4}
        base_cfg = {
            "feature_channels": 512,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "dropout": 0.1,
            "encoder_type": "temporal",
            "device": "cpu",
        }

        # 1. Old checkpoint without attention_type should load as "additive"
        old_model = VideoCaptionModel(
            vocab_size=len(vocab),
            feature_dim=512,
            embed_dim=32,
            hidden_dim=64,
            encoder_type="temporal",
            attention_type="additive",
        )
        old_ckpt_path = tmp_path / "old_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": old_model.state_dict(),
                "config": base_cfg,
                "vocab": vocab,
                "epoch": 1,
                "best_val_loss": 1.0,
            },
            old_ckpt_path,
        )

        loaded_old, cfg_old, _, _, _ = load_model(
            device=torch.device("cpu"), checkpoint_path=old_ckpt_path
        )
        assert loaded_old.decoder.attention_type == "additive"
        assert loaded_old.decoder.attention is not None

        # 2. Checkpoint with attention_type="mean"
        mean_cfg = dict(base_cfg)
        mean_cfg["attention_type"] = "mean"
        mean_model = VideoCaptionModel(
            vocab_size=len(vocab),
            feature_dim=512,
            embed_dim=32,
            hidden_dim=64,
            encoder_type="temporal",
            attention_type="mean",
        )
        mean_ckpt_path = tmp_path / "mean_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": mean_model.state_dict(),
                "config": mean_cfg,
                "vocab": vocab,
                "epoch": 1,
                "best_val_loss": 1.0,
                "attention_type": "mean",
            },
            mean_ckpt_path,
        )

        loaded_mean, cfg_mean, _, _, _ = load_model(
            device=torch.device("cpu"), checkpoint_path=mean_ckpt_path
        )
        assert loaded_mean.decoder.attention_type == "mean"
        assert loaded_mean.decoder.attention is None


def test_efficiency_helpers():
    vgg_count = count_frozen_vgg16_parameters()
    assert vgg_count == 14714688

    cpu_model = get_cpu_model()
    assert isinstance(cpu_model, str)
    assert len(cpu_model) > 0

    sampler = ProcessMemorySampler(interval_sec=0.005)
    sampler.start()
    _ = [np.zeros((100, 100)) for _ in range(10)]
    peak_rss = sampler.stop()
    assert peak_rss > 0


def test_visualize_attention_plot_helpers():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Heatmap plot
        words = ["a", "person", "is", "walking"]
        attn_matrix = np.full((len(words), 8), 0.125, dtype=np.float32)
        heatmap_out = tmp_path / "test_heatmap.png"
        plot_frame_attention_heatmap(
            attn_matrix=attn_matrix,
            words=words,
            video_id="test_vid",
            caption="a person is walking",
            out_path=heatmap_out,
        )
        assert heatmap_out.is_file()
        assert heatmap_out.stat().st_size > 0

        # 2. Spatiotemporal overlay plot
        reshaped = np.random.rand(len(words), 8, 7, 7).astype(np.float32)
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(8)]
        indices = list(range(8))
        overlay_out = tmp_path / "test_overlay.png"
        plot_spatiotemporal_spatial_overlay(
            reshaped_weights=reshaped,
            frames_rgb=frames,
            frame_indices=indices,
            words=words,
            video_id="test_vid",
            caption="a person is walking",
            out_path=overlay_out,
        )
        assert overlay_out.is_file()
        assert overlay_out.stat().st_size > 0
