import pytest
import torch
from pathlib import Path

from model import (
    VideoCaptionModel,
    TemporalFeatureEncoder,
    TemporalBiGRUFeatureEncoder,
    SpatioTemporalFeatureEncoder,
    SpatioTemporalAttentionFeatureEncoder,
    AdditiveAttention,
    AdditiveTemporalAttention,
    reshape_attention_weights,
)


def test_encoder_output_shapes():
    batch_size = 2
    num_frames = 8
    channels = 512
    height, width = 7, 7
    features = torch.randn(batch_size, num_frames, channels, height, width)

    # 1. Temporal
    temporal_enc = TemporalFeatureEncoder(feature_dim=512, hidden_dim=512)
    out_temporal = temporal_enc(features)
    assert out_temporal.shape == (batch_size, num_frames, 512)

    # 2. Temporal BiGRU
    bigru_enc = TemporalBiGRUFeatureEncoder(feature_dim=512, hidden_dim=512)
    out_bigru = bigru_enc(features)
    assert out_bigru.shape == (batch_size, num_frames, 512)

    # 3. Spatiotemporal alias
    alias_enc = SpatioTemporalFeatureEncoder(feature_dim=512, hidden_dim=512)
    out_alias = alias_enc(features)
    assert out_alias.shape == (batch_size, num_frames, 512)

    # 4. Spatiotemporal Attention
    spa_enc = SpatioTemporalAttentionFeatureEncoder(
        feature_dim=512, hidden_dim=512, num_frames=8, num_spatial=49
    )
    out_spa = spa_enc(features)
    assert out_spa.shape == (batch_size, 392, 512)


def test_video_caption_model_forward():
    vocab_size = 50
    seq_len = 10
    batch_size = 2
    features = torch.randn(batch_size, 8, 512, 7, 7)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    for enc_type, expected_n in [
        ("temporal", 8),
        ("temporal_bigru", 8),
        ("spatiotemporal", 8),
        ("spatiotemporal_attention", 392),
    ]:
        model = VideoCaptionModel(
            vocab_size=vocab_size,
            feature_dim=512,
            embed_dim=128,
            hidden_dim=256,
            dropout=0.1,
            encoder_type=enc_type,
        )
        logits, attn = model(features, captions)
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert attn.shape == (batch_size, seq_len, expected_n)


def test_attention_weights_sum_to_one():
    attention = AdditiveAttention(feature_dim=256, hidden_dim=256, attention_dim=256)
    batch_size = 3
    num_positions = 392
    encoder_out = torch.randn(batch_size, num_positions, 256)
    hidden_state = torch.randn(batch_size, 256)

    context, weights = attention(encoder_out, hidden_state)
    assert context.shape == (batch_size, 256)
    assert weights.shape == (batch_size, num_positions)
    # Check sum to 1
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Check precomputed_W1 yields identical results
    precomputed_w1 = attention.W1(encoder_out)
    context_pre, weights_pre = attention(
        encoder_out, hidden_state, precomputed_W1=precomputed_w1
    )
    assert torch.allclose(context, context_pre, atol=1e-6)
    assert torch.allclose(weights, weights_pre, atol=1e-6)


def test_reshape_attention_weights():
    weights = torch.randn(2, 10, 392)
    reshaped = reshape_attention_weights(weights, num_frames=8, height=7, width=7)
    assert reshaped.shape == (2, 10, 8, 7, 7)

    with pytest.raises(ValueError):
        reshape_attention_weights(torch.randn(2, 10, 100))


def test_old_checkpoints_loading():
    checkpoints_dir = Path("checkpoints")
    temporal_ckpt_path = checkpoints_dir / "temporal_attention_continued_best.pt"
    spatiotemporal_ckpt_path = checkpoints_dir / "spatiotemporal_vgg_best.pt"

    device = torch.device("cpu")

    if temporal_ckpt_path.exists():
        ckpt = torch.load(temporal_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = VideoCaptionModel(
            vocab_size=len(ckpt["vocab"]),
            feature_dim=cfg["feature_channels"],
            embed_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            encoder_type=cfg.get("encoder_type", "temporal"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    if spatiotemporal_ckpt_path.exists():
        ckpt = torch.load(spatiotemporal_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = VideoCaptionModel(
            vocab_size=len(ckpt["vocab"]),
            feature_dim=cfg["feature_channels"],
            embed_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            encoder_type=cfg.get("encoder_type", "spatiotemporal"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
