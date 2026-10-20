import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import feature_dir_for, is_valid_feature_shape, MSVDDataset
from model import VideoCaptionModel, TemporalFeatureEncoder, TemporalBiGRUFeatureEncoder
from generate_caption import greedy_decode, beam_search_decode, load_model
from extract_features_clip import load_completed, EXPECTED_FEATURE_SHAPE


def test_feature_dir_for():
    vgg_dir = feature_dir_for("vgg16_spatial")
    assert vgg_dir.name == "vgg16_spatial"
    assert vgg_dir.parent.name == "features"

    clip_dir = feature_dir_for("clip_vitb32")
    assert clip_dir.name == "clip_vitb32"
    assert clip_dir.parent.name == "features"

    default_dir = feature_dir_for(None)
    assert default_dir.name == "vgg16_spatial"


def test_is_valid_feature_shape():
    # Valid shapes: (8, 512, H, W) with H*W in {1, 49}
    assert is_valid_feature_shape((8, 512, 7, 7))
    assert is_valid_feature_shape((8, 512, 1, 1))

    # Invalid channel counts
    assert not is_valid_feature_shape((8, 256, 1, 1))
    assert not is_valid_feature_shape((8, 1024, 1, 1))

    # Invalid frame counts
    assert not is_valid_feature_shape((16, 512, 1, 1))
    assert not is_valid_feature_shape((4, 512, 7, 7))

    # Invalid spatial dimensions
    assert not is_valid_feature_shape((8, 512, 2, 2))  # 4 not in {1, 49}
    assert not is_valid_feature_shape((8, 512, 3, 3))  # 9 not in {1, 49}
    assert not is_valid_feature_shape((8, 512, 14, 14))  # 196 not in {1, 49}

    # Invalid rank
    assert not is_valid_feature_shape((8, 512, 1))
    assert not is_valid_feature_shape((8, 512))


def test_dataset_clip_feature_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)

        vid = "test_vid_001"
        feat_valid = np.random.randn(8, 512, 1, 1).astype(np.float32)
        np.save(feature_dir / f"{vid}.npy", feat_valid)

        ann_path = tmp_path / "annotations.json"
        vocab_path = tmp_path / "vocab.json"

        records = [
            {
                "video_id": vid,
                "video": f"{vid}.avi",
                "caption": ["a cat is playing with a ball"],
            }
        ]
        ann_path.write_text(json.dumps(records), encoding="utf-8")

        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4, "cat": 5}
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

        ds = MSVDDataset(
            split="train",
            annotation_path=ann_path,
            feature_dir=feature_dir,
            feature_type="clip_vitb32",
            vocab_path=vocab_path,
        )

        item = ds[0]
        assert item["video_id"] == vid
        assert item["features"].shape == (8, 512, 1, 1)
        assert item["features"].dtype == torch.float32

        # Invalid shape raises ValueError
        feat_invalid = np.random.randn(8, 512, 3, 3).astype(np.float32)
        np.save(feature_dir / f"{vid}.npy", feat_invalid)
        with pytest.raises(ValueError, match="Unexpected feature shape"):
            _ = ds[0]


def test_model_forward_with_clip_features():
    batch_size = 2
    clip_features = torch.randn(batch_size, 8, 512, 1, 1)

    # 1. TemporalFeatureEncoder
    enc_temporal = TemporalFeatureEncoder(feature_dim=512, hidden_dim=256)
    out_temp = enc_temporal(clip_features)
    assert out_temp.shape == (batch_size, 8, 256)

    # 2. TemporalBiGRUFeatureEncoder
    enc_bigru = TemporalBiGRUFeatureEncoder(feature_dim=512, hidden_dim=256)
    out_bigru = enc_bigru(clip_features)
    assert out_bigru.shape == (batch_size, 8, 256)

    # 3. VideoCaptionModel forward pass
    vocab_size = 30
    model = VideoCaptionModel(
        vocab_size=vocab_size,
        feature_dim=512,
        embed_dim=64,
        hidden_dim=128,
        encoder_type="temporal",
        attention_type="additive",
    )
    captions = torch.randint(0, vocab_size, (batch_size, 5))
    logits, attn = model(clip_features, captions)
    assert logits.shape == (batch_size, 5, vocab_size)
    assert attn.shape == (batch_size, 5, 8)


def test_decoding_with_clip_features():
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "dog": 4, "runs": 5}
    id_to_word = {i: w for w, i in vocab.items()}

    model = VideoCaptionModel(
        vocab_size=len(vocab),
        feature_dim=512,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        encoder_type="temporal",
        attention_type="additive",
    )
    model.eval()

    clip_feat = np.random.randn(8, 512, 1, 1).astype(np.float32)

    caption_greedy, attn_greedy = greedy_decode(
        model=model,
        features=clip_feat,
        id_to_word=id_to_word,
        max_length=8,
        device=torch.device("cpu"),
    )
    assert isinstance(caption_greedy, str)
    assert len(attn_greedy) > 0
    assert attn_greedy[0].shape == (8,)

    caption_beam, _ = beam_search_decode(
        model=model,
        features=clip_feat,
        id_to_word=id_to_word,
        max_length=8,
        device=torch.device("cpu"),
        beam_width=1,
        length_penalty=0.0,
    )
    assert caption_beam == caption_greedy


def test_clip_extraction_helpers_no_network():
    # 1. Test L2 normalisation logic
    raw_embs = torch.randn(8, 512)
    normed = raw_embs / (raw_embs.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    norms = normed.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # 2. Reshape to (8, 512, 1, 1) float32
    features = normed.view(8, 512, 1, 1).numpy().astype(np.float32)
    assert features.shape == EXPECTED_FEATURE_SHAPE
    assert features.dtype == np.float32

    # 3. Test load_completed resume logic
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_path = tmp_path / "manifest.jsonl"
        feat_dir = tmp_path / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)

        # Valid feature file
        vid_ok = "vid_ok"
        np.save(feat_dir / f"{vid_ok}.npy", features)

        # Corrupt feature file (wrong shape)
        vid_corrupt = "vid_corrupt"
        np.save(feat_dir / f"{vid_corrupt}.npy", np.zeros((8, 512, 7, 7), dtype=np.float32))

        records = [
            {"video_id": vid_ok, "feature_file": f"{vid_ok}.npy"},
            {"video_id": vid_corrupt, "feature_file": f"{vid_corrupt}.npy"},
        ]
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        completed = load_completed(manifest_path, feat_dir)
        assert vid_ok in completed
        assert vid_corrupt not in completed


def test_checkpoint_feature_type_compatibility():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

        # 1. Legacy checkpoint without feature_type
        legacy_model = VideoCaptionModel(
            vocab_size=len(vocab),
            feature_dim=512,
            embed_dim=32,
            hidden_dim=64,
            encoder_type="temporal",
        )
        legacy_path = tmp_path / "legacy.pt"
        torch.save(
            {
                "model_state_dict": legacy_model.state_dict(),
                "config": {
                    "feature_channels": 512,
                    "embedding_dim": 32,
                    "hidden_dim": 64,
                    "dropout": 0.1,
                    "encoder_type": "temporal",
                    "device": "cpu",
                },
                "vocab": vocab,
                "epoch": 1,
                "best_val_loss": 1.0,
            },
            legacy_path,
        )

        _, cfg_legacy, _, _, ckpt_legacy = load_model(
            device=torch.device("cpu"), checkpoint_path=legacy_path
        )
        resolved_feat_type = cfg_legacy.get(
            "feature_type", ckpt_legacy.get("feature_type", "vgg16_spatial")
        )
        assert resolved_feat_type == "vgg16_spatial"

        # 2. CLIP checkpoint with feature_type="clip_vitb32"
        clip_path = tmp_path / "clip.pt"
        torch.save(
            {
                "model_state_dict": legacy_model.state_dict(),
                "config": {
                    "feature_channels": 512,
                    "embedding_dim": 32,
                    "hidden_dim": 64,
                    "dropout": 0.1,
                    "encoder_type": "temporal",
                    "feature_type": "clip_vitb32",
                    "device": "cpu",
                },
                "vocab": vocab,
                "epoch": 1,
                "best_val_loss": 1.0,
                "feature_type": "clip_vitb32",
            },
            clip_path,
        )

        _, cfg_clip, _, _, ckpt_clip = load_model(
            device=torch.device("cpu"), checkpoint_path=clip_path
        )
        resolved_clip_type = cfg_clip.get(
            "feature_type", ckpt_clip.get("feature_type", "vgg16_spatial")
        )
        assert resolved_clip_type == "clip_vitb32"
