import random
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
VOCAB_PATH = ANNOTATION_DIR / "vocab.json"

MAX_CAPTION_LENGTH = 30
EXPECTED_FEATURE_SHAPE = (8, 512, 7, 7)


def feature_dir_for(feature_type="vgg16_spatial"):
    """Return feature directory Path for given feature_type."""
    if feature_type in (None, "vgg16_spatial"):
        return PROJECT_ROOT / "data" / "features" / "vgg16_spatial"
    elif feature_type == "clip_vitb32":
        return PROJECT_ROOT / "data" / "features" / "clip_vitb32"
    else:
        return PROJECT_ROOT / "data" / "features" / feature_type


def is_valid_feature_shape(shape):
    """Return True if shape matches (8, 512, H, W) with H*W in {1, 49}."""
    return (
        len(shape) == 4
        and shape[0] == 8
        and shape[1] == 512
        and (shape[2] * shape[3]) in {1, 49}
    )


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def load_vocab(path=VOCAB_PATH):
    with Path(path).open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    required = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    for token, expected_id in required.items():
        if vocab.get(token) != expected_id:
            raise ValueError(
                f"Invalid vocabulary ID for {token}: "
                f"expected {expected_id}, got {vocab.get(token)}"
            )

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be unique and contiguous.")

    return vocab


class MSVDDataset(Dataset):
    def __init__(
        self,
        split,
        annotation_path=None,
        feature_dir=None,
        feature_type=None,
        vocab_path=VOCAB_PATH,
        max_caption_length=MAX_CAPTION_LENGTH,
        captions_per_video=1,
        seed=42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        if max_caption_length < 2:
            raise ValueError(
                "max_caption_length must allow BOS and EOS"
            )

        if int(captions_per_video) < 0:
            raise ValueError("captions_per_video must be >= 0 (0 means all captions)")

        if annotation_path is None:
            annotation_path = ANNOTATION_DIR / f"msvd_{split}.json"

        self.split = split
        if feature_dir is not None:
            self.feature_dir = Path(feature_dir)
        elif feature_type is not None:
            self.feature_dir = feature_dir_for(feature_type)
        else:
            self.feature_dir = FEATURE_DIR
        self.feature_type = feature_type
        self.max_caption_length = max_caption_length
        self.captions_per_video = int(captions_per_video)
        self.seed = int(seed)
        self.vocab = load_vocab(vocab_path)

        self.pad_idx = self.vocab["<pad>"]
        self.bos_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        self.unk_idx = self.vocab["<unk>"]

        with Path(annotation_path).open(
            "r", encoding="utf-8"
        ) as file:
            self.records = json.load(file)

        if not isinstance(self.records, list):
            raise ValueError("Annotations must contain a JSON list.")

        seen_ids = set()

        for record in self.records:
            required_fields = {"video_id", "video", "caption"}
            missing_fields = required_fields - set(record)

            if missing_fields:
                raise ValueError(
                    f"Record is missing fields: {missing_fields}"
                )

            video_id = record["video_id"]

            if video_id in seen_ids:
                raise ValueError(
                    f"Duplicate video ID in {split}: {video_id}"
                )
            seen_ids.add(video_id)

            if not isinstance(record["caption"], list):
                raise ValueError(
                    f"Captions must be a list for {video_id}"
                )

            if not record["caption"]:
                raise ValueError(
                    f"No reference captions for {video_id}"
                )

            feature_path = self.feature_dir / f"{video_id}.npy"

            if not feature_path.is_file():
                raise FileNotFoundError(
                    f"Missing features for {video_id}: {feature_path}"
                )

        self.video_ids = [record["video_id"] for record in self.records]

        # Multi-caption setup
        if self.split == "train":
            self.samples = []
            self.resample(epoch=0)
        elif self.split == "val":
            if self.captions_per_video == 0:
                # All reference captions for complete validation loss
                self.samples = [
                    (record, cap)
                    for record in self.records
                    for cap in record["caption"]
                ]
            else:
                self.samples = [
                    (record, cap)
                    for record in self.records
                    for cap in record["caption"][:self.captions_per_video]
                ]
        else:
            self.samples = None

    def resample(self, epoch):
        """Resample captions per video for training using a dedicated deterministic RNG."""
        if self.split != "train":
            return

        rng = random.Random(self.seed * 1000003 + epoch)
        self.samples = []

        for record in self.records:
            caps = record["caption"]
            if self.captions_per_video == 0:
                for cap in caps:
                    self.samples.append((record, cap))
            else:
                sample_k = min(self.captions_per_video, len(caps))
                sampled_caps = rng.sample(caps, sample_k)
                for cap in sampled_caps:
                    self.samples.append((record, cap))

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return len(self.records)

    def encode_caption(self, caption):
        tokens = tokenize(caption)

        # Reserve two positions for BOS and EOS.
        tokens = tokens[: self.max_caption_length - 2]
        token_ids = [self.bos_idx]

        token_ids.extend(
            self.vocab.get(token, self.unk_idx)
            for token in tokens
        )
        token_ids.append(self.eos_idx)

        token_ids = token_ids[: self.max_caption_length]
        token_ids.extend(
            [self.pad_idx]
            * (self.max_caption_length - len(token_ids))
        )

        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, index):
        if self.samples is not None:
            record, selected_caption = self.samples[index]
            caption_ids = self.encode_caption(selected_caption)
        else:
            record = self.records[index]
            caption_ids = torch.empty(0, dtype=torch.long)

        video_id = record["video_id"]
        feature_path = self.feature_dir / f"{video_id}.npy"

        features_np = np.load(feature_path, allow_pickle=False)

        if not is_valid_feature_shape(features_np.shape):
            raise ValueError(
                f"Unexpected feature shape for {video_id}: "
                f"{features_np.shape}; expected (8, 512, H, W) with H*W in {{1, 49}}"
            )

        if features_np.dtype != np.float32:
            raise TypeError(
                f"Expected float32 features for {video_id}, "
                f"got {features_np.dtype}"
            )

        if not np.isfinite(features_np).all():
            raise ValueError(
                f"Non-finite feature values for {video_id}"
            )

        features = torch.from_numpy(
            np.ascontiguousarray(features_np)
        )

        return {
            "video_id": video_id,
            "features": features,
            "caption_ids": caption_ids,
            "references": record["caption"],
        }


def collate_video_batch(batch):
    return {
        "video_id": [item["video_id"] for item in batch],
        "features": torch.stack(
            [item["features"] for item in batch]
        ),
        "caption_ids": torch.stack(
            [item["caption_ids"] for item in batch]
        ),
        "references": [item["references"] for item in batch],
    }
