import pytest
from dataset import MSVDDataset


def test_dataset_resample_differ_and_union():
    k = 2
    seed = 42
    dataset = MSVDDataset("train", captions_per_video=k, seed=seed)

    # Record captions for first video across epochs
    first_vid = dataset.records[0]["video_id"]
    total_refs = len(dataset.records[0]["caption"])
    assert total_refs > k, "Test video should have more than k captions for meaningful test"

    seen_captions_epoch0 = [
        cap for record, cap in dataset.samples if record["video_id"] == first_vid
    ]
    assert len(seen_captions_epoch0) == k

    # Union over 10 epochs
    union_captions = set(seen_captions_epoch0)
    captions_by_epoch = [seen_captions_epoch0]

    for epoch in range(1, 10):
        dataset.resample(epoch)
        epoch_caps = [
            cap for record, cap in dataset.samples if record["video_id"] == first_vid
        ]
        captions_by_epoch.append(epoch_caps)
        union_captions.update(epoch_caps)

    # 1. Captions differ across epochs (not all epochs have identical sampled captions)
    assert any(c != captions_by_epoch[0] for c in captions_by_epoch[1:]), (
        "Captions across epochs should differ"
    )

    # 2. Union over 10 epochs exceeds k
    assert len(union_captions) > k, (
        f"Union over 10 epochs ({len(union_captions)}) should exceed k={k}"
    )


def test_dataset_resample_reproducibility():
    k = 3
    seed = 12345
    dataset1 = MSVDDataset("train", captions_per_video=k, seed=seed)
    dataset2 = MSVDDataset("train", captions_per_video=k, seed=seed)

    for epoch in [0, 3, 7]:
        dataset1.resample(epoch)
        dataset2.resample(epoch)

        samples1 = [cap for _, cap in dataset1.samples]
        samples2 = [cap for _, cap in dataset2.samples]
        assert samples1 == samples2, f"Samples for (seed={seed}, epoch={epoch}) must be identical"


def test_dataset_captions_per_video_zero_all_captions():
    dataset = MSVDDataset("train", captions_per_video=0, seed=42)
    expected_total = sum(len(r["caption"]) for r in dataset.records)
    assert len(dataset) == expected_total
