import pytest
from metrics import evaluate_predictions, compute_all_metrics


def test_metrics_identical_hyp_ref():
    refs = {
        "vid_1": ["a person is playing the piano", "a man plays a piano keyboard"],
        "vid_2": ["a cat is jumping onto a sofa", "a cute kitten jumps on couch"],
        "vid_3": ["a dog runs through a grassy park", "a puppy is running in the grass"],
    }
    hyps = {
        "vid_1": "a person is playing the piano",
        "vid_2": "a cat is jumping onto a sofa",
        "vid_3": "a dog runs through a grassy park",
    }

    results = evaluate_predictions(refs, hyps)

    assert results["scale"] == "x100"
    # When predictions exactly match references, BLEU-4 and CIDEr are high
    assert results["BLEU-4"] >= 95.0
    assert results["CIDEr"] > 0.0
    assert results["METEOR"] > 90.0
    assert results["ROUGE-L"] >= 95.0

    # Verify per-video scores
    assert "per_video" in results
    assert set(results["per_video"].keys()) == {"vid_1", "vid_2", "vid_3"}
    for vid in ["vid_1", "vid_2", "vid_3"]:
        assert results["per_video"][vid]["BLEU-4"] >= 90.0
        assert results["per_video"][vid]["CIDEr"] > 0.0


def test_metrics_empty_hypothesis():
    refs = {
        "vid_1": ["a woman is slicing vegetables in the kitchen"],
        "vid_2": ["a boy is kicking a soccer ball"],
    }
    hyps = {
        "vid_1": "",
        "vid_2": "a boy kicks a ball",
    }

    # Must not crash
    results = evaluate_predictions(refs, hyps)
    assert results["scale"] == "x100"
    assert results["per_video"]["vid_1"]["BLEU-4"] == 0.0
    assert results["per_video"]["vid_1"]["METEOR"] == 0.0
    assert results["per_video"]["vid_1"]["CIDEr"] == 0.0


def test_metrics_video_id_keys_consistency():
    refs = {
        "vid_a": ["reference caption one"],
        "vid_b": ["reference caption two"],
    }
    hyps_mismatch = {
        "vid_a": "reference caption one",
        "vid_c": "reference caption two",
    }

    with pytest.raises(ValueError):
        evaluate_predictions(refs, hyps_mismatch)


def test_compute_all_metrics_backwards_compatible():
    records = [
        {"video_id": "v1", "caption": ["a person cooks dinner in the kitchen"]},
        {"video_id": "v2", "caption": ["a car drives down a highway"]},
    ]
    predictions = [
        "a person cooks food",
        "a car drives on road",
    ]

    out = compute_all_metrics(records, predictions)
    assert "BLEU-4" in out
    assert "CIDEr" in out
    assert "METEOR" in out
    assert "ROUGE-L" in out
    assert "distinct_1" in out
    assert "repeated_bigram_caption_rate" in out
