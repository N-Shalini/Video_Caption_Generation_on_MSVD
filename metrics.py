"""Standard COCO caption evaluation metrics for Video Captioning.

Uses pycocoevalcap (Bleu-1..4, METEOR, ROUGE-L, CIDEr) with PTBTokenizer.
Legacy custom metric implementations are in metrics_legacy.py.
"""

import math
import re
import shutil
from collections import Counter
import numpy as np

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def ngram_counts(tokens, n):
    return Counter(
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    )


def caption_statistics(captions):
    """Computes diversity and repetition statistics across generated captions."""
    total_tokens = 0
    unique_tokens = set()
    ngram_totals = {1: 0, 2: 0}
    ngram_unique = {1: set(), 2: set()}
    repeated_bigram_captions = 0
    repeated_trigram_captions = 0
    empty_captions = 0

    for caption in captions:
        tokens = tokenize(caption)
        total_tokens += len(tokens)
        unique_tokens.update(tokens)

        if not tokens:
            empty_captions += 1

        counts2 = ngram_counts(tokens, 2)
        counts3 = ngram_counts(tokens, 3)

        if any(count > 1 for count in counts2.values()):
            repeated_bigram_captions += 1
        if any(count > 1 for count in counts3.values()):
            repeated_trigram_captions += 1

        for n in (1, 2):
            counts = ngram_counts(tokens, n)
            ngram_totals[n] += sum(counts.values())
            ngram_unique[n].update(counts.keys())

    number = len(captions)

    return {
        "caption_count": number,
        "empty_caption_count": empty_captions,
        "mean_caption_tokens": round(total_tokens / number, 3) if number else 0.0,
        "distinct_1": round(len(ngram_unique[1]) / ngram_totals[1], 4) if ngram_totals[1] else 0.0,
        "distinct_2": round(len(ngram_unique[2]) / ngram_totals[2], 4) if ngram_totals[2] else 0.0,
        "captions_with_repeated_bigrams": repeated_bigram_captions,
        "captions_with_repeated_trigrams": repeated_trigram_captions,
        "repeated_bigram_caption_rate": round(repeated_bigram_captions / number, 4) if number else 0.0,
        "repeated_trigram_caption_rate": round(repeated_trigram_captions / number, 4) if number else 0.0,
        "unique_token_count": len(unique_tokens),
    }


def evaluate_predictions(refs, hyps):
    """Evaluate candidate captions against reference captions using standard COCO metrics.

    Args:
        refs: dict mapping video_id to list of raw reference strings, e.g. {vid: [ref1, ref2, ...]}
        hyps: dict mapping video_id to single raw hypothesis string, e.g. {vid: hyp}

    Returns:
        dict containing:
            BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, CIDEr (scaled x100, rounded to 2 decimals)
            "scale": "x100"
            "per_video": dict mapping video_id to per-video metric dict
    """
    if not shutil.which("java"):
        raise RuntimeError(
            "Java is required for pycocoevalcap standard evaluation but 'java' was not found in PATH."
        )

    if set(refs.keys()) != set(hyps.keys()):
        raise ValueError(
            f"Video ID mismatch between references ({len(refs)}) and hypotheses ({len(hyps)})."
        )

    video_ids = list(hyps.keys())
    if not video_ids:
        return {
            "BLEU-1": 0.0,
            "BLEU-2": 0.0,
            "BLEU-3": 0.0,
            "BLEU-4": 0.0,
            "METEOR": 0.0,
            "ROUGE-L": 0.0,
            "CIDEr": 0.0,
            "scale": "x100",
            "per_video": {},
        }

    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    refs_formatted = {
        vid: [{"caption": str(cap)} for cap in refs[vid]]
        for vid in video_ids
    }
    hyps_formatted = {
        vid: [{"caption": str(hyps[vid]) if hyps[vid] is not None else ""}]
        for vid in video_ids
    }

    tokenizer = PTBTokenizer()
    refs_tok = tokenizer.tokenize(refs_formatted)
    hyps_tok = tokenizer.tokenize(hyps_formatted)

    # Ensure all video IDs are preserved even if hypothesis or reference was blank
    for vid in video_ids:
        if vid not in refs_tok or not refs_tok[vid]:
            refs_tok[vid] = [""]
        if vid not in hyps_tok or not hyps_tok[vid]:
            hyps_tok[vid] = [""]

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    overall_results = {"scale": "x100"}
    per_video = {vid: {} for vid in video_ids}

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs_tok, hyps_tok)
        if isinstance(method, list):
            for n, m_name in enumerate(method):
                overall_results[m_name] = round(float(score[n]) * 100.0, 2)
                for idx, vid in enumerate(video_ids):
                    per_video[vid][m_name] = round(float(scores[n][idx]) * 100.0, 2)
        else:
            overall_results[method] = round(float(score) * 100.0, 2)
            for idx, vid in enumerate(video_ids):
                per_video[vid][method] = round(float(scores[idx]) * 100.0, 2)

    overall_results["per_video"] = per_video
    return overall_results


def compute_all_metrics(records, predictions):
    """Computes BLEU (1-4), METEOR, ROUGE-L, CIDEr, and caption diversity statistics.

    Maintains backwards compatibility by taking records list and predictions list,
    delegating to evaluate_predictions and appending diversity statistics.
    """
    refs = {}
    hyps = {}

    for idx, record in enumerate(records):
        vid = record.get("video_id", str(idx))
        ref_list = record.get("references") or record.get("caption") or []
        refs[vid] = ref_list
        hyps[vid] = predictions[idx] if idx < len(predictions) else ""

    results = evaluate_predictions(refs, hyps)

    # Add diversity and repetition statistics
    results.update(caption_statistics(predictions))

    return results
