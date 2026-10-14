"""Legacy metric scorers (deprecated, retained for reference and backward-compatibility)."""

import math
import re
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def lcs(seq1, seq2):
    """Compute length of Longest Common Subsequence between seq1 and seq2."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for elem1 in seq1:
        prev = 0
        for j, elem2 in enumerate(seq2):
            temp = dp[j + 1]
            if elem1 == elem2:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp
    return dp[n]


def sentence_rouge_l(candidate_tokens, reference_tokens_list, beta=1.2):
    """Compute multi-reference ROUGE-L score for a candidate against a list of references."""
    if not candidate_tokens:
        return 0.0

    c_len = len(candidate_tokens)
    best_f1 = 0.0

    for ref in reference_tokens_list:
        r_len = len(ref)
        if r_len == 0:
            continue
        lcs_len = lcs(candidate_tokens, ref)
        if lcs_len == 0:
            continue
        prec = lcs_len / c_len
        rec = lcs_len / r_len
        denom = rec + (beta ** 2) * prec
        if denom > 0:
            f1 = ((1 + beta ** 2) * prec * rec) / denom
            if f1 > best_f1:
                best_f1 = f1

    return best_f1


class CIDErScorer:
    """Computes CIDEr score (Vedantam et al., CVPR 2015) for video captioning."""

    def __init__(self, references_list, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.ref_len = len(references_list)
        self.doc_freq = [Counter() for _ in range(n)]

        for refs in references_list:
            seen_ngrams = [set() for _ in range(n)]
            for ref in refs:
                for k in range(1, n + 1):
                    for i in range(len(ref) - k + 1):
                        seen_ngrams[k - 1].add(tuple(ref[i:i + k]))
            for k in range(n):
                for ngram in seen_ngrams[k]:
                    self.doc_freq[k][ngram] += 1

    def _get_tfidf_vector(self, tokens, k):
        counts = Counter(
            tuple(tokens[i:i + k])
            for i in range(len(tokens) - k + 1)
        )
        total = sum(counts.values())
        if total == 0:
            return {}

        vec = {}
        for ngram, count in counts.items():
            df = self.doc_freq[k - 1].get(ngram, 0)
            idf = math.log(max(1.0, self.ref_len / (df + 1e-8)))
            vec[ngram] = (count / total) * idf
        return vec

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        dot = sum(v * vec2.get(k, 0.0) for k, v in vec1.items())
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 <= 0 or norm2 <= 0:
            return 0.0
        return dot / (norm1 * norm2)

    def score(self, hypothesis_tokens, reference_tokens_list):
        if not hypothesis_tokens or not reference_tokens_list:
            return 0.0

        scores_by_n = []
        hyp_len = len(hypothesis_tokens)

        ref_lens = [len(r) for r in reference_tokens_list if len(r) > 0]
        ref_len_mean = np.mean(ref_lens) if ref_lens else 1.0

        for k in range(1, self.n + 1):
            hyp_vec = self._get_tfidf_vector(hypothesis_tokens, k)
            sims = [
                self._cosine_similarity(hyp_vec, self._get_tfidf_vector(ref, k))
                for ref in reference_tokens_list
            ]
            scores_by_n.append(np.mean(sims) if sims else 0.0)

        length_penalty = math.exp(-((hyp_len - ref_len_mean) ** 2) / (2 * self.sigma ** 2))
        return 10.0 * float(np.mean(scores_by_n)) * length_penalty
