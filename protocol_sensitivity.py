import collections
import json
import sys

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from dataset import tokenize
from metrics import evaluate_predictions

recs = json.load(open(sys.argv[1], encoding="utf-8"))
ids = [r["video_id"] for r in recs]
refs_raw = {r["video_id"]: r["references"] for r in recs}
refs_tok = [[tokenize(x) for x in refs_raw[v]] for v in ids]
sm1 = SmoothingFunction().method1
sm4 = SmoothingFunction().method4
W = {n: tuple([1.0 / n] * n + [0.0] * (4 - n)) for n in (1, 4)}


def safe_meteor(refs, hyp):
    return meteor_score(refs, hyp) if hyp else 0.0


def table(hyps_raw):
    hyps_tok = [tokenize(hyps_raw[v]) for v in ids]
    n = len(ids)
    rows = []
    coco = evaluate_predictions(refs_raw, hyps_raw)
    rows.append(("COCO toolkit, corpus-level, all refs (standard)",
                 coco["BLEU-1"], coco["BLEU-4"], coco["METEOR"]))
    rows.append(("NLTK corpus-level BLEU, all refs",
                 100 * corpus_bleu(refs_tok, hyps_tok, weights=W[1], smoothing_function=sm1),
                 100 * corpus_bleu(refs_tok, hyps_tok, weights=W[4], smoothing_function=sm1), None))

    def sent(refs_fn, k):
        return 100 * np.mean([sentence_bleu(refs_fn(i), hyps_tok[i], weights=W[k],
                                            smoothing_function=sm4) for i in range(n)])

    rows.append(("NLTK sentence-level BLEU averaged, all refs",
                 sent(lambda i: refs_tok[i], 1), sent(lambda i: refs_tok[i], 4), None))
    rows.append(("NLTK sentence-level BLEU averaged, first ref only",
                 sent(lambda i: refs_tok[i][:1], 1), sent(lambda i: refs_tok[i][:1], 4), None))

    def best(k):
        return 100 * np.mean([max(sentence_bleu([r], hyps_tok[i], weights=W[k],
                                                smoothing_function=sm4) for r in refs_tok[i])
                              for i in range(n)])

    rows.append(("NLTK sentence-level BLEU, best single ref (oracle)", best(1), best(4), None))
    try:
        m_all = 100 * np.mean([safe_meteor(refs_tok[i], hyps_tok[i]) for i in range(n)])
        m_first = 100 * np.mean([safe_meteor(refs_tok[i][:1], hyps_tok[i]) for i in range(n)])
        rows.append(("NLTK METEOR, all refs (max over refs)", None, None, m_all))
        rows.append(("NLTK METEOR, first ref only", None, None, m_first))
    except LookupError:
        print("NLTK needs WordNet: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
    return rows


def show(title, rows):
    print("\n" + title)
    print("%-52s %8s %8s %8s" % ("protocol", "BLEU-1", "BLEU-4", "METEOR"))
    for name, b1, b4, m in rows:
        f = lambda x: "%8.2f" % x if x is not None else "%8s" % "-"
        print("%-52s %s %s %s" % (name, f(b1), f(b4), f(m)))


model_hyps = {r["video_id"]: r["prediction"] for r in recs}
show("MODEL PREDICTIONS (" + sys.argv[1] + ")", table(model_hyps))

train = json.load(open("data/annotations/msvd_train.json", encoding="utf-8"))
const = collections.Counter(c.lower().strip() for r in train for c in r["caption"]).most_common(1)[0][0]
print("\nconstant caption used for every video:", repr(const))
show("CONSTANT-CAPTION BASELINE", table({v: const for v in ids}))