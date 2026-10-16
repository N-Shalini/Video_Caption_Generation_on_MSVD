import json, collections
from dataset import tokenize

tr = json.load(open("data/annotations/msvd_train.json", encoding="utf-8"))
occ, fin = collections.Counter(), collections.Counter()
for r in tr:
    for c in r["caption"]:
        t = tokenize(c)
        if not t:
            continue
        occ.update(t)
        fin[t[-1]] += 1
block = sorted(w for w, n in occ.items() if n >= 50 and fin[w] / n <= 0.002)
print(len(block), "blocked tokens")
print(block)
for w in ("a", "an", "the", "on", "with", "in", "of", "to", "and", "is", "are", "while"):
    print("%-6s occurrences=%6d final_rate=%.4f %s" % (w, occ[w], fin[w] / max(occ[w], 1), "BLOCKED" if w in block else ""))