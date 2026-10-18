import collections
import glob
import json
import re
import statistics

rows = collections.defaultdict(list)
for path in sorted(glob.glob("results/*_v2_s[0-9]*_best_test_evaluation.json")):
    m = re.search(r"results[\\/](.+)_v2_s(\d+)_best_test_evaluation\.json", path)
    enc, seed = m.group(1), int(m.group(2))
    d = json.load(open(path, encoding="utf-8"))
    for row, res in d["ablation_results"].items():
        rows[(enc, row)].append((seed, res))

metrics = ("CIDEr", "BLEU-4", "METEOR", "ROUGE-L")
print("%-24s %-8s %-8s " % ("encoder", "decoding", "seeds") + " ".join("%-17s" % x for x in metrics))
for (enc, row), items in sorted(rows.items()):
    cells = []
    for name in metrics:
        v = [r[name] for _, r in items]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        cells.append("%6.2f +/- %-6.2f" % (statistics.mean(v), sd))
    print("%-24s %-8s %-8s " % (enc, row, ",".join(str(s) for s, _ in items)) + " ".join(cells))