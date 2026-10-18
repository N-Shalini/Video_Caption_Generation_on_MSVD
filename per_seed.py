import glob
import json
import re

rows = []
for path in sorted(glob.glob("results/*_v2_s[0-9]*_best_test_evaluation.json")):
    m = re.search(r"results[\\/](.+)_v2_s(\d+)_best_test_evaluation\.json", path)
    enc, seed = m.group(1), int(m.group(2))
    d = json.load(open(path, encoding="utf-8"))["ablation_results"]["beam_5"]
    rows.append((enc, seed, d["CIDEr"], d["BLEU-4"], d["METEOR"], d["ROUGE-L"]))
print("%-26s %-5s %-8s %-8s %-8s %-8s" % ("encoder", "seed", "CIDEr", "BLEU-4", "METEOR", "ROUGE-L"))
for r in rows:
    print("%-26s %-5d %-8.2f %-8.2f %-8.2f %-8.2f" % r)