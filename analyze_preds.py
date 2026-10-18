import collections
import json
import statistics
import sys

path = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else "chosen"
recs = json.load(open(path, encoding="utf-8"))
if key == "chosen":
    caps = [r["prediction"] for r in recs]
else:
    caps = [r["predictions"][key] for r in recs]
ends = {"a", "an", "the", "of", "with", "and", "to"}
c = collections.Counter(caps)

print("decoding row:", key)
print("videos:", len(caps),
      "| unique captions:", len(c),
      "| unique words:", len({w for x in caps for w in x.split()}))
print("mean length:", round(statistics.mean(len(x.split()) for x in caps), 2),
      "| ending in a function word:", sum(1 for x in caps if x.split() and x.split()[-1] in ends))
print("top captions:")
for cap, n in c.most_common(8):
    print("  %3d  %s" % (n, cap))

if key == "chosen":
    cid = [r["scores"].get("CIDEr", 0) for r in recs]
    print("per-video CIDEr median:", statistics.median(cid),
          "| share scoring 0:", round(sum(1 for x in cid if x == 0) / len(cid), 3))
    order = sorted(range(len(recs)), key=lambda i: cid[i])
    print("worst 5 by CIDEr:")
    for i in order[:5]:
        print("  ", recs[i]["video_id"], "|", caps[i], "| ref:", recs[i]["references"][0])
    print("best 5 by CIDEr:")
    for i in order[-5:]:
        print("  ", recs[i]["video_id"], "|", caps[i], "| ref:", recs[i]["references"][0])