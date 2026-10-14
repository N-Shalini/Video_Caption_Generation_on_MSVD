import json

with open("results/test_set_predictions.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

total = len(predictions)
print(f"Total test videos: {total}")

for j in range(15):
    i = round(j * (total - 1) / 14)
    item = predictions[i]

    print("\n" + "=" * 70)
    print(f"INDEX: {i} | VIDEO: {item['video_id']}")
    preds = item["predictions"]
    if "1.2" in preds:
        print(f"  GREEDY (1.2): {preds['1.2']}")
    if "beam_5" in preds:
        print(f"  BEAM (k=5):   {preds['beam_5']}")
    elif "beam_3" in preds:
        print(f"  BEAM (k=3):   {preds['beam_3']}")
    print("  REFERENCES:")
    for caption in item["references"][:3]:
        print(f"    - {caption}")