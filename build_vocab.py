
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATION_PATH = PROJECT_ROOT / "data" / "annotations" / "msvd_train.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "vocab.json"

MIN_WORD_FREQUENCY = 2
MAX_CAPTION_LENGTH = 30

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def main():
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    word_counts = Counter()
    total_captions = 0
    truncated_captions = 0
    total_tokens = 0

    for record in records:
        for caption in record["caption"]:
            tokens = tokenize(caption)

            if not tokens:
                continue

            total_captions += 1
            total_tokens += len(tokens)

            # Reserve two positions for <bos> and <eos>.
            if len(tokens) > MAX_CAPTION_LENGTH - 2:
                truncated_captions += 1

            word_counts.update(tokens)

    vocabulary_words = sorted(
        word
        for word, count in word_counts.items()
        if count >= MIN_WORD_FREQUENCY
    )

    # Special-token IDs remain fixed and documented.
    vocab = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    for word in vocabulary_words:
        if word not in vocab:
            vocab[word] = len(vocab)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)

    print("Vocabulary build: PASS")
    print("Training records:", len(records))
    print("Captions counted:", total_captions)
    print("Total caption tokens:", total_tokens)
    print("Unique words before filtering:", len(word_counts))
    print("Minimum word frequency:", MIN_WORD_FREQUENCY)
    print("Maximum caption length including special tokens:",
          MAX_CAPTION_LENGTH)
    print("Captions requiring truncation:", truncated_captions)
    print("Special tokens:", SPECIAL_TOKENS)
    print("Final vocabulary size:", len(vocab))
    print("PAD ID:", vocab["<pad>"])
    print("BOS ID:", vocab["<bos>"])
    print("EOS ID:", vocab["<eos>"])
    print("UNK ID:", vocab["<unk>"])
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
