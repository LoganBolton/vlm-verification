"""Download a subset of the CharXiv dataset for VLM verification experiments.

CharXiv (https://huggingface.co/datasets/princeton-nlp/CharXiv) pairs scientific
chart images (from arXiv papers) with hand-curated question/answer pairs. Each row
carries one open-ended *reasoning* question (`reasoning_q` / `reasoning_a`) plus four
*descriptive* questions. We use the reasoning question, because it is a self-contained
free-text Q&A over the chart -- the direct analog of CountBenchQA's "one question per
image" setup (see scripts/download_countbench.py).

This script pulls the first N examples (default 100), saves each chart image to disk,
and writes a metadata.jsonl with one record per example:

    {
        "id": 0,
        "image": "images/000.jpg",   # path relative to the output dir
        "question": "Which model shows a greater decline in accuracy ...?",
        "answer": "Joint-CNN",        # ground-truth free-text answer
        "category": "cs",             # arXiv subject area
        "year": "20",                 # arXiv year
        "original_id": "2004.10956"   # source arXiv paper id
    }
"""

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, default="princeton-nlp/CharXiv")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/charxiv")
    args = parser.parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Loading {args.hf_dataset} [{args.split}] ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    n = min(args.num_examples, len(ds))
    print(f"Dataset has {len(ds)} rows; taking first {n}.")

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            image = ex["image"].convert("RGB")
            rel_path = os.path.join("images", f"{i:03d}.jpg")
            image.save(os.path.join(args.output_dir, rel_path), format="JPEG")

            record = {
                "id": i,
                "image": rel_path,
                "question": ex["reasoning_q"],
                "answer": ex["reasoning_a"],
                "category": ex["category"],
                "year": ex["year"],
                "original_id": ex["original_id"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {n} images to {images_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
