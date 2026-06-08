"""Download a subset of the CountBenchQA dataset for VLM verification experiments.

CountBenchQA (https://huggingface.co/datasets/vikhyatk/CountBenchQA) pairs natural
images with a counting question ("How many X are there in the image?") and the
ground-truth integer count.

This script pulls the first N examples (default 100), saves each image to disk, and
writes a metadata.jsonl with one record per example:

    {
        "id": 0,
        "image": "images/000.jpg",   # path relative to the output dir
        "question": "How many headsets are there in the image?",
        "answer": 10,                  # ground-truth count
        "text": "We review the ten best gaming headsets in the market"  # original caption
    }
"""

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, default="vikhyatk/CountBenchQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/countbench")
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
                "question": ex["question"],
                "answer": int(ex["number"]),
                "text": ex["text"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {n} images to {images_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
