#!/usr/bin/env python3
"""Export all VLM conversation logs under vlm/result/ into a HuggingFace dataset.

Produces two configs, pushed to `loganbolton/vlm-verification-logs`:

  * "logs"   - one row per (solver / verifier / agentic / rejection / self-consistency)
               conversation, with a unified superset schema plus the full raw record
               and file metadata preserved as JSON strings.
  * "images" - the 1491 unique source images (charxiv + countbench) as real HF Image
               features, keyed by (dataset, id). The logs table references these via
               `image_path` / (`dataset`, `id`) rather than duplicating the bytes.

Idempotent-ish: re-running rebuilds from disk and re-pushes.
"""
import json
import glob
import os
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, Features, Value, Image, Sequence

REPO = os.environ.get("HF_DATASET_REPO", "loganbolton/vlm-verification-logs")
ROOT = Path(__file__).resolve().parents[2]            # repo root
RESULT_DIR = ROOT / "vlm" / "result"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# image id -> relative path lookup, built from disk (authoritative)
# ---------------------------------------------------------------------------
def build_image_index():
    idx = {}          # (dataset, int_id) -> relative posix path
    for ds in ("charxiv", "countbench"):
        img_dir = DATA_DIR / ds / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                key = int(p.stem)
            except ValueError:
                key = p.stem
            idx[(ds, key)] = str(p.relative_to(ROOT).as_posix())
    return idx


IMG_INDEX = build_image_index()


def rel_image_path(abs_or_rel, dataset, rid):
    """Normalise an image reference to a repo-relative path, or synthesize from id."""
    if abs_or_rel:
        p = str(abs_or_rel)
        # make relative to repo root if it lives under it
        marker = "/data/"
        if marker in p:
            return "data/" + p.split(marker, 1)[1]
        if p.startswith("data/"):
            return p
    # synthesize from (dataset, id)
    if dataset:
        try:
            key = int(rid)
        except (TypeError, ValueError):
            key = rid
        return IMG_INDEX.get((dataset, key))
    return None


def dataset_of(name):
    if isinstance(name, dict):
        name = name.get("name")
    if not name:
        return None
    n = str(name).lower()
    if "charxiv" in n:
        return "charxiv"
    if "countbench" in n:
        return "countbench"
    return None


def short_model(m):
    if not m:
        return None
    if isinstance(m, dict):
        m = m.get("name") or m.get("model") or json.dumps(m)
    return str(m).split("/")[-1]


def js(obj):
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def s(x):
    """Coerce to string, preserving None (source answers/verdicts may be int/bool)."""
    if x is None:
        return None
    return x if isinstance(x, str) else str(x)


# ---------------------------------------------------------------------------
# file classification + dedup
# ---------------------------------------------------------------------------
def classify(path):
    b = os.path.basename(path)
    if b == "metrics.json":
        return "metrics"
    if b.endswith("_scores.json"):
        return "scores"
    if b == "records.json":
        return "records"
    if b.startswith("verify"):
        return "verify"
    return "solver"


def kept_files():
    all_json = sorted(glob.glob(str(RESULT_DIR / "**" / "*.json"), recursive=True))
    have_scores = {f[: -len("_scores.json")] + ".json"
                   for f in all_json if f.endswith("_scores.json")}
    kept = []
    for f in all_json:
        fam = classify(f)
        if fam == "metrics":
            continue                    # aggregate-only, not a conversation
        if fam == "solver" and f in have_scores:
            continue                    # raw solver is a lossless subset of its _scores twin
        kept.append((fam, f))
    return kept


def record_type_of(fam, meta):
    if fam == "verify":
        return "verify"
    if fam in ("solver", "scores"):
        return "solver"
    # fam == "records": disambiguate by metadata keys
    if "max_attempts" in meta:
        return "rejection"
    if "n_samples" in meta and "keys_scorer" not in meta:
        # self-consistency records carry 'keys' per record; agentic carries 'turns'
        return "self_consistency"
    if "max_crops" in meta:
        return "agentic"
    return "records"


# fields consumed into first-class columns; everything else per record -> `extra`
CORE_REC_KEYS = {
    "id", "image", "question", "answer", "text_prompt", "rendered_prompt",
    "solver_full_output", "solver_extracted_answer", "solver_correct",
    "verifier_prompt", "verifier_rendered_prompt", "verifier_response",
    "verifier_verdict",
}

COLUMNS = [
    "experiment", "record_type", "source_file", "dataset",
    "solver_model", "verifier_model", "id", "image_path", "image_exists",
    "question", "gold_answer", "system_prompt", "task_prompt", "rendered_prompt",
    "model_output", "extracted_answer", "correct",
    "verifier_prompt", "verifier_rendered_prompt", "verifier_output", "verifier_verdict",
    "generation_params", "extra", "raw_metadata", "raw_record",
]


def blank_row():
    return {c: None for c in COLUMNS}


def row_generator():
    for fam, f in kept_files():
        rel = os.path.relpath(f, RESULT_DIR)
        experiment = rel.split(os.sep)[0]
        try:
            doc = json.load(open(f))
        except Exception as e:
            print(f"  !! skip unreadable {rel}: {e}")
            continue
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        records = doc["records"] if isinstance(doc, dict) and "records" in doc else doc
        if not isinstance(records, list):
            continue
        rtype = record_type_of(fam, meta)

        solver_model = short_model(meta.get("solver_model") or meta.get("model"))
        verifier_model = short_model(meta.get("verifier_model"))
        ds = dataset_of(meta.get("dataset") or experiment)
        # dataset field may be a dict {"name": ...}
        if isinstance(meta.get("dataset"), dict):
            ds = dataset_of(meta["dataset"].get("name")) or ds
        prompt_meta = meta.get("prompt") or {}
        system_prompt = prompt_meta.get("template") if isinstance(prompt_meta, dict) else None
        gen_params = meta.get("generation_params") or meta.get("verification_params")
        meta_json = js(meta)

        for r in records:
            if not isinstance(r, dict):
                continue
            row = blank_row()
            rid = r.get("id")
            imgp = rel_image_path(r.get("image"), ds, rid)
            extra = {k: v for k, v in r.items() if k not in CORE_REC_KEYS}

            row.update(
                experiment=experiment,
                record_type=rtype,
                source_file=rel,
                dataset=ds,
                solver_model=solver_model,
                verifier_model=verifier_model,
                id=str(rid) if rid is not None else None,
                image_path=imgp,
                image_exists=bool(imgp and (ROOT / imgp).exists()),
                question=s(r.get("question")),
                gold_answer=s(r.get("answer")),
                system_prompt=s(system_prompt),
                task_prompt=s(r.get("text_prompt")),
                rendered_prompt=s(r.get("rendered_prompt")),
                model_output=s(r.get("solver_full_output")),
                extracted_answer=s(r.get("solver_extracted_answer")),
                correct=r.get("solver_correct") if isinstance(r.get("solver_correct"), bool) else None,
                verifier_prompt=s(r.get("verifier_prompt")),
                verifier_rendered_prompt=s(r.get("verifier_rendered_prompt")),
                verifier_output=s(r.get("verifier_response")),
                verifier_verdict=s(r.get("verifier_verdict")),
                generation_params=js(gen_params),
                extra=js(extra) if extra else None,
                raw_metadata=meta_json,
                raw_record=js(r),
            )
            yield row


LOGS_FEATURES = Features({
    "experiment": Value("string"),
    "record_type": Value("string"),
    "source_file": Value("string"),
    "dataset": Value("string"),
    "solver_model": Value("string"),
    "verifier_model": Value("string"),
    "id": Value("string"),
    "image_path": Value("string"),
    "image_exists": Value("bool"),
    "question": Value("string"),
    "gold_answer": Value("string"),
    "system_prompt": Value("string"),
    "task_prompt": Value("string"),
    "rendered_prompt": Value("string"),
    "model_output": Value("string"),
    "extracted_answer": Value("string"),
    "correct": Value("bool"),
    "verifier_prompt": Value("string"),
    "verifier_rendered_prompt": Value("string"),
    "verifier_output": Value("string"),
    "verifier_verdict": Value("string"),
    "generation_params": Value("string"),
    "extra": Value("string"),
    "raw_metadata": Value("string"),
    "raw_record": Value("string"),
})


def build_images_dataset():
    rows = {"dataset": [], "id": [], "image_path": [], "image": []}
    for (ds, key), path in sorted(IMG_INDEX.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        rows["dataset"].append(ds)
        rows["id"].append(str(key))
        rows["image_path"].append(path)
        rows["image"].append(str(ROOT / path))
    feats = Features({
        "dataset": Value("string"),
        "id": Value("string"),
        "image_path": Value("string"),
        "image": Image(),
    })
    return Dataset.from_dict(rows, features=feats)


def main():
    private = os.environ.get("HF_PRIVATE", "1") != "0"
    # which configs to (re)push; images are large & rarely change, so allow skipping
    configs = {c.strip() for c in os.environ.get("HF_CONFIGS", "logs,images").split(",") if c.strip()}
    print(f"Building dataset -> {REPO} (private={private}, configs={sorted(configs)})")

    if "logs" in configs:
        # from_generator fingerprints the generator's *code*, not the files it reads,
        # so it will silently reuse a stale build after new result files land. Force a
        # throwaway cache dir every run so we always rebuild from current disk.
        tmp_cache = tempfile.mkdtemp(prefix="vlm_export_")
        try:
            logs = Dataset.from_generator(
                row_generator, features=LOGS_FEATURES, cache_dir=tmp_cache
            )
            print(f"  logs rows: {len(logs):,}")
            logs.push_to_hub(REPO, config_name="logs", private=private)
            print("  pushed logs config")
        finally:
            shutil.rmtree(tmp_cache, ignore_errors=True)

    if "images" in configs:
        images = build_images_dataset()
        print(f"  images rows: {len(images):,}")
        images.push_to_hub(REPO, config_name="images", private=private)
        print("  pushed images config")


if __name__ == "__main__":
    main()
