#!/usr/bin/env python3
"""Vision-encoder FLOP accounting, to complement the LLM-side `solver_gflops`.

The existing compute number prices every token (text + image-placeholder + generated) at
`2 * llm_params_B`. That captures the LLM decoder's cost of *attending over* image tokens, but
NOT the vision encoder's own forward pass, which runs the raw image patches through a ViT before
the LLM ever sees them. Ignoring it biases the comparison against models with a big encoder or
aggressive image tiling (many patches) relative to their LLM. This module adds that term:

    vision_gflops = vit_params_B * num_vit_patches * 2      (per image, per encode)

Two ingredients, both measured from the *actual* local artifacts (no hand-typed guesses):
  * `vit_params(model)`      — summed from the model's safetensors headers (vision tower only).
  * `patches_per_image(ds)`  — avg ViT sequence length, by running each model's real HF image
                               processor over a sample of that dataset's images (captures
                               dynamic-resolution / tiling differences across families).

Results are cached to vlm/result/vision_flops.json so the tradeoff scripts read them cheaply.

Run once to (re)build the cache:  .venv-vllm/bin/python vlm/analysis/vision_flops.py
"""
import json, os, struct, glob, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
VLM_DIR = os.path.dirname(HERE)
CACHE = os.path.join(VLM_DIR, "result", "vision_flops.json")
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

# tensor-name substrings that identify the vision *encoder* (exclude the mm projector, which is
# a small LLM-side adapter, not part of the ViT forward pass we're pricing).
VIS_MATCH = ("visual", "vision_tower", "vision_model")
VIS_EXCLUDE = ("projector", "mlp1", "merger")   # merger/projector map ViT->LLM dim, priced LLM-side

# HF repo id per short solver_model key used in MODEL_SIZES (rejection_sampling.py).
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def _snapshot_dir(repo_id):
    d = os.path.join(HF_HUB, "models--" + repo_id.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")))
    return snaps[-1] if snaps else None


def _iter_safetensors_header(path):
    """Yield (name, dtype, shape) for every tensor in a .safetensors file, reading only the
    JSON header (no weights loaded)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        yield name, meta["dtype"], meta["shape"]


def vit_params(repo_id):
    """Total vision-encoder parameter count (in billions) for a local model, from safetensors headers."""
    snap = _snapshot_dir(repo_id)
    if not snap:
        return None
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    if not files:
        return None
    total = 0
    for fp in files:
        for name, _dt, shape in _iter_safetensors_header(fp):
            if any(m in name for m in VIS_MATCH) and not any(x in name for x in VIS_EXCLUDE):
                numel = 1
                for s in shape:
                    numel *= s
                total += numel
    return total / 1e9


# ---- patch counting: run the real processor over dataset images -----------------------------

def _count_vit_patches(ip, image):
    """Number of patches the ViT actually processes for one image, from the image processor
    output alone (bypasses the full processor's text/image-token validation)."""
    import numpy as np
    kw = {}
    if getattr(ip, "crop_to_patches", None) is False:   # InternVL: reproduce dynamic tiling
        kw["crop_to_patches"] = True                    # (max_patches uses the processor default)
    out = ip(images=image, return_tensors="np", **kw)
    # Qwen family exposes an explicit patch grid.
    if "image_grid_thw" in out:
        g = np.array(out["image_grid_thw"]).reshape(-1, 3)
        return int(g.prod(axis=1).sum())                # t*h*w summed over images
    pv = np.array(out["pixel_values"])
    if pv.ndim == 3:                                    # gemma: [crops, num_patches, patch_dim]
        return int(pv.shape[0] * pv.shape[1])          # already patchified -> patches = crops*npatch
    if pv.ndim == 5:                                    # [batch, tiles, C, H, W]
        pv = pv.reshape((-1,) + pv.shape[-3:])
    if pv.ndim != 4:                                    # [tiles, C, H, W]
        return None
    tiles, _c, h, w = pv.shape
    ps = _patch_size(ip)
    return int(tiles * (h // ps) * (w // ps))


def _patch_size(ip):
    v = getattr(ip, "patch_size", None)
    if isinstance(v, int):
        return v
    if isinstance(v, dict):                 # some processors store {'height':..,'width':..}
        return v.get("height", 14)
    return 14   # CLIP/SigLIP default


def patches_per_image(repo_id, image_paths, sample=160, seed=0):
    """Average ViT patch count per image over a random sample of the given dataset images."""
    from transformers import AutoImageProcessor
    from PIL import Image
    ip = AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
    rng = random.Random(seed)
    paths = list(image_paths)
    rng.shuffle(paths)
    paths = paths[:sample]
    counts = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            n = _count_vit_patches(ip, img)
            if n:
                counts.append(n)
        except Exception as e:                          # noqa: BLE001
            print(f"    ! {os.path.basename(p)}: {e}", file=sys.stderr)
    return sum(counts) / len(counts) if counts else None


def _dataset_images(ds):
    meta = os.path.join(os.path.dirname(VLM_DIR), "data", ds, "metadata.jsonl")
    root = os.path.dirname(meta)
    paths = []
    for line in open(meta):
        r = json.loads(line)
        paths.append(os.path.abspath(os.path.join(root, r["image"])))
    return paths


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def build(datasets=("charxiv", "countbench")):
    """Measure vit_params + patches_per_image for every model in MODEL_SIZES; cache to disk."""
    sys.path.insert(0, os.path.join(VLM_DIR, "pipeline"))
    from rejection_sampling import MODEL_SIZES
    cache = load_cache()
    imgs = {ds: _dataset_images(ds) for ds in datasets}
    for repo_id in MODEL_SIZES:
        rec = cache.get(repo_id, {})
        if "vit_params_B" not in rec:
            vp = vit_params(repo_id)
            rec["vit_params_B"] = vp
            print(f"[{repo_id}] vit_params = {vp and round(vp,4)} B")
        rec.setdefault("patches", {})
        for ds in datasets:
            if ds in rec["patches"]:
                continue
            print(f"[{repo_id}] measuring patches on {ds} ...")
            try:
                n = patches_per_image(repo_id, imgs[ds])
            except Exception as e:                       # noqa: BLE001
                print(f"    ! processor failed: {e}", file=sys.stderr)
                n = None
            rec["patches"][ds] = n
            print(f"    -> {n and round(n,1)} patches/image")
        cache[repo_id] = rec
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(cache, open(CACHE, "w"), indent=2)
    return cache


def vision_gflops_total(repo_id, ds, n_problems, cache=None):
    """Total vision-encoder GFLOPs to encode the whole dataset once (one pass over all images)."""
    cache = cache or load_cache()
    rec = cache.get(repo_id)
    if not rec:
        return None
    vp = rec.get("vit_params_B")
    pat = (rec.get("patches") or {}).get(ds)
    if vp is None or pat is None:        # vp == 0.0 is valid: unified model, no separate ViT
        return None
    return vp * pat * 2 * n_problems


if __name__ == "__main__":
    build()
    print(f"\nwrote {CACHE}")
