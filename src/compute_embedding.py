from pprint import pprint
import math
import argparse
import json
import os

import torch
torch._inductor.config.autotune_local_cache = False
from datasets import Dataset
from transformers import set_seed

from solver_cache import SolverCacheManager
from inference import MODEL_SIZES, DATA_SPECIFIC_FUNCTIONS
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Inception Distance between two multivariate Gaussians"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Means have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different shapes"

    diff = mu1 - mu2

    # Add regularization to avoid numerical issues
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding regularization"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model similarity computation.

    Returns:
        Parsed argument namespace containing model names, dataset configuration,
        and output settings.
    """
    parser = argparse.ArgumentParser()

    # model and prompts
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Prompt directory, should contain inference_prompt.md and verification_prompt.md")

    # dataset
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_subset_ratio", type=float, default=1.0)

    # solver vllm sampling params (vllm native default values, except max_new_tokens)
    parser.add_argument("--solver_max_new_tokens", type=int, default=8192)
    parser.add_argument("--solver_temperature", type=float, default=0.7)
    parser.add_argument("--solver_top_k", type=int, default=-1)
    parser.add_argument("--solver_top_p", type=float, default=0.9)
    parser.add_argument("--solver_n_samples", type=int, default=1)

    # dataset generation configurations
    parser.add_argument("--data_generation_kwargs", type=str, default="", help="""
                        Generation kwargs as comma-separated key=value pairs.
                        Examples: 'sat_type=2,num_samples=1000,min_vars=3' for SAT,
                                  'size=4,num_samples=500,min_empty=6' for Sudoku,
                                  'size=6,num_samples=300,min_val=-5,max_val=5' for Matrix Multiplication""")

    # miscellaneous
    parser.add_argument("--output_dir", type=str, default='utils/similarity_maps', help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    # solver caching
    parser.add_argument("--solver_cache_root", type=str, default="solver_cache", help="Root directory for solver cache")

    # just check cache existence
    parser.add_argument("--dry_run", action='store_true')

    args = parser.parse_args()

    assert all(n in MODEL_SIZES for n in args.model_names)

    if args.solver_n_samples > 1:
        assert args.solver_temperature > 0.0

    return args



if __name__ == "__main__":
    ####################################### SETUP #######################################
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print('==============================================================================')
    pprint(vars(args))
    print('==============================================================================')

    # dataset functions
    process_data_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['process_data']
    extract_answer_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['extract_answer']
    oracle_verifier_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['oracle_verifier']
    dataset_validator_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['dataset_validator']

    if args.dataset_name in ["sat", "sudoku", "matmul"]:
        dataset = process_data_fn(args.output_dir, args.data_generation_kwargs)
    else:
        dataset = process_data_fn()
    dataset = dataset.shuffle(seed=args.seed)
    assert isinstance(dataset, Dataset) and set(['question', 'answer']).issubset(set(dataset.column_names))
    assert dataset_validator_fn(dataset)
    if args.dataset_subset_ratio < 1.0:
        subset_size = math.ceil(len(dataset) * args.dataset_subset_ratio)
        dataset = dataset.select(range(subset_size))
    assert len(dataset) > 0
    print(f"Dataset size: {len(dataset)}")

    # prompts
    with open(f"{args.prompt_dir}/inference_prompt.md", "r") as f:
        inference_prompt = f.read()
    with open(f"{args.prompt_dir}/verification_prompt.md", "r") as f:
        verification_prompt = f.read()

    solver_cache = SolverCacheManager(cache_root=args.solver_cache_root)
    model_name_to_embeddings = {}

    model = None
    if not args.dry_run:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    for model_name in args.model_names:
        cache_params = {
            'solver_model_name': model_name,
            'dataset_name': args.dataset_name,
            'dataset_subset_ratio': args.dataset_subset_ratio,
            'data_generation_kwargs': args.data_generation_kwargs,
            'solver_temperature': args.solver_temperature,
            'solver_max_new_tokens': args.solver_max_new_tokens,
            'solver_top_k': args.solver_top_k,
            'solver_top_p': args.solver_top_p,
            'solver_n_samples': args.solver_n_samples,
            'seed': args.seed,
            'inference_prompt': inference_prompt,
            'dataset': dataset,
        }
        outputs, solver_total_tokens = solver_cache.load(cache_params)
        assert outputs != None, cache_params
        assert all(isinstance(s, str) for s in outputs)

        if args.dry_run:
            continue

        print('computing embeddings of', model_name)
        embeddings = model.encode(outputs)
        model_name_to_embeddings[model_name] = embeddings

    if args.dry_run:
        exit()

    # Compute pairwise similarities
    similarity_matrix = [[None] * len(args.model_names) for _ in range(len(args.model_names))]
    for i1, m1 in enumerate(args.model_names):
        for i2, m2 in enumerate(args.model_names):
            if i1 == i2:
                similarity_matrix[i1][i2] = 1.0
            elif i1 < i2:
                e1 = model_name_to_embeddings[m1]
                e2 = model_name_to_embeddings[m2]
                assert e1.shape == e2.shape
                similarities = model.similarity_pairwise(e1, e2)
                similarity_matrix[i1][i2] = float(similarities.mean())
                similarity_matrix[i2][i1] = float(similarities.mean())

    similarity_array = np.array(similarity_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_array,
                annot=True,
                fmt='.3f',
                xticklabels=args.model_names,
                yticklabels=args.model_names,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Similarity Score'})

    plt.title('Model Similarity Heatmap')
    plt.xlabel('Models')
    plt.ylabel('Models')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_file = os.path.join(args.output_dir, f'{args.dataset_name}_heatmap.png')
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'model_names': args.model_names,
        'similarity_matrix': similarity_matrix,
    }
    output_file = os.path.join(args.output_dir, f'{args.dataset_name}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")
    print(f"Heatmap saved to {heatmap_file}")
