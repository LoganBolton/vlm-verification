from pprint import pprint
import math
import argparse
import json
import gc
import os

import torch
torch._inductor.config.autotune_local_cache = False
from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM

from inference import MODEL_SIZES, DATA_SPECIFIC_FUNCTIONS, process_prompts, inference
from solver_cache import SolverCacheManager
from answer_extractors import extract_verifier_answer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the rejection sampling pipeline.

    Returns:
        Parsed argument namespace containing model names, dataset configuration,
        sampling parameters, and rejection sampling settings.
    """
    parser = argparse.ArgumentParser()

    # model and prompts
    parser.add_argument("--solver_model_name", type=str, required=True, help="Model for initial solving")
    parser.add_argument("--verifier_model_name", type=str, required=True, help="Model for verification")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Prompt directory, should contain inference_prompt.md and verification_prompt.md")

    # dataset
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_subset_ratio", type=float, default=1.0)

    # vllm server initialization
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # solver vllm sampling params (vllm native default values, except max_new_tokens)
    parser.add_argument("--solver_max_new_tokens", type=int, default=8192)
    parser.add_argument("--solver_temperature", type=float, default=0.7)
    parser.add_argument("--solver_top_k", type=int, default=-1)
    parser.add_argument("--solver_top_p", type=float, default=0.9)

    # verifier vllm sampling params
    parser.add_argument("--verifier_max_new_tokens", type=int, default=None)
    parser.add_argument("--verifier_temperature", type=float, default=None)
    parser.add_argument("--verifier_top_k", type=int, default=None)
    parser.add_argument("--verifier_top_p", type=float, default=None)

    # rejection sampling specific
    parser.add_argument("--max_attempts", type=int, default=5, help="Maximum number of solver attempts per problem")

    # dataset generation configurations
    parser.add_argument("--data_generation_kwargs", type=str, default="", help="""
                        Generation kwargs as comma-separated key=value pairs.
                        Examples: 'sat_type=2,num_samples=1000,min_vars=3' for SAT,
                                  'size=4,num_samples=500,min_empty=6' for Sudoku,
                                  'size=6,num_samples=300,min_val=-5,max_val=5' for Matrix Multiplication""")

    # miscellaneous
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    # Solver caching (only for first iteration)
    parser.add_argument("--solver_cache_root", type=str, default="solver_cache", help="Root directory for solver cache")
    parser.add_argument("--no_load_solver_cache", action='store_true', help='Disable loading from solver cache')

    parser.add_argument("--oracle_verifier", action='store_true')

    args = parser.parse_args()

    # verifier sampling params default to solver's
    if args.verifier_max_new_tokens is None:
        args.verifier_max_new_tokens = args.solver_max_new_tokens
    if args.verifier_temperature is None:
        args.verifier_temperature = args.solver_temperature
    if args.verifier_top_k is None:
        args.verifier_top_k = args.solver_top_k
    if args.verifier_top_p is None:
        args.verifier_top_p = args.solver_top_p

    assert args.solver_model_name in MODEL_SIZES and args.verifier_model_name in MODEL_SIZES

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
    cache_params = {
        'solver_model_name': args.solver_model_name,
        'dataset_name': args.dataset_name,
        'dataset_subset_ratio': args.dataset_subset_ratio,
        'data_generation_kwargs': args.data_generation_kwargs,
        'solver_temperature': args.solver_temperature,
        'solver_max_new_tokens': args.solver_max_new_tokens,
        'solver_top_k': args.solver_top_k,
        'solver_top_p': args.solver_top_p,
        'solver_n_samples': 1,
        'seed': args.seed,
        'inference_prompt': inference_prompt,
        'dataset': dataset,
    }

    global_active_solver_indices = list(range(len(dataset)))
    solver_records_first_attempt_no_bad = {}
    all_iteration_metrics = {-1: {
        "total_in_original_data": len(dataset),
    }}

    ####################################### REJECTION SAMPLING LOOP #######################################
    for attempt in range(args.max_attempts):
        print(f"\n============= ATTEMPT {attempt + 1}/{args.max_attempts} with {len(global_active_solver_indices)} Active Problems =============")
        iteration_seed = args.seed + attempt
        set_seed(iteration_seed)

        ####################################### SOLVER #######################################
        outputs, solver_total_tokens = None, None

        if attempt == 0 and not args.no_load_solver_cache:
            outputs, solver_total_tokens = solver_cache.load(cache_params)

        run_solver = (outputs == None)
        if run_solver:
            solver_model = LLM(
                model=args.solver_model_name,
                dtype=torch.bfloat16,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=iteration_seed,
            )
            solver_tokenizer = AutoTokenizer.from_pretrained(args.solver_model_name)

            prompts = [[{"role": "user", "content": inference_prompt.format(question=dataset[idx]['question'])}]
                      for idx in global_active_solver_indices]
            prompts = process_prompts(solver_tokenizer, prompts)
            outputs, solver_total_tokens = inference(
                model=solver_model,
                prompts=prompts,
                temperature=args.solver_temperature,
                max_new_tokens=args.solver_max_new_tokens,
                top_k=args.solver_top_k,
                top_p=args.solver_top_p,
                n_samples=1,
                seed=iteration_seed,
            )
            assert len(outputs) == len(global_active_solver_indices), (len(outputs), len(global_active_solver_indices))
            del solver_model, solver_tokenizer
            gc.collect()

        ####################################### EVALUATE SOLVER #######################################
        bad_solve_count = 0
        verifier_records = []
        active_verifier_indices = []

        for output, idx in zip(outputs, global_active_solver_indices):
            extracted_answer = extract_answer_fn(output)

            if extracted_answer == None:
                bad_solve_count += 1
            else:
                # evaluate answer
                is_correct = oracle_verifier_fn(
                    data_row=dataset[idx],
                    solver_extracted_answer=extracted_answer,
                )
                assert isinstance(is_correct, bool)

                if attempt == 0:
                    assert idx not in solver_records_first_attempt_no_bad
                else:
                    assert idx in solver_records_first_attempt_no_bad
                solver_records_first_attempt_no_bad[idx] = {
                    "data_row": dataset[idx],
                    "solver_correct": is_correct,
                    "solver_full_output": output,
                    'solver_extracted_answer': extracted_answer,
                }

                verifier_records.append({
                    "question": dataset[idx]['question'],
                    "solver_full_output": output,
                    "oracle_is_correct": is_correct
                })
                active_verifier_indices.append(idx)

        if attempt == 0:
            global_active_solver_indices = sorted(solver_records_first_attempt_no_bad.keys())
            print(f"First attempt filters out problems with malformed outputs, {len(global_active_solver_indices)} remaining")

        iteration_metrics = {
            "attempt": attempt,
            "solver": {
                "total": len(solver_records_first_attempt_no_bad),
                "accuracy": sum(record['solver_correct'] for record in solver_records_first_attempt_no_bad.values()) / len(solver_records_first_attempt_no_bad),
                "total_this_iteration": len(outputs),
                "bad_count_this_iteration": bad_solve_count,
                "gflops": solver_total_tokens * 2 * MODEL_SIZES[args.solver_model_name],
            },
        }

        if attempt == args.max_attempts - 1 or len(active_verifier_indices) == 0:
            print(f"\n============================ Iteration {attempt + 1} Metrics ============================")
            pprint(iteration_metrics)
            all_iteration_metrics[attempt] = iteration_metrics
            break

        ####################################### VERIFIER #######################################
        set_seed(iteration_seed)

        if not args.oracle_verifier:
            model = LLM(
                model=args.verifier_model_name,
                dtype=torch.bfloat16,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=iteration_seed,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_name)

            prompts = [[{
                "role": "user",
                "content": verification_prompt.format(
                    question=record['question'],
                    response=record['solver_full_output'],
                )
            }] for record in verifier_records]
            prompts = process_prompts(tokenizer, prompts)
            outputs, total_tokens = inference(
                model=model,
                prompts=prompts,
                temperature=args.verifier_temperature,
                max_new_tokens=args.verifier_max_new_tokens,
                top_k=args.verifier_top_k,
                top_p=args.verifier_top_p,
                n_samples=1,
                seed=iteration_seed,
            )
            assert len(outputs) == len(verifier_records) == len(active_verifier_indices), (len(outputs), len(verifier_records), len(active_verifier_indices))
            del model, tokenizer
            gc.collect()
        else:
            outputs = ['dummy'] * len(verifier_records)
            total_tokens = 0

        problems_newly_accepted = 0
        for output, record, idx in zip(outputs, verifier_records, active_verifier_indices):
            if (args.oracle_verifier and record['oracle_is_correct']) or (not args.oracle_verifier and extract_verifier_answer(output)):
                global_active_solver_indices.remove(idx)
                problems_newly_accepted += 1

        iteration_metrics.update({
            "verifier": {
                "gflops": total_tokens * 2 * MODEL_SIZES[args.verifier_model_name],
                "problems_newly_accepted": problems_newly_accepted,
            },
        })

        print(f"\n============================ Iteration {attempt + 1} Metrics ============================")
        pprint(iteration_metrics)
        all_iteration_metrics[attempt] = iteration_metrics

        if len(global_active_solver_indices) == 0:
            print("No more problems to solve!")
            break

    # Save and log metrics
    print("\n============================ Final Metrics ============================")
    pprint(all_iteration_metrics)
    metrics_path = f"{args.output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_iteration_metrics, f, indent=4)
    print(f'saved metrics to {metrics_path}')
