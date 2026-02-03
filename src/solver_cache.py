import hashlib
import json
import os
from typing import Dict, List, Tuple, Any, Optional


class SolverCacheManager:
    """Automatic caching system for solver outputs."""

    def __init__(self, cache_root: str):
        """Initialize cache with root directory for storing cached results."""
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)

    def _generate_cache_dir(self, cache_params: Dict[str, Any]) -> str:
        """Generate a cache directory path based on a hash of all parameters that affect solver output."""
        questions_str = json.dumps(list(cache_params['dataset']['question']))
        answer_str = json.dumps(list(cache_params['dataset']['answer']))
        dataset_hash = hashlib.sha256((questions_str + answer_str).encode()).hexdigest()[:16]
        prompt_hash = hashlib.sha256(cache_params['inference_prompt'].encode()).hexdigest()[:16]

        key_params = {
            'solver_model_name': cache_params['solver_model_name'],
            'dataset_name': cache_params['dataset_name'],
            'dataset_subset_ratio': cache_params['dataset_subset_ratio'],
            'data_generation_kwargs': cache_params['data_generation_kwargs'],
            'solver_temperature': cache_params['solver_temperature'],
            'solver_max_new_tokens': cache_params['solver_max_new_tokens'],
            'solver_top_k': cache_params['solver_top_k'],
            'solver_top_p': cache_params['solver_top_p'],
            'solver_n_samples': cache_params['solver_n_samples'],
            'seed': cache_params['seed'],
            'dataset_hash': dataset_hash,
            'prompt_hash': prompt_hash,
        }

        params_str = json.dumps(key_params, sort_keys=True)
        cache_key = hashlib.sha256(params_str.encode()).hexdigest()
        return os.path.join(self.cache_root, cache_key)

    def load(self, cache_params: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[int]]:
        """
        Load cached solver results if they exist.

        Returns:
            Tuple of (outputs, solver_total_tokens) if cache hit, None if cache miss
        """
        cache_dir = self._generate_cache_dir(cache_params)
        outputs_file = os.path.join(cache_dir, "solver_outputs.jsonl")
        metadata_file = os.path.join(cache_dir, "metadata.json")
        outputs_file_exists = os.path.exists(outputs_file)
        metadata_file_exists = os.path.exists(metadata_file)
        assert outputs_file_exists == metadata_file_exists

        if not outputs_file_exists:
            print(f"[CACHE MISS] Cannot find solver results from: {cache_dir}")
            return None, None

        outputs = []
        with open(outputs_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                outputs.append(data['output'])

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        solver_total_tokens = metadata['solver_total_tokens']

        print(f"[CACHE HIT] Loading solver results from: {cache_dir}")
        return outputs, solver_total_tokens

    def save(
        self,
        cache_params: Dict,
        solver_outputs: List[str],
        solver_total_tokens: int,
    ) -> None:
        """Save solver results to cache in JSONL format."""
        cache_dir = self._generate_cache_dir(cache_params)
        outputs_file = os.path.join(cache_dir, "solver_outputs.jsonl")
        metadata_file = os.path.join(cache_dir, "metadata.json")
        outputs_file_exists = os.path.exists(outputs_file)
        metadata_file_exists = os.path.exists(metadata_file)
        assert outputs_file_exists == metadata_file_exists

        metadata = {
            'solver_total_tokens': solver_total_tokens,
            'cache_params': {
                'solver_model_name': cache_params['solver_model_name'],
                'dataset_name': cache_params['dataset_name'],
                'dataset_subset_ratio': cache_params['dataset_subset_ratio'],
                'data_generation_kwargs': cache_params['data_generation_kwargs'],
                'solver_temperature': cache_params['solver_temperature'],
                'solver_max_new_tokens': cache_params['solver_max_new_tokens'],
                'solver_top_k': cache_params['solver_top_k'],
                'solver_top_p': cache_params['solver_top_p'],
                'solver_n_samples': cache_params['solver_n_samples'],
                'seed': cache_params['seed'],
                'dataset_size': len(cache_params['dataset']),
            }
        }

        if outputs_file_exists:
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            metadata_match = (metadata == existing_metadata)

            existing_outputs = []
            with open(outputs_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    existing_outputs.append(data['output'])
            outputs_match = (solver_outputs == existing_outputs)

            if not metadata_match:
                print(f"[CACHE WARNING] Output mismatch detected!")
            elif not outputs_match:
                print(f"[CACHE WARNING] Output mismatch detected!")
            else:
                print(f"[CACHE HIT WHEN SAVING] Cache is consistent with existing version, skipping.")
                return

        os.makedirs(cache_dir, exist_ok=True)
        with open(outputs_file, 'w') as f:
            for i, output in enumerate(solver_outputs):
                output_data = {
                    'index': i,
                    'output': output
                }
                f.write(json.dumps(output_data) + '\n')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"[CACHE SAVE] Saved solver results to: {cache_dir}")
