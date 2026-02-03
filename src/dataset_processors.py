from datasets import load_dataset, concatenate_datasets, Dataset
from answer_extractors import *
import json
import random
import subprocess
import os
from typing import Dict, Any


MMLU_STEM = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning"
]
MMLU_SOCIAL_SCIENCES = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy"
]
MMLU_HUMANITIES = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions"
]
MMLU_OTHER = [
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology"
]
MMLU_ALL = MMLU_STEM + MMLU_SOCIAL_SCIENCES + MMLU_HUMANITIES + MMLU_OTHER
assert len(set(MMLU_STEM).union(set(MMLU_SOCIAL_SCIENCES)).union(set(MMLU_HUMANITIES)).union(set(MMLU_OTHER))) == 57
assert len(MMLU_STEM) + len(MMLU_SOCIAL_SCIENCES) + len(MMLU_HUMANITIES) + len(MMLU_OTHER) == 57


def process_generated_data(script_name: str, output_dir: str, data_generation_kwargs: str) -> Dataset:
    """Generate data on the fly using the specified generation script.

    Args:
        script_name: Name of the generation script (e.g., 'sat', 'sudoku', 'matmul').
        output_dir: Directory for temporary file storage.
        data_generation_kwargs: Comma-separated key=value pairs for generation parameters.

    Returns:
        Dataset with generated problems and solutions.
    """
    kwargs = {}
    if data_generation_kwargs:
        for pair in data_generation_kwargs.split(','):
            assert "=" in pair, f"Invalid kwarg format: {pair}"
            key, value = pair.split('=')
            key, value = key.strip(), value.strip()
            try:
                kwargs[key] = int(value)
            except:
                try:
                    kwargs[key] = float(value)
                except:
                    kwargs[key] = value

    temp_file = os.path.join(output_dir, f"{script_name}_temp.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    cmd = ['python', f'src/generate_problems/{script_name}.py', '--output_path', temp_file]
    for k, v in kwargs.items():
        cmd.extend([f"--{k}", str(v)])

    print(f"Generating {script_name} dataset with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    data = []
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    dataset = Dataset.from_list(data)

    os.remove(temp_file)
    return dataset


def process_gsm() -> Dataset:
    """Load and process the GSM8K math reasoning dataset.

    Returns:
        Dataset with 'question' and numeric 'answer' columns.
    """
    ds = load_dataset("openai/gsm8k", "main")['test']
    ds = ds.map(lambda x: {'answer': extract_gt_answer_gsm8k(x['answer'])})
    return ds


def process_aime() -> Dataset:
    """Load and process the AIME competition math dataset.

    Returns:
        Dataset with 'question' and numeric 'answer' columns.
    """
    ds = load_dataset("TianHongZXY/aime-1983-2025", split='test')
    ds = ds.map(lambda x: {'answer': extract_gt_answer_aime(x['answer'])})
    return ds.rename_columns({'problem': 'question'})


def process_mmlu(supercategory: str) -> Dataset:
    """Load and process MMLU benchmark dataset for a given supercategory.

    Args:
        supercategory: One of 'all', 'stem', 'social_sciences', 'humanities', or 'other'.

    Returns:
        Dataset with 'question' (including formatted choices) and 'answer' (choice index) columns.
    """

    def format(example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single MMLU example into multiple-choice format."""
        assert len(example['choices']) == 4
        choices_display = ""
        for i, choice in enumerate(example['choices']):
            choices_display += f"Option {i}: {choice}\n"
        return {
            'question': example['question'] + '\n' + choices_display,
            'answer': example['answer']
        }

    if supercategory == 'all':
        ds = [load_dataset("cais/mmlu", c) for c in MMLU_ALL]
    elif supercategory == 'stem':
        ds = [load_dataset("cais/mmlu", c) for c in MMLU_STEM]
    elif supercategory == 'social_sciences':
        ds = [load_dataset("cais/mmlu", c) for c in MMLU_SOCIAL_SCIENCES]
    elif supercategory == 'humanities':
        ds = [load_dataset("cais/mmlu", c) for c in MMLU_HUMANITIES]
    elif supercategory == 'other':
        ds = [load_dataset("cais/mmlu", c) for c in MMLU_OTHER]
    else:
        raise NotImplementedError()

    ds = concatenate_datasets([x['test'] for x in ds])
    ds = ds.map(
        format,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    return ds


def process_csqa() -> Dataset:
    """Load and process the CommonsenseQA dataset.

    Returns:
        Dataset with 'question' (including formatted choices) and 'answer' (choice index) columns.
    """

    def format(example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single CommonsenseQA example into multiple-choice format."""
        assert example['choices']['label'] == ['A', 'B', 'C', 'D', 'E']
        assert len(example['choices']['text']) == 5

        choices_display = ""
        for i, choice in enumerate(example['choices']['text']):
            choices_display += f"Option {i}: {choice}\n"
        return {
            'question': example['question'] + '\n' + choices_display,
            'answer': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}[example['answerKey']],
        }

    ds = load_dataset("tau/commonsense_qa", split="validation")
    ds = ds.map(
        format,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    return ds


def process_gpqa() -> Dataset:
    """Load and process the GPQA Diamond dataset.

    Shuffles answer choices randomly for each question.

    Returns:
        Dataset with 'question' (including formatted choices) and 'answer' (choice index) columns.
    """

    def format(example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single GPQA example into multiple-choice format with shuffled options."""
        question = example["Question"]
        answer = example["Correct Answer"]
        incorrect1 = example["Incorrect Answer 1"]
        incorrect2 = example["Incorrect Answer 2"]
        incorrect3 = example["Incorrect Answer 3"]
        assert all(isinstance(s, str) for s in [question, answer, incorrect1, incorrect2, incorrect3])
        assert answer != incorrect1 and answer != incorrect2 and answer != incorrect3

        all_choices = [answer, incorrect1, incorrect2, incorrect3]
        random.shuffle(all_choices)
        correct_answer_index = all_choices.index(answer)

        choices_display = ""
        for i, choice in enumerate(all_choices):
            choices_display += f"Option {i}: {choice}\n"
        return {
            'question': question + '\n' + choices_display,
            'answer': correct_answer_index,
        }

    ds = load_dataset('Idavidrein/gpqa', "gpqa_diamond")['train']
    ds = ds.map(
        format,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    return ds


def process_sat(n_sat: int) -> Dataset:
    """Load a pre-generated SAT problem dataset from file.

    Args:
        n_sat: The SAT type (2 or 3) indicating literals per clause.

    Returns:
        Dataset with SAT problems and their satisfying assignments.
    """
    data = []
    with open(f"src/generate_problems/{n_sat}sat.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return Dataset.from_list(data)


def process_sudoku(size: int) -> Dataset:
    """Load a pre-generated Sudoku puzzle dataset from file.

    Args:
        size: The grid size (4 or 9) of the Sudoku puzzles.

    Returns:
        Dataset with Sudoku puzzles and their solutions.
    """
    data = []
    with open(f"src/generate_problems/{size}x{size}sudoku.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return Dataset.from_list(data)


def process_matmul(size: int) -> Dataset:
    """Load a pre-generated matrix multiplication problem dataset from file.

    Args:
        size: The dimension of the square matrices.

    Returns:
        Dataset with matrix multiplication problems and their products.
    """
    data = []
    with open(f"src/generate_problems/{size}x{size}matmul.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return Dataset.from_list(data)
