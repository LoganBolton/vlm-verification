import os
import argparse
import random
import json
from typing import List, Tuple, Set
from tqdm import tqdm
from transformers import set_seed


class MatrixMultiplicationGenerator:
    """Generator for matrix multiplication problems with solutions."""

    def __init__(self):
        """Initialize the generator."""
        self.generated_problems: Set[str] = set()

    def normalize_matrices(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> str:
        """Normalize matrices for uniqueness comparison."""
        return str((matrix1, matrix2))

    def generate_matrix(self, size: int, min_val: int, max_val: int) -> List[List[int]]:
        """Generate a random square matrix with given size and value bounds."""
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                value = random.randint(min_val, max_val)
                row.append(value)
            matrix.append(row)
        return matrix

    def multiply_matrices(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        """Multiply two square matrices."""
        size = len(matrix1)
        result = [[0 for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]

        return result

    def format_matrix(self, matrix: List[List[int]]) -> str:
        """Format matrix as a string with whitespace-separated values."""
        return '\n'.join(' '.join(str(x) for x in row) for row in matrix)

    def generate_unique_problem(self, size: int, min_val: int, max_val: int, max_attempts: int = 100000) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Generate a unique matrix multiplication problem that hasn't been generated before."""
        for _ in range(max_attempts):
            matrix1 = self.generate_matrix(size, min_val, max_val)
            matrix2 = self.generate_matrix(size, min_val, max_val)

            problem_signature = self.normalize_matrices(matrix1, matrix2)
            if problem_signature not in self.generated_problems:
                self.generated_problems.add(problem_signature)
                product = self.multiply_matrices(matrix1, matrix2)
                return matrix1, matrix2, product

        raise ValueError("Cannot generate any more unique matrix multiplication problems")

    def generate_problem(self, size: int, min_val: int, max_val: int) -> Tuple[str, List[List[int]], List[List[int]], List[List[int]], str]:
        """Generate a single matrix multiplication problem with solution."""
        matrix1, matrix2, product = self.generate_unique_problem(size, min_val, max_val)

        assert len(matrix1) == len(matrix2) == len(product) == size
        assert all(len(row) == size for row in matrix1)
        assert all(len(row) == size for row in matrix2)
        assert all(len(row) == size for row in product)

        expected_product = self.multiply_matrices(matrix1, matrix2)
        assert product == expected_product, "Matrix multiplication verification failed"

        question = f"""## Matrix Multiplication Problem

**Matrix Multiplication** is a fundamental operation in linear algebra where we compute the product of two matrices. For two square matrices A and B of size {size}x{size}, the product C = A x B is computed as:

C[i][j] = Σ(k=0 to {size-1}) A[i][k] x B[k][j]

## The Problem

Compute the product of the following two {size}x{size} matrices:

**Matrix A:**
{self.format_matrix(matrix1)}

**Matrix B:**
{self.format_matrix(matrix2)}

## Instructions

Provide your answer as the resulting {size}x{size} matrix C = A x B, formatted with each row on a separate line and numbers separated by spaces.

For example, a 2x2 result matrix is formatted like:
\\boxed{{
1 2
3 4
}}
"""

        answer = self.format_matrix(product)

        return question, matrix1, matrix2, product, answer


def main() -> None:
    """Generate matrix multiplication problems and save them to a JSONL file.

    Parses command-line arguments to configure problem generation parameters
    including matrix size, value range, and sample count.
    """
    parser = argparse.ArgumentParser(description="Generate matrix multiplication problems and solutions")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--size", type=int, required=True, help="Size of the square matrices (e.g., 2 for 2x2, 3 for 3x3)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of problems to generate")
    parser.add_argument("--min_val", type=int, default=-10, help="Minimum value for matrix elements")
    parser.add_argument("--max_val", type=int, default=10, help="Maximum value for matrix elements")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    assert args.size > 0, "Matrix size must be positive"
    assert args.min_val <= args.max_val, "min_val must be <= max_val"

    set_seed(args.seed)

    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    generator = MatrixMultiplicationGenerator()

    questions = []
    matrix1s = []
    matrix2s = []
    products = []
    answers = []
    print(f"Generating {args.num_samples} {args.size}x{args.size} matrix multiplication problems...")
    for _ in tqdm(range(args.num_samples)):
        question, matrix1, matrix2, product, answer = generator.generate_problem(
            size=args.size,
            min_val=args.min_val,
            max_val=args.max_val,
        )
        questions.append(question)
        matrix1s.append(matrix1)
        matrix2s.append(matrix2)
        products.append(product)
        answers.append(answer)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for question, matrix1, matrix2, product, answer in zip(questions, matrix1s, matrix2s, products, answers):
            json.dump({
                "question": question,
                "matrix1": matrix1,
                "matrix2": matrix2,
                "product": product,
                "answer": answer
            }, f, ensure_ascii=False)
            f.write('\n')
    print(f"Dataset saved to {args.output_path}")

    print("\nSample problem:")
    print("=" * 50)
    print(questions[0])
    print("\nSample answer:")
    print("=" * 50)
    print(answers[0])


if __name__ == "__main__":
    main()