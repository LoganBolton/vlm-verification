import os
import argparse
import random
import json
from typing import List, Tuple, Set
from tqdm import tqdm
from transformers import set_seed


class SudokuGenerator:
    """Generator for Sudoku problems (4x4 and 9x9) with solutions."""

    def __init__(self):
        """Initialize the generator."""
        self.generated_puzzles: Set[str] = set()

    def normalize_grid(self, grid: List[List[int]]) -> str:
        """Normalize grid for uniqueness comparison."""
        return str(grid)

    def is_valid(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        size = len(grid)

        # Check row
        if num in grid[row]:
            return False

        # Check column
        for i in range(size):
            if grid[i][col] == num:
                return False

        # Check box
        box_size = int(size ** 0.5)
        start_row = (row // box_size) * box_size
        start_col = (col // box_size) * box_size

        for i in range(start_row, start_row + box_size):
            for j in range(start_col, start_col + box_size):
                if grid[i][j] == num:
                    return False

        return True

    def solve_sudoku(self, grid: List[List[int]]) -> bool:
        """Solve sudoku using backtracking."""
        size = len(grid)

        for row in range(size):
            for col in range(size):
                if grid[row][col] == 0:
                    for num in range(1, size + 1):
                        if self.is_valid(grid, row, col, num):
                            grid[row][col] = num

                            if self.solve_sudoku(grid):
                                return True

                            grid[row][col] = 0

                    return False

        return True

    def generate_complete_grid(self, size: int) -> List[List[int]]:
        """Generate a complete valid Sudoku grid."""
        grid = [[0 for _ in range(size)] for _ in range(size)]

        # Fill diagonal boxes first
        box_size = int(size ** 0.5)
        for box_start in range(0, size, box_size):
            nums = list(range(1, size + 1))
            random.shuffle(nums)

            idx = 0
            for i in range(box_start, box_start + box_size):
                for j in range(box_start, box_start + box_size):
                    grid[i][j] = nums[idx]
                    idx += 1

        # Solve the rest
        self.solve_sudoku(grid)
        return grid

    def remove_numbers(self, grid: List[List[int]], num_remove: int) -> List[List[int]]:
        """Remove numbers from complete grid to create puzzle."""
        size = len(grid)
        puzzle = [row[:] for row in grid]

        positions = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(positions)

        removed = 0
        for row, col in positions:
            if removed >= num_remove:
                break

            # Try removing this number
            original = puzzle[row][col]
            puzzle[row][col] = 0

            test_grid = [row[:] for row in puzzle]
            if self.has_unique_solution(test_grid):
                removed += 1
            else:
                puzzle[row][col] = original

        return puzzle

    def has_unique_solution(self, grid: List[List[int]]) -> bool:
        """Check if puzzle has exactly one solution."""
        solutions = []
        self.find_all_solutions(grid, solutions, max_solutions=2)
        return len(solutions) == 1

    def find_all_solutions(self, grid: List[List[int]], solutions: List[List[List[int]]], max_solutions: int = 2):
        """Find all solutions up to max_solutions."""
        if len(solutions) >= max_solutions:
            return

        size = len(grid)

        for row in range(size):
            for col in range(size):
                if grid[row][col] == 0:
                    for num in range(1, size + 1):
                        if self.is_valid(grid, row, col, num):
                            grid[row][col] = num
                            self.find_all_solutions(grid, solutions, max_solutions)
                            grid[row][col] = 0
                    return

        solutions.append([row[:] for row in grid])

    def generate_unique_puzzle(self, size: int, min_empty: int, max_empty: int, max_attempts: int = 100000) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate a unique Sudoku puzzle that hasn't been generated before."""
        for _ in range(max_attempts):
            # Generate complete solution
            solution = self.generate_complete_grid(size)

            if not any(0 in row for row in solution):
                num_remove = random.randint(min_empty, max_empty)
                puzzle = self.remove_numbers(solution, num_remove)

                puzzle_signature = self.normalize_grid(puzzle)
                if puzzle_signature not in self.generated_puzzles:
                    self.generated_puzzles.add(puzzle_signature)
                    return puzzle, solution

        raise ValueError("Cannot generate any more unique puzzles")

    def format_grid(self, grid: List[List[int]], empty_char: str = '_') -> str:
        """Format grid as a string."""
        return '\n'.join(' '.join(str(x if x != 0 else empty_char) for x in row) for row in grid)

    def generate_problem(self, size: int, min_empty: int, max_empty: int) -> Tuple[str, List[List[int]], str]:
        """Generate a single Sudoku problem with solution."""
        puzzle, solution = self.generate_unique_puzzle(size, min_empty, max_empty)

        assert all(0 not in row for row in solution)
        assert len(puzzle) == len(solution) == size
        for row1, row2 in zip(puzzle, solution):
            assert len(row1) == len(row2) == size
            for x, y in zip(row1, row2):
                assert x == 0 or x == y

        question = f"""## Sudoku Problem

**Sudoku** is a logic-based number-placement puzzle. The objective is to fill a {size}x{size} grid with numbers so that each column, each row, and each of the {'2x2' if size == 4 else '3x3'} sub-grids contains all of the numbers from 1 to {size}.

## The Puzzle

Complete the following {size}x{size} Sudoku grid (empty cells are marked with '_'):

{self.format_grid(puzzle)}

## Instructions

Provide your answer as a completed {size}x{size} grid with all numbers filled in, formatted exactly like the puzzle above but with numbers instead of underscores.

For example, a completed 4x4 grid should look like:
\\boxed{{
1 2 3 4
3 4 1 2
2 3 4 1
4 1 2 3
}}
"""

        answer = self.format_grid(solution)

        return question, puzzle, answer


def main() -> None:
    """Generate Sudoku puzzles and save them to a JSONL file.

    Parses command-line arguments to configure puzzle generation parameters
    including grid size, difficulty (number of empty cells), and sample count.
    """
    parser = argparse.ArgumentParser(description="Generate Sudoku problems and solutions")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--size", type=int, choices=[4, 9], required=True, help="Size of Sudoku grid (4 for 4x4, 9 for 9x9)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of problems to generate")
    parser.add_argument("--min_empty", type=int, default=4, help="Minimum number of empty cells")
    parser.add_argument("--max_empty", type=int, default=8, help="Maximum number of empty cells")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    generator = SudokuGenerator()

    questions = []
    puzzles = []
    answers = []
    print(f"Generating {args.num_samples} {args.size}x{args.size} Sudoku problems...")
    for _ in tqdm(range(args.num_samples)):
        question, puzzle, answer = generator.generate_problem(
            size=args.size,
            min_empty=args.min_empty,
            max_empty=args.max_empty,
        )
        questions.append(question)
        puzzles.append(puzzle)
        answers.append(answer)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for question, puzzle, answer in zip(questions, puzzles, answers):
            json.dump({
                "question": question,
                "puzzle": puzzle,
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