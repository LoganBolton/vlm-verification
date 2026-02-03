from typing import Any, Dict, Optional, List


def exact_match(
    data_row: Dict[str, Any],
    solver_extracted_answer: Optional[Any],
) -> bool:
    """Check if the solver's answer exactly matches the ground truth.

    Args:
        data_row: Dictionary containing the ground truth 'answer' field.
        solver_extracted_answer: The answer extracted from the solver's response.

    Returns:
        True if answers match exactly, False otherwise (including if answer is None).
    """
    if solver_extracted_answer == None:
        return False

    return solver_extracted_answer == data_row['answer']


def sat_is_correct(
    data_row: Dict[str, Any],
    solver_extracted_answer: Optional[Dict[str, bool]],
) -> bool:
    """Verify if a variable assignment satisfies the CNF formula.

    Checks that every clause in the CNF is satisfied by the given assignment.

    Args:
        data_row: Dictionary containing the 'cnf' formula as a list of clauses.
        solver_extracted_answer: Variable assignment mapping names to boolean values.

    Returns:
        True if the assignment satisfies all clauses, False otherwise.
    """
    if solver_extracted_answer == None:
        return False

    cnf = data_row['cnf']
    assignment = solver_extracted_answer

    for clause in cnf:
        clause_satisfied = False
        for literal in clause:
            if literal.startswith("~"):
                var = literal[1:]
                if var not in assignment:
                    return False
                if not assignment[var]:
                    clause_satisfied = True
            else:
                if literal not in assignment:
                    return False
                if assignment[literal]:
                    clause_satisfied = True
        if not clause_satisfied:
            return False

    return True


def sudoku_is_correct(
    data_row: Dict[str, Any],
    solver_extracted_answer: Optional[List[List[int]]],
) -> bool:
    """Verify if a Sudoku solution is valid for the given puzzle.

    Checks that the solution has correct dimensions, preserves given numbers,
    and satisfies all Sudoku constraints (rows, columns, and boxes).

    Args:
        data_row: Dictionary containing the 'puzzle' as a 2D grid with 0s for empty cells.
        solver_extracted_answer: The completed Sudoku grid.

    Returns:
        True if the solution is valid, False otherwise.
    """
    if solver_extracted_answer == None:
        return False

    puzzle = data_row['puzzle']
    size = len(puzzle)
    grid = solver_extracted_answer

    # correct size
    if len(grid) != len(puzzle) or any(len(r1) != len(r2) for r1, r2 in zip(grid, puzzle)):
        return False

    # did not replace numbers already in incomplete grid
    for r1, r2 in zip(grid, puzzle):
        for x, y in zip(r1, r2):
            if not (y == 0 or y == x):
                return False

    # check rows
    valid_numbers = set(range(1, size + 1))
    if not (set(row) == valid_numbers for row in grid):
        return False

    # check cols
    for col in range(size):
        if set([grid[row][col] for row in range(size)]) != valid_numbers:
            return False

    # Check boxes - each box should contain all numbers 1 to size exactly once
    box_size = int(size ** 0.5)
    for box_row in range(0, size, box_size):
        for box_col in range(0, size, box_size):
            box_numbers = []
            for r in range(box_row, box_row + box_size):
                for c in range(box_col, box_col + box_size):
                    box_numbers.append(grid[r][c])
            if set(box_numbers) != valid_numbers:
                return False

    return True


def matmul_is_correct(
    data_row: Dict[str, Any],
    solver_extracted_answer: Optional[List[List[int]]],
) -> bool:
    """Verify if a matrix multiplication result is correct.

    Args:
        data_row: Dictionary containing the expected 'product' matrix.
        solver_extracted_answer: The computed product matrix from the solver.

    Returns:
        True if the answer matches the expected product, False otherwise.
    """
    if solver_extracted_answer == None:
        return False

    return solver_extracted_answer == data_row['product']
