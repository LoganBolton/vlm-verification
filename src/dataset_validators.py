from datasets import Dataset


def trivial_validator(ds: Dataset) -> bool:
    """A no-op validator that always returns True.

    Used for datasets that don't require custom validation.

    Args:
        ds: The dataset to validate.

    Returns:
        Always True.
    """
    return True


def validate_sat_dataset(ds: Dataset) -> bool:
    """Validate that all SAT problems have correct satisfying assignments.

    Checks that each assignment in the dataset actually satisfies its CNF formula.

    Args:
        ds: Dataset containing 'cnf' and 'assignment' columns.

    Returns:
        True if all assignments are valid, False otherwise.
    """
    for i in range(len(ds)):
        ex = ds[i]
        cnf = ex['cnf']
        assignment = ex['assignment']

        vars_in_cnf = set()
        for clause in cnf:
            clause_satisfied = False
            for literal in clause:
                if literal.startswith("~"):
                    var = literal[1:]
                    vars_in_cnf.add(var)
                    if not (var in assignment): return False
                    if not assignment[var]:
                        clause_satisfied = True
                else:
                    vars_in_cnf.add(literal)
                    if not (literal in assignment): return False
                    if assignment[literal]:
                        clause_satisfied = True
            if not clause_satisfied:
                return False

        if not vars_in_cnf.issubset(set(assignment.keys())): return False
        for var, value in assignment.items():
            if var not in vars_in_cnf:
                if value != None: return False
        ex['assignment'] = {var: value for var, value in assignment.items() if value != None}

    return True


def validate_sudoku_dataset(ds: Dataset) -> bool:
    """Validate that all Sudoku puzzles have correct solutions.

    Checks that each solution in the dataset is valid for its puzzle.

    Args:
        ds: Dataset containing puzzle and answer columns.

    Returns:
        True if all solutions are valid, False otherwise.
    """
    from oracle_verifiers import sudoku_is_correct

    for i in range(len(ds)):
        ex = ds[i]
        answer_text = ex['answer']
        grid = [[int(x) for x in line.split()] for line in answer_text.split('\n')]
        if not sudoku_is_correct(ex, grid):
            return False

    return True


def validate_matmul_dataset(ds: Dataset) -> bool:
    """Validate that all matrix multiplication problems have correct products.

    Checks that each product in the dataset is the correct result of multiplying
    its input matrices.

    Args:
        ds: Dataset containing matrix inputs and product columns.

    Returns:
        True if all products are correct, False otherwise.
    """
    from oracle_verifiers import matmul_is_correct

    for i in range(len(ds)):
        ex = ds[i]
        answer_text = ex['answer']
        product = [[int(x) for x in line.split()] for line in answer_text.split('\n')]
        if not matmul_is_correct(ex, product):
            return False

    return True
