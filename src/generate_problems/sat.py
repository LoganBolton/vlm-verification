import os
import argparse
import random
import json
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm


class SATGenerator:
    """Generator for SAT problems (2SAT and 3SAT) with solutions."""

    def __init__(self):
        """Initialize the SAT problem generator."""
        self.generated_cnfs: Set[str] = set()

    def normalize_cnf(self, cnf: List[List[str]]) -> str:
        """Normalize CNF formula for uniqueness comparison.

        Sorts literals within each clause and sorts clauses to create a
        canonical representation.
        """
        normalized_clauses = []
        for clause in cnf:
            sorted_clause = sorted(clause)
            normalized_clauses.append(tuple(sorted_clause))
        normalized_clauses.sort()
        return str(normalized_clauses)

    def generate_clause(self, variables: List[str], clause_size: int) -> List[str]:
        """Generate a random clause of specified size."""
        assert len(variables) >= clause_size
        selected_vars = random.sample(variables, clause_size)
        clause = []
        for var in selected_vars:
            if random.choice([True, False]):
                clause.append(var)
            else:
                clause.append(f"~{var}")

        return clause

    def generate_unique_cnf(self, num_vars: int, num_clauses: int, clause_size: int) -> Optional[List[List[str]]]:
        """Generate a unique CNF formula that hasn't been generated before."""
        variables = [chr(ord('a') + i) for i in range(num_vars)]
        cnf = [self.generate_clause(variables, clause_size) for _ in range(num_clauses)]
        cnf_signature = self.normalize_cnf(cnf)

        if cnf_signature not in self.generated_cnfs:
            self.generated_cnfs.add(cnf_signature)
            return cnf

        return None

    def is_satisfying_assignment(self, assignment: Dict[str, bool], cnf: List[List[str]]) -> bool:
        """Check if an assignment satisfies the CNF formula."""
        for clause in cnf:
            clause_satisfied = False
            for literal in clause:
                if literal.startswith("~"):
                    var = literal[1:]
                    if not assignment[var]:
                        clause_satisfied = True
                        break
                else:
                    if assignment[literal]:
                        clause_satisfied = True
                        break

            if not clause_satisfied:
                return False

        return True

    def solve_sat(self, cnf: List[List[str]], max_attempts: int = 100000) -> Optional[Dict[str, bool]]:
        """Find a satisfying assignment for the CNF formula using random search."""
        variables = set()
        for clause in cnf:
            for literal in clause:
                if literal.startswith("~"):
                    variables.add(literal[1:])
                else:
                    variables.add(literal)
        variables = list(variables)

        for _ in range(max_attempts):
            assignment = {var: random.choice([True, False]) for var in sorted(variables)}
            if self.is_satisfying_assignment(assignment, cnf):
                return assignment

        return None

    def format_cnf(self, cnf: List[List[str]]) -> str:
        """Format CNF formula as a readable string."""
        clause_strings = []
        for clause in cnf:
            clause_str = "(" + " ∨ ".join(clause) + ")"
            clause_strings.append(clause_str)

        return " ∧ ".join(clause_strings)

    def format_assignment(self, assignment: Dict[str, bool]) -> str:
        """Format assignment as answer string with variables in alphabetical order."""
        lines = []
        for var in sorted(assignment.keys()):
            value = "T" if assignment[var] else "F"
            lines.append(f"{var} {value}")

        return "\n".join(lines)

    def generate_problem(
        self,
        min_vars: int,
        min_clauses: int,
        max_vars: int,
        max_clauses: int,
        clause_size: int
    ) -> Tuple[List[List[str]], Dict[str, bool], str, str]:

        """Generate a single SAT problem with solution."""
        assignment = None
        for _ in range(10000):
            num_vars = random.randint(min_vars, max_vars)
            num_clauses = random.randint(min_clauses, min(max_clauses, num_vars * 2))
            cnf = self.generate_unique_cnf(num_vars, num_clauses, clause_size)
            if cnf == None:
                continue
            assert all(len(c) == clause_size for c in cnf)
            assert len(cnf) <= num_clauses

            assignment = self.solve_sat(cnf)
            if assignment != None:
                assert set(assignment.keys()).issubset(set([chr(ord('a') + i) for i in range(num_vars)]))
                break

        if assignment is None:
            raise ValueError("Cannot find valid satisfiable CNF after 10000 attempts")

        # Format problem description
        sat_type = f"{clause_size}SAT"
        cnf_str = self.format_cnf(cnf)

        question = f"""## Problem Definition

**SAT (Boolean Satisfiability Problem)** is a fundamental problem in computer science where we need to determine if there exists an assignment of Boolean values (True/False) to variables that makes a given Boolean formula evaluate to True.
**Variables**: In this problem, variables are named as single letters. Each variable can be assigned either True (T) or False (F).
**Literals**: A literal is either a variable (like a) or its negation (like ~a, meaning "not a"). If a is True, then ~a is False, and vice versa.
**Clauses**: A clause is a disjunction (OR operation) of literals. A clause is satisfied (True) if at least one of its literals is True. For example, the clause (a ∨ ~b) is True if either a is True OR b is False (or both).
**CNF (Conjunctive Normal Form)**: The Boolean formula is given in CNF, which is a conjunction (AND operation) of multiple clauses. The entire formula is satisfied only if ALL clauses are satisfied simultaneously.
**{sat_type}**: This is a special case of SAT where every clause contains exactly {clause_size} literals.

## The Problem

Find a satisfying assignment for the following CNF formula: {cnf_str}

## Instructions

Provide your answer as a list of variable assignments, one per line, in the format "variable_name T" or "variable_name F". For example:
\\boxed{{
a T
b F
}}
This means a=True, b=False.

Another example answer is
\\boxed{{
a F
b T
}}
This means a=False, b=True.

Output and only output the T/F values for the variables that appear in the provided CNF formula.
"""

        answer = self.format_assignment(assignment)

        return cnf, assignment, question, answer


def main() -> None:
    """Generate SAT problems and save them to a JSONL file.

    Parses command-line arguments to configure problem generation parameters
    including SAT type (2SAT/3SAT), number of variables, clauses, and samples.
    """
    parser = argparse.ArgumentParser(description="Generate SAT problems and solutions")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--sat_type", type=int, choices=[2, 3], required=True, help="Type of SAT problem (2SAT or 3SAT)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of problems to generate")
    parser.add_argument("--min_vars", type=int, default=3, help="Minimum number of variables")
    parser.add_argument("--max_vars", type=int, default=8, help="Maximum number of variables (limited to 26)")
    parser.add_argument("--min_clauses", type=int, default=2, help="Minimum number of clauses")
    parser.add_argument("--max_clauses", type=int, default=8, help="Maximum number of clauses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    assert args.min_vars <= args.max_vars <= 26

    random.seed(args.seed)

    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    generator = SATGenerator()

    cnfs = []
    assignments = []
    questions = []
    answers = []
    print(f"Generating {args.num_samples} {args.sat_type}SAT problems...")
    for _ in tqdm(range(args.num_samples)):
        cnf, assignment, question, answer = generator.generate_problem(
            min_vars=args.min_vars,
            max_vars=args.max_vars,
            min_clauses=args.min_clauses,
            max_clauses=args.max_clauses,
            clause_size=args.sat_type,
        )
        cnfs.append(cnf)
        assignments.append(assignment)
        questions.append(question)
        answers.append(answer)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for cnf, assignment, question, answer in zip(cnfs, assignments, questions, answers):
            json.dump({
                "cnf": cnf,
                "assignment": assignment,
                "question": question,
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
