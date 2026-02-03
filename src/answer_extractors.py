from typing import Optional, Dict, List
import re


def extract_gt_answer_gsm8k(text: str) -> float:
    """Extract the numeric answer from GSM8K dataset ground truth text.

    GSM8K answers are formatted with '#### ' prefix on the final line.

    Args:
        text: Raw answer text from the GSM8K dataset.

    Returns:
        The numeric answer as a float.
    """
    final_line = text.split('\n')[-1]
    assert final_line.startswith('#### ')
    final_line = final_line[5:].strip()

    numbers_only = re.sub(r'[^\d.]', '', final_line)
    return float(numbers_only)


def extract_gt_answer_aime(text: str) -> float:
    """Extract the numeric answer from AIME dataset ground truth text.

    Args:
        text: Raw answer text from the AIME dataset.

    Returns:
        The numeric answer as a float.
    """
    numbers_only = re.sub(r'[^\d.]', '', text)
    return float(numbers_only)


def get_final_box_match(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{...} occurrence in the text.

    Args:
        text: Input text potentially containing \\boxed{...} expressions.

    Returns:
        The content inside the last \\boxed{} if found, None otherwise.
    """
    boxed_pattern = r'boxed\{([^}]*)\}'
    boxed_matches = re.findall(boxed_pattern, text, re.DOTALL)
    if not boxed_matches:
        return None
    return boxed_matches[-1].strip()


def extract_verifier_answer(text: str) -> Optional[bool]:
    """Extract the verifier's verdict (correct/incorrect) from the response.

    Looks for 'correct' or 'incorrect' in the boxed content first, then falls
    back to searching the entire text for these keywords.

    Args:
        text: The verifier's response text.

    Returns:
        True if verdict is 'correct', False if 'incorrect', None if extraction fails.
    """
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    if boxed_content.lower() in ['incorrect', 'correct']:
        return boxed_content.lower() == 'correct'

    if ('wrong' in text.lower()) or ('incorrect' in text.lower()):
        return False
    elif 'correct' in text.lower():
        return True

    print(f"[WARNING] Failed extracting answer: {boxed_content}")
    return None


def extract_float_answer(text: str) -> Optional[float]:
    """Extract a numeric answer from the boxed content.

    Parses the content inside \\boxed{} as a float, stripping non-numeric
    characters except digits and decimal points.

    Args:
        text: The solver's response text containing a boxed answer.

    Returns:
        The extracted float value, or None if extraction/parsing fails.
    """
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        numbers_only = re.sub(r'[^\d.]', '', boxed_content)
        return float(numbers_only)
    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_sat_answer(text: str) -> Optional[Dict[str, bool]]:
    """Extract a SAT variable assignment from the boxed content.

    Parses variable assignments in the format 'variable T/F' (one per line)
    from the boxed content.

    Args:
        text: The solver's response text containing a boxed SAT assignment.

    Returns:
        Dictionary mapping variable names to boolean values, or None if parsing fails.
    """
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        assignments = {}
        for line in boxed_content.split('\n'):
            var, value = line.strip().split()
            var, value = var.lower(), value.upper()
            if not (len(var) == 1 and var.isalpha() and value in ['T', 'F']):
                continue
            assignments[var] = (value == 'T')
        return assignments

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_sudoku_answer(text: str) -> Optional[List[List[int]]]:
    """Extract a Sudoku grid from the boxed content.

    Parses a grid of digits from the boxed content, where each line represents
    a row of the Sudoku grid.

    Args:
        text: The solver's response text containing a boxed Sudoku grid.

    Returns:
        2D list of integers representing the Sudoku grid, or None if parsing fails.
    """
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        grid = []
        for line in boxed_content.split('\n'):
            clean_line = re.sub(r'\s+', '', line)
            # Parse each character as a digit
            row = []
            for char in clean_line:
                if char.isdigit():
                    row.append(int(char))
            if len(row) > 0:  # Only add non-empty rows
                grid.append(row)

        return grid

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_matmul_answer(text: str) -> Optional[List[List[int]]]:
    """Extract a matrix from the boxed content.

    Parses a matrix of integers from the boxed content, where each line
    represents a row with space-separated values.

    Args:
        text: The solver's response text containing a boxed matrix.

    Returns:
        2D list of integers representing the matrix, or None if parsing fails.
    """
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        matrix = []
        for line in boxed_content.split('\n'):
            row = []
            for num_str in line.strip().split():
                if num_str.lstrip('-').isdigit():
                    row.append(int(num_str))
            if len(row) > 0:
                matrix.append(row)
        return matrix

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None
