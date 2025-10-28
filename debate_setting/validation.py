import os
from typing import List, Tuple

def _join(data_dir: str, filename: str) -> str:
    return os.path.join(data_dir, filename)

def load_lines(path: str) -> List[str]:
    \"\"\"Load non-empty, well-formed lines from a file.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file contains no valid lines or a malformed line
    \"\"\"
    if not os.path.exists(path):
        raise FileNotFoundError(f\"Data file not found: {path}\")

    lines: List[str] = []
    with open(path, \"r\", encoding=\"utf-8\") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                # skip blank lines but note could be an issue upstream
                continue
            # Basic format validation: must contain at least one alphanumeric character
            if not any(c.isalnum() for c in line):
                raise ValueError(f\"Malformed line {i} in {path}: no alphanumeric characters present\")
            lines.append(line)

    if not lines:
        raise ValueError(f\"No valid entries found in {path}\")

    return lines

def validate_data_files(data_dir: str = \"data\") -> Tuple[List[str], List[str]]:
    \"\"\"Validate required data files exist and have matching, well-formed entries.

    Returns:
        Tuple of (questions, arguments) as lists of strings.

    Raises:
        FileNotFoundError: if either file is missing
        ValueError: for malformed lines or mismatched counts
    \"\"\"
    qpath = _join(data_dir, \"questions.txt\")
    apath = _join(data_dir, \"arguments.txt\")

    questions = load_lines(qpath)
    arguments = load_lines(apath)

    if len(questions) != len(arguments):
        raise ValueError(
            f\"Data mismatch: '{qpath}' contains {len(questions)} entries but '{apath}' contains {len(arguments)} entries.\\n\"
            f\"Please ensure both files have the same number of non-empty lines and that lines correspond pairwise.\"
        )

    return questions, arguments
