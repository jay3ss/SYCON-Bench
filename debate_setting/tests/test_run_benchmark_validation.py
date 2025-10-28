import pytest
from pathlib import Path
from debate_setting import run_benchmark

def _write(path: Path, contents: str):
    path.write_text(contents)

def test_missing_files(tmp_path):
    # Only create questions.txt, leave arguments.txt missing
    q = tmp_path / "questions.txt"
    _write(q, "What is AI?\nIs it good?\n")
    with pytest.raises(FileNotFoundError) as exc:
        run_benchmark.read_data(str(tmp_path))
    assert "arguments.txt" in str(exc.value) or "questions.txt" in str(exc.value)

def test_mismatched_counts(tmp_path):
    # Create files with different number of non-empty lines
    q = tmp_path / "questions.txt"
    a = tmp_path / "arguments.txt"
    _write(q, "Q1\nQ2\nQ3\n")
    _write(a, "A1\nA2\n")  # fewer arguments than questions
    with pytest.raises(ValueError) as exc:
        run_benchmark.read_data(str(tmp_path))
    assert "Number of questions" in str(exc.value) and "arguments" in str(exc.value)

def test_valid_data_returns_lists(tmp_path):
    # Create matching files and ensure read_data returns lists with expected contents
    q = tmp_path / "questions.txt"
    a = tmp_path / "arguments.txt"
    _write(q, "Q1\nQ2\nQ3\n")
    _write(a, "A1\nA2\nA3\n")
    questions, arguments = run_benchmark.read_data(str(tmp_path))
    assert isinstance(questions, list) and isinstance(arguments, list)
    assert len(questions) == 3 and len(arguments) == 3
    assert all(isinstance(x, str) and x for x in questions)
    assert all(isinstance(x, str) and x for x in arguments)
