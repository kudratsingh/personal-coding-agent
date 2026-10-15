from tools.baselines.humaneval import normalize, truncate_completion


def test_humaneval_mapping_and_stop_words() -> None:
    rows = [
        {
            "task_id": "synthetic/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "canonical_solution": "    return a+b\n",
            "test": "def check(candidate): assert candidate(1,2) == 3",
        }
    ]
    task = normalize(rows)[0]
    assert task.split == "eval"
    assert task.hidden_tests.endswith("check(add)")
    assert task.solution == "def add(a, b):\n    return a+b\n"
    assert "candidate(1,2)" not in task.public_prompt()
    assert truncate_completion("    return a+b\n\ndef other(): pass") == "    return a+b\n"
