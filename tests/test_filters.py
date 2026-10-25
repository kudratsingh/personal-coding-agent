from scratchcoder.data.filters import (
    ContaminationIndex,
    Document,
    FilterConfig,
    document_split,
    filter_documents,
    rule_reason,
)


def test_planted_benchmark_removed_from_both_domains() -> None:
    held_out = "Implement a function that returns the sum of all distinct positive elements"
    documents = [
        Document("a", "web", f"article prefix {held_out.upper()} trailing prose", "web"),
        Document("b", "code", f"# {held_out}\ndef answer(): return 1", "python"),
        Document("c", "web", "An unrelated document about rivers and mountain valleys.", "web"),
    ]
    kept, report = filter_documents(documents, ContaminationIndex([held_out]))
    assert [item.id for item in kept] == ["c"]
    assert report["web"]["contamination"] == report["code"]["contamination"] == 1


def test_filter_accounting_and_cross_source_duplicates() -> None:
    text = "def compute(value):\n    return value + 1\n"
    docs = [
        Document("1", "one", text, "python"),
        Document("2", "two", text, "python"),
        Document("3", "one", "# auto-generated\n" + text, "python"),
        Document("4", "one", "x", "web"),
        Document("5", "one", "x" * 100, "python"),
    ]
    kept, report = filter_documents(docs, ContaminationIndex([]), FilterConfig(max_line_length=60))
    assert len(kept) == 1
    assert report["two"]["duplicate"] == 1
    assert report["one"]["generated"] == 1
    assert report["one"]["too_short"] == 1
    assert report["one"]["long_line"] == 1
    for counts in report.values():
        assert counts["seen"] == sum(count for reason, count in counts.items() if reason != "seen")


def test_hash_splits_are_stable_and_disjoint() -> None:
    first = {str(i): document_split(str(i), web=True) for i in range(10000)}
    second = {str(i): document_split(str(i), web=True) for i in reversed(range(10000))}
    assert first == second
    assert set(first.values()) == {"stage1", "stage2", "validation"}


def test_rule_reason_is_the_per_document_part_of_filter_documents() -> None:
    config = FilterConfig(max_line_length=60)
    docs = [
        Document("1", "s", "def compute(value):\n    return value + 1\n", "python"),
        Document("2", "s", "# auto-generated\ndef compute(value):\n    return 2\n", "python"),
        Document("3", "s", "x", "web"),
        Document("4", "s", "x" * 100, "python"),
        Document("5", "s", "A plain English paragraph about rivers and valleys.", "other"),
    ]
    decisions: list[str] = []
    filter_documents(
        docs, ContaminationIndex([]), config, on_decision=lambda _, why: decisions.append(why)
    )
    assert [rule_reason(doc, config) for doc in docs] == decisions
    assert decisions == ["kept", "generated", "too_short", "long_line", "unknown_domain"]


def test_min_alpha_tokens_keeps_number_runs_from_matching() -> None:
    numbers = " ".join(["1"] * 20)
    statement = "Write a function that returns the sum of all distinct positive elements in a list"
    plain = ContaminationIndex([numbers, statement])
    worded = ContaminationIndex([numbers, statement], min_alpha_tokens=3)
    assert plain.contains("input: " + numbers) and not worded.contains("input: " + numbers)
    assert worded.contains("Task: " + statement.upper() + " please")
