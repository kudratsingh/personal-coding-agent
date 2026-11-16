"""The development task-family rule on synthetic agent instances and replay rows.

Reference tests compare the port (:mod:`scratchcoder.agent.synth.dev_family`) with copies of
the functions it ports (``dev_family_ids`` and ``scan``, in ``dev_family_reference``) on the
cross product of each node's prompts and solutions. Nothing generated is executed."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from dev_family_fixtures import (
    dev_family_argv,
    meta_line,
    pool_task,
    register,
    words,
    write_build,
    write_instance,
    write_jsonl,
    write_pool,
    write_specs,
)
from dev_family_reference import dev_family_ids, scan

from scratchcoder.agent import cases
from scratchcoder.agent.synth import dev_family
from scratchcoder.agent.synth.dev_family import Instance, Node, assess, task_node
from scratchcoder.data.clusters import (
    DEFAULT_RULES,
    Item,
    UnionFind,
    canonical_key,
    components,
    find_edges,
)
from scratchcoder.data.pool import shingles
from scratchcoder.eval.tasks import Task
from tools import agent_sft_data

WORDS = [f"alpha{index}" for index in range(20)]
DEV_PROMPT = " ".join(WORDS)


def prompt(shared: int, tail: str) -> str:
    """The first ``shared`` of the dev prompt's 20 words, then distinct filler words."""
    return " ".join(WORDS[:shared] + [f"{tail}{index}" for index in range(20 - shared)])


DEV_SOLUTION = (
    "def secretfn(values):\n    total = 0\n    for value in values:\n"
    "        total += value * 3\n    return total\n"
)
SECOND_SOLUTION = "def second(x):\n    return x ** 5 + 11\n"
HELPER = "def shared_helper(z):\n    return z + 41\n"
LONELY = "def lonely(q):\n    return q * 13 - 2\n"

DEV = [
    pool_task("d0", DEV_PROMPT, DEV_SOLUTION, "dev"),
    pool_task("d1", words("zeta", 20), SECOND_SOLUTION, "dev"),
]
TRAIN = [
    # A paraphrase of a dev prompt: in a dev family already, without any synthetic node.
    pool_task("t-adjudicated", prompt(16, "pool"), "def poolfn(w):\n    return w % 9\n", "train"),
    # Shares only its program with a rejected synthetic instance.
    pool_task("t-new", words("lonelyp", 20), LONELY, "train"),
    pool_task("t-other", words("plain", 20), "def plain(s):\n    return s.upper()\n", "train"),
    # A shared name with a dev task at the flag level: a pending flag edge, kept.
    pool_task("t-named", prompt(6, "tn"), "def secretfn(v):\n    return max(v)\n", "train"),
]
COPY_MODULE = "def other(n):\n    return n\n\n\n" + DEV_SOLUTION.replace(
    "secretfn", "tally"
).replace("total", "acc").replace("value", "item")


def instance(identifier: str, task: str, reference: str, *, template: str, **extra: str) -> Node:
    item = Instance(
        id=identifier,
        template_id=template,
        family="fam",
        task=task,
        reference_target=reference,
        original_target=reference.replace("return", "return 1 +", 1),
        modules=(("toolbox.py", extra.get("module", reference)),),
    )
    return dev_family.instance_node(item, [extra.get("spec", words(f"spec{template}", 12))], [])


SUBJECTS = [
    instance("s-clear", words("clr", 15), "def fine(s):\n    return s.lower()\n", template="tc"),
    # Linked to dev only through one top-level function of its fixed module (renamed).
    instance(
        "s-copy",
        words("cp", 15),
        "def other(n):\n    return n\n",
        template="tx",
        module=COPY_MODULE,
    ),  # fmt: skip
    # Linked to dev only through the rejected instance b-rej (same helper program).
    instance("s-kept", words("kept", 15), HELPER, template="ta", spec=words("specta1", 12)),
    instance(
        "s-sibling",
        words("sib", 15),
        "def sib(a):\n    return a * 5\n",
        template="ta",
        spec=words("specta2", 12),
    ),  # fmt: skip
    instance(
        "s-flag", prompt(6, "flag"), "def secretfn(v):\n    return sorted(v)\n", template="tf"
    ),  # fmt: skip
    # Only its second prompt (the specification text) sits at the named-paraphrase level.
    instance(
        "s-named",
        words("nm", 15),
        "def secretfn(v):\n    return min(v)\n",
        template="tn",
        spec=prompt(10, "named"),
    ),  # fmt: skip
]
BRIDGES = [
    Node("synth:b-rej", (prompt(16, "rej"),), (HELPER,)),
    # A second program equal to d1's up to renaming joins t-new's family to d1.
    Node("synth:b-rej2", (words("rtwo", 15),), (LONELY, SECOND_SOLUTION.replace("x", "y"))),
]
REPLAY = [task.id for task in TRAIN]


def _assess(bridges: Sequence[Node] = BRIDGES) -> dev_family.Assessment:
    return assess(
        [task_node(task) for task in DEV],
        [task_node(task) for task in TRAIN],
        SUBJECTS,
        bridges,
        replay=REPLAY,
    )


def test_fixture_prompts_sit_where_the_rules_put_them() -> None:
    def jac(a: str, b: str) -> float:
        x, y = shingles(a, 3), shingles(b, 3)
        return len(x & y) / len(x | y)

    assert jac(prompt(16, "a"), DEV_PROMPT) >= 0.5
    assert 0.25 <= jac(prompt(10, "a"), DEV_PROMPT) < 0.5
    assert 0.10 <= jac(prompt(6, "a"), DEV_PROMPT) < 0.25
    unit = dev_family.top_level_units(COPY_MODULE)[1]
    assert canonical_key(unit) == canonical_key(DEV_SOLUTION)
    assert canonical_key(COPY_MODULE) != canonical_key(DEV_SOLUTION)


def test_each_subject_gets_the_status_and_reason_of_the_rule() -> None:
    found = {item.id: (item.status, item.reason) for item in _assess().subjects}
    assert found == {
        "synth:s-clear": ("clear", None),
        "synth:s-copy": ("excluded", "dev_family"),
        "synth:s-kept": ("excluded", "dev_family"),
        "synth:s-sibling": ("clear", None),
        "synth:s-flag": ("flagged", "dev_named_flag"),
        "synth:s-named": ("excluded", "dev_named_paraphrase"),
    }


def test_a_kept_instance_reaches_dev_only_through_a_rejected_instance() -> None:
    """Regression (a): kept synthetic -> rejected synthetic -> dev task."""
    without = {item.id: item.status for item in _assess(bridges=()).subjects}
    assert without["synth:s-kept"] == "clear"
    assert {item.id: item.status for item in _assess().subjects}["synth:s-kept"] == "excluded"


def test_a_replay_row_reaching_dev_through_a_rejected_instance_is_newly_excluded() -> None:
    """Regression (b): replay row -> rejected synthetic -> dev task. Families that held a dev
    task without synthetic nodes keep their adjudicated outcome; flag edges are reported."""
    found = _assess()
    assert found.replay_excluded == ["t-new"]
    assert found.replay_in_dev_family == ["t-adjudicated"]
    assert found.replay_named == ["t-named"]
    assert _assess(bridges=()).replay_excluded == []


def test_template_expansion_drops_every_instance_of_a_dropped_template() -> None:
    """Regression (c): s-sibling is clear, but its template holds an excluded instance."""
    prefix = len(dev_family.NAMESPACE)
    judged = [
        dev_family.Judgement(item.id[prefix:], item.status, item.reason)
        for item in _assess().subjects
    ]
    template = {"s-clear": "tc", "s-copy": "tx", "s-kept": "ta", "s-sibling": "ta"}
    template |= {"s-flag": "tf", "s-named": "tn"}
    assert dev_family.expand_templates(judged, template) == {
        "s-copy": "dev_family",
        "s-kept": "dev_family",
        "s-sibling": "template",
        "s-flag": "dev_named_flag",
        "s-named": "dev_named_paraphrase",
    }


def test_two_nodes_under_one_instance_id_are_joined() -> None:
    """Two instance sets reusing an ID: the intake joins repeated IDs."""
    other = Node("synth:s-clear", (prompt(16, "again"),), ("def unrelated(v):\n    return v\n",))
    found = assess([task_node(t) for t in DEV], [], SUBJECTS[:1], [other])
    assert [item.status for item in found.subjects] == ["excluded"]


# -- Reference: the port decides as the functions it ports do, item by item ---------------


def _tasks(items: Sequence[Item], identifier: str | None = None) -> list[Task]:
    return [pool_task(identifier or item.id, item.prompt, item.solution, "train") for item in items]


def _connected(items: Sequence[Item]) -> bool:
    groups = components(len(items), find_edges(items))
    return len(groups) == 1


def test_the_port_matches_dev_family_ids_on_the_cross_product() -> None:
    """Each subject's first item is the candidate; its other items and every bridge item
    repeat their node's ID, which ``dev_family_ids`` joins to the holder."""
    for bridges in (BRIDGES, ()):
        candidates: list[Task] = []
        range_bridges: list[Task] = []
        for node in SUBJECTS:
            items = dev_family.items_of(node, "pilot")
            candidates += _tasks(items[:1])
            range_bridges += _tasks(items[1:], node.id)
        for node in bridges:
            range_bridges += _tasks(dev_family.items_of(node, "bridge"), node.id)
        expected = dev_family_ids(candidates, DEV, TRAIN, range_bridges)
        assert expected == ({"synth:s-copy", "synth:s-kept"} if bridges else {"synth:s-copy"})
        found = assess(
            [task_node(t) for t in DEV], [task_node(t) for t in TRAIN], SUBJECTS, bridges
        )
        assert {j.id for j in found.subjects if j.reason == "dev_family"} == expected


def test_the_port_matches_the_integrity_scan_on_the_cross_product() -> None:
    """Every item of a subject is a scan subject; the subject takes its worst status."""
    rank = {"clear": 0, "flagged": 1, "excluded": 2}
    for bridges in (BRIDGES, ()):
        subject_items = {node.id: dev_family.items_of(node, "s") for node in SUBJECTS}
        assert all(_connected(items) for items in subject_items.values())
        results, _ = scan(
            [Item(t.id, "dev", t.source, t.prompt, t.solution) for t in DEV],
            {"s": [item for items in subject_items.values() for item in items]},
            bridges={
                "train": [Item(t.id, "train", t.source, t.prompt, t.solution) for t in TRAIN],
                "b": [item for node in bridges for item in dev_family.items_of(node, "b")],
            },
        )
        worst: dict[str, str] = {}
        for result in results:
            node = result.id.split("#")[0]
            if rank[result.status] >= rank[worst.get(node, "clear")]:
                worst[node] = result.status
        assert set(worst.values()) == {"clear", "flagged", "excluded"}
        found = assess(
            [task_node(t) for t in DEV], [task_node(t) for t in TRAIN], SUBJECTS, bridges
        )
        assert {j.id: j.status for j in found.subjects} == worst


def _partition(size: int, links: Sequence[tuple[int, int]]) -> set[frozenset[int]]:
    joined = UnionFind(size)
    for a, b in links:
        joined.union(a, b)
    groups: dict[int, set[int]] = {}
    for index in range(size):
        groups.setdefault(joined.find(index), set()).add(index)
    return {frozenset(group) for group in groups.values()}


PAIRWISE = [
    Item("MBPP/3", "train", "s", words("u", 10), "def f(a):\n    return a\n"),
    Item("mbpp/3", "train", "s", words("v", 10), None),  # equal upstream ID ignoring case
    Item("p1", "train", "s", words("w", 10), "def g(b):\n    '''Doc.'''\n    return b + 1\n"),
    Item("p2", "train", "s", words("x", 10), "def g(b):\n    return b + 1\n"),  # docstring
    Item("p3", "train", "s", words("y", 10), "def h(c):\n    return c + 1\n"),  # renamed
    Item("p4", "train", "s", words("w", 9) + " other", "def k(d):\n    return d * 9\n"),
    Item("p5", "train", "s", "two words", "def broken(:"),
    Item("p6", "train", "s", "two words", "def m(e):\n    return e - 2\n"),  # same short prompt
    Item("p7", "train", "s", words("z", 10), "def n(f):\n    return f // 7\n"),
    Item("p8", "train", "s", words("q", 10), "def q(g):\n    return g % 3\n"),
]


def test_family_links_give_the_components_of_find_edges() -> None:
    nodes = [Node(i.id, (i.prompt,), (i.solution,) if i.solution else ()) for i in PAIRWISE]
    expected = {frozenset(group) for group in components(len(PAIRWISE), find_edges(PAIRWISE))}
    assert _partition(len(nodes), dev_family.family_links(nodes)) == expected
    assert len(expected) == 5  # Every union signal fires; p7 and p8 stay alone.


def test_links_among_the_first_nodes_alone_give_their_own_families() -> None:
    nodes = [Node(i.id, (i.prompt,), (i.solution,) if i.solution else ()) for i in PAIRWISE]
    added = [Node("synth:x", (words("u", 10), words("q", 10)), ("def n(f):\n    return f // 7\n",))]
    graph = dev_family.FamilyGraph(nodes, added)
    base = {frozenset(g) for g in components(len(PAIRWISE), find_edges(PAIRWISE))}
    groups: dict[int, set[int]] = {}
    for index in range(len(nodes)):
        groups.setdefault(graph.base.find(index), set()).add(index)
    assert {frozenset(group) for group in groups.values()} == base
    # The added node joins three base families that stay apart without it.
    assert graph.full.find(0) == graph.full.find(8) == graph.full.find(9)
    assert len({graph.base.find(0), graph.base.find(8), graph.base.find(9)}) == 3


def test_named_scores_take_the_strongest_prompt_of_a_shared_name() -> None:
    target = [Node("d", (DEV_PROMPT,), (DEV_SOLUTION,))]
    weak = Node("a", (prompt(6, "w"),), ("def secretfn(v):\n    return v\n",))
    strong = Node("b", (prompt(6, "w"), prompt(10, "s")), ("def secretfn(v):\n    return v\n",))
    none = Node("c", (DEV_PROMPT,), ("def elsewhere(v):\n    return v\n",))
    low, high, missing = dev_family.named_scores(target, [weak, strong, none])
    assert low is not None and DEFAULT_RULES.named_flag <= low < DEFAULT_RULES.named_paraphrase
    assert high is not None and high >= DEFAULT_RULES.named_paraphrase
    assert missing is None


# -- The ``dev-family`` subcommand and the instance reader --------------------------------


def _setup(tmp_path: Path) -> dict[str, Path]:
    """Two dev tasks, train tasks, one judged build (with a rejected instance) and one
    superseded set; rows for every accepted instance; a replay index."""
    pool = tmp_path / "pool"
    dev = write_pool(pool / "dev.jsonl", DEV)
    train = write_pool(pool / "train.jsonl", TRAIN)
    judged = tmp_path / "instances-a"
    write_instance(judged, "k-clear", template="tc", task=words("clr", 15),
                   reference="def fine(s):\n    return s.lower()\n")  # fmt: skip
    write_instance(judged, "k-kept", template="ta", task=words("kept", 15), reference=HELPER)
    write_instance(judged, "k-sibling", template="ta", task=words("sib", 15),
                   reference="def sib(a):\n    return a * 5\n")  # fmt: skip
    write_instance(judged, "k-flag", template="tf", task=prompt(6, "flag"),
                   reference="def secretfn(v):\n    return sorted(v)\n")  # fmt: skip
    write_instance(judged, "k-flag-sib", template="tf", task=words("fsib", 15),
                   reference="def fsib(a):\n    return a - 5\n")  # fmt: skip
    write_instance(judged, "k-rejected", template="tr", task=prompt(16, "rej"), reference=HELPER)
    register(judged)
    accepted = ("k-clear", "k-kept", "k-sibling", "k-flag", "k-flag-sib")
    build = write_build(
        tmp_path / "build-a", {**dict.fromkeys(accepted, True), "k-rejected": False}
    )
    old = tmp_path / "instances-old"
    write_instance(old, "o-bridge", template="tr", task=words("rtwo", 15), reference=LONELY,
                   others={"extra.py": SECOND_SOLUTION.replace("x", "y")})  # fmt: skip
    register(old)
    specs = write_specs(
        tmp_path / "specs.json",
        {key: words(f"spec{key}", 12) for key in ("tc", "ta", "tf", "tr")},
    )
    rows = []
    for name, template in (("k-clear", "tc"), ("k-kept", "ta"), ("k-flag-sib", "tf")):
        for kind in ("edit", "repair-1", "repair-2"):
            rows.append(meta_line(name, template, kind, words(f"{name}{kind}", 12), "x = 1"))
    rows.append(meta_line("k-flag", "tf", "edit", words("flagrow", 12), "x = 2"))
    rows.append(meta_line("k-sibling", "ta", "edit", words("sibrow", 12), "x = 3"))
    meta = write_jsonl(tmp_path / "rows-meta.jsonl", rows)
    decoded = write_jsonl(
        tmp_path / "decoded-index.jsonl",
        [{"id": task.id, "kind": "code", "user": [], "assistant": []} for task in TRAIN]
        + [{"id": "lang-1", "kind": "language", "user": [], "assistant": []}],
    )
    return {
        "dev": dev,
        "train": train,
        "judged": judged,
        "build": build,
        "old": old,
        "specs": specs,
        "meta": meta,
        "decoded": decoded,
    }


def _argv(paths: dict[str, Path], out: Path, *, bridges: bool = True) -> list[str]:
    return dev_family_argv(
        paths["dev"],
        paths["train"],
        [(paths["judged"], paths["build"])],
        [paths["specs"]],
        paths["meta"],
        paths["decoded"],
        out,
        bridges=[paths["old"]] if bridges else [],
    )


def test_the_subcommand_writes_a_bound_status_only_result(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    out = tmp_path / "dev-family.json"
    assert agent_sft_data.main(_argv(paths, out)) == 0
    result = json.loads(out.read_text())
    assert result["dev_family"] == "task-clusters-v1"
    assert result["rules"] == {
        "version": "task-clusters-v1",
        "width": 3,
        "paraphrase": 0.5,
        "named_paraphrase": 0.25,
        "named_flag": 0.1,
    }
    agent = result["agent"]
    assert agent["input_kind"] == "rows_meta"
    assert agent["input_sha256"] == dev_family.sha_file(paths["meta"])
    # k-sibling shares its template's specification text with k-kept, so it is in the same
    # family; k-flag-sib is clear on its own and dropped by template expansion (c).
    assert agent["instances"]["dropped"] == {
        "k-kept": "dev_family",
        "k-sibling": "dev_family",
        "k-flag": "dev_named_flag",  # Regression (d): a flag drops the instance.
        "k-flag-sib": "template",
    }
    every = ("edit", "repair-1", "repair-2")
    assert agent["rejected"] == sorted(
        [f"agent-sft1/{name}/{kind}" for name in ("k-kept", "k-flag-sib") for kind in every]
        + ["agent-sft1/k-flag/edit", "agent-sft1/k-sibling/edit"]
    )
    assert agent["rows_dropped_instance_level"] == 5
    assert agent["rows_dropped_by_reason"]["template"] == 3
    assert agent["instances"]["by_status_count"] == {"excluded": 2, "flagged": 1, "clear": 2}
    assert agent["surviving"]["rows"] == 3 and agent["surviving"]["templates"] == 1
    replay = result["replay"]
    assert replay["input_kind"] == "decoded" and replay["kinds"] == ["code"]
    assert replay["rejected"] == ["t-new"]  # Through the superseded set's instance.
    assert replay["kept_in_dev_family_without_synthetic"] == 1
    assert replay["kept_direct_named_edge_without_synthetic"] == 1
    assert replay["not_judged_other_kinds"] == 1
    assert result["pool"]["dev_sha256"] == dev_family.sha_file(paths["dev"])
    assert result["synthetic"]["judged"][0]["rejected"] == 1
    assert result["synthetic"]["bridges"][0]["count"] == 1
    # Status-only: no prompt, solution or dev ID appears in the result.
    text = out.read_text()
    assert "alpha0" not in text and "secretfn" not in text and '"d0"' not in text
    # Without the superseded set, the replay row has no chain to dev.
    again = tmp_path / "without.json"
    assert agent_sft_data.main(_argv(paths, again, bridges=False)) == 0
    assert json.loads(again.read_text())["replay"]["rejected"] == []
    assert agent_sft_data.main(_argv(paths, out)) == 2  # The output exists.


def test_the_subcommand_refuses_unresolved_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    # A sealed split in place of the development split is refused before anything is read.
    sealed = write_pool(tmp_path / "sealed" / "test.jsonl", DEV)
    assert agent_sft_data.main(_argv({**paths, "dev": sealed}, tmp_path / "a.json")) == 2
    # Rows of an instance no build accepted.
    rows = paths["meta"].read_text().replace('"k-clear"', '"k-rejected"')
    bad = tmp_path / "bad-meta.jsonl"
    bad.write_text(rows.replace('"tc"', '"tr"'))
    assert agent_sft_data.main(_argv({**paths, "meta": bad}, tmp_path / "b.json")) == 2
    assert "which no build accepted" in capsys.readouterr().err
    # A replay code row that is not a pool training task cannot be judged.
    unknown = write_jsonl(tmp_path / "unknown-index.jsonl", [{"id": "elsewhere", "kind": "code"}])
    assert agent_sft_data.main(_argv({**paths, "decoded": unknown}, tmp_path / "c.json")) == 2
    assert "not pool training tasks" in capsys.readouterr().err
    # A template without specification text.
    specs = write_specs(tmp_path / "few-specs.json", {"tc": words("spectc", 12)})
    assert agent_sft_data.main(_argv({**paths, "specs": specs}, tmp_path / "d.json")) == 2
    assert "no specification text" in capsys.readouterr().err
    # A registered file changed after registration.
    (paths["judged"] / "k-clear" / "synth.json").write_text("{}")
    assert agent_sft_data.main(_argv(paths, tmp_path / "e.json")) == 2
    assert "changed since registration" in capsys.readouterr().err
    assert not any((tmp_path / f"{name}.json").exists() for name in "abcde")


def test_the_reader_takes_generator_output_as_text(tmp_path: Path) -> None:
    """Instances from ``generate`` (static; nothing runs) and the ``specs`` export: a dev task
    whose reference equals one instance's target excludes that instance and, by template
    expansion, every other instance of its template."""
    from synth_fixtures import Chunks, families_json

    def loader(path: Path) -> tuple[Chunks, Chunks]:
        encoder = Chunks()
        return encoder, encoder

    (tmp_path / "tokenizer.json").write_text("{}")
    out = tmp_path / "generated"
    argv = ["generate", "--out", str(out), "--tokenizer", str(tmp_path / "tokenizer.json")]
    argv += ["--families", str(families_json(tmp_path / "families.json"))]
    assert agent_sft_data.main([*argv, "--templates", "stock1", "grid3"], loader=loader) == 0
    specs = tmp_path / "specs.json"
    assert agent_sft_data.main(["specs", "--instances", str(out), "--out", str(specs)]) == 0
    found, registry = dev_family.read_instances(out)
    assert registry == cases.registry_sha256(out)
    by_template = {item.template_id for item in found}
    assert by_template == {"stock1", "grid3"}
    first = next(item for item in found if item.template_id == "stock1")
    assert any(first.reference_target.split("(")[0] in text for _, text in first.modules)
    node = dev_family.instance_node(first, ["a spec text here"], [])
    assert first.reference_target in node.solutions and first.task in node.prompts
    dev = write_pool(
        tmp_path / "pool" / "dev.jsonl",
        [pool_task("d0", words("unrelated", 20), first.reference_target, "dev")],
    )
    train = write_pool(tmp_path / "pool" / "train.jsonl", TRAIN)
    build = write_build(tmp_path / "build", {item.id: True for item in found})
    meta = write_jsonl(
        tmp_path / "rows-meta.jsonl",
        [meta_line(item.id, item.template_id, "edit", words("row", 12), "x") for item in found],
    )
    decoded = write_jsonl(tmp_path / "index.jsonl", [{"id": "t-other", "kind": "code"}])
    result_path = tmp_path / "dev-family.json"
    argv = dev_family_argv(dev, train, [(out, build)], [specs], meta, decoded, result_path)
    assert agent_sft_data.main(argv) == 0
    dropped = json.loads(result_path.read_text())["agent"]["instances"]["dropped"]
    assert dropped[first.id] == "dev_family"
    stock = {item.id for item in found if item.template_id == "stock1"}
    assert set(dropped) >= stock
    assert {dropped[i] for i in stock - {first.id}} <= {"template", "dev_family"}
