"""The development task-family rule on synthetic agent instances and replay rows.

Reference tests compare the port (:mod:`scratchcoder.agent.synth.dev_family`) with copies of
the functions it ports (``dev_family_ids`` and ``scan``, in ``dev_family_reference``) on the
cross product of each node's prompts and solutions. Nothing generated is executed."""

from __future__ import annotations

import json
import string
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from dev_family_fixtures import (
    case_sha,
    dev_family_argv,
    meta_line,
    phrased,
    pool_task,
    register,
    words,
    write_build,
    write_flag_manifest,
    write_instance,
    write_jsonl,
    write_pool,
    write_replay_record,
    write_specs,
)
from dev_family_reference import dev_family_ids, scan

from scratchcoder.agent import cases
from scratchcoder.agent import prompt as harness_prompt
from scratchcoder.agent.synth import dev_family, instances
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


def instance(identifier: str, text: str, reference: str, *, template: str, **extra: str) -> Node:
    """An instance whose template's specification texts are ``text`` and, when given,
    ``spec``; its generated task statement names nothing else."""
    item = Instance(
        id=identifier,
        template_id=template,
        family="fam",
        task=phrased(words(f"{identifier}-st", 4)),
        reference_target=reference,
        original_target=reference.replace("return", "return 1 +", 1),
        modules=(("toolbox.py", extra.get("module", reference)),),
    )
    specs = [text, *([extra["spec"]] if "spec" in extra else [])]
    return dev_family.instance_node(item, specs, [])


SUBJECTS = [
    instance("s-clear", words("clr", 15), "def fine(s):\n    return s.lower()\n", template="tc"),
    # Its reference target is the dev solution up to renaming: one family with it.
    instance("s-direct", words("dr", 15), COPY_MODULE.split("\n\n\n")[1], template="td"),
    # One top-level function of its repository (shown, not trained) is the dev solution up to
    # renaming: a prompt exposure, not a family link.
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
        "synth:s-direct": ("excluded", "dev_family"),
        "synth:s-copy": ("excluded", "prompt_code_exposure"),
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


def test_a_replay_row_joined_to_a_further_dev_task_is_newly_excluded() -> None:
    """A replay row already in d0's family (kept by the replay stage) is joined to d1 through
    a synthetic node: that link was never adjudicated, so the row is dropped."""
    program = "def rprog(w):\n    return w % 7 + 3\n"
    row = pool_task("r", prompt(16, "rp"), program, "train")
    joiner = Node("synth:j", (words("jn", 15),), (program, SECOND_SOLUTION.replace("x", "y")))
    found = assess(
        [task_node(t) for t in DEV], [task_node(row)], [], [joiner], replay=["r"]
    )  # fmt: skip
    assert found.replay_excluded == ["r"] and found.replay_new_dev_link == ["r"]
    assert found.replay_new_chain == [] and found.replay_in_dev_family == []
    alone = assess([task_node(t) for t in DEV], [task_node(row)], [], [], replay=["r"])
    assert alone.replay_excluded == [] and alone.replay_in_dev_family == ["r"]
    # The regression (b) row is a new chain, not a new link.
    assert _assess().replay_new_chain == ["t-new"] and _assess().replay_new_dev_link == []


def test_a_bridge_with_a_named_edge_to_dev_drops_its_template() -> None:
    """A rejected instance that defines a dev task's function name at the named-paraphrase
    level, while its accepted sibling has neither signal: the bridge is not judged, but its
    template is dropped."""
    sibling = Node("synth:a", (words("sa", 15),), ("def sib(a):\n    return a * 5\n",))
    bridge = Node("synth:b", (prompt(10, "br"),), ("def secretfn(v):\n    return min(v)\n",))
    weak = Node("synth:w", (prompt(6, "wk"),), ("def secretfn(v):\n    return max(v)\n",))
    found = assess([task_node(t) for t in DEV], [], [sibling], [bridge, weak])
    assert [(j.id, j.status) for j in found.subjects] == [("synth:a", "clear")]
    assert found.bridges_hit == {"synth:b": "dev_named_paraphrase", "synth:w": "dev_named_flag"}
    judged = [dev_family.Judgement("a", "clear", None)]
    assert dev_family.expand_templates(judged, {"a": "T"}) == {}
    assert dev_family.expand_templates(judged, {"a": "T"}, ["T"]) == {"a": "template"}


def test_template_expansion_drops_every_instance_of_a_dropped_template() -> None:
    """Regression (c): s-sibling is clear, but its template holds an excluded instance."""
    prefix = len(dev_family.NAMESPACE)
    judged = [
        dev_family.Judgement(item.id[prefix:], item.status, item.reason)
        for item in _assess().subjects
    ]
    template = {"s-clear": "tc", "s-copy": "tx", "s-kept": "ta", "s-sibling": "ta"}
    template |= {"s-flag": "tf", "s-named": "tn", "s-direct": "td"}
    assert dev_family.expand_templates(judged, template) == {
        "s-direct": "dev_family",
        "s-copy": "prompt_code_exposure",
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
        assert expected == ({"synth:s-direct", "synth:s-kept"} if bridges else {"synth:s-direct"})
        found = assess(
            [task_node(t) for t in DEV], [task_node(t) for t in TRAIN], SUBJECTS, bridges
        )
        assert {j.id for j in found.subjects if j.reason == "dev_family"} == expected


def test_the_port_matches_the_integrity_scan_on_the_cross_product() -> None:
    """Every family item of a subject is a scan subject; the subject takes its worst status.
    Exposure items are direct-only checks the scan does not have, so they are left out."""
    rank = {"clear": 0, "flagged": 1, "excluded": 2}
    family_only = [replace(node, shown_code=(), shown_text=()) for node in SUBJECTS]
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
            [task_node(t) for t in DEV], [task_node(t) for t in TRAIN], family_only, bridges
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
    write_instance(judged, "k-flag", template="tf", task=words("flg", 15),
                   reference="def secretfn(v):\n    return sorted(v)\n")  # fmt: skip
    write_instance(judged, "k-flag-sib", template="tf", task=words("fsib", 15),
                   reference="def fsib(a):\n    return a - 5\n")  # fmt: skip
    write_instance(judged, "k-rejected", template="tr", task=words("rej", 15), reference=HELPER)
    register(judged)
    accepted = ("k-clear", "k-kept", "k-sibling", "k-flag", "k-flag-sib")
    build = write_build(
        tmp_path / "build-a", {**dict.fromkeys(accepted, True), "k-rejected": False}, judged
    )
    old = tmp_path / "instances-old"
    # A superseded instance whose specification paraphrases d0 and whose target is t-new's
    # program: a valid family relationship it still bridges. Its repository also shows d1's
    # solution (renamed): an exposure hit of the bridge, never a family link.
    write_instance(old, "o-bridge", template="to", task=words("ob", 15), reference=LONELY,
                   others={"extra.py": SECOND_SOLUTION.replace("x", "y")})  # fmt: skip
    register(old)
    # Specification texts carry the descriptions: tr's paraphrases d0 (k-rejected joins
    # d0's family), tf's sits at the flag level beside k-flag's dev function name, and to's
    # paraphrases d0 (o-bridge's valid relationship).
    texts = {key: words(f"spec{key}", 12) for key in ("tc", "ta")}
    texts |= {"tf": prompt(6, "flag"), "tr": prompt(16, "rej"), "to": prompt(16, "ob")}
    specs = write_specs(tmp_path / "specs.json", texts)
    rows = []
    for name, template in (("k-clear", "tc"), ("k-kept", "ta"), ("k-flag-sib", "tf")):
        for kind in ("edit", "repair-1", "repair-2"):
            rows.append(
                meta_line(
                    name,
                    template,
                    kind,
                    words(f"{name}{kind}", 12),
                    "x = 1",
                    spec_sha256=case_sha(judged, name),
                )
            )
    for name, template, text, reply in (
        ("k-flag", "tf", words("flagrow", 12), "x = 2"),
        ("k-sibling", "ta", words("sibrow", 12), "x = 3"),
    ):
        rows.append(
            meta_line(name, template, "edit", text, reply, spec_sha256=case_sha(judged, name))
        )
    meta = write_jsonl(tmp_path / "rows-meta.jsonl", rows)
    decoded = write_jsonl(
        tmp_path / "decoded-index.jsonl",
        [{"id": task.id, "kind": "code", "user": [], "assistant": []} for task in TRAIN]
        + [{"id": "lang-1", "kind": "language", "user": [], "assistant": []}],
    )
    # Pending flag edges: t-named's is unresolved; t-other's has a not-equivalent decision on
    # exactly its pair (the unclear decision on t-named's pair resolves nothing).
    flags = write_flag_manifest(
        tmp_path / "clusters.json",
        [("t-named", "d0", "pending"), ("t-other", "d1", "pending")],
    )
    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            {"train": "t-other", "dev": "d1", "final": "not_equivalent"},
            {"train": "t-named", "dev": "d0", "final": "unclear"},
            {"train": "t-named", "dev": "d1", "final": "not_equivalent"},
        ],
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
        "flags": flags,
        "decisions": decisions,
        # The replay stage counts both flag-edge rows (t-named and t-other).
        "stage": write_replay_record(tmp_path / "replay-record.json", 2),
    }


def _argv(
    paths: dict[str, Path],
    out: Path,
    *,
    bridges: bool = True,
    more_bridges: Sequence[Path] = (),
    stage: Path | None = None,
) -> list[str]:
    argv = dev_family_argv(
        paths["dev"],
        paths["train"],
        [(paths["judged"], paths["build"])],
        [paths["specs"]],
        paths["meta"],
        paths["decoded"],
        out,
        bridges=[*([paths["old"]] if bridges else []), *more_bridges],
        flags=paths["flags"],
        stage=stage or paths["stage"],
    )
    return [*argv, "--flag-decisions", str(paths["decisions"])]


def test_the_subcommand_writes_a_bound_status_only_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    out = tmp_path / "dev-family.json"
    stage = write_replay_record(tmp_path / "stage-more.json", 2, rows_on_pending_flags=7)
    assert agent_sft_data.main(_argv(paths, out, stage=stage)) == 0
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
    # t-new through the superseded set's instance; t-named on an unresolved flag edge.
    assert replay["rejected"] == ["t-named", "t-new"]
    assert replay["rejected_by_reason"] == {
        "dev_family_new_chain": 1,
        "dev_family_new_link": 0,
        "pending_flag_unresolved": 1,
    }
    pending = replay["pending_flags"]
    assert pending["applied"] and pending["edges_to_dev"] == 2
    assert pending["candidate_ids"] == ["t-named", "t-other"]
    assert pending["kept_resolved_decision_lines"] == {"t-other": [1]}
    assert pending["excluded"] == 1 and pending["candidates_by_kind"] == {"code": 2}
    assert pending["decisions_sha256"] == dev_family.sha_file(paths["decisions"])
    assert pending["manifest_sha256"] == dev_family.sha_file(paths["flags"])
    assert replay["rejected_new_chain"] == 1 and replay["rejected_new_dev_link"] == 0
    assert replay["kept_in_dev_family_without_synthetic"] == 1
    assert replay["kept_direct_named_edge_without_synthetic"] == 1
    assert replay["not_judged_other_kinds"] == 1
    # The replay stage's own pending-flag figure, with its record's hash, sits beside it.
    assert replay["replay_stage"] == {
        "record": str(stage),
        "record_sha256": dev_family.sha_file(stage),
        "format": "replay-stage-v1",
        "pending_flag_rows_reported": {
            "R2.rows_on_pending_flags": 7,
            f"R2.{PENDING_KEY}": 2,
        },
    }
    # k-kept is excluded only through the rejected instance; k-sibling joins it by template
    # text, so both are reached only through bridges.
    assert agent["instances"]["only_through_bridges"] == {
        "judged_not_clear": ["k-kept", "k-sibling"],
        "dropped": ["k-kept", "k-sibling"],
    }
    # The superseded instance shows a dev solution: its (subject-less) template is dropped.
    # The rejected instance's repository shows HELPER, which k-kept (in d0's family) trains;
    # never its own target alone.
    assert agent["instances"]["bridges_with_direct_hit"] == {
        "synth:k-rejected": "prompt_code_exposure",
        "synth:o-bridge": "prompt_code_exposure",
    }
    assert agent["instances"]["templates_dropped_through_bridge_hits"] == ["to", "tr"]
    assert result["synthetic"]["item_mapping"]["version"] == dev_family.MAPPING
    assert agent["instances"]["owner_drops"]["templates"] == []
    # The earlier mapping made links this one does not; every one has a category.
    changes = result["link_changes"]
    evidence = json.loads(Path(changes["evidence"]).read_text())
    assert changes["evidence_sha256"] == dev_family.sha_file(Path(changes["evidence"]))
    assert changes["removed"]["links"] > 0
    assert set(changes["removed"]["by_category"]) <= set(dev_family.CATEGORIES.values())
    assert "repo_module_as_solution" in changes["removed"]["by_category"]
    assert evidence["removed_links"]["summary"] == changes["removed"]
    assert agent["surviving"]["rows_by_kind"] == {"edit": 1, "repair": 2}
    assert result["synthetic"]["judged"][0]["build_record_sha256"] == dev_family.sha_file(
        paths["build"] / "record.json"
    )
    assert result["pool"]["dev_sha256"] == dev_family.sha_file(paths["dev"])
    assert result["synthetic"]["judged"][0]["rejected"] == 1
    assert result["synthetic"]["bridges"][0]["count"] == 1
    # Status-only: no prompt, solution or dev ID appears in the result.
    text = out.read_text()
    assert "alpha0" not in text and "secretfn" not in text and '"d0"' not in text
    # Without the superseded set, the replay row has no chain to dev.
    capsys.readouterr()
    again = tmp_path / "without.json"
    assert agent_sft_data.main(_argv(paths, again, bridges=False)) == 0
    assert json.loads(again.read_text())["replay"]["rejected"] == ["t-named"]
    assert "warning: no --bridge-instances" in capsys.readouterr().err
    assert agent_sft_data.main(_argv(paths, out)) == 2  # The output exists.


def test_a_bridge_named_edge_drops_its_template_in_the_subcommand(tmp_path: Path) -> None:
    """A superseded instance of k-clear's template defines a dev function name at the
    named-paraphrase level; k-clear has no signal of its own and is dropped (``template``)."""
    paths = _setup(tmp_path)
    extra = tmp_path / "instances-named"
    # Its target's docstring (a family prompt) sits at the named-paraphrase level.
    documented = f'def secretfn(v):\n    """{prompt(10, "nb")}"""\n    return min(v)\n'
    write_instance(extra, "n-bridge", template="tc", task=words("nb", 15), reference=documented)
    register(extra)
    out = tmp_path / "dev-family.json"
    assert agent_sft_data.main(_argv(paths, out, more_bridges=[extra])) == 0
    instances = json.loads(out.read_text())["agent"]["instances"]
    assert instances["bridges_with_direct_hit"]["synth:n-bridge"] == "dev_named_paraphrase"
    assert instances["templates_dropped_through_bridge_hits"] == ["tc", "to", "tr"]
    assert instances["dropped"]["k-clear"] == "template"
    assert "k-clear" in instances["only_through_bridges"]["dropped"]
    assert "k-clear" not in instances["only_through_bridges"]["judged_not_clear"]


def test_a_judged_pair_must_belong_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A superseded set that reuses the instance IDs cannot stand in for the set the rows
    were built from: the build record names the registry, and each row names its instance's
    registered case.json."""
    paths = _setup(tmp_path)
    superseded = tmp_path / "instances-superseded"
    templates = {"k-clear": "tc", "k-kept": "ta", "k-sibling": "ta", "k-flag": "tf"}
    templates |= {"k-flag-sib": "tf", "k-rejected": "tr"}
    for name, template in templates.items():
        write_instance(superseded, name, template=template, task=words(f"old{name}", 15),
                       reference="def fine(s):\n    return s.lower()\n")  # fmt: skip
    register(superseded)
    swapped = {**paths, "judged": superseded}
    assert agent_sft_data.main(_argv(swapped, tmp_path / "a.json")) == 2
    assert "was not built from" in capsys.readouterr().err
    # A build record that names the superseded registry: the rows still name other instances.
    built = (paths["build"] / "instances.jsonl").read_text().splitlines()
    outcomes = {line["instance"]: line["accepted"] for line in map(json.loads, built)}
    relabelled = write_build(tmp_path / "build-relabelled", outcomes, superseded)
    swapped = {**swapped, "build": relabelled}
    assert agent_sft_data.main(_argv(swapped, tmp_path / "b.json")) == 2
    assert "were not built from the judged instance" in capsys.readouterr().err
    # A row without the instance's case.json hash.
    lines = paths["meta"].read_text().splitlines()
    first = json.loads(lines[0])
    del first["spec_sha256"]
    bare = tmp_path / "bare-meta.jsonl"
    bare.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    assert agent_sft_data.main(_argv({**paths, "meta": bare}, tmp_path / "c.json")) == 2
    assert "were not built from the judged instance" in capsys.readouterr().err
    assert not any((tmp_path / f"{name}.json").exists() for name in "abc")


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
    # Family items: the spec, the target's docstrings and the reference target alone; the
    # generated task statement is exposure text.
    assert node.solutions == (first.reference_target,)
    assert node.prompts == ("a spec text here", *dev_family.docstrings(first.reference_target))
    assert first.task in node.shown_text
    # Exposure items: the original target, every wrong attempt, both repositories' modules.
    assert first.original_target in node.shown_code and first.mutants
    assert set(first.mutants) <= set(node.shown_code)
    assert {text for _, text in first.original_modules} <= set(node.shown_code)
    assert dict(first.names)["restock"] == first.reference_target[4:].split("(")[0]
    dev = write_pool(
        tmp_path / "pool" / "dev.jsonl",
        [pool_task("d0", words("unrelated", 20), first.reference_target, "dev")],
    )
    train = write_pool(tmp_path / "pool" / "train.jsonl", TRAIN)
    build = write_build(tmp_path / "build", {item.id: True for item in found}, out)
    meta = write_jsonl(
        tmp_path / "rows-meta.jsonl",
        [
            meta_line(
                item.id,
                item.template_id,
                "edit",
                words("row", 12),
                "x",
                spec_sha256=item.spec_sha256,
            )
            for item in found
        ],
    )
    decoded = write_jsonl(tmp_path / "index.jsonl", [{"id": "t-other", "kind": "code"}])
    result_path = tmp_path / "dev-family.json"
    flags = write_flag_manifest(tmp_path / "clusters.json")
    stage = write_replay_record(tmp_path / "replay-record.json", 0)
    argv = dev_family_argv(
        dev, train, [(out, build)], [specs], meta, decoded, result_path, flags=flags, stage=stage
    )
    assert agent_sft_data.main(argv) == 0
    dropped = json.loads(result_path.read_text())["agent"]["instances"]["dropped"]
    assert dropped[first.id] == "dev_family"
    stock = {item.id for item in found if item.template_id == "stock1"}
    assert set(dropped) >= stock
    assert {dropped[i] for i in stock - {first.id}} <= {"template", "dev_family"}


# -- The item mapping: family items, exposure items and scaffolding ----------------------

BOUNDED = "def bounded(v, low, high):\n    return min(max(v, low), high)\n"
"""A generic helper two templates' repositories both hold (shared scaffolding)."""


def _repo_instance(
    identifier: str,
    template: str,
    reference: str,
    *,
    original: str | None = None,
    helper: str = BOUNDED,
    mutants: Sequence[str] = (),
) -> Instance:
    """An instance whose one module holds the target and ``helper``; the target's function
    name and the helper's are its recorded names (roles ``target`` and ``helper``)."""
    original = original or reference.replace("return", "return 1 +", 1)
    target_name = reference[4:].split("(")[0]
    helper_name = helper[4:].split("(")[0]
    return Instance(
        id=identifier,
        template_id=template,
        family="fam",
        task=phrased(words(f"{identifier}task", 15)),
        reference_target=reference,
        original_target=original,
        modules=(("lib.py", reference + "\n\n" + helper),),
        original_modules=(("lib.py", original + "\n\n" + helper),),
        mutants=tuple(mutants),
        names=(("target", target_name), ("helper", helper_name)),
    )


ROLES = {
    template: {"target": "target_unit", "helper": "shared_unit"}
    for template in ("ta", "tb", "tc", "te")
}
RENAMED_DEV = DEV_SOLUTION.replace("secretfn", "tally").replace("total", "acc")


def _spec(template: str) -> list[str]:
    return [words(f"spec{template}", 12)]


def test_shared_scaffolding_is_not_a_family_link() -> None:
    """Owner regression (i), a false family link: templates ta and tb share only a generic
    helper. ta's target is the dev solution (renamed), so ta is in d0's family; tb reaches
    d0 only through the helper. The earlier mapping joined tb to that family through the
    helper and excluded it; under this mapping the helper is shown code, checked directly
    (it is no dev solution), and tb is clear. The evidence names the removed link's
    category, its template pair and the helper's name."""
    first = _repo_instance("a", "ta", RENAMED_DEV)
    second = _repo_instance("b", "tb", "def flip(s):\n    return s[::-1]\n")
    dev = [task_node(t) for t in DEV]
    current = [
        dev_family.instance_node(item, _spec(item.template_id), []) for item in (first, second)
    ]
    found = {j.id: j.reason for j in assess(dev, [], current).subjects}
    assert found == {"synth:a": "dev_family", "synth:b": None}
    legacy = [
        dev_family.legacy_items(item, _spec(item.template_id), [], ROLES)
        for item in (first, second)
    ]
    before = {j.id: j.reason for j in assess(dev, [], [item.node() for item in legacy]).subjects}
    assert before == {"synth:a": "dev_family", "synth:b": "dev_family"}
    base = [dev_family.pool_labeled(t) for t in DEV]
    new = [dev_family.current_items(item, _spec(item.template_id)) for item in (first, second)]
    changes = dev_family.link_changes([*base, *legacy], [*base, *new], len(base))
    removed = changes["removed"]
    assert removed["summary"]["by_category"]["shared_scaffolding_code"] == 1
    assert removed["summary"]["by_pair_kind"]["cross_template"] == 1
    cross = [row for row in removed["rows"] if row["groups"] == ["template:ta", "template:tb"]]
    assert cross and "bounded" in cross[0]["defined_names"]
    assert cross[0]["pairs_by_category"]["shared_scaffolding_code"] == 1
    assert changes["added"]["summary"]["links"] == 0


def test_a_dev_solution_shown_in_a_prompt_is_excluded() -> None:
    """Owner regression (ii), a real dev-solution exposure inside a prompt: instance c's
    original (buggy) target is the dev solution up to renaming, so its rows show it; it is
    excluded (``prompt_code_exposure``) though no family item links it, and its template
    sibling c2 is dropped with it. Shown code is checked against pool tasks in a dev family
    too (e's helper is t-adjudicated's program) and inside row user texts (f's snippet)."""
    fixed = RENAMED_DEV.replace("* 3", "* 4")
    shown = _repo_instance("c", "tc", fixed, original=RENAMED_DEV)
    sibling = _repo_instance("c2", "tc", "def other_fn(s):\n    return s.strip()\n")
    pool_copy = _repo_instance(
        "e",
        "te",
        "def lower_fn(s):\n    return s.lower()\n",
        helper="def m(w):\n    return w % 9\n",
    )
    snippet = (
        f"{phrased('x')}\n\nMAP\nlib.py: tally\n\nSNIPPET lib.py::tally\n{RENAMED_DEV}\n"
        f"RESULT run_tests: failed (1 of 2 tests)\nFAIL test_x\n\n{harness_prompt.INSTRUCTION}"
    )
    in_text = _repo_instance("f", "tf2", "def keep_fn(s):\n    return s.title()\n")
    nodes = [
        dev_family.instance_node(shown, _spec("tc"), []),
        dev_family.instance_node(sibling, _spec("tc2"), []),
        dev_family.instance_node(pool_copy, _spec("te"), []),
        dev_family.instance_node(in_text, _spec("tf2"), [snippet]),
    ]
    assert dev_family.code_blocks(snippet) == [RENAMED_DEV.rstrip("\n")]
    found = assess([task_node(t) for t in DEV], [task_node(t) for t in TRAIN], nodes)
    reasons = {j.id: j.reason for j in found.subjects}
    assert reasons == {
        "synth:c": "prompt_code_exposure",
        "synth:c2": None,
        "synth:e": "prompt_code_exposure",
        "synth:f": "prompt_code_exposure",
    }
    assert found.exposure_hits["prompt_code_exposure"] == 3
    judged = [dev_family.Judgement(j.id[6:], j.status, j.reason) for j in found.subjects]
    templates = {"c": "tc", "c2": "tc", "e": "te", "f": "tf2"}
    assert dev_family.expand_templates(judged, templates) == {
        "c": "prompt_code_exposure",
        "c2": "template",
        "e": "prompt_code_exposure",
        "f": "prompt_code_exposure",
    }


def test_shown_text_and_names_are_checked_directly() -> None:
    """A row's user text that paraphrases a dev prompt, and a shown helper that defines a
    dev task's function name beside a similar prompt, exclude the instance; neither joins a
    family (the sibling sharing that helper stays clear)."""
    para = _repo_instance("p", "tp", "def p_fn(s):\n    return s * 2\n")
    named = _repo_instance(
        "n", "tn2", "def n_fn(s):\n    return s * 3\n", helper="def secretfn(v):\n    return v\n"
    )
    sibling = _repo_instance(
        "n2", "tn3", "def n2_fn(s):\n    return s * 4\n", helper="def secretfn(v):\n    return v\n"
    )
    nodes = [
        dev_family.instance_node(para, _spec("tp"), [prompt(18, "row")]),
        dev_family.instance_node(named, _spec("tn2"), [prompt(10, "row")]),
        dev_family.instance_node(sibling, _spec("tn3"), [words("plainrow", 20)]),
    ]
    found = assess([task_node(t) for t in DEV], [], nodes)
    assert [(j.status, j.reason) for j in found.subjects] == [
        ("excluded", "prompt_text_exposure"),
        ("excluded", "prompt_named_exposure"),
        ("clear", None),
    ]


def test_strip_task_removes_every_generator_phrasing() -> None:
    """Each phrasing's fixed text is removed and the values it filled in are kept; the
    traceback's header and frame framing are removed; other text is refused."""
    trace = "\n".join(
        [
            instances.TRACEBACK_HEADER,
            instances.FRAME_LINE.format(file="test_lib.py", function="test_run_one"),
            instances.FRAME_LINE.format(file="lib.py", function="run"),
            "IndexError: list index out of range",
        ]
    )
    values = {
        "call": "run([1, 2])",
        "got": "3",
        "want": "4",
        "test": "test_run_one",
        "exc": "IndexError",
        "name": "run",
        "file": "lib.py",
        "traceback": trace,
    }
    for phrase in instances.TASK_PHRASES:
        text = phrase.format(**values)
        stripped = dev_family.strip_task(text)
        for literal, field_name, _, _ in string.Formatter().parse(phrase):
            if field_name is not None and field_name != "traceback":
                assert values[field_name] in stripped
            for word in ("Fix", "instead", "Implement", "Traceback", "File"):
                if word in literal:
                    assert word not in stripped, (phrase, stripped)
        if "{traceback}" in phrase:
            assert "test_lib.py test_run_one\nlib.py run\nIndexError" in stripped
    with pytest.raises(ValueError, match="no generator phrasing"):
        dev_family.strip_task("Please rewrite this function.")


def test_pending_flag_rows_are_kept_only_on_a_record_that_resolves_each_edge() -> None:
    edges = [
        ("t1", "d0", "pending"),
        ("t2", "d0", "pending"),
        ("t2", "d1", "pending"),
        ("t3", "d0", "not_equivalent"),
        ("t4", "d1", "pending"),
    ]
    decisions = [
        {"train": "t2", "dev": "d0", "final": "not_equivalent"},
        {"train": "t2", "dev": "d1", "final": "not_equivalent"},
        {"train": "t4", "dev": "d0", "final": "not_equivalent"},  # Another pair.
        {"train": "t1", "dev": "d0", "final": "unclear"},
    ]
    found = dev_family.pending_flags(edges, decisions, ["t1", "t2", "t3", "t4", "t5"])
    assert found.candidates == ["t1", "t2", "t3", "t4"]
    assert found.kept == {"t2": [1, 2], "t3": []}
    assert found.excluded == ["t1", "t4"]
    manifest = {
        "flag_edges": [
            {"a": "t1", "b": "d0", "splits": ["train", "dev"], "review": "pending"},
            {"a": "d1", "b": "t9", "splits": ["dev", "train"]},
            {"a": "t1", "b": "t2", "splits": ["train", "train"], "review": "pending"},
        ]
    }
    assert dev_family.flag_edges(manifest) == [("t1", "d0", "pending"), ("t9", "d1", "pending")]


def test_an_owner_ruling_drops_a_whole_template() -> None:
    judged = [
        dev_family.Judgement("j1", "clear", None),
        dev_family.Judgement("j2", "excluded", "dev_family"),
        dev_family.Judgement("k1", "clear", None),
        dev_family.Judgement("k2", "flagged", "dev_named_flag"),
        dev_family.Judgement("m1", "clear", None),
    ]
    templates = {"j1": "tj", "j2": "tj", "k1": "tk", "k2": "tk", "m1": "tm"}
    assert dev_family.expand_templates(judged, templates, (), ["tj"]) == {
        "j1": "owner_ruling",
        "j2": "owner_ruling",
        "k2": "dev_named_flag",
        "k1": "template",
    }


def test_link_changes_refuse_a_removed_link_without_a_category() -> None:
    """A link the earlier mapping made only through items both mappings hold must be made
    by the later one too; if not, the comparison is wrong and says so."""
    old = [
        dev_family.Labeled("synth:x", "template:tx", (("spec", words("same", 12)),)),
        dev_family.Labeled("synth:y", "template:ty", (("spec", words("same", 12)),)),
    ]
    new = [
        dev_family.Labeled("synth:x", "template:tx", (("spec", words("same", 12)),)),
        dev_family.Labeled("synth:y", "template:ty", (("spec", words("other", 12)),)),
    ]
    with pytest.raises(ValueError, match="rests only on items both mappings hold"):
        dev_family.link_changes(old, new, 0)
    with pytest.raises(ValueError, match="same nodes"):
        dev_family.link_changes(old, new[:1], 0)


def test_generator_output_scaffolding_comes_from_the_catalog(tmp_path: Path) -> None:
    """Two templates of one family from ``generate``: the catalog gives each non-target role
    its kind, and every link the earlier mapping made between their instances and this one
    does not has a category."""
    from synth_fixtures import Chunks, families_json

    from scratchcoder.agent.synth.catalog import catalog

    def loader(path: Path) -> tuple[Chunks, Chunks]:
        encoder = Chunks()
        return encoder, encoder

    (tmp_path / "tokenizer.json").write_text("{}")
    out = tmp_path / "generated"
    argv = ["generate", "--out", str(out), "--tokenizer", str(tmp_path / "tokenizer.json")]
    argv += ["--families", str(families_json(tmp_path / "families.json"))]
    assert agent_sft_data.main([*argv, "--templates", "stock1", "stock3"], loader=loader) == 0
    found, _ = dev_family.read_instances(out)
    roles = dev_family.unit_kinds(catalog())
    assert roles["stock3"]["restock"] == "shared_unit"  # stock1's target, a stock3 helper.
    assert roles["stock1"]["restock"] == "target_unit"
    old = [dev_family.legacy_items(item, _spec(item.template_id), [], roles) for item in found]
    kinds = {kind for item in old for kind, _ in item.solutions}
    assert "shared_unit" in kinds and "unresolved_unit" not in kinds
    new = [dev_family.current_items(item, _spec(item.template_id)) for item in found]
    for labeled, item in zip(new, found, strict=True):
        node = dev_family.instance_node(item, _spec(item.template_id), [])
        assert labeled.node().prompts == node.prompts
        assert labeled.node().solutions == node.solutions
    changes = dev_family.link_changes(old, new, 0)
    categories = set(changes["removed"]["summary"]["by_category"])
    assert categories <= set(dev_family.CATEGORIES.values())
    assert "shared_scaffolding_code" in categories


def test_the_subcommand_applies_an_owner_drop_with_its_ruling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    out = tmp_path / "dev-family.json"
    assert agent_sft_data.main([*_argv(paths, out), "--drop-template", "tc"]) == 2
    assert "needs --drop-ruling" in capsys.readouterr().err
    ruling = tmp_path / "ruling.md"
    ruling.write_text("Drop the template.\n")
    argv = [*_argv(paths, out), "--drop-template", "nope", "--drop-ruling", str(ruling)]
    assert agent_sft_data.main(argv) == 2
    assert "names templates no instance has" in capsys.readouterr().err
    argv = [*_argv(paths, out), "--drop-template", "tc", "--drop-ruling", str(ruling)]
    assert agent_sft_data.main(argv) == 0
    instances_block = json.loads(out.read_text())["agent"]["instances"]
    assert instances_block["dropped"]["k-clear"] == "owner_ruling"
    assert instances_block["owner_drops"] == {
        "templates": ["tc"],
        "ruling": str(ruling),
        "ruling_sha256": dev_family.sha_file(ruling),
        "instances": 1,
        "rows": 3,
    }
    assert instances_block["dropped_by_reason"]["owner_ruling"] == 1


def test_the_subcommand_refuses_a_pending_count_the_replay_stage_disagrees_with(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The replay stage's own count binds the flag manifest: a record reporting another
    count (a stale or wrong manifest) or none at all is refused, and the record is
    required."""
    paths = _setup(tmp_path)
    for count, code in ((5, 2), (2, 0)):
        stage = write_replay_record(tmp_path / f"stage-{count}.json", count)
        out = tmp_path / f"family-{count}.json"
        assert agent_sft_data.main(_argv(paths, out, stage=stage)) == code
    assert "the replay stage reports 5 rows on pending flag edges" in capsys.readouterr().err
    block = json.loads((tmp_path / "family-2.json").read_text())["replay"]["pending_flags"]
    assert block["replay_stage_reported"] == 2
    # A manifest without these rows' edges: the record's count no longer matches.
    empty = write_flag_manifest(tmp_path / "empty-clusters.json")
    out = tmp_path / "family-empty.json"
    assert agent_sft_data.main(_argv({**paths, "flags": empty}, out)) == 2
    assert "gives 0 among the replay rows" in capsys.readouterr().err
    silent = write_replay_record(tmp_path / "stage-none.json", None, other_count=3)
    assert agent_sft_data.main(_argv(paths, tmp_path / "family-none.json", stage=silent)) == 2
    assert f"reports no {PENDING_KEY} count" in capsys.readouterr().err
    argv = _argv(paths, tmp_path / "family-unset.json")
    del argv[argv.index("--replay-record") : argv.index("--replay-record") + 2]
    with pytest.raises(SystemExit):
        agent_sft_data.main(argv)
    assert not any((tmp_path / f"family-{n}.json").exists() for n in ("empty", "none", "unset"))


PENDING_KEY = agent_sft_data.PENDING_FLAG_KEY


def _w1(call: str, got: str, want: str) -> str:
    return instances.W1_PHRASES[0].format(call=call, got=got, want=want)


def test_shared_literal_values_in_task_statements_are_not_a_family_link() -> None:
    """The generated task statement is scaffolding (a phrasing filled in with test values),
    not a description: two templates whose statements share only digit runs stay apart.
    With the phrasing removed those values alone reach the paraphrase level, so a statement
    that were a family item would join tb to ta's development family."""
    first = replace(
        _repo_instance("a", "ta", RENAMED_DEV),
        task=_w1("widest([[1, 1, 1], [0, 0, 0], [0, 1, 1]])", "4", "6"),
    )
    second = replace(
        _repo_instance("b", "tb", "def flip(s):\n    return s[::-1]\n"),
        task=_w1("bits_of(9)", "[1, 0, 0, 0]", "[1, 1, 1, 0]"),
    )
    stripped = [shingles(dev_family.strip_task(item.task), 3) for item in (first, second)]
    assert len(stripped[0] & stripped[1]) / len(stripped[0] | stripped[1]) >= 0.5
    dev = [task_node(t) for t in DEV]
    nodes = [
        dev_family.instance_node(item, _spec(item.template_id), []) for item in (first, second)
    ]
    assert all(
        item.task in node.shown_text for item, node in zip((first, second), nodes, strict=True)
    )
    assert not any(
        item.task in node.prompts for item, node in zip((first, second), nodes, strict=True)
    )
    found = {j.id: j.reason for j in assess(dev, [], nodes).subjects}
    assert found == {"synth:a": "dev_family", "synth:b": None}
    as_family = [
        replace(node, prompts=(*node.prompts, dev_family.strip_task(item.task)))
        for item, node in zip((first, second), nodes, strict=True)
    ]
    linked = {j.id: j.reason for j in assess(dev, [], as_family).subjects}
    assert linked == {"synth:a": "dev_family", "synth:b": "dev_family"}


def test_code_another_dev_family_instance_trains_is_exposure_when_shown() -> None:
    """Template tc's specification paraphrases d0, so tc's instance (target R) is in d0's
    family, and so is ta's instance, which also trains R. tb's rows show R as a helper while
    training an unrelated target: tb is excluded (``prompt_code_exposure``) as it would be if
    R were a pool task's program, and its sibling with it. A rejected instance of tb is a
    bridge with a direct hit; an instance in a dev family showing only its own target is
    not."""
    program = "def top_up(level, need):\n    return max(need - level, 0)\n"
    paraphrase = _repo_instance("c", "tc", program)
    trains = _repo_instance("a", "ta", program)
    shows = _repo_instance("b", "tb", "def flip(s):\n    return s[::-1]\n", helper=program)
    sibling = _repo_instance("b2", "tb", "def upper_fn(s):\n    return s.upper()\n")
    specs = {"tc": [prompt(16, "tcspec")], "ta": _spec("ta"), "tb": _spec("tb")}
    node = {
        item.id: dev_family.instance_node(item, specs[item.template_id], [])
        for item in (paraphrase, trains, shows, sibling)
    }
    dev = [task_node(t) for t in DEV]
    found = assess(dev, [], [node["c"], node["a"], node["b"], node["b2"]])
    reasons = {j.id: j.reason for j in found.subjects}
    assert reasons == {
        "synth:c": "dev_family",
        "synth:a": "dev_family",
        "synth:b": "prompt_code_exposure",
        "synth:b2": None,
    }
    assert found.families["exposure_targets_synthetic"] == 2
    judged = [dev_family.Judgement(j.id[6:], j.status, j.reason) for j in found.subjects]
    templates = {"c": "tc", "a": "ta", "b": "tb", "b2": "tb"}
    assert dev_family.expand_templates(judged, templates)["b2"] == "template"
    as_bridges = assess(dev, [], [node["b2"]], [node["c"], node["b"]])
    assert as_bridges.bridges_hit == {"synth:b": "prompt_code_exposure"}
    alone = assess(dev, [], [], [node["c"]])
    assert alone.families["exposure_targets_synthetic"] == 1 and alone.bridges_hit == {}


def test_pending_flags_fail_closed_on_conflicting_or_malformed_records() -> None:
    """A pair with any decision other than not equivalent stays unresolved, whatever the
    order of its lines; an edge without two endpoints and two splits is refused."""
    edges = [("t1", "d0", "pending"), ("t2", "d0", "pending")]
    decisions = [
        {"train": "t1", "dev": "d0", "final": "not_equivalent"},
        {"train": "t1", "dev": "d0", "final": "confirmed_equivalent"},
        {"train": "t2", "dev": "d0", "final": "unclear"},
        {"train": "t2", "dev": "d0", "final": "not_equivalent"},
    ]
    found = dev_family.pending_flags(edges, decisions, ["t1", "t2"])
    assert found.kept == {} and found.excluded == ["t1", "t2"]
    agreed = dev_family.pending_flags(edges, [decisions[0], decisions[0]], ["t1"])
    assert agreed.kept == {"t1": [1, 2]}
    for edge in (
        {"a": "t1", "b": "d0", "review": "pending"},
        {"a": "t1", "b": "d0", "splits": ["dev"]},
        {"a": "t1", "splits": ["train", "dev"]},
        {"a": "t1", "b": "d0", "splits": ["train", None]},
    ):
        with pytest.raises(ValueError, match="two endpoints and two splits"):
            dev_family.flag_edges({"flag_edges": [edge]})
    with pytest.raises(ValueError, match="no flag_edges"):
        dev_family.flag_edges({})


def test_private_records_are_refused_in_a_work_tree_that_does_not_ignore_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The evidence file names defined names from development solutions; it and the
    result may sit only outside every Git work tree or where Git ignores them."""
    import subprocess

    paths = _setup(tmp_path)
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("runs/\n")
    out = tmp_path / "family.json"
    for place in (repo / "docs" / "evidence.json", repo / "evidence.json"):
        argv = dev_family_argv(
            paths["dev"],
            paths["train"],
            [(paths["judged"], paths["build"])],
            [paths["specs"]],
            paths["meta"],
            paths["decoded"],
            out,
            flags=paths["flags"],
            stage=paths["stage"],
            evidence=place,
        )
        assert agent_sft_data.main(argv) == 2
        assert "and not ignored" in capsys.readouterr().err
        assert not place.exists() and not out.exists()
    tracked_out = repo / "family.json"
    argv = _argv(paths, tmp_path / "unused.json")
    argv[argv.index("--out") + 1] = str(tracked_out)
    assert agent_sft_data.main(argv) == 2
    assert "and not ignored" in capsys.readouterr().err
    ignored = repo / "runs" / "v3" / "evidence.json"
    argv = dev_family_argv(
        paths["dev"],
        paths["train"],
        [(paths["judged"], paths["build"])],
        [paths["specs"]],
        paths["meta"],
        paths["decoded"],
        repo / "runs" / "v3" / "family.json",
        flags=paths["flags"],
        stage=paths["stage"],
        evidence=ignored,
    )
    assert agent_sft_data.main(argv) == 0 and ignored.exists()
