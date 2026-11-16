"""``dev-family``: the development task-family rule on synthetic agent rows and replay rows.

The rule is the frozen task-family rule of :mod:`scratchcoder.data.clusters`
(``task-clusters-v1``, default thresholds), applied the way the pilot intake's
``dev_family_ids`` and the holdout integrity scan's ``scan`` apply it, with the pool's
development split as the target. Static text and AST signals only: nothing here runs any
code, and tests are never a signal.

**Nodes.** The family graph holds every pool development task (the targets), every pool
training task, and every synthetic instance (:class:`Node`). A pool task is one node with
one prompt and one solution. A synthetic instance is one node with several prompts (task
statement, the template's specification text, the target function's docstring, every
training row's user text) and several solutions (the reference target, the original target,
every module of the fixed repository and every top-level function or class of those modules
on its own). The node stands for every (prompt, solution) pair of these, joined into one
family, so any link of any pair marks the whole instance (:func:`items_of`).

Synthetic node IDs are ``synth:<instance>``, so their upstream identity never equals a pool
ID. Two synthetic nodes under one instance ID (two instance sets reusing an ID) are joined,
as the intake joins repeated IDs.

**Links** (union signals only; :func:`family_links`): equal docstring-stripped solution AST,
equal identifier-canonicalized solution AST, equal upstream ID, and 3-word prompt shingle
Jaccard at or above 0.5. Families are the connected components. A shared defined name never
links two nodes.

**Judging** (:func:`assess`). Only *subjects* (the accepted instances, whose rows are
trained on) are judged. A subject is

``excluded``
    when its family contains any development task (``dev_family``), or when it shares a
    defined function or class name with a development task at prompt Jaccard at or above
    0.25 (``dev_named_paraphrase``);
``flagged``
    when it is not excluded but shares a defined name with a development task at prompt
    Jaccard at or above 0.10 (``dev_named_flag``);
``clear``
    otherwise.

*Bridges* (rejected and superseded instances) and pool training tasks are never judged, but
they carry links, so a subject joined to a development task only through a rejected row is
still excluded. Excluded and flagged subjects stay in the graph as nodes.

**Replay rows** are pool training tasks. One is newly excluded when its family, with the
synthetic nodes added, contains a development task while its family without them did not:
a new chain through synthetic nodes. Families that already held a development task were
judged when the replay set was built; this check reports them and does not overturn that
outcome.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases, workspace
from scratchcoder.data.adapters import canonical_id
from scratchcoder.data.clusters import (
    DEFAULT_RULES,
    RULES_VERSION,
    ClusterRules,
    Item,
    UnionFind,
    canonical_key,
    defined_names,
    jaccard,
    similar_pairs,
    solution_key,
)
from scratchcoder.data.pool import shingles
from scratchcoder.eval.tasks import Task

SCHEMA = "agent-dev-family-v1"
RULE = RULES_VERSION
"""The identity key every result carries (``dev_family``) and ``mix`` requires."""
NAMESPACE = "synth:"
STATUSES = ("excluded", "flagged", "clear")
REASONS = ("dev_family", "dev_named_paraphrase", "dev_named_flag", "template")
"""Reason classes: the first three judge an instance; ``template`` drops every instance of a
template that holds an excluded or flagged instance."""
METHOD = (
    "task-family rule task-clusters-v1 against the pool development split: union-signal "
    "families over pool development and training tasks and every synthetic instance "
    "(rejected and superseded instances as unjudged bridges); an accepted instance is "
    "dropped when its family holds a development task or it shares a defined name with one "
    "(prompt Jaccard at least 0.10, flagged ones included), and every instance of a template "
    "with a dropped instance is dropped; replay rows are dropped on a new chain through "
    "synthetic nodes"
)


@dataclass(frozen=True)
class Node:
    """One graph node: every (prompt, solution) pair of it is one task, all joined."""

    id: str
    prompts: tuple[str, ...] = ()
    solutions: tuple[str, ...] = ()


def task_node(task: Task) -> Node:
    return Node(task.id, (task.prompt,), (task.solution,) if task.solution else ())


def items_of(node: Node, split: str) -> list[Item]:
    """The task items a node stands for: the cross product of its prompts and solutions
    (a node without solutions stands for its prompts alone). IDs after the first carry a
    ``#n`` suffix; callers join them to the first."""
    prompts = node.prompts or ("",)
    solutions: Sequence[str | None] = node.solutions or (None,)
    found = []
    for number, (prompt, solution) in enumerate(
        (prompt, solution) for prompt in prompts for solution in solutions
    ):
        identifier = node.id if number == 0 else f"{node.id}#{number}"
        found.append(Item(identifier, split, "synthetic", prompt, solution))
    return found


def family_links(
    nodes: Sequence[Node], rules: ClusterRules = DEFAULT_RULES
) -> list[tuple[int, int]]:
    """Links ``(a, b)``, ``a < b``, whose connected components are the union-signal families
    of the nodes' items (``clusters.find_edges`` with the union signals, then union-find).

    Nodes that share a key are chained in index order instead of linked pairwise; nodes with
    an identical nonempty shingle set are chained (their Jaccard is 1) and every other
    prompt pair is found once among the distinct sets. Each link joins the lowest-index
    holders, so the links among the first ``k`` nodes alone give the families of those
    ``k`` nodes alone (:class:`FamilyGraph` uses this for the graph without synthetic
    nodes)."""
    links: list[tuple[int, int]] = []

    def chain(groups: Iterable[Sequence[int]]) -> None:
        for members in groups:
            ordered = sorted(set(members))
            links.extend(zip(ordered, ordered[1:], strict=False))

    solution_groups: dict[Hashable, list[int]] = defaultdict(list)
    canonical_groups: dict[Hashable, list[int]] = defaultdict(list)
    upstream_groups: dict[Hashable, list[int]] = defaultdict(list)
    prompt_groups: dict[frozenset[tuple[str, ...]], list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        upstream_groups[canonical_id(node.id)].append(index)
        for text in node.solutions:
            for key, groups in (
                (solution_key(text), solution_groups),
                (canonical_key(text), canonical_groups),
            ):
                if key is not None:
                    groups[key].append(index)
        for text in node.prompts:
            found = shingles(text, rules.width)
            if found:
                prompt_groups[found].append(index)
    for groups in (solution_groups, canonical_groups, upstream_groups):
        chain(groups.values())
    chain(prompt_groups.values())
    distinct = list(prompt_groups)
    for i, j, _ in similar_pairs(distinct, rules.paraphrase):
        a, b = prompt_groups[distinct[i]][0], prompt_groups[distinct[j]][0]
        if a != b:
            links.append((min(a, b), max(a, b)))
    return links


class FamilyGraph:
    """Families of ``base + added`` nodes, and of the ``base`` nodes alone."""

    def __init__(
        self, base: Sequence[Node], added: Sequence[Node], rules: ClusterRules = DEFAULT_RULES
    ) -> None:
        self.nodes = [*base, *added]
        self.base_size = len(base)
        self.full = UnionFind(len(self.nodes))
        self.base = UnionFind(len(base))
        for a, b in family_links(self.nodes, rules):
            self.full.union(a, b)
            if b < self.base_size:
                self.base.union(a, b)


def _names(node: Node) -> frozenset[str]:
    return frozenset(name for text in node.solutions for name in defined_names(text))


def named_scores(
    targets: Sequence[Node], others: Sequence[Node], rules: ClusterRules = DEFAULT_RULES
) -> list[float | None]:
    """For each of ``others``, the highest prompt Jaccard with a target it shares a defined
    name with (``None`` when it shares none). Over the cross product of a node's items this
    is the score of its strongest direct shared-name edge to a target."""
    index: dict[str, list[int]] = defaultdict(list)
    for position, target in enumerate(targets):
        for name in _names(target):
            index[name].append(position)
    target_prompts = [[shingles(text, rules.width) for text in t.prompts] for t in targets]
    found: list[float | None] = []
    for node in others:
        candidates = sorted({t for name in _names(node) for t in index.get(name, ())})
        if not candidates:
            found.append(None)
            continue
        own = [shingles(text, rules.width) for text in node.prompts] or [frozenset()]
        found.append(
            max(
                jaccard(a, b)
                for t in candidates
                for a in own
                for b in target_prompts[t] or [frozenset()]
            )
        )
    return found


@dataclass(frozen=True)
class Judgement:
    id: str
    status: str
    reason: str | None


@dataclass
class Assessment:
    subjects: list[Judgement]
    replay_excluded: list[str]
    """Replay rows newly in a development family through synthetic nodes."""
    replay_in_dev_family: list[str]
    """Replay rows whose family held a development task without synthetic nodes (kept)."""
    replay_named: list[str]
    """Replay rows outside every development family, with or without synthetic nodes, that
    have a direct shared-name edge to a development task at the flag level or above (kept:
    pending flag edges stay as the replay stage reported them)."""
    families: dict[str, int]


def _status(in_family: bool, score: float | None, rules: ClusterRules) -> tuple[str, str | None]:
    if in_family:
        return "excluded", "dev_family"
    if score is not None and score >= rules.named_paraphrase:
        return "excluded", "dev_named_paraphrase"
    if score is not None and score >= rules.named_flag:
        return "flagged", "dev_named_flag"
    return "clear", None


def assess(
    dev: Sequence[Node],
    train: Sequence[Node],
    subjects: Sequence[Node],
    bridges: Sequence[Node] = (),
    *,
    replay: Sequence[str] = (),
    rules: ClusterRules = DEFAULT_RULES,
) -> Assessment:
    """Judge every subject and every replay row (pool training IDs) against ``dev``."""
    base = [*dev, *train]
    if len({node.id for node in base}) != len(base):
        raise ValueError("development and training task IDs must be unique")
    graph = FamilyGraph(base, [*subjects, *bridges], rules)
    full_dev = {graph.full.find(index) for index in range(len(dev))}
    base_dev = {graph.base.find(index) for index in range(len(dev))}
    first_subject = len(base)
    scores = named_scores(dev, subjects, rules)
    judged = []
    roots: dict[int, int] = defaultdict(int)
    for offset, (node, score) in enumerate(zip(subjects, scores, strict=True)):
        root = graph.full.find(first_subject + offset)
        roots[root] += 1
        status, reason = _status(root in full_dev, score, rules)
        judged.append(Judgement(node.id, status, reason))
    position = {node.id: index for index, node in enumerate(train, start=len(dev))}
    unknown = [identifier for identifier in replay if identifier not in position]
    if unknown:
        raise ValueError(f"{len(unknown)} replay rows are not pool training tasks")
    newly, already = [], []
    for identifier in replay:
        index = position[identifier]
        if graph.base.find(index) in base_dev:
            already.append(identifier)
        elif graph.full.find(index) in full_dev:
            newly.append(identifier)
    judged_already = {*already, *newly}
    outside = [identifier for identifier in replay if identifier not in judged_already]
    named = named_scores(dev, [base[position[i]] for i in outside], rules)
    replay_named = [
        identifier
        for identifier, score in zip(outside, named, strict=True)
        if score is not None and score >= rules.named_flag
    ]
    families = {
        "subject_families": len(roots),
        "largest_subject_family": max(roots.values(), default=0),
        "subject_families_with_dev": sum(1 for root in roots if root in full_dev),
    }
    return Assessment(judged, sorted(newly), sorted(already), sorted(replay_named), families)


def expand_templates(judged: Sequence[Judgement], template_of: Mapping[str, str]) -> dict[str, str]:
    """Every dropped subject and its reason class: each excluded or flagged subject, and
    every other subject of a template that holds one (``template``)."""
    dropped = {item.id: str(item.reason) for item in judged if item.status != "clear"}
    templates = {template_of[identifier] for identifier in dropped}
    for item in judged:
        if item.id not in dropped and template_of[item.id] in templates:
            dropped[item.id] = "template"
    return dropped


# -- Synthetic instances on disk (read as text, never imported or run) -------------------


@dataclass(frozen=True)
class Instance:
    id: str
    template_id: str
    family: str
    task: str
    reference_target: str
    original_target: str
    modules: tuple[tuple[str, str], ...]
    """The fixed repository's source modules (tests left out): name and text."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(directory: Path) -> dict[str, Path]:
    found = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in an instance directory: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
            found[str(path.relative_to(directory))] = path
    return found


def read_instances(root: Path) -> tuple[list[Instance], str]:
    """Every registered instance under ``root``, each file checked against its registry
    hash first, and the registry's SHA-256. Only text is read; nothing is imported or
    validated by running it."""
    if any(cases.forbidden_part(part) for part in root.resolve().parts):
        raise ValueError("refusing a held-out location")
    raw = (root / cases.REGISTRY_NAME).read_bytes()
    registry = json.loads(raw)
    if registry.get("schema") != cases.REGISTRY_SCHEMA:
        raise ValueError(f"{root}: unknown registry schema")
    found = []
    for entry in registry["cases"]:
        directory = (root / str(entry["directory"])).resolve()
        if directory.parent != root.resolve():
            raise ValueError(f"{entry['id']}: not a direct child of {root}")
        present = _files(directory)
        if set(present) != set(entry["files"]):
            raise ValueError(f"{entry['id']}: files differ from the registry")
        texts = {}
        for name, digest in entry["files"].items():
            data = present[name].read_bytes()
            if _sha(data) != digest:
                raise ValueError(f"{entry['id']}: {name} changed since registration")
            texts[name] = data
        spec = json.loads(texts["case.json"])
        synth = json.loads(texts[cases.SYNTHETIC_MARKER])
        if spec["id"] != entry["id"] or synth["id"] != entry["id"]:
            raise ValueError(f"{entry['id']}: case.json or synth.json names another instance")
        repo = str(spec["repo"])
        modules = {
            Path(name).name: texts[name].decode("utf-8")
            for name in texts
            if Path(name).parent == Path(repo)
            and name.endswith(".py")
            and not workspace.is_test_file(Path(name).name)
        }
        for module, relative in spec["reference"].items():
            modules[str(module)] = texts[str(relative)].decode("utf-8")
        found.append(
            Instance(
                id=str(entry["id"]),
                template_id=str(synth["template_id"]),
                family=str(synth["family"]),
                task=str(spec["task"]),
                reference_target=str(synth["reference_target"]),
                original_target=str(synth["original_target"]),
                modules=tuple(sorted(modules.items())),
            )
        )
    return found, _sha(raw)


def _parse(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def docstrings(source: str) -> list[str]:
    """Docstrings of the functions and classes ``source`` defines (it is parsed, not run)."""
    tree = _parse(source)
    if tree is None:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node)
            if text:
                found.append(text)
    return found


def top_level_units(source: str) -> list[str]:
    """Each module-level function and class of ``source`` on its own, as source text."""
    tree = _parse(source)
    if tree is None:
        return []
    return [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def instance_node(instance: Instance, specs: Iterable[str], user_texts: Iterable[str]) -> Node:
    """The node of one instance (module docstring: prompts and solutions)."""
    prompts = [
        instance.task,
        *specs,
        *docstrings(instance.reference_target),
        *user_texts,
    ]
    solutions = [instance.reference_target, instance.original_target]
    for _, text in instance.modules:
        solutions.append(text)
        solutions.extend(top_level_units(text))
    return Node(
        NAMESPACE + instance.id,
        tuple(dict.fromkeys(text for text in prompts if text.strip())),
        tuple(dict.fromkeys(text for text in solutions if text.strip())),
    )


def load_specs(paths: Sequence[Path]) -> dict[str, list[str]]:
    """Template ID to its specification texts, over every ``specs.json`` given."""
    found: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for template in document["templates"]:
            text = str(template["spec"])
            if text not in found[str(template["template_id"])]:
                found[str(template["template_id"])].append(text)
    return dict(found)


def rules_record(rules: ClusterRules = DEFAULT_RULES) -> dict[str, Any]:
    return asdict(rules)


def require_result(
    result: Mapping[str, Any],
    *,
    agent_meta: Path,
    replay_index: Path,
    replay_kinds: Iterable[str],
    dev_sha256: Iterable[str | None] = (),
) -> None:
    """Refuse a dev-family result unless it applies this rule, with its default thresholds,
    to exactly ``agent_meta`` and ``replay_index`` as they are now, covers every replay kind
    mixed, and judged against the same development file as any n-gram check that names one
    (``dev_sha256``)."""
    if result.get("schema") != SCHEMA or result.get("dev_family") != RULE:
        raise ValueError(f"not a {RULE} dev-family result")
    if result.get("rules") != rules_record():
        raise ValueError("the dev-family result used other thresholds")
    for side, path, kind in (
        ("agent", agent_meta, "rows_meta"),
        ("replay", replay_index, "decoded"),
    ):
        part = result.get(side) or {}
        if part.get("input_kind") != kind:
            raise ValueError(f"the dev-family result's {side} side checked another kind of input")
        if part.get("input_sha256") != sha_file(path):
            raise ValueError(f"the dev-family result was not computed on {path} as it is")
    missing = set(replay_kinds) - set(result["replay"].get("kinds", ()))
    if missing:
        raise ValueError(f"the dev-family result did not judge replay kinds {sorted(missing)}")
    pool_dev = result.get("pool", {}).get("dev_sha256")
    if not pool_dev:
        raise ValueError("the dev-family result names no development file")
    if any(other is not None and other != pool_dev for other in dev_sha256):
        raise ValueError("the dev-family result used another development file")


def load_result(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a dev-family result")
    return dict(raw)
