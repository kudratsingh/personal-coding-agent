"""Task-family candidate clusters from static signals, and the adjudication manifest.

Two tasks belong to one *family* when they ask for the same behaviour. Equality of
behaviour is decided later, by running each reference against the partner's tests in the
isolated sandbox and by a reviewer (``tools.task_equivalence``). This module only finds
*candidates* and records why, so nothing here confirms that two tasks are equivalent.

Signals, each recorded on the edge it creates (``docs/walkthroughs/task-clusters.md``):

``solution_ast``
    SHA-256 of the reference solution's AST, dumped without positions, after module,
    class and function docstrings are removed. Identifiers are kept. Equal keys mean the
    references are the same program up to whitespace, comments and docstrings.
``canonical_ast``
    The same after every name the program binds (functions, classes, parameters,
    assigned and loop names, exception names) is renamed by order of first appearance and
    annotations are dropped. Keyword arguments are renamed only when they name a parameter
    of a function the program defines. Builtins, imported modules and attribute names are
    kept.
    A candidate signal only: two programs that differ only in naming match.
``prompt_paraphrase``
    Jaccard similarity of 3-word prompt shingles at or above 0.5 (the build's
    holdout-paraphrase threshold, ``tools.task_pool.HoldoutParaphrases``).
``prompt_named_paraphrase`` / ``prompt_named_flag``
    A shared defined function or class name with shingle Jaccard at or above 0.25 (the
    build's named-paraphrase rule) or at or above 0.10. *Flag only*: recorded for a
    reviewer, never merged into a cluster. Common names (``Book``, ``add``, ``area``)
    chain unrelated prompts together at these levels; measured on the pool, merging them
    joined hundreds of tasks through train-internal links.
``upstream_id``
    Equal upstream identity (``canonical_id``) under two records.

Every edge also carries the comparison fields a reviewer needs: shared defined names,
prompt Jaccard, and whether source, mode and module-level names agree. Clusters are the
connected components of the union edges (union-find). Rules are versioned; a change to a
threshold or signal is a new ``RULES_VERSION``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scratchcoder.data.adapters import canonical_id
from scratchcoder.data.pool import shingles

RULES_VERSION = "task-clusters-v1"
MANIFEST_FORMAT = "task-cluster-manifest-v1"
UNION_SIGNALS = (
    "solution_ast",
    "canonical_ast",
    "upstream_id",
    "prompt_paraphrase",
)
FLAG_SIGNALS = ("prompt_named_paraphrase", "prompt_named_flag")
SIGNALS = UNION_SIGNALS + FLAG_SIGNALS
GENERIC_NAMES = frozenset({"main", "solve", "solution"})
"""Defined names too common to tie two prompts together (``tools.task_pool``'s list)."""
ADJUDICATION_STATUSES = ("pending", "confirmed_equivalent", "not_equivalent", "unclear_quarantined")
MEMBER_STATUSES = ("confirmed_equivalent", "not_equivalent", "unclear_quarantined")
"""Per-member decisions inside an ``unclear_quarantined`` (mixed) cluster."""
BEHAVIOR_OUTCOMES = ("pending", "equivalent", "not_equivalent", "unclear")


@dataclass(frozen=True)
class ClusterRules:
    """The thresholds of one rule version."""

    version: str = RULES_VERSION
    width: int = 3
    paraphrase: float = 0.5
    named_paraphrase: float = 0.25
    named_flag: float = 0.10

    def __post_init__(self) -> None:
        if self.width < 1 or not 0 < self.named_flag <= self.named_paraphrase <= self.paraphrase:
            raise ValueError("thresholds must satisfy 0 < flag <= named <= paraphrase")
        if self.paraphrase > 1:
            raise ValueError("a Jaccard threshold is at most 1")


DEFAULT_RULES = ClusterRules()


@dataclass(frozen=True)
class Item:
    """One task as the cluster scan sees it. ``roles`` are labels such as ``sft_train``."""

    id: str
    split: str
    source: str
    prompt: str
    solution: str | None
    mode: str = "function"
    entry_point: str | None = None
    roles: tuple[str, ...] = ()


@dataclass
class Edge:
    """A pair of item indices ``a < b`` and the evidence found for it."""

    a: int
    b: int
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def union(self) -> bool:
        return any(name in self.signals for name in UNION_SIGNALS)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _parse(source: str | None) -> ast.Module | None:
    if not source or not source.strip():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # Invalid escapes still parse.
            return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _digest(tree: ast.AST) -> str:
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def solution_key(source: str | None) -> str | None:
    """Docstring-stripped AST key, or None when the text does not parse."""
    tree = _parse(source)
    return None if tree is None else _digest(_strip_docstrings(tree))


def _bound_names(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


class _AnnotationsDropped(ast.NodeTransformer):
    """``x: T = v`` becomes ``x = v``; a bare ``x: T`` (no runtime effect) is removed."""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        if getattr(node, "body", None) == []:
            node.body = [ast.Pass()]  # type: ignore[attr-defined]
        return node


def canonical_key(source: str | None) -> str | None:
    """Identifier-canonicalized AST key (candidate signal), or None when unparsable."""
    tree = _parse(source)
    if tree is None:
        return None
    _strip_docstrings(tree)
    tree = _AnnotationsDropped().visit(tree)
    bound = _bound_names(tree)
    parameters = {
        node.arg
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for node in ast.walk(function.args)
        if isinstance(node, ast.arg)
    }
    mapping: dict[str, str] = {}

    def rename(name: str) -> str:
        if name not in bound:
            return name
        return mapping.setdefault(name, f"_v{len(mapping)}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = rename(node.name)
            node.returns = None
        elif isinstance(node, ast.ClassDef):
            node.name = rename(node.name)
        elif isinstance(node, ast.arg):
            node.arg = rename(node.arg)
            node.annotation = None
        elif isinstance(node, ast.Name):
            node.id = rename(node.id)
        elif isinstance(node, ast.keyword) and node.arg in parameters:
            node.arg = rename(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            node.name = rename(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            node.names = [rename(name) for name in node.names]
    return _digest(tree)


def defined_names(source: str | None) -> frozenset[str]:
    """Function and class names a reference defines, minus generic and dunder names."""
    tree = _parse(source)
    if tree is None:
        return frozenset()
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name not in GENERIC_NAMES
        and not (node.name.startswith("__") and node.name.endswith("__"))
    )


def top_level_names(source: str | None) -> tuple[str, ...]:
    """Module-level function and class names, in definition order."""
    tree = _parse(source)
    if tree is None:
        return ()
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    return len(a & b) / max(len(a | b), 1)


def similar_pairs(sets: Sequence[frozenset[Any]], threshold: float) -> list[tuple[int, int, float]]:
    """Every pair ``i < j`` with Jaccard >= ``threshold``, by exact check after a prefix filter.

    Elements are ordered rarest first; two sets reaching the threshold must share an
    element among the first ``len - ceil(threshold * len) + 1`` of each (prefix filtering).
    Empty sets never match.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    frequency: Counter[Any] = Counter(element for value in sets for element in value)
    index: dict[Any, list[int]] = defaultdict(list)
    found: list[tuple[int, int, float]] = []
    for j, value in enumerate(sets):
        if not value:
            continue
        ordered = sorted(value, key=lambda element: (frequency[element], repr(element)))
        prefix = ordered[: len(ordered) - math.ceil(threshold * len(ordered)) + 1]
        candidates = {i for element in prefix for i in index[element]}
        for i in sorted(candidates):
            score = jaccard(sets[i], value)
            if score >= threshold:
                found.append((i, j, score))
        for element in prefix:
            index[element].append(j)
    return found


def find_edges(items: Sequence[Item], rules: ClusterRules = DEFAULT_RULES) -> list[Edge]:
    """All candidate edges among ``items``, each with every signal that fired for it."""
    if len({item.id for item in items}) != len(items):
        raise ValueError("item IDs must be unique")
    edges: dict[tuple[int, int], Edge] = {}

    def add(i: int, j: int, name: str, value: Any) -> None:
        a, b = min(i, j), max(i, j)
        edges.setdefault((a, b), Edge(a, b)).signals[name] = value

    def grouped(keys: Sequence[str | None], name: str) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            if key is not None:
                groups[key].append(index)
        for key, members in groups.items():
            for x, i in enumerate(members):
                for j in members[x + 1 :]:
                    add(i, j, name, key[:16])

    grouped([solution_key(item.solution) for item in items], "solution_ast")
    grouped([canonical_key(item.solution) for item in items], "canonical_ast")
    grouped([canonical_id(item.id) for item in items], "upstream_id")
    prompts = [shingles(item.prompt, rules.width) for item in items]
    for i, j, score in similar_pairs(prompts, rules.paraphrase):
        add(i, j, "prompt_paraphrase", round(score, 4))
    names = [defined_names(item.solution) for item in items]
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(names):
        for name in value:
            by_name[name].append(index)
    seen: set[tuple[int, int]] = set()
    for members in by_name.values():
        for x, i in enumerate(members):
            for j in members[x + 1 :]:
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                score = jaccard(prompts[i], prompts[j])
                if score >= rules.named_paraphrase:
                    add(i, j, "prompt_named_paraphrase", round(score, 4))
                elif score >= rules.named_flag:
                    add(i, j, "prompt_named_flag", round(score, 4))
    top = [top_level_names(item.solution) for item in items]
    for edge in edges.values():
        first, second = items[edge.a], items[edge.b]
        edge.signals["evidence"] = {
            "shared_names": sorted(names[edge.a] & names[edge.b]),
            "prompt_jaccard": round(jaccard(prompts[edge.a], prompts[edge.b]), 4),
            "same_source": first.source == second.source,
            "same_mode": first.mode == second.mode,
            "same_top_level_names": top[edge.a] == top[edge.b],
        }
    return [edges[key] for key in sorted(edges)]


class UnionFind:
    """Disjoint sets over ``0..n-1`` with path halving and union by size."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def cluster_id(member_ids: Iterable[str]) -> str:
    """Stable identity of a member set: ``tc-`` and 16 hex digits of a SHA-256."""
    return "tc-" + hashlib.sha256("\n".join(sorted(member_ids)).encode()).hexdigest()[:16]


def components(
    size: int, edges: Iterable[Edge], signals: Collection[str] = UNION_SIGNALS
) -> list[list[int]]:
    """Connected components of the edges carrying any of ``signals`` (default: the union
    signals), members sorted, singletons included."""
    if not set(signals) <= set(UNION_SIGNALS):
        raise ValueError("flag-only signals never join a cluster")
    found = UnionFind(size)
    for edge in edges:
        if any(name in edge.signals for name in signals):
            found.union(edge.a, edge.b)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        groups[found.find(index)].append(index)
    return sorted(groups.values(), key=lambda members: members[0])


def family_map(items: Sequence[Item], rules: ClusterRules = DEFAULT_RULES) -> dict[str, str]:
    """Each item's family (cluster) ID under the union signals; singletons get their own."""
    groups = components(len(items), find_edges(items, rules))
    return {
        items[index].id: cluster_id(items[member].id for member in members)
        for members in groups
        for index in members
    }


def member_record(item: Item) -> dict[str, Any]:
    return {
        "split": item.split,
        "source": item.source,
        "id": item.id,
        "mode": item.mode,
        "entry_point": item.entry_point,
        "top_level_names": list(top_level_names(item.solution)),
        "roles": sorted(item.roles),
    }


def edge_record(items: Sequence[Item], edge: Edge) -> dict[str, Any]:
    signals = {key: value for key, value in edge.signals.items() if key != "evidence"}
    return {
        "a": items[edge.a].id,
        "b": items[edge.b].id,
        "splits": [items[edge.a].split, items[edge.b].split],
        "union": edge.union,
        "signals": signals,
        "evidence": edge.signals["evidence"],
    }


def pending_adjudication() -> dict[str, Any]:
    return {"status": "pending", "reviewer": None, "decided_at": None, "rationale": None}


def pending_behavior(a: str, b: str) -> dict[str, Any]:
    return {"a": a, "b": b, "outcome": "pending", "directions": [], "run": None}


def cluster_record(
    items: Sequence[Item],
    members: Sequence[int],
    edges: Sequence[Edge],
    *,
    categories: Sequence[str] = (),
) -> dict[str, Any]:
    """One cluster's manifest entry: static evidence, pending behaviour and adjudication.

    A cluster whose union edges join a function task to an I/O task cannot be compared by
    running one reference against the other's tests, so it starts ``unclear_quarantined``
    with that uncertainty written down; every other cluster starts ``pending``.
    """
    ids = [items[index].id for index in members]
    inside = set(members)
    own = [edge for edge in edges if edge.a in inside and edge.b in inside]
    uncertainty: list[str] = []
    adjudication = pending_adjudication()
    union_edges = [edge for edge in own if edge.union]
    if any(not edge.signals["evidence"]["same_mode"] for edge in union_edges):
        uncertainty.append(
            "a union edge joins a function task and an I/O task: references cannot be run "
            "against each other's tests, so behaviour is not comparable by the harness"
        )
        adjudication["status"] = "unclear_quarantined"
    joined = [set(UNION_SIGNALS) & set(edge.signals) for edge in union_edges]
    if any(names == {"canonical_ast"} for names in joined):
        uncertainty.append(
            "some edges rest only on the identifier-canonicalized AST (a candidate signal)"
        )
    if any(not names & {"solution_ast", "canonical_ast"} for names in joined):
        uncertainty.append("some edges rest only on prompt similarity, not on reference code")
    return {
        "cluster_id": cluster_id(ids),
        "members": [member_record(items[index]) for index in members],
        "categories": sorted(categories),
        "edges": [edge_record(items, edge) for edge in own],
        "behavior": [
            pending_behavior(items[edge.a].id, items[edge.b].id)
            for edge in union_edges
            if items[edge.a].split != items[edge.b].split
        ],
        "adjudication": adjudication,
        "uncertainty": uncertainty,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse a manifest whose statuses, members or reviewer records are inconsistent.

    ``confirmed_equivalent`` needs every behaviour record of the cluster ``equivalent``;
    ``not_equivalent`` needs every one ``not_equivalent``; both need a reviewer, a date and
    a rationale. ``pending`` carries no reviewer. ``unclear_quarantined`` needs at least one
    written uncertainty note. No member may be a sealed-test or holdout task.

    An ``unclear_quarantined`` cluster may also carry ``member_decisions`` (``member_rules``);
    any member without one stays quarantined with the cluster.
    """
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("not a task-cluster manifest of a known format")
    seen: set[str] = set()
    for cluster in manifest["clusters"]:
        ids = [member["id"] for member in cluster["members"]]
        if cluster["cluster_id"] != cluster_id(ids):
            raise ValueError("cluster_id does not match its members")
        if cluster["cluster_id"] in seen:
            raise ValueError("duplicate cluster_id")
        seen.add(cluster["cluster_id"])
        if any(member["split"] not in {"train", "dev"} for member in cluster["members"]):
            raise ValueError("cluster members must come from pool train or dev only")
        known = set(ids)
        for edge in cluster["edges"]:
            if edge["a"] not in known or edge["b"] not in known:
                raise ValueError("an edge names a task outside its cluster")
            if not set(edge["signals"]) <= set(SIGNALS) or not edge["signals"]:
                raise ValueError("an edge must name at least one known signal")
        outcomes = []
        for record in cluster["behavior"]:
            if record["outcome"] not in BEHAVIOR_OUTCOMES:
                raise ValueError("unknown behaviour outcome")
            outcomes.append(record["outcome"])
        decision = cluster["adjudication"]
        status = decision["status"]
        if status not in ADJUDICATION_STATUSES:
            raise ValueError("unknown adjudication status")
        decided = all(decision.get(key) for key in ("reviewer", "decided_at", "rationale"))
        if status == "pending" and decision.get("reviewer") is not None:
            raise ValueError("a pending cluster carries no reviewer")
        if status in {"confirmed_equivalent", "not_equivalent"}:
            needed = "equivalent" if status == "confirmed_equivalent" else "not_equivalent"
            if not decided:
                raise ValueError(f"{status} needs a reviewer, a date and a rationale")
            if not outcomes or any(outcome != needed for outcome in outcomes):
                raise ValueError(f"{status} needs every behaviour result to be {needed}")
        if status == "unclear_quarantined" and not cluster["uncertainty"]:
            raise ValueError("an unclear cluster must keep its uncertainty notes")
        member_rules(cluster)


def _touching(cluster: Mapping[str, Any], member: str) -> list[str | None]:
    """The behaviour outcome of each joining edge that touches ``member`` (``None``: untested)."""
    outcomes = {
        frozenset((record["a"], record["b"])): record["outcome"] for record in cluster["behavior"]
    }
    return [
        outcomes.get(frozenset((edge["a"], edge["b"])))
        for edge in cluster["edges"]
        if edge["union"] and member in (edge["a"], edge["b"])
    ]


def member_rules(cluster: Mapping[str, Any]) -> None:
    """Check the per-member decisions of one cluster.

    They refine an ``unclear_quarantined`` (mixed) cluster only, and each needs a reviewer,
    a date and a rationale. A member is released (``not_equivalent``) only when every
    joining edge that touches it was run and found ``not_equivalent``: an untested edge (a
    same-split edge has no behaviour record) keeps it quarantined. A member is
    ``confirmed_equivalent`` only when at least one touching edge was run and every run one
    is ``equivalent``. ``unclear_quarantined`` keeps it out, like an undecided member.
    """
    decisions = cluster.get("member_decisions")
    if not decisions:
        return
    if cluster["adjudication"]["status"] != "unclear_quarantined":
        raise ValueError("member decisions refine an unclear_quarantined cluster only")
    known = {member["id"] for member in cluster["members"]}
    for member, decision in decisions.items():
        if member not in known:
            raise ValueError("a member decision names a task outside its cluster")
        status = decision.get("status")
        if status not in MEMBER_STATUSES:
            raise ValueError("unknown member decision status")
        if not all(decision.get(key) for key in ("reviewer", "decided_at", "rationale")):
            raise ValueError("a member decision needs a reviewer, a date and a rationale")
        touching = _touching(cluster, member)
        run = [outcome for outcome in touching if outcome not in (None, "pending")]
        if status == "not_equivalent" and (
            not touching or any(outcome != "not_equivalent" for outcome in touching)
        ):
            raise ValueError(
                "a member is released only when every joining edge touching it was run "
                "and found not_equivalent"
            )
        if status == "confirmed_equivalent" and (
            not run or any(outcome != "equivalent" for outcome in run)
        ):
            raise ValueError(
                "a confirmed_equivalent member needs every run edge touching it equivalent"
            )


def adjudicate(
    manifest: dict[str, Any],
    cluster: str,
    *,
    status: str,
    reviewer: str,
    decided_at: str,
    rationale: str,
    uncertainty: str | None = None,
    member: str | None = None,
) -> dict[str, Any]:
    """Record one reviewer decision if the result still passes ``validate_manifest``.

    With ``member``, the decision is about that member of an ``unclear_quarantined``
    cluster (``member_rules``); otherwise it is the cluster's. An ``unclear_quarantined``
    decision appends ``uncertainty`` (required) to the notes; no decision removes an
    existing note.
    """
    if status == "pending":
        raise ValueError("a decision cannot return a cluster to pending")
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("a decision needs a reviewer and a rationale")
    if status == "unclear_quarantined" and not (uncertainty and uncertainty.strip()):
        raise ValueError("an unclear decision must state the uncertainty")
    target: dict[str, Any] | None = next(
        (item for item in manifest["clusters"] if item["cluster_id"] == cluster), None
    )
    if target is None:
        raise KeyError(cluster)
    decision = {
        "status": status,
        "reviewer": reviewer,
        "decided_at": decided_at,
        "rationale": rationale,
    }
    updated = copy.deepcopy(target)
    if member is None:
        updated["adjudication"] = decision
    else:
        updated.setdefault("member_decisions", {})[member] = decision
    if uncertainty:
        updated["uncertainty"].append(uncertainty)
    # Checked on a copy: a refused decision leaves the manifest as it was.
    validate_manifest(
        {**manifest, "clusters": [updated if c is target else c for c in manifest["clusters"]]}
    )
    target.clear()
    target.update(updated)
    return target
