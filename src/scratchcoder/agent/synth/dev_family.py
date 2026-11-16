"""``dev-family``: the development task-family rule on synthetic agent rows and replay rows.

The rule is the frozen task-family rule of :mod:`scratchcoder.data.clusters`
(``task-clusters-v1``, default thresholds), applied the way the pilot intake's
``dev_family_ids`` and the holdout integrity scan's ``scan`` apply it, with the pool's
development split as the target. Static text and AST signals only: nothing here runs any
code, and tests are never a signal.

**Nodes.** The family graph holds every pool development task (the targets), every pool
training task, and every synthetic instance (:class:`Node`). A pool task is one node with
one prompt and one solution. A synthetic instance carries two kinds of items
(:data:`MAPPING`, :func:`instance_node`):

*Family items* join families (union signals), as a pool task's prompt and solution do. They
are what the instance's rows train the model to do: the prompts are the template's
specification text and the target function's docstring, the descriptions of the task; the
one solution is the reference target, the function every row's reply is. The node stands for
every (prompt, solution) pair of these, joined into one family (:func:`items_of`).

*Exposure items* are what the rows show without training it as the answer: the original
(buggy or stub) target, the wrong attempts, every module of the original and the fixed
repository and each top-level function or class of those on its own, any code inside a
row's user text, and the user texts and the task statement as text. They never join
families; they are checked directly (*exposure*, below), so buggy and helper code that a
prompt shows still meets the AST and overlap checks.

What the generator emits for more than one template, or fills in from a template's test
values, is *scaffolding* and is never a family item. The task statement is one of the
generator's phrasings (``instances.TASK_PHRASES``, with the traceback framing
``TRACEBACK_HEADER`` and ``FRAME_LINE``) filled in with a call, literal values, test,
exception, helper file and function names: none of it describes the task, and literal
values alone (digit runs, lists) reach the paraphrase level between unrelated templates. It
is exposure text only; :func:`strip_task` refuses a statement the generator did not write.
The harness prompt's fixed blocks (map, snippet headers, changes-so-far block, results,
instruction) and the catalog's non-target functions, which most templates of a family share,
are scaffolding too. Only the target function is the trained unit (every case has one
``module::function`` target), so no whole module is a family item.

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
    when its family contains any development task (``dev_family``); when it shares a
    defined function or class name with a development task at prompt Jaccard at or above
    0.25 (``dev_named_paraphrase``); when an exposure code item has the solution AST or
    canonical AST of a node in a family with a development task (that task, a pool task, or
    another accepted or bridge instance; never the node itself) (``prompt_code_exposure``);
    when an exposure text reaches prompt Jaccard 0.5 with a prompt of such a node
    (``prompt_text_exposure``); or when a name defined in its family or
    exposure code is shared with a development task at prompt Jaccard at or above 0.25 over
    its family and exposure texts (``prompt_named_exposure``);
``flagged``
    when it is not excluded but one of the two shared-name signals reaches 0.10
    (``dev_named_flag``, then ``prompt_named_exposure``);
``clear``
    otherwise.

*Bridges* (rejected and superseded instances) and pool training tasks are never judged, but
they carry links under the same mapping, so a subject joined to a development task only
through a rejected row is still excluded. A bridge's own direct signals (shared-name edges
and every exposure check) are scored (``bridges_hit``): the caller drops every subject of a
template that holds such a bridge.

**Replay rows** are pool training tasks. One is newly excluded when its family, with the
synthetic nodes added, holds a development task that its family without them did not: a new
chain through synthetic nodes, either into a first development task or into one more. Rows
whose family already held a development task and gains none were judged when the replay set
was built; this check reports them and does not overturn that outcome. Rows on pending
(unreviewed) shared-name flag edges to development tasks are dropped unless an existing
adjudication record resolves every such edge as not equivalent (:func:`pending_flags`).

**Evidence** (:func:`link_changes`): every direct link of the earlier mapping
(:data:`LEGACY_MAPPING`, :func:`legacy_items`) that this mapping no longer makes, with the
category of the item that made it (:data:`CATEGORIES`), and every link this mapping adds. A
removed link that only retained items made is a bug and raises.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases, workspace
from scratchcoder.agent import prompt as harness_prompt
from scratchcoder.agent.synth import instances as generator
from scratchcoder.agent.synth.template import Catalog
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

SCHEMA = "agent-dev-family-v2"
RULE = RULES_VERSION
"""The identity key every result carries (``dev_family``) and ``mix`` requires."""
MAPPING = "synthetic-items-v3"
"""The synthetic item mapping: family items (specification text, target docstring; the
reference target) and direct-only exposure items (everything else a row shows, the task
statement included)."""
LEGACY_MAPPING = "synthetic-items-v2"
"""The earlier mapping, kept only to document what changed: every prompt (task statement,
specification text, target docstring, every row's user text) and every solution (reference
and original target, every fixed-repository module and each of its top-level functions and
classes) was a family item."""
NAMESPACE = "synth:"
STATUSES = ("excluded", "flagged", "clear")
JUDGED_REASONS = (
    "dev_family",
    "dev_named_paraphrase",
    "prompt_code_exposure",
    "prompt_text_exposure",
    "prompt_named_exposure",
    "dev_named_flag",
)
"""Reasons that judge one instance, in precedence order."""
REASONS = (*JUDGED_REASONS, "template", "owner_ruling")
"""Reason classes of a dropped instance: the judged ones; ``template`` drops every instance
of a template that holds a judged (excluded or flagged) instance, or a bridge with a direct
signal; ``owner_ruling`` drops every instance of a template the owner named."""
REPLAY_REASONS = ("dev_family_new_chain", "dev_family_new_link", "pending_flag_unresolved")
EXPOSURE_REASONS = ("prompt_code_exposure", "prompt_text_exposure", "prompt_named_exposure")
METHOD = (
    "task-family rule task-clusters-v1 against the pool development split: union-signal "
    "families over pool development and training tasks and every synthetic instance "
    "(rejected and superseded instances as unjudged bridges), where a synthetic instance's "
    "family items are its template's specification text, its target docstring and its "
    "reference target; every other code and text its rows show, the generated task "
    "statement included, is checked directly (solution and canonical AST against every "
    "other node in a family with a development task, prompt Jaccard 0.5 against their "
    "prompts, shared defined names against development tasks); an accepted instance "
    "is dropped when its family holds a development task or any direct signal reaches the "
    "flag level, and every instance of a template with a dropped instance, a bridge with a "
    "direct signal, or an owner ruling is dropped; replay rows are dropped when synthetic "
    "nodes add a development task to their family or a pending shared-name flag edge to a "
    "development task is not resolved as not equivalent by an adjudication record"
)


@dataclass(frozen=True)
class Node:
    """One graph node: every (prompt, solution) pair of it is one task, all joined. The
    ``shown`` items are exposure items: checked directly, never a family link."""

    id: str
    prompts: tuple[str, ...] = ()
    solutions: tuple[str, ...] = ()
    shown_code: tuple[str, ...] = ()
    shown_text: tuple[str, ...] = ()


def task_node(task: Task) -> Node:
    return Node(task.id, (task.prompt,), (task.solution,) if task.solution else ())


def items_of(node: Node, split: str) -> list[Item]:
    """The task items a node stands for: the cross product of its prompts and solutions
    (a node without solutions stands for its prompts alone). IDs after the first carry a
    ``#n`` suffix; callers join them to the first. Exposure items are not task items."""
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


def _keys(text: str) -> Iterator[tuple[str, str]]:
    for kind, key in (("solution_ast", solution_key(text)), ("canonical_ast", canonical_key(text))):
        if key is not None:
            yield kind, key


@dataclass(frozen=True)
class Exposure:
    """The direct signals of one node's exposure items."""

    code: bool = False
    """An exposure code item has the solution or canonical AST of an exposure target."""
    text: bool = False
    """An exposure text reaches the paraphrase level with an exposure target's prompt."""
    named: float | None = None
    """The strongest shared-name score over family and exposure items (``named_scores``)."""


def exposures(
    targets: Sequence[Node],
    named_targets: Sequence[Node],
    nodes: Sequence[Node],
    rules: ClusterRules = DEFAULT_RULES,
) -> list[Exposure]:
    """The direct signals of each node's exposure items: code against the solutions of
    ``targets`` (every node in a family with a development task: pool tasks, accepted and
    bridge instances), text against their prompts, and shared names against
    ``named_targets`` (development tasks). A node's items are never checked against a
    target under its own ID (an instance's repository holds its own target)."""
    key_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    set_owners: dict[frozenset[tuple[str, ...]], set[str]] = defaultdict(set)
    for target in targets:
        for text in target.solutions:
            for key in _keys(text):
                key_owners[key].add(target.id)
        for text in target.prompts:
            if found := shingles(text, rules.width):
                set_owners[found].add(target.id)
    shown_sets = list(
        dict.fromkeys(
            found
            for node in nodes
            for text in node.shown_text
            if (found := shingles(text, rules.width))
        )
    )
    target_sets = list(set_owners)
    # Shown sets after the target sets, identical ones included, so each pair is scored.
    everything = [*target_sets, *shown_sets]
    first_shown = len(target_sets)
    reached: dict[frozenset[tuple[str, ...]], set[str]] = defaultdict(set)
    for i, j, _ in similar_pairs(everything, rules.paraphrase):
        if i < first_shown <= j:  # ``i < j``: a target set, then a shown set.
            reached[everything[j]] |= set_owners[everything[i]]
    widened = [
        Node(
            node.id,
            (*node.prompts, *node.shown_text),
            (*node.solutions, *node.shown_code),
        )
        for node in nodes
    ]
    named = named_scores(named_targets, widened, rules)
    signals = []
    for node, score in zip(nodes, named, strict=True):
        own = {node.id}
        code_hit = any(
            key_owners.get(key, own) - own for code in node.shown_code for key in _keys(code)
        )
        text_hit = any(
            reached.get(shingles(item, rules.width), own) - own for item in node.shown_text
        )
        signals.append(Exposure(code_hit, text_hit, score))
    return signals


@dataclass(frozen=True)
class Judgement:
    id: str
    status: str
    reason: str | None


@dataclass
class Assessment:
    subjects: list[Judgement]
    replay_excluded: list[str]
    """Replay rows whose family gains a development task through synthetic nodes (dropped):
    both kinds below."""
    replay_new_chain: list[str]
    """Of those, rows whose family held no development task without synthetic nodes."""
    replay_new_dev_link: list[str]
    """Of those, rows whose family already held a development task without synthetic nodes
    and is joined to a further one through them (that link was never adjudicated)."""
    replay_in_dev_family: list[str]
    """Replay rows whose family held a development task without synthetic nodes and gains
    none through them (kept: the replay stage's outcome stands)."""
    replay_named: list[str]
    """Replay rows outside every development family, with or without synthetic nodes, that
    have a direct shared-name edge to a development task at the flag level or above (an
    in-graph stand-in for the pending flag edges, which :func:`pending_flags` applies)."""
    bridges_hit: dict[str, str]
    """Bridges with a direct signal (a shared-name edge to a development task at the flag
    level or above, or an exposure hit) and its reason. Bridges are not judged; the caller
    drops every subject of their templates."""
    families: dict[str, int]
    exposure_hits: dict[str, int] = field(default_factory=dict)
    """Subjects with each direct exposure signal, whatever their judged reason."""


_RANK = {reason: rank for rank, reason in enumerate(JUDGED_REASONS)}


def _status(
    in_family: bool, score: float | None, exposure: Exposure, rules: ClusterRules
) -> tuple[str, str | None]:
    if in_family:
        return "excluded", "dev_family"
    if score is not None and score >= rules.named_paraphrase:
        return "excluded", "dev_named_paraphrase"
    if exposure.code:
        return "excluded", "prompt_code_exposure"
    if exposure.text:
        return "excluded", "prompt_text_exposure"
    if exposure.named is not None and exposure.named >= rules.named_paraphrase:
        return "excluded", "prompt_named_exposure"
    if score is not None and score >= rules.named_flag:
        return "flagged", "dev_named_flag"
    if exposure.named is not None and exposure.named >= rules.named_flag:
        return "flagged", "prompt_named_exposure"
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
    # Development tasks per family, with and without the synthetic nodes.
    full_dev: dict[int, set[int]] = defaultdict(set)
    base_dev: dict[int, set[int]] = defaultdict(set)
    for index in range(len(dev)):
        full_dev[graph.full.find(index)].add(index)
        base_dev[graph.base.find(index)].add(index)
    first_subject = len(base)
    # Exposure targets: every node in a family with a development task (development and
    # training tasks, accepted and bridge instances), so code a prompt shows that another
    # instance in such a family trains on is checked as a pool task's program would be.
    in_dev = [index for index in range(len(graph.nodes)) if graph.full.find(index) in full_dev]
    targets = [graph.nodes[index] for index in in_dev]
    synthetic_targets = sum(1 for index in in_dev if index >= first_subject)
    synthetic = [*subjects, *bridges]
    exposure = exposures(targets, dev, synthetic, rules)
    scores = named_scores(dev, synthetic, rules)
    judged = []
    roots: dict[int, int] = defaultdict(int)
    hits: Counter[str] = Counter()
    for offset, node in enumerate(subjects):
        root = graph.full.find(first_subject + offset)
        roots[root] += 1
        found = exposure[offset]
        status, reason = _status(root in full_dev, scores[offset], found, rules)
        judged.append(Judgement(node.id, status, reason))
        hits["prompt_code_exposure"] += found.code
        hits["prompt_text_exposure"] += found.text
        if found.named is not None and found.named >= rules.named_flag:
            hits["prompt_named_exposure"] += 1
    bridges_hit: dict[str, str] = {}
    for offset, node in enumerate(bridges, start=len(subjects)):
        _, reason = _status(False, scores[offset], exposure[offset], rules)
        if reason is not None:
            # A bridge under the ID of another bridge keeps its strongest reason.
            held = bridges_hit.get(node.id)
            if held is None or _RANK[reason] < _RANK[held]:
                bridges_hit[node.id] = reason
    position = {node.id: index for index, node in enumerate(train, start=len(dev))}
    unknown = [identifier for identifier in replay if identifier not in position]
    if unknown:
        raise ValueError(f"{len(unknown)} replay rows are not pool training tasks")
    new_chain: list[str] = []
    new_link: list[str] = []
    already: list[str] = []
    for identifier in replay:
        index = position[identifier]
        before = base_dev.get(graph.base.find(index), set())
        after = full_dev.get(graph.full.find(index), set())
        if len(after) > len(before):  # ``after`` always contains ``before``.
            (new_link if before else new_chain).append(identifier)
        elif before:
            already.append(identifier)
    judged_already = {*already, *new_chain, *new_link}
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
        "exposure_targets": len(targets) - synthetic_targets,
        "exposure_targets_synthetic": synthetic_targets,
    }
    return Assessment(
        subjects=judged,
        replay_excluded=sorted([*new_chain, *new_link]),
        replay_new_chain=sorted(new_chain),
        replay_new_dev_link=sorted(new_link),
        replay_in_dev_family=sorted(already),
        replay_named=sorted(replay_named),
        bridges_hit=dict(sorted(bridges_hit.items())),
        families=families,
        exposure_hits={reason: hits[reason] for reason in EXPOSURE_REASONS},
    )


def expand_templates(
    judged: Sequence[Judgement],
    template_of: Mapping[str, str],
    also_templates: Iterable[str] = (),
    owner_templates: Iterable[str] = (),
) -> dict[str, str]:
    """Every dropped subject and its reason class: each subject of a template in
    ``owner_templates`` (``owner_ruling``, whatever its judgement); each other excluded or
    flagged subject; and every other subject of a template that holds one or is in
    ``also_templates`` (the templates of bridges with a direct signal): ``template``."""
    owner = set(owner_templates)
    dropped = {item.id: "owner_ruling" for item in judged if template_of[item.id] in owner}
    for item in judged:
        if item.id not in dropped and item.status != "clear":
            dropped[item.id] = str(item.reason)
    templates = {template_of[identifier] for identifier in dropped} | set(also_templates)
    for item in judged:
        if item.id not in dropped and template_of[item.id] in templates:
            dropped[item.id] = "template"
    return dropped


# -- Pending flag edges of replay rows ------------------------------------------------------


@dataclass
class PendingFlags:
    candidates: list[str]
    """Replay rows that are an endpoint of a flag edge to a development task."""
    kept: dict[str, list[int]]
    """Candidates every one of whose edges an adjudication record resolves as not
    equivalent, with the decision lines (1-based) that resolve them."""
    excluded: list[str]
    """Every other candidate (``pending_flag_unresolved``)."""
    edges: int
    """Flag edges to development tasks in the manifest."""


NOT_EQUIVALENT = "not_equivalent"


def flag_edges(manifest: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(row, development task, review)`` for every manifest flag edge with exactly one
    development-split endpoint. Refuses a manifest without ``flag_edges`` and an edge
    without two endpoints and two splits: a malformed edge is never skipped silently."""
    if not isinstance(manifest.get("flag_edges"), list):
        raise ValueError("the flag manifest has no flag_edges list")
    found = []
    for number, edge in enumerate(manifest["flag_edges"], start=1):
        splits = edge.get("splits") if isinstance(edge, Mapping) else None
        if (
            not isinstance(splits, list)
            or len(splits) != 2
            or not all(isinstance(split, str) and split for split in splits)
            or not edge.get("a")
            or not edge.get("b")
        ):
            raise ValueError(f"flag edge {number} lacks two endpoints and two splits")
        ends = [str(edge["a"]), str(edge["b"])]
        if splits.count("dev") != 1:
            continue
        dev_side = splits.index("dev")
        found.append((ends[1 - dev_side], ends[dev_side], str(edge.get("review", "pending"))))
    return found


def pending_flags(
    edges: Sequence[tuple[str, str, str]],
    decisions: Sequence[Mapping[str, Any]],
    rows: Iterable[str],
) -> PendingFlags:
    """Which replay ``rows`` stand on flag edges to development tasks, and which of those an
    existing record resolves: an edge is resolved only when its own review is
    ``not_equivalent`` or the decisions on exactly that (training task, development task)
    pair are all final ``not_equivalent`` (one that says anything else leaves it
    unresolved). A row is kept only when every one of its edges is resolved."""
    finals: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for line, decision in enumerate(decisions, start=1):
        pair = (str(decision.get("train")), str(decision.get("dev")))
        finals[pair].append((line, str(decision.get("final"))))
    resolving = {
        pair: [line for line, _ in lines]
        for pair, lines in finals.items()
        if all(final == NOT_EQUIVALENT for _, final in lines)
    }
    by_row: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row, dev, review in edges:
        by_row[row].append((dev, review))
    candidates = sorted({row for row in rows if row in by_row})
    kept: dict[str, list[int]] = {}
    excluded = []
    for row in candidates:
        lines: list[int] = []
        resolved = True
        for dev, review in by_row[row]:
            if review == NOT_EQUIVALENT:
                continue
            cited = resolving.get((row, dev))
            if not cited:
                resolved = False
                break
            lines.extend(cited)
        if resolved:
            kept[row] = sorted(set(lines))
        else:
            excluded.append(row)
    return PendingFlags(candidates, kept, excluded, len(edges))


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
    spec_sha256: str = ""
    """The registered SHA-256 of its ``case.json`` (what each row's ``spec_sha256`` names)."""
    original_modules: tuple[tuple[str, str], ...] = ()
    """The original repository's source modules (tests left out), as the rows show them."""
    mutants: tuple[str, ...] = ()
    """The wrong attempts' sources (each a version of the target)."""
    names: tuple[tuple[str, str], ...] = ()
    """The generator's rendered names: placeholder (function role) and identifier."""


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
        original = {
            Path(name).name: texts[name].decode("utf-8")
            for name in texts
            if Path(name).parent == Path(repo)
            and name.endswith(".py")
            and not workspace.is_test_file(Path(name).name)
        }
        modules = dict(original)
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
                spec_sha256=str(entry["files"]["case.json"]),
                original_modules=tuple(sorted(original.items())),
                mutants=tuple(str(item["source"]) for item in synth.get("mutants", ())),
                names=tuple(sorted((str(k), str(v)) for k, v in synth.get("names", {}).items())),
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


_UNITS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def top_level_units(source: str) -> list[str]:
    """Each module-level function and class of ``source`` on its own, as source text."""
    tree = _parse(source)
    if tree is None:
        return []
    return [ast.unparse(node) for node in tree.body if isinstance(node, _UNITS)]


def _defines(text: str) -> bool:
    tree = _parse(text)
    return tree is not None and any(isinstance(node, _UNITS) for node in tree.body)


@cache
def _phrasings() -> tuple[tuple[re.Pattern[str], tuple[str, ...], int], ...]:
    """Each generator phrasing as a pattern, its field names and its fixed-text length,
    longest fixed text first (the most specific phrasing wins)."""
    found = []
    for order, phrase in enumerate(generator.TASK_PHRASES):
        pattern, fields, fixed = "", [], 0
        for literal, name, _, _ in string.Formatter().parse(phrase):
            pattern += re.escape(literal)
            fixed += len(literal)
            if name is not None:
                pattern += "(.+?)"
                fields.append(name)
        found.append((order, re.compile(pattern, re.DOTALL), tuple(fields), fixed))
    found.sort(key=lambda item: (-item[3], item[0]))
    return tuple((pattern, fields, fixed) for _, pattern, fields, fixed in found)


_FRAME = re.compile(
    "^"
    + re.escape(generator.FRAME_LINE)
    .replace(re.escape("{file}"), "(.*)")
    .replace(re.escape("{function}"), "(.*)")
    + "$"
)


def _strip_traceback(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        if line == generator.TRACEBACK_HEADER:
            continue
        frame = _FRAME.match(line)
        lines.append(" ".join(frame.groups()) if frame else line)
    return "\n".join(lines)


def strip_task(task: str) -> str:
    """The task statement without the generator's fixed phrasing: the values it filled in
    (call, values, test, exception, function and file names, traceback frames and error
    line), one per line. Raises when no generator phrasing matches: a statement written
    some other way is not stripped by guesswork. :func:`instance_node` uses it only to refuse
    a statement the generator did not write (the statement is exposure text, never a family
    item)."""
    for pattern, fields, _ in _phrasings():
        match = pattern.fullmatch(task)
        if match is None:
            continue
        values = []
        for name, value in zip(fields, match.groups(), strict=True):
            values.append(_strip_traceback(value) if name == "traceback" else value)
        return "\n".join(values)
    raise ValueError("a task statement matches no generator phrasing")


_HEADERS = ("SNIPPET ", harness_prompt.PREVIOUS_HEADER, "RESULT ", "MAP\n")


def code_blocks(user_text: str) -> list[str]:
    """The Python sources a row's user text shows: each snippet block's function (its
    header line removed; a snippet runs to the next block header or the instruction), and
    any other blank-line-separated block that parses and defines a top-level function or
    class. Text is parsed, never run."""
    found: list[str] = []
    current: list[str] | None = None
    for part in user_text.split("\n\n"):
        boundary = part.startswith(_HEADERS) or part.strip() == harness_prompt.INSTRUCTION
        if current is not None and not boundary:
            current.append(part)
            continue
        if current is not None:
            found.append("\n\n".join(current))
            current = None
        if part.startswith("SNIPPET "):
            current = [part.partition("\n")[2]]
        elif _defines(part):
            found.append(part)
    if current is not None:
        found.append("\n\n".join(current))
    return [code for code in found if _defines(code)]


def _unique(texts: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for text in texts if text.strip()))


def instance_node(instance: Instance, specs: Iterable[str], user_texts: Iterable[str]) -> Node:
    """The node of one instance under :data:`MAPPING` (module docstring: family and
    exposure items). Refuses a task statement the generator did not write."""
    user_texts = tuple(user_texts)
    strip_task(instance.task)  # Generated text, scaffolding only: refused otherwise.
    prompts = [*specs, *docstrings(instance.reference_target)]
    shown = [instance.original_target, *instance.mutants]
    for _, text in (*instance.original_modules, *instance.modules):
        shown.append(text)
        shown.extend(top_level_units(text))
    for text in user_texts:
        for code in code_blocks(text):
            shown.append(code)
            shown.extend(top_level_units(code))
    return Node(
        NAMESPACE + instance.id,
        _unique(prompts),
        _unique([instance.reference_target]),
        _unique(shown),
        _unique([instance.task, *user_texts]),
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


# -- Evidence: the links the earlier mapping made and this one does not ---------------------


CATEGORIES = {
    "original_target": "original_target_as_solution",
    "repo_module": "repo_module_as_solution",
    "shared_unit": "shared_scaffolding_code",
    "own_unit": "non_target_function",
    "unresolved_unit": "non_target_code_unresolved",
    "user_text": "harness_text",
    "task": "generated_task_statement",
}
"""Item kind -> the category of a link it alone made in one mapping and not the other.
``shared_unit`` is a non-target function the catalog puts in more than one template;
``own_unit`` one only this template has; ``unresolved_unit`` one whose role the generator's
recorded names do not give; ``task`` the generated task statement (phrasing and values)."""
RETAINED = frozenset(
    {"reference_target", "target_unit", "spec", "docstring", "pool_prompt", "pool_solution"}
)
"""Item kinds both mappings hold as the same family item (a ``target_unit`` is the reference
target as the fixed module prints it, the same program)."""


@dataclass(frozen=True)
class Labeled:
    """A node's family items with their kinds, for comparing two mappings."""

    id: str
    group: str
    """``template:<id>`` for a synthetic node, ``pool:<split>`` for a pool task."""
    prompts: tuple[tuple[str, str], ...] = ()
    solutions: tuple[tuple[str, str], ...] = ()

    def node(self) -> Node:
        return Node(
            self.id,
            _unique(text for _, text in self.prompts),
            _unique(text for _, text in self.solutions),
        )


def pool_labeled(task: Task) -> Labeled:
    return Labeled(
        task.id,
        f"pool:{task.split}",
        (("pool_prompt", task.prompt),),
        (("pool_solution", task.solution),) if task.solution else (),
    )


def unit_kinds(catalog: Catalog) -> dict[str, dict[str, str]]:
    """Template ID -> function role -> ``target_unit``, ``shared_unit`` (a non-target role
    another template of the catalog also holds) or ``own_unit``: the generator source's
    definition of scaffolding code."""
    holders: dict[tuple[str, str], set[str]] = defaultdict(set)
    for template in catalog.templates:
        for role in template.roles():
            holders[(template.family.id, role)].add(template.id)
    found: dict[str, dict[str, str]] = {}
    for template in catalog.templates:
        kinds = {}
        for role in template.roles():
            if role == template.target:
                kinds[role] = "target_unit"
            elif holders[(template.family.id, role)] - {template.id}:
                kinds[role] = "shared_unit"
            else:
                kinds[role] = "own_unit"
        found[template.id] = kinds
    return found


def _first_name(source: str) -> str | None:
    tree = _parse(source)
    if tree is None or not tree.body or not isinstance(tree.body[0], _UNITS):
        return None
    return tree.body[0].name


def _unit_kind(instance: Instance, unit: str, roles: Mapping[str, Mapping[str, str]]) -> str:
    """A fixed-module unit's kind: the target, or the kind :func:`unit_kinds` gives its role
    (found through the instance's recorded names), else ``unresolved_unit``."""
    name = _first_name(unit)
    if name is not None and name == _first_name(instance.reference_target):
        return "target_unit"
    role_of = {identifier: role for role, identifier in instance.names}
    role = role_of.get(name or "")
    return roles.get(instance.template_id, {}).get(role or "", "unresolved_unit")


def legacy_items(
    instance: Instance,
    specs: Iterable[str],
    user_texts: Iterable[str],
    roles: Mapping[str, Mapping[str, str]],
) -> Labeled:
    """The instance's family items under :data:`LEGACY_MAPPING`, with their kinds."""
    prompts = [("task", instance.task), *(("spec", text) for text in specs)]
    prompts += [("docstring", text) for text in docstrings(instance.reference_target)]
    prompts += [("user_text", text) for text in user_texts]
    solutions = [("reference_target", instance.reference_target)]
    solutions.append(("original_target", instance.original_target))
    for _, text in instance.modules:
        solutions.append(("repo_module", text))
        solutions += [(_unit_kind(instance, unit, roles), unit) for unit in top_level_units(text)]
    return Labeled(
        NAMESPACE + instance.id,
        f"template:{instance.template_id}",
        _labeled_unique(prompts),
        _labeled_unique(solutions),
    )


def current_items(instance: Instance, specs: Iterable[str]) -> Labeled:
    """The instance's family items under :data:`MAPPING`, with their kinds (the same texts
    as :func:`instance_node`'s prompts and solutions)."""
    prompts = [("spec", text) for text in specs]
    prompts += [("docstring", text) for text in docstrings(instance.reference_target)]
    return Labeled(
        NAMESPACE + instance.id,
        f"template:{instance.template_id}",
        _labeled_unique(prompts),
        _labeled_unique([("reference_target", instance.reference_target)]),
    )


def _labeled_unique(items: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Items with nonblank, distinct texts; a repeated text keeps its first kind."""
    seen: dict[str, str] = {}
    for kind, text in items:
        if text.strip() and text not in seen:
            seen[text] = kind
    return tuple((kind, text) for text, kind in seen.items())


class _Direct:
    """The direct links of a labeled node list: key groups, prompt sets and similar pairs."""

    def __init__(self, nodes: Sequence[Labeled], rules: ClusterRules) -> None:
        self.size = len(nodes)
        self.groups: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.group_names: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.keys: list[set[tuple[str, str]]] = [set() for _ in nodes]
        self.upstream = [canonical_id(node.id) for node in nodes]
        index: dict[frozenset[tuple[str, ...]], int] = {}
        self.sets: list[frozenset[tuple[str, ...]]] = []
        self.holders: list[dict[int, set[str]]] = []
        self.node_sets: list[set[int]] = [set() for _ in nodes]
        for position, node in enumerate(nodes):
            for kind, text in node.solutions:
                names = defined_names(text)
                for key in _keys(text):
                    self.groups[key][position].add(kind)
                    self.group_names[key] |= names
                    self.keys[position].add(key)
            for kind, text in node.prompts:
                found = shingles(text, rules.width)
                if not found:
                    continue
                if found not in index:
                    index[found] = len(self.sets)
                    self.sets.append(found)
                    self.holders.append(defaultdict(set))
                number = index[found]
                self.holders[number][position].add(kind)
                self.node_sets[position].add(number)
        self.similar = [(i, j) for i, j, _ in similar_pairs(self.sets, rules.paraphrase)]
        self.similar_to: dict[int, set[int]] = defaultdict(set)
        for i, j in self.similar:
            self.similar_to[i].add(j)
            self.similar_to[j].add(i)

    def linked(self, a: int, b: int) -> bool:
        if self.upstream[a] == self.upstream[b] or self.keys[a] & self.keys[b]:
            return True
        own, other = self.node_sets[a], self.node_sets[b]
        if own & other:
            return True
        return any(self.similar_to.get(number, set()) & other for number in own)

    def pairs(self) -> Iterator[tuple[str, int, int, set[str], set[str], frozenset[str]]]:
        """Every directly linked node pair once per signal group: (signal, a, b, the kinds
        of ``a``'s and of ``b``'s items that made it, the defined names involved)."""
        for key, holders in self.groups.items():
            members = sorted(holders)
            names = frozenset(self.group_names[key])
            for x, a in enumerate(members):
                for b in members[x + 1 :]:
                    yield key[0], a, b, holders[a], holders[b], names
        for holders in self.holders:
            members = sorted(holders)
            for x, a in enumerate(members):
                for b in members[x + 1 :]:
                    yield "prompt_paraphrase", a, b, holders[a], holders[b], frozenset()
        empty: set[str] = set()
        for i, j in self.similar:
            first, second = self.holders[i], self.holders[j]
            for a in first:
                for b in second:
                    if a == b:
                        continue
                    kinds_a = first[a] | second.get(a, empty)
                    kinds_b = second[b] | first.get(b, empty)
                    if a > b:
                        a, b, kinds_a, kinds_b = b, a, kinds_b, kinds_a
                    yield "prompt_paraphrase", a, b, kinds_a, kinds_b, frozenset()


def _pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _cause(kinds_a: set[str], kinds_b: set[str]) -> set[str]:
    """The categories of a link one mapping makes and the other does not: the kinds on each
    side that holds no retained item (a side with one would make the link in both)."""
    found: set[str] = set()
    for kinds in (kinds_a, kinds_b):
        if not kinds & RETAINED:
            found |= {CATEGORIES[kind] for kind in kinds}
    return found


def _changes(
    made: _Direct,
    other: _Direct,
    groups: Sequence[str],
    base_size: int,
    direction: str,
) -> dict[str, Any]:
    pairs: dict[int, set[str]] = {}
    detail: dict[tuple[str, str], dict[str, set[str]]] = {}
    size = made.size
    for signal, a, b, kinds_a, kinds_b, names in made.pairs():
        if b < base_size or other.linked(a, b):
            continue
        categories = _cause(kinds_a, kinds_b)
        if not categories:
            raise ValueError(
                f"a {direction} link between {groups[a]} and {groups[b]} rests only on items "
                "both mappings hold"
            )
        pairs.setdefault(a * size + b, set()).update(categories)
        row = detail.setdefault(
            _pair(groups[a], groups[b]), {"signals": set(), "kinds": set(), "names": set()}
        )
        row["signals"].add(signal)
        row["kinds"].update(kinds_a | kinds_b)
        row["names"].update(names)
    by_category: Counter[str] = Counter()
    by_set: Counter[str] = Counter()
    by_groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    kinds_of_pair: Counter[str] = Counter()
    for code, categories in pairs.items():
        a, b = divmod(code, size)
        label = "+".join(sorted(categories))
        by_category.update(categories)
        by_set[label] += 1
        by_groups[_pair(groups[a], groups[b])][label] += 1
        if groups[a].startswith("pool:") or groups[b].startswith("pool:"):
            kinds_of_pair["synthetic_pool"] += 1
        elif groups[a] == groups[b]:
            kinds_of_pair["same_template"] += 1
        else:
            kinds_of_pair["cross_template"] += 1
    rows = [
        {
            "groups": list(pair),
            "pairs_by_category": dict(sorted(by_groups[pair].items())),
            "signals": sorted(detail[pair]["signals"]),
            "item_kinds": sorted(detail[pair]["kinds"]),
            "defined_names": sorted(detail[pair]["names"]),
        }
        for pair in sorted(by_groups)
    ]
    summary = {
        "links": len(pairs),
        "by_category": dict(sorted(by_category.items())),
        "by_category_set": dict(sorted(by_set.items())),
        "by_pair_kind": {
            kind: kinds_of_pair[kind]
            for kind in ("synthetic_pool", "same_template", "cross_template")
        },
    }
    return {"summary": summary, "rows": rows}


def link_changes(
    old: Sequence[Labeled],
    new: Sequence[Labeled],
    base_size: int,
    rules: ClusterRules = DEFAULT_RULES,
) -> dict[str, Any]:
    """Every direct node link (pairwise, any union signal) of ``old`` that ``new`` lacks
    (``removed``) and of ``new`` that ``old`` lacks (``added``), node lists aligned by index,
    pool-only pairs (the first ``base_size`` nodes) left out. Each link carries the
    categories of the items that made it (:data:`CATEGORIES`); one that only items both
    mappings hold (:data:`RETAINED`) made is a bug and raises ``ValueError``."""
    if [node.id for node in old] != [node.id for node in new]:
        raise ValueError("the two mappings must hold the same nodes in the same order")
    groups = [node.group for node in old]
    before, after = _Direct(old, rules), _Direct(new, rules)
    return {
        "removed": _changes(before, after, groups, base_size, "removed"),
        "added": _changes(after, before, groups, base_size, "added"),
    }


# -- Results ---------------------------------------------------------------------------------


def rules_record(rules: ClusterRules = DEFAULT_RULES) -> dict[str, Any]:
    return asdict(rules)


def require_result(
    result: Mapping[str, Any],
    *,
    agent_meta: Path,
    replay_index: Path,
    replay_kinds: Iterable[str],
    dev_sha256: Iterable[str | None] = (),
    bridge_registries: Iterable[str] = (),
    dropped_templates: Iterable[str] = (),
    flag_manifest_sha256: str | None = None,
) -> None:
    """Refuse a dev-family result unless it applies this rule, with its default thresholds
    and this item mapping, to exactly ``agent_meta`` and ``replay_index`` as they are now,
    covers every replay kind mixed, applied the pending flag edges of a manifest whose
    in-graph count of rows on them the replay stage's record confirmed (and, when
    ``flag_manifest_sha256`` is given, of exactly that manifest), judged against the same
    development file as any n-gram check that names one (``dev_sha256``), held every
    instance set named by ``bridge_registries`` (registry SHA-256) in its graph, as a judged
    set or a bridge set, and dropped every template in ``dropped_templates`` by owner ruling."""
    if result.get("schema") != SCHEMA or result.get("dev_family") != RULE:
        raise ValueError(f"not a {SCHEMA} result of {RULE}")
    synthetic = result.get("synthetic") or {}
    if (synthetic.get("item_mapping") or {}).get("version") != MAPPING:
        raise ValueError(f"the dev-family result did not use the item mapping {MAPPING}")
    held = {
        str(part.get("registry_sha256"))
        for part in [*synthetic.get("judged", ()), *synthetic.get("bridges", ())]
    }
    absent = sorted(set(bridge_registries) - held)
    if absent:
        raise ValueError(f"the dev-family result did not hold instance sets {absent}")
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
    pending = result["replay"].get("pending_flags") or {}
    if not pending.get("applied"):
        raise ValueError("the dev-family result did not apply the pending flag edges")
    reported = pending.get("replay_stage_reported")
    if reported is None or reported != pending.get("candidates"):
        raise ValueError(
            "the dev-family result's pending flag rows were not confirmed by the replay stage's "
            "record"
        )
    if not pending.get("manifest_sha256"):
        raise ValueError("the dev-family result names no flag manifest")
    if flag_manifest_sha256 is not None and pending["manifest_sha256"] != flag_manifest_sha256:
        raise ValueError("the dev-family result applied another flag manifest")
    owner = set((result["agent"]["instances"].get("owner_drops") or {}).get("templates", ()))
    undropped = sorted(set(dropped_templates) - owner)
    if undropped:
        raise ValueError(f"the dev-family result did not drop templates {undropped}")
    pool_dev = result.get("pool", {}).get("dev_sha256")
    if not pool_dev:
        raise ValueError("the dev-family result names no development file")
    if any(other is not None and other != pool_dev for other in dev_sha256):
        raise ValueError("the dev-family result used another development file")


def load_result(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a dev-family result of schema {SCHEMA}")
    return dict(raw)
