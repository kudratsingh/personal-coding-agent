"""Checkpoint interfaces: how a checkpoint resolves to its prompt and extraction protocol."""

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
from eval_fixtures import TOKENIZER, chat_tokens, pool_task, save_task_file

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data import chat
from scratchcoder.data.evalplus import origin_task, release
from scratchcoder.data.loader import MixedLoader
from scratchcoder.engine.generation import Completion
from scratchcoder.eval import command, prompt, suites
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.completions import prompt_sha256, read_completions
from scratchcoder.eval.generate import generate_completions
from scratchcoder.eval.harness import evaluate
from scratchcoder.eval.interface import (
    BASE_INTERFACE,
    CHAT_INTERFACE,
    DECLARED,
    FROM_TRAINER,
    FROM_WEIGHTS_V1,
    UNDECLARED,
    CheckpointInterface,
    InterfaceMismatchError,
    checkpoint_interface,
    read_checkpoint_interface,
    resolve_legacy_mode,
    resolve_protocol,
)
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.sandbox.runner import Limits, Result, Runtime
from scratchcoder.train.trainer import Trainer

MODEL = ModelConfig(vocab_size=TOKENIZER.vocab_size, width=16, layers=1, heads=2, context=128)
CHAT = CheckpointInterface(CHAT_INTERFACE, DECLARED)
BASE = CheckpointInterface(BASE_INTERFACE, FROM_TRAINER)
NONE = CheckpointInterface(None, UNDECLARED)


def sft_checkpoint(root: Path) -> Path:
    """Post-training weights as the SFT stage writes them (``save_weights``)."""
    torch.manual_seed(0)
    path = root / "sft.pt"
    save_weights(path, GPT(MODEL), tokenizer_hash=TOKENIZER.sha256, run_id="sft", step=4)
    return path


def historical_sft_checkpoint(root: Path) -> Path:
    """A weights-v1 file in the exact layout written before the interface key existed."""
    torch.manual_seed(0)
    path = root / "historical-sft.pt"
    torch.save(
        {
            "format": "weights-v1",
            "config": {"model": asdict(MODEL)},
            "tokenizer_hash": TOKENIZER.sha256,
            "run_id": "posttrain-1b-sft",
            "step": 212,
            "metadata": {"quarter": 4, "dev_loss": 1.0},
            "model": GPT(MODEL).state_dict(),
        },
        path,
    )
    return path


def trainer_checkpoint(
    root: Path,
    *,
    special_rows_init: str = "none",
    loss_normalization: str = "capacity",
    source: str = "web",
) -> Path:
    """A full checkpoint written by the trainer itself (pretraining with the defaults)."""
    config = RunConfig(
        model=MODEL,
        train=TrainConfig(
            steps=1,
            sequence_length=8,
            batch_size=2,
            special_rows_init=special_rows_init,
            loss_normalization=loss_normalization,
        ),
        data_mix={source: 1.0},
    )
    loader = MixedLoader(
        {source: [[256, 1, 2, 3] * 8]}, {source: 1.0}, sequence_length=8, batch_size=2, seed=0
    )
    name = f"{source}-{special_rows_init}-{loss_normalization}"
    trainer = Trainer(
        config,
        loader,
        run_id=f"trainer-{name}",
        run_root=root / "runs",
        data_hash="fixture-data",
        tokenizer_hash=TOKENIZER.sha256,
    )
    path = root / f"trainer-{name}.pt"
    trainer.checkpoint(path)
    return path


def undeclared_checkpoint(root: Path) -> Path:
    path = root / "undeclared.pt"
    torch.save(
        {
            "model": GPT(MODEL).state_dict(),
            "config": {"model": asdict(MODEL)},
            "tokenizer_hash": TOKENIZER.sha256,
        },
        path,
    )
    return path


# The rules ----------------------------------------------------------------------------


def test_each_checkpoint_kind_resolves_by_its_recorded_fields(tmp_path: Path) -> None:
    assert read_checkpoint_interface(sft_checkpoint(tmp_path)) == CHAT
    assert read_checkpoint_interface(historical_sft_checkpoint(tmp_path)) == CheckpointInterface(
        CHAT_INTERFACE, FROM_WEIGHTS_V1
    )
    assert read_checkpoint_interface(trainer_checkpoint(tmp_path)) == BASE
    special = trainer_checkpoint(tmp_path, special_rows_init="ordinary-mean-noise")
    assert read_checkpoint_interface(special) == CheckpointInterface(CHAT_INTERFACE, FROM_TRAINER)
    assert read_checkpoint_interface(undeclared_checkpoint(tmp_path)) == NONE


def test_trainer_checkpoints_of_post_training_runs_resolve_to_chat(tmp_path: Path) -> None:
    # Run 1's SFT configuration: no special-row redraw, but SFT data and target-normalized
    # loss. The more-SFT control trains on rl-references the same way. Any one mark is enough.
    sft_trainer = CheckpointInterface(CHAT_INTERFACE, FROM_TRAINER)
    run_1_style = trainer_checkpoint(tmp_path, loss_normalization="update_targets", source="sft")
    assert read_checkpoint_interface(run_1_style) == sft_trainer
    control = trainer_checkpoint(
        tmp_path, loss_normalization="update_targets", source="rl-references"
    )
    assert read_checkpoint_interface(control) == sft_trainer
    assert read_checkpoint_interface(trainer_checkpoint(tmp_path, source="sft")) == sft_trainer
    normalized = trainer_checkpoint(tmp_path, loss_normalization="update_targets")
    assert read_checkpoint_interface(normalized) == sft_trainer
    # Pretraining domains with the pretraining defaults stay base.
    assert read_checkpoint_interface(trainer_checkpoint(tmp_path, source="python")) == BASE


def test_declared_key_wins_and_unknown_values_are_refused() -> None:
    weights = {"format": "weights-v1", "interface": "base"}
    assert checkpoint_interface(weights) == CheckpointInterface(BASE_INTERFACE, DECLARED)
    with pytest.raises(ValueError, match="unknown interface"):
        checkpoint_interface({"interface": "chat-v2"})
    # The shared random initialization records no interface.
    assert checkpoint_interface({"config": {}, "run_id": "rehearsal-init"}) == NONE
    # A trainer checkpoint that predates special_rows_init is base.
    old = {"version": 1, "optimizer": {}, "config": {"model": {}, "train": {"steps": 1}}}
    assert checkpoint_interface(old) == BASE


def test_save_weights_records_its_interface_and_old_files_still_load(tmp_path: Path) -> None:
    path = sft_checkpoint(tmp_path)
    assert torch.load(path, weights_only=True)["interface"] == "chat-v1"
    base = tmp_path / "base-weights.pt"
    save_weights(base, GPT(MODEL), tokenizer_hash="t", run_id="r", step=0, interface="base")
    assert read_checkpoint_interface(base) == CheckpointInterface(BASE_INTERFACE, DECLARED)
    with pytest.raises(ValueError, match="interface must be one of"):
        save_weights(
            tmp_path / "x.pt", GPT(MODEL), tokenizer_hash="t", run_id="r", step=0, interface="chat"
        )
    historical = historical_sft_checkpoint(tmp_path)
    before = historical.read_bytes()
    model, _ = load_checkpoint(historical, _tokenizer_file(tmp_path))
    assert model.config == MODEL
    read_checkpoint_interface(historical)
    assert historical.read_bytes() == before  # Resolution never rewrites a checkpoint.


@pytest.mark.parametrize(
    "interface,family,expected",
    [
        (CHAT, "pool", "chat-v1"),
        (CHAT, "benchmark", "chat-v1"),
        (BASE, "pool", "base-pool-completion-v1"),
        (BASE, "benchmark", "base-completion-rstrip-v1"),
    ],
)
def test_an_omitted_protocol_resolves_from_the_interface(
    interface: CheckpointInterface, family: str, expected: str
) -> None:
    assert resolve_protocol(interface, family, None) == expected


def test_named_protocols_must_fit_unless_deliberately_crossed() -> None:
    with pytest.raises(InterfaceMismatchError, match="--allow-interface-mismatch"):
        resolve_protocol(CHAT, "pool", "base-pool-completion-v1")
    with pytest.raises(InterfaceMismatchError, match="interface is base"):
        resolve_protocol(BASE, "benchmark", "chat-v1")
    assert resolve_protocol(CHAT, "pool", "base-pool-completion-v1", allow_mismatch=True) == (
        "base-pool-completion-v1"
    )
    assert resolve_protocol(NONE, "pool", "chat-v1") == "chat-v1"
    with pytest.raises(InterfaceMismatchError, match="choose the protocol explicitly"):
        resolve_protocol(NONE, "pool", None)
    with pytest.raises(InterfaceMismatchError, match="needs an explicit --protocol"):
        resolve_protocol(CHAT, "pool", None, allow_mismatch=True)


def test_legacy_modes_fit_base_checkpoints_only() -> None:
    assert resolve_legacy_mode(BASE, None) == "base"
    assert resolve_legacy_mode(BASE, "instruction") == "instruction"
    assert resolve_legacy_mode(NONE, "base") == "base"
    with pytest.raises(InterfaceMismatchError, match="eval-generate"):
        resolve_legacy_mode(CHAT, None)
    with pytest.raises(InterfaceMismatchError, match="--allow-interface-mismatch"):
        resolve_legacy_mode(CHAT, "instruction")
    assert resolve_legacy_mode(CHAT, "instruction", allow_mismatch=True) == "instruction"
    with pytest.raises(InterfaceMismatchError, match="--prompt-mode base or instruction"):
        resolve_legacy_mode(NONE, None)


# Prompt IDs and extraction through eval-generate ---------------------------------------


def _tokenizer_file(root: Path) -> Path:
    path = root / "tokenizer.json"
    if not path.exists():
        TOKENIZER.save(path)
    return path


def _answering(program_tokens: dict[str, tuple[int, ...]]) -> Any:
    """A generator stand-in that answers each prompt kind with fixed tokens."""
    seen: list[list[int]] = []

    def run(model: GPT, prompts: list[list[int]], **options: Any) -> list[Completion]:
        out = []
        for ids in prompts:
            seen.append(ids)
            budget = min(options["max_new_tokens"], model.config.context - len(ids))
            kind = "chat" if ids[-1] == chat.ASSISTANT_START else "base"
            tokens = list(program_tokens[kind])[:budget]
            if kind == "base":  # A base answer has no stop token: it spends its budget.
                tokens = (tokens + [ord(" ")] * budget)[:budget]
            out.append(Completion(tokens, "stop" if kind == "chat" else "length", budget))
        return out

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_sft_and_base_checkpoints_resolve_to_their_prompt_ids_and_extraction(
    tmp_path: Path,
) -> None:
    task = pool_task("t/1")
    tasks = save_task_file(tmp_path / "dev.jsonl", [task])
    program = "def add_one(x):\n    return x + 1"
    base_completion = program + "\nassert add_one(1) == 2\n"
    answers = {"chat": chat_tokens(program), "base": tuple(TOKENIZER.encode(base_completion))}
    expected = {
        "sft": ("chat-v1", chat.task_prompt(TOKENIZER, task)),
        "base": ("base-pool-completion-v1", TOKENIZER.encode(prompt.pool_prompt(task))),
    }
    checkpoints = {"sft": sft_checkpoint(tmp_path), "base": trainer_checkpoint(tmp_path)}
    for name, checkpoint in checkpoints.items():
        generator = _answering(answers)
        output = tmp_path / f"{name}.jsonl"
        header, _ = generate_completions(
            checkpoint=checkpoint,
            tokenizer_path=_tokenizer_file(tmp_path),
            suite="pool",
            split="dev",
            tasks_path=tasks,
            protocol=None,
            output=output,
            generate=generator,
        )
        protocol, prompt_ids = expected[name]
        assert header.protocol == protocol
        assert generator.seen == [prompt_ids]
        row = read_completions(output).row("t/1", 0)
        assert row.prompt_sha256 == prompt_sha256(prompt_ids)
        assert row.program == program
    assert expected["sft"][1][-1] == chat.ASSISTANT_START
    assert not any(chat.is_special(token) for token in expected["base"][1])


def test_eval_generate_refuses_implicit_and_mismatched_protocols(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = save_task_file(tmp_path / "dev.jsonl", [pool_task("t/1")])
    common = ["--tokenizer", str(_tokenizer_file(tmp_path)), "--suite", "pool"]
    common += ["--split", "dev", "--tasks", str(tasks)]
    cases = [
        (
            sft_checkpoint(tmp_path),
            ["--protocol", "base-pool-completion-v1"],
            "interface is chat-v1",
        ),
        (trainer_checkpoint(tmp_path), ["--protocol", "chat-v1"], "interface is base"),
        (undeclared_checkpoint(tmp_path), [], "choose the protocol explicitly"),
    ]
    for index, (checkpoint, extra, message) in enumerate(cases):
        output = tmp_path / f"refused-{index}.jsonl"
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "eval-generate",
                    "--checkpoint",
                    str(checkpoint),
                    *common,
                    *extra,
                    "--output",
                    str(output),
                ]
            )
        assert raised.value.code == 2
        assert message in capsys.readouterr().err
        assert not output.exists()


# The legacy eval command --------------------------------------------------------------


def _legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub execution (no code runs) and record the prompts the legacy path generates from."""
    seen: list[str] = []

    def generated(model: GPT, prompts: torch.Tensor, **options: Any) -> torch.Tensor:
        seen.append(TOKENIZER.decode(prompts[0].tolist()))
        tokens = TOKENIZER.encode("\n    return x + 1\n\ndef later(): pass")[
            : options["max_new_tokens"]
        ]
        return torch.cat((prompts, torch.tensor([tokens])), dim=1)

    monkeypatch.setattr(command, "generate", generated)
    monkeypatch.setattr(command, "evaluate", partial(evaluate, execute=lambda *_: Result("passed")))
    return seen


def _legacy_args(tmp_path: Path, checkpoint: Path, tasks: Path, name: str) -> list[str]:
    return [
        "eval", "--checkpoint", str(checkpoint), "--tokenizer", str(_tokenizer_file(tmp_path)),
        "--tasks", str(tasks), "--output", str(tmp_path / name),
        "--scorecard", str(tmp_path / "scorecard.jsonl"), "--max-new-tokens", "16",
    ]  # fmt: skip


def test_legacy_eval_refuses_chat_and_undeclared_checkpoints_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _legacy(tmp_path, monkeypatch)
    tasks = save_task_file(tmp_path / "dev.jsonl", [pool_task("t/1")])
    sft = sft_checkpoint(tmp_path)
    cases = [
        (sft, [], "eval-generate"),
        (historical_sft_checkpoint(tmp_path), [], "weights-v1 rule"),
        (sft, ["--prompt-mode", "instruction"], "--allow-interface-mismatch"),
        (undeclared_checkpoint(tmp_path), [], "--prompt-mode base or instruction"),
    ]
    for index, (checkpoint, extra, message) in enumerate(cases):
        with pytest.raises(SystemExit) as raised:
            main([*_legacy_args(tmp_path, checkpoint, tasks, f"refused-{index}"), *extra])
        assert raised.value.code == 2
        assert message in capsys.readouterr().err
        assert not (tmp_path / f"refused-{index}").exists()
    assert not (tmp_path / "scorecard.jsonl").exists()


def test_legacy_eval_needs_an_explicit_scorecard(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--checkpoint", "c", "--tokenizer", "t", "--tasks", "x", "--output", "o"])
    assert "--scorecard" in capsys.readouterr().err


def test_explicit_legacy_modes_still_reproduce_and_name_what_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _legacy(tmp_path, monkeypatch)
    base = trainer_checkpoint(tmp_path)
    task = pool_task("t/1", prompt="def add_one(x):\n  \t\n", entry_point="add_one")
    pool = save_task_file(tmp_path / "dev.jsonl", [task])
    benchmark = save_task_file(tmp_path / "bench.jsonl", [replace(task, split="eval")])
    runs = [
        ("implicit-pool", pool, [], "legacy-pool-rstrip-v1", "checkpoint interface"),
        ("base-pool", pool, ["--prompt-mode", "base"], "legacy-pool-rstrip-v1", "requested"),
        ("base-bench", benchmark, ["--prompt-mode", "base"], "base-completion-rstrip-v1",
         "requested"),
        ("instruction", pool, ["--prompt-mode", "instruction"], "legacy-instruction-v1",
         "requested"),
    ]  # fmt: skip
    for name, tasks, extra, protocol, source in runs:
        assert main([*_legacy_args(tmp_path, base, tasks, name), *extra]) == 0
        report = json.loads((tmp_path / name / "report.json").read_text())
        assert report["protocol"]["name"] == protocol
        assert report["checkpoint_interface"] == {
            "name": "base",
            "source": "trainer-checkpoint",
            "prompt_mode_source": source,
            "interface_mismatch_allowed": False,
        }
    # Base prompts are the stripped raw prompt; instruction keeps the public-tests prompt.
    assert seen == [task.prompt.rstrip()] * 3 + [task.public_prompt()]
    # The renamed pool record differs from the benchmark protocol in name and scope only.
    pool_record = prompt.prompt_protocol("legacy-pool")["protocol"]
    benchmark_record = prompt.prompt_protocol("base")["protocol"]
    assert {k: v for k, v in pool_record.items() if k not in {"name", "applies_to"}} == {
        k: v for k, v in benchmark_record.items() if k != "name"
    }
    rows = (tmp_path / "scorecard.jsonl").read_text().splitlines()
    assert len(rows) == len(runs)


def test_a_deliberate_legacy_mismatch_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _legacy(tmp_path, monkeypatch)
    task = pool_task("t/1")
    tasks = save_task_file(tmp_path / "dev.jsonl", [task])
    args = _legacy_args(tmp_path, sft_checkpoint(tmp_path), tasks, "crossed")
    assert main([*args, "--prompt-mode", "instruction", "--allow-interface-mismatch"]) == 0
    report = json.loads((tmp_path / "crossed" / "report.json").read_text())
    assert report["protocol"]["name"] == "legacy-instruction-v1"
    assert report["checkpoint_interface"] == {
        "name": "chat-v1",
        "source": "declared",
        "prompt_mode_source": "requested",
        "interface_mismatch_allowed": True,
    }
    assert seen == [task.public_prompt()]


def test_evalplus_checkpoint_scoring_refuses_chat_weights(tmp_path: Path) -> None:
    spec = replace(release("humaneval-plus"), task_count=1)
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a,b):\n",
        "canonical_solution": "    return a+b\n",
        "entry_point": "add",
        "test": "def check(candidate): assert candidate(2,3)==5",
    }
    with pytest.raises(InterfaceMismatchError, match="interface is chat-v1"):
        suites.evaluate_suite(
            [origin_task(row, spec)],
            spec,
            execute=lambda *_: Result("passed"),
            runtime=Runtime("sha256:" + "a" * 64, True),
            limits=Limits(),
            output=tmp_path / "run",
            checkpoint=sft_checkpoint(tmp_path),
            tokenizer_path=_tokenizer_file(tmp_path),
        )
    assert not (tmp_path / "run" / "checkpoint").exists()
