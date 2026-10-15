import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from scratchcoder.common.config import RunConfig
from scratchcoder.data.fixtures import prepare_fixture
from scratchcoder.train.command import train_from_files


def test_explicit_fixture_training_writes_checkpoint_and_record(tmp_path: Path) -> None:
    data = tmp_path / "data"
    tokenizer, _, metadata = prepare_fixture(data, vocab_size=512)
    original = RunConfig.load(Path("configs/smoke.yaml"))
    config = replace(original, model=replace(original.model, vocab_size=tokenizer.vocab_size))
    config_path = tmp_path / "config.yaml"
    config.save(config_path)
    run = partial(
        train_from_files,
        config_path=config_path,
        sources_path=data / "stage1-sources.json",
        tokenizer_path=data / "tokenizer.json",
        output_root=tmp_path / "runs",
        run_id="explicit",
    )
    run()
    assert (tmp_path / "runs/explicit/checkpoint.pt").exists()
    record = json.loads((tmp_path / "runs/explicit/record.json").read_text())
    assert record["status"] == "completed"
    assert record["tokenizer_hash"] == tokenizer.sha256
    with pytest.raises(ValueError, match="already exists"):
        run()
    ids = metadata["split_document_ids"]
    assert set(ids["web-stage1"]).isdisjoint(ids["web-stage2"])
    assert set(ids["web-stage1"]).isdisjoint(ids["web-validation"])
