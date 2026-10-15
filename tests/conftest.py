"""Avoid thread oversubscription in tiny CPU reference tests."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def cpu_threads() -> None:
    import torch

    torch.set_num_threads(1)
