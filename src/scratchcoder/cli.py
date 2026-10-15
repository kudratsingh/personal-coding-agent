"""Developer entry point; pipeline stages are added in later milestones."""

import argparse
import sys
from collections.abc import Sequence

from scratchcoder.common.smoke import run_smoke


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch only implemented commands and return a process exit status."""
    parser = argparse.ArgumentParser(prog="scratchcoder")
    parser.add_argument("command", choices=("smoke",))
    parser.parse_args(argv)
    return run_smoke(sys.stderr)
