"""Measure fast/reference identity and synthetic encoding throughput, offline."""

import argparse
import json
import time

from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megabytes", type=int, default=1)
    args = parser.parse_args()
    if args.megabytes < 1:
        parser.error("megabytes must be positive")
    unit = "def add(a, b):\n    return a + b\nEnglish words and spaces.\n" * 10
    tokenizer = ByteBPE.train([unit], 320)
    encoder = FastEncoder(tokenizer)
    count = max(1, args.megabytes * 1_000_000 // len(unit.encode()))
    total_bytes = count * len(unit.encode())
    results: dict[str, float] = {}
    for name, encode in (("reference", tokenizer.encode), ("fast", encoder.encode)):
        started = time.perf_counter()
        for _ in range(count):
            if encode(unit) != encoder.encode(unit):
                raise RuntimeError("encoder identity failed")
        results[f"{name}_mb_per_second"] = total_bytes / 1_000_000 / (time.perf_counter() - started)
    print(json.dumps({"bytes": total_bytes, "synthetic": True, **results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
