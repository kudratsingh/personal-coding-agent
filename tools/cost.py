"""Estimate two-stage compute from explicitly supplied rates."""

import argparse
import json
import math
from dataclasses import asdict

from scratchcoder.common.cost import estimate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-tokens", type=int, required=True)
    parser.add_argument("--stage2-tokens", type=int, required=True)
    parser.add_argument("--tokens-per-second-per-gpu", type=float, required=True)
    parser.add_argument("--dollars-per-gpu-hour", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--layer-parameters", type=int)
    parser.add_argument("--reference-parameters", type=int)
    args = parser.parse_args()
    try:
        stages = [
            estimate(
                tokens,
                args.tokens_per_second_per_gpu,
                args.dollars_per_gpu_hour,
                gpus=args.gpus,
                slowdown=args.slowdown,
                layer_parameters=args.layer_parameters,
                reference_parameters=args.reference_parameters,
            )
            for tokens in (args.stage1_tokens, args.stage2_tokens)
        ]
        total = {key: sum(asdict(item)[key] for item in stages) for key in asdict(stages[0])}
        if any(not math.isfinite(value) for value in total.values()):
            raise ValueError("combined estimate must be finite and representable")
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "stages": [asdict(item) for item in stages],
                "total": total,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
