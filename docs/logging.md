# Debugging and experiment logs

`EventLogger` appends timestamped JSON lines locally before optional tracker forwarding. Use it as a context manager so streams and trackers close correctly:

```python
from pathlib import Path
from scratchcoder.common.logging import EventLogger

with EventLogger(Path("runs/local/events.jsonl"), "local") as log:
    log.event("started", stage="data")
    log.metrics({"loss": 1.25}, step=0)
```

For diagnostic tracebacks, `debug_logger(Path("runs/local/debug.jsonl"))` returns an isolated standard-library logger with JSON formatting and no changes to root logging handlers. Close its handlers when done.

Logs are created with owner-only file permissions. Nested credential-like keys and known credential patterns are redacted, but redaction is a defense in depth: do not log credentials, raw training text, hidden tests, reference solutions, or environment dumps.

Hosted tracking is disabled by default. The optional `tracking` extra provides W&B; constructing `WandbTracker` explicitly enables online tracking and requires the owner's account setup. Tests use a mock tracker and make no network requests. Local metrics survive tracker logging failures.
