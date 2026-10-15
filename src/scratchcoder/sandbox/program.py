"""Trusted supervisor source, executed ONLY inside the isolated container."""

import json
import os
import secrets
import selectors
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any


def supervise(payload: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + payload["timeout"]
    cases = (
        [{"input": "", "output": ""}]
        if payload["mode"] == "function"
        else json.loads(payload["tests"])
    )
    last_output = ""
    for case in cases:
        marker = "__scratchcoder_completed_" + secrets.token_hex(24)
        program = payload["code"]
        if payload["mode"] == "function":
            program += "\n" + payload["tests"] + "\nprint(" + repr(marker) + ")\n"
        Path("/tmp/candidate.py").write_text(program, encoding="utf-8")
        with TemporaryFile() as input_stream:
            input_stream.write(case["input"].encode())
            input_stream.seek(0)
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "/tmp/candidate.py"],
                stdin=input_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            captured = bytearray()
            status = "failed"
            try:
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        status = "timeout"
                        break
                    for key, _ in selector.select(timeout=0.05):
                        block = os.read(key.fd, 4096)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        captured.extend(block)
                        if len(captured) > payload["output_bytes"]:
                            status = "output_limit"
                            break
                    if status == "output_limit":
                        break
                if status in {"timeout", "output_limit"}:
                    process.kill()
                process.wait(timeout=1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=1)
                selector.close()
                process.stdout.close()
            last_output = bytes(captured[: payload["output_bytes"]]).decode(
                "utf-8", errors="replace"
            )
            if status in {"timeout", "output_limit"}:
                return {"status": status, "stdout": last_output, "exit_code": process.returncode}
            if process.returncode != 0:
                return {"status": "failed", "stdout": last_output, "exit_code": process.returncode}
            if payload["mode"] == "function":
                if not last_output.rstrip().endswith(marker):
                    return {"status": "failed", "stdout": last_output, "exit_code": 0}
                last_output = last_output.rsplit(marker, 1)[0]
            elif last_output.strip() != case["output"].strip():
                return {"status": "failed", "stdout": last_output, "exit_code": 0}
    return {"status": "passed", "stdout": last_output, "exit_code": 0}


if __name__ == "__main__":
    try:
        result = supervise(json.load(sys.stdin))
    except BaseException as error:
        result = {
            "status": "failed",
            "stdout": "",
            "exit_code": 1,
            "error_type": type(error).__name__,
        }
    print(json.dumps(result))
