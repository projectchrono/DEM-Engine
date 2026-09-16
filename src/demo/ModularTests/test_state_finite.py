"""Run GPU state guards in fresh processes: python3 test_state_finite.py build/bin/DEMTest_StateFinite."""

import os
import tempfile
import subprocess
import sys
from pathlib import Path


def main():
    # A trap poisons the child's CUDA context, so never reuse a process between injection cases.
    executable = str(Path(sys.argv[1]).resolve())
    # Share discovered headers only within this test run to keep fresh-process GPU cases affordable.
    with tempfile.TemporaryDirectory(prefix="deme-state-finite-") as cache_dir:
        env = dict(os.environ, DEME_PERSISTENT_JITIFY_CACHE=str(Path(cache_dir) / "headers.bin"))
        run_cases(executable, env)


def run_cases(executable, env):
    for margin in ("auto", "fixed", "no-angular"):
        for kind, value in (("healthy", "nan"), ("linear", "nan"), ("angular", "nan"),
                            ("linear", "inf"), ("angular", "inf")):
            result = subprocess.run([executable, kind, margin, value], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=600, env=env)
            expected = "PASS: healthy no-contact state" if kind == "healthy" else f"Non-finite {kind} velocity"
            if expected not in result.stdout or (kind == "healthy" and result.returncode != 0):
                raise RuntimeError(f"FAIL: {kind}/{margin}/{value}, exit {result.returncode}\n{result.stdout}")
            if kind != "healthy" and result.returncode == 0:
                raise RuntimeError(f"FAIL: invalid case returned success: {kind}/{margin}/{value}")
            print(f"PASS: {kind}/{margin}/{value}", flush=True)


if __name__ == "__main__":
    main()
