"""
Testes de caracterizacao (golden master): garantem que a refatoracao nao muda o que o
framework faz -- os comandos executados, a saida no terminal e os CSVs gerados.

    python3 -m unittest discover -s tests -v          # compara com os snapshots
    UPDATE_GOLDEN=1 python3 -m unittest discover -s tests   # regrava os snapshots

So regrave os snapshots quando a mudanca de comportamento for intencional.
"""
import difflib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "golden" / "harness.py"
SNAPSHOTS = HERE / "golden" / "snapshots"
SCENARIOS = ["docker", "podman", "rapl_only_package", "giveup"]


def run_scenario(name):
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "result.json"
        subprocess.run([sys.executable, str(HARNESS), name, str(output)], check=True, timeout=300)
        return output.read_text()


class GoldenMasterTest(unittest.TestCase):
    maxDiff = None


def make_test(name):
    def test(self):
        result = run_scenario(name)
        snapshot = SNAPSHOTS / f"{name}.json"
        if os.environ.get("UPDATE_GOLDEN"):
            snapshot.write_text(result)
            self.skipTest("snapshot atualizado")
        expected = snapshot.read_text()
        if result != expected:
            diff = "".join(list(difflib.unified_diff(
                expected.splitlines(True), result.splitlines(True), "snapshot", "atual"))[:200])
            self.fail(f"saida do cenario '{name}' mudou:\n{diff}")

    return test


for _scenario in SCENARIOS:
    setattr(GoldenMasterTest, f"test_{_scenario}", make_test(_scenario))

if __name__ == "__main__":
    unittest.main()
