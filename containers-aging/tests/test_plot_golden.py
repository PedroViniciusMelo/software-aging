"""
Golden master dos scripts de plot/: mesmos arquivos gerados (por sha256) e mesmo stdout.

Precisa das libs de plot/requirements.txt. Por padrao usa o Python atual; para usar outro
(ex.: um venv so' com as libs de analise) defina PLOT_PYTHON=/caminho/do/python.

    PLOT_PYTHON=venv/bin/python python3 -m unittest tests.test_plot_golden
    UPDATE_GOLDEN=1 PLOT_PYTHON=... python3 -m unittest tests.test_plot_golden

O snapshot depende das versoes de pandas/matplotlib usadas para grava-lo.
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
HARNESS = HERE / "plot_golden" / "harness.py"
SNAPSHOT = HERE / "plot_golden" / "snapshot.json"
PYTHON = os.environ.get("PLOT_PYTHON", sys.executable)


def has_plot_dependencies():
    check = "import pandas, matplotlib, scipy, sklearn, pymannkendall"
    return subprocess.run([PYTHON, "-c", check], capture_output=True).returncode == 0


@unittest.skipUnless(has_plot_dependencies(), "libs de plot/ ausentes (defina PLOT_PYTHON)")
class PlotGoldenMasterTest(unittest.TestCase):
    def test_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            subprocess.run([PYTHON, str(HARNESS), str(output)], check=True, timeout=900)
            result = output.read_text()

        if os.environ.get("UPDATE_GOLDEN"):
            SNAPSHOT.write_text(result)
            self.skipTest("snapshot atualizado")

        expected = SNAPSHOT.read_text()
        if result != expected:
            diff = "".join(list(difflib.unified_diff(
                expected.splitlines(True), result.splitlines(True), "snapshot", "atual"))[:200])
            self.fail(f"saida de plot/ mudou:\n{diff}")


if __name__ == "__main__":
    unittest.main()
