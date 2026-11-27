"""
Teste de Mann-Kendall (tendencia) de memoria e CPU por cenario e por periodo do experimento.

    python3 plot/mannkendall.py [base_dir] [--output arquivo.csv]
    (padrao: base_dir "D:/final", saida "<base_dir>/mann_kendall_results.csv")
"""
import argparse
from pathlib import Path

import pandas as pd
from pymannkendall import original_test

import metrics
from phases import phase_windows
from scenario import extract_categorical_info

DEFAULT_BASE_DIR = "D:/final"
MIN_SAMPLES = 3  # o teste precisa de pelo menos 3 pontos numa janela
RESOURCES = {"memory": (metrics.memory, ["used", "swap_used"]), "cpu": (metrics.cpu, ["sys", "usr"])}


def analyze_folder(folder: Path) -> list:
    """Uma linha por (janela, recurso): "Full" (usado pelo mannkendall_summary.py) e cada fase do runs.csv."""
    print(f"Analyzing folder: {folder.name}")
    system, software, version = extract_categorical_info(folder.name)
    results = []
    for resource, (compute, columns) in RESOURCES.items():
        df = compute(folder)
        if df is None:
            print(f"  {resource}: no data")
            continue
        t0 = df.index[0]
        df.index = (df.index - t0).total_seconds() / 3600
        windows = {"Full": (0, float("inf")), **phase_windows(folder, t0)}
        for label, (start, end) in windows.items():
            subset = df[(df.index >= start) & (df.index <= end)]
            if len(subset) < MIN_SAMPLES:  # ex.: o rejuvenescimento dura segundos (1 amostra)
                continue
            for col in columns:
                test = original_test(subset[col])
                results.append({"Interval": label, "Resource": f"{resource} - {col}", "Trend": test.trend,
                                "p-value": test.p, "z": test.z, "System": system, "Software": software,
                                "Software Version": version})
    return results


def main(base_directory, output_csv_path):
    rows = [row for folder in Path(base_directory).iterdir() if folder.is_dir() for row in analyze_folder(folder)]
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    print(f"Mann-Kendall results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", nargs="?", default=DEFAULT_BASE_DIR,
                        help="pasta com uma subpasta por cenario (padrao: %(default)s)")
    parser.add_argument("--output", help="CSV de saida (padrao: <base_dir>/mann_kendall_results.csv)")
    args = parser.parse_args()
    main(args.base_dir, args.output or f"{args.base_dir}/mann_kendall_results.csv")
