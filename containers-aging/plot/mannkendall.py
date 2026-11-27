"""
Teste de Mann-Kendall (tendencia) de memoria e CPU por cenario e por periodo do experimento.

    python3 plot/mannkendall.py [base_dir] [--output arquivo.csv]
    (padrao: base_dir "D:/final", saida "<base_dir>/mann_kendall_results.csv")
"""
import argparse
from pathlib import Path

import pandas as pd
from pymannkendall import original_test

from phases import phase_windows
from scenario import extract_categorical_info

DEFAULT_BASE_DIR = "D:/final"
MIN_SAMPLES = 3  # o teste precisa de pelo menos 3 pontos numa janela


def perform_mann_kendall_analysis(data, columns, intervals):
    """Perform Mann-Kendall analysis for selected columns and time intervals."""
    results = []
    for label, (start, end) in intervals.items():
        subset = data[(data.index >= start) & (data.index <= end)]
        if len(subset) < MIN_SAMPLES:  # ex.: o rejuvenescimento dura segundos (1 amostra)
            continue
        for col in columns:
            if col not in subset.columns:
                continue
            mk_result = original_test(subset[col])
            results.append({
                "Interval": label,
                "Resource": col,
                "Trend": mk_result.trend,
                "p-value": mk_result.p,
                "z": mk_result.z
            })
    return results


def analyze_folder(folder_path, output_data):
    """Analyze Mann-Kendall for memory.csv and cpu.csv files in the folder."""
    folder_name = folder_path.name
    print(f"Analyzing folder: {folder_name}")

    # Extract system, software, and version info
    system, software, version = extract_categorical_info(folder_name)

    # Analyze files
    files_to_columns = {
        "memory.csv": ["used", "swap"],
        "cpu.csv": ["sys", "usr"]
    }

    for file_name, columns in files_to_columns.items():
        file_path = folder_path / file_name
        if not file_path.exists():
            print(f"File not found: {file_name} in {folder_name}")
            continue

        print(f"Processing file: {file_name} in {folder_name}")

        # Load data
        df = pd.read_csv(file_path, sep=';', parse_dates=['date_time'])

        # Replace commas with dots for numeric conversion
        df = df.replace(',', '.', regex=True)

        t0 = df['date_time'].min()
        df['elapsed_hours'] = (df['date_time'] - t0).dt.total_seconds() / 3600
        df = df.set_index('elapsed_hours')
        df = df.drop(columns=['date_time'], errors='ignore')

        # Convert columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        if "swap_used" in df.columns:  # memory.csv novo: swap em uso
            df["swap"] = df["swap_used"]
        elif "swap" in df.columns:  # antigo: swap livre, supondo 8 GB de swap total
            df["swap"] = (df["swap"] - 8192) * -1

        # "Full" (usado pelo mannkendall_summary.py) + cada fase real do runs.csv ("Run 1 - rest", ...)
        intervals = {"Full": (0, float('inf')), **phase_windows(folder_path, t0)}
        mk_results = perform_mann_kendall_analysis(df, columns, intervals)


        # Append results with metadata
        for res in mk_results:
            resource = res["Resource"]

            res.update({
                "System": system,
                "Resource":  file_name.replace(".csv", "") + " - " + resource,
                "Software": software,
                "Software Version": version,
            })
            output_data.append(res)
        print(f"Completed analysis for {file_name} in {folder_name}")



def main(base_directory, output_csv_path):
    """Run the Mann-Kendall analysis for all subfolders."""
    base_path = Path(base_directory)
    output_data = []

    print(f"Starting analysis in base directory: {base_directory}")
    for folder in base_path.iterdir():
        if folder.is_dir():
            analyze_folder(folder, output_data)

    # Save all results to a CSV
    print(f"Saving results to {output_csv_path}")
    results_df = pd.DataFrame(output_data)
    results_df.to_csv(output_csv_path, index=False)
    print("Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", nargs="?", default=DEFAULT_BASE_DIR,
                        help="pasta com uma subpasta por cenario (padrao: %(default)s)")
    parser.add_argument("--output", help="CSV de saida (padrao: <base_dir>/mann_kendall_results.csv)")
    args = parser.parse_args()
    output_csv_path = args.output or f"{args.base_dir}/mann_kendall_results.csv"

    main(args.base_dir, output_csv_path)
    print(f"Mann-Kendall results saved to {output_csv_path}")
