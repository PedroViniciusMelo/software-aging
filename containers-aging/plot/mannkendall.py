import pandas as pd
from pymannkendall import original_test
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def extract_categorical_info(folder_name):
    """Extract System, Software, and Software Version based on folder name."""
    system = None
    software = None
    version = None

    if "debian" in folder_name:
        system = "Debian 12"
    elif "ubuntu22" in folder_name:
        system = "Ubuntu 22.04"
    elif "ubuntu24" in folder_name:
        system = "Ubuntu 24.04"

    if "docker" in folder_name:
        software = "Docker"
        version = "27.2" if "novo" in folder_name else "23"
    elif "podman" in folder_name:
        software = "Podman"
        version = "5.1.3"

    return system, software, version


def perform_mann_kendall_analysis(data, columns, intervals):
    """Perform Mann-Kendall analysis for selected columns and time intervals."""
    results = []
    for label, (start, end) in intervals.items():
        subset = data[(data.index >= start) & (data.index <= end)]
        if subset.empty:
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


def normalize_columns(df, columns):
    """Apply Min-Max normalization to each column independently."""
    for column in columns:
        if column in df.columns:
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return df


def analyze_folder(folder_path, output_data):
    """Analyze Mann-Kendall for memory.csv and cpu.csv files in the folder."""
    folder_name = folder_path.name
    print(f"Analyzing folder: {folder_name}")

    # Extract system, software, and version info
    system, software, version = extract_categorical_info(folder_name)

    intervals = {
        "Full": (0, float('inf')),  # Intervalo completo (todas as atividades)

        # Primeiro período (0 a 3.5 dias)
        "0 to 0.5 days": (0, 12),  # Apenas Workload 1
        "0.5 days to 3.5 days": (12, 84),  # Workloade Stressload 1

        # Segundo período (3.5 a 7 dias)
        "3.5 to 4 days": (84, 96),  # Apenas Workload 2
        "4 days to 7 days": (96, 168),  # Workload e Stressload 2

        # Terceiro período (7 a 10.5 dias)
        "7 to 7.5 days": (168, 180),  # Apenas Workload 3
        "7.5 days to 10.5 days": (180, 252),  # Workload e Stressload 3

        # Quarto período (10.5 a 14 dias)
        "10.5 to 11 days": (252, 264),  # Apenas Workload 4
        "11 days to 14 days": (264, 336),  # Workload e Stressload 4
    }

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

        df['elapsed_hours'] = (df['date_time'] - df['date_time'].min()).dt.total_seconds() / 3600
        df = df.set_index('elapsed_hours')
        df = df.drop(columns=['date_time'], errors='ignore')

        # Convert columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        if "swap" in df.columns:
            df["swap"] = (df["swap"] - 8192) * -1

        # Apply Min-Max normalization
        #df = normalize_columns(df, columns)

        # Perform Mann-Kendall analysis
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


# Execute the main function
base_directory = "D:/final"  # Substitua pelo caminho correto
output_csv_path = "D:/final/mann_kendall_results.csv"
main(base_directory, output_csv_path)

print(f"Mann-Kendall results saved to {output_csv_path}")
