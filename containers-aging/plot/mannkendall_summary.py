"""
Resume o resultado do mannkendall.py: filtra o intervalo "Full", ordena os cenarios pela
media do z e gera o grafico de barras Mann-Kendall.svg.

    python3 plot/mannkendall_summary.py [folder]      # padrao: D:/final
    (le <folder>/mann_kendall_results.csv e grava os resultados na mesma pasta)
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_FOLDER = "D:/final"


def simplify_labels(system, software, version):
    """Rotulo do eixo x: sistema na primeira linha, runtime e versao na segunda."""
    return f"{system}\n{software} {version}"


def main(folder=DEFAULT_FOLDER):
    # Load the dataset
    # versao como texto: com versoes como "27" o pandas leria a coluna como numero
    df = pd.read_csv(f"{folder}/mann_kendall_results.csv", dtype={"Software Version": str})

    # Filter for "Full" interval
    df = df.loc[df["Interval"] == "Full"]

    # Ensure the 'z' column is numeric and drop invalid rows
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df = df.dropna(subset=['z'])  # Drop rows with NaN in the 'z' column

    # Save the filtered data for future use
    df.to_csv(f"{folder}/mann_kendall_results_full.csv", index=False)

    # Group data by the categorical variables and calculate key statistics for trends (e.g., mean z-score)
    grouped_data2 = df.groupby(['System', 'Software', 'Software Version']).agg(
        mean_z=('z', 'mean'),
        count=('z', 'count')
    ).reset_index()

    # Sort by mean z-score to identify the scenarios with the highest average trend
    grouped_data_sorted = grouped_data2.sort_values(by='mean_z', ascending=False)

    grouped_data_sorted.to_csv(f"{folder}/mann_kendall_results_sorted.csv", index=False)

    # Group by system, software, version, and resource
    grouped_data = df.groupby(['System', 'Software', 'Software Version', 'Resource']).mean(numeric_only=True).reset_index()

    # Filter for the most relevant metrics (CPU and memory usage)
    relevant_metrics = grouped_data[grouped_data['Resource'].str.contains('cpu|memory', case=False)]

    # Pivot the data for easier plotting
    pivot_data = relevant_metrics.pivot_table(
        index=['System', 'Software', 'Software Version'],
        columns='Resource',
        values='z',
        aggfunc='mean'
    )

    # Create the bar chart with adjusted bar width
    ax = pivot_data.plot(kind='bar', figsize=(15, 10), width=0.9, edgecolor='black', alpha=0.8)  # Adjust `width` to control bar thickness
    plt.ylabel("Mann-Kendall Z-Score", fontsize=18)
    plt.xlabel("System and Software Version", fontsize=18)

    # Update x-axis labels with simplified text
    new_labels = [simplify_labels(system, software, version) for system, software, version in pivot_data.index]
    ax.set_xticklabels(new_labels)

    # Add values to the center of each bar
    for bars in ax.containers:
        for bar in bars:
            height = bar.get_height()
            if height != 0:  # Avoid placing text on zero-height bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # X position at the center of the bar
                    height / 2,  # Y position at the middle of the bar
                    f'{height:.2f}',  # Format the value with 2 decimal places
                    ha='center', va='center', fontsize=18, rotation=90, color="black", weight="bold"  # Center-align with rotation
                )

    # Customize ticks and legend
    plt.xticks(rotation=0, ha='center', fontsize=17)  # Horizontal orientation with larger font
    plt.yticks(fontsize=18)
    plt.legend(title="Resource Metric", loc='upper left', fontsize=18, title_fontsize=18)  # Set font sizes for legend and title

    # Adjust layout for better visibility
    plt.tight_layout()
    plt.savefig(f"{folder}/Mann-Kendall.svg", bbox_inches='tight', dpi=300, format="svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=DEFAULT_FOLDER,
                        help="pasta com mann_kendall_results.csv (padrao: %(default)s)")
    main(parser.parse_args().folder)
