"""
Theil-Sen: inclinacao (tendencia, unidade/dia) de memoria (MB) e CPU (%) em cada cenario e um indice
de envelhecimento a partir das inclinacoes normalizadas (min-max entre os cenarios). Salva o CSV
e os PNGs no diretorio atual.

    python3 plot/theilsen.py [logs_dir]      # padrao: ./logs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

import metrics
from scenario import extract_categorical_info

UNKNOWN_SCENARIO = ("Unknown OS", "Unknown Software", "N/A")
DEFAULT_LOGS_DIR = "./logs"
# coluna de resultado -> (funcao de metrics.py, coluna)
SLOPES = {
    "mem_used_slope": (metrics.memory, "used"),
    "mem_swap_slope": (metrics.memory, "swap_used"),
    "cpu_usr_slope": (metrics.cpu, "usr"),
    "cpu_sys_slope": (metrics.cpu, "sys"),
}


def slope_per_day(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    days = (series.index - series.index[0]).total_seconds() / 86400
    try:
        return theilslopes(series.to_numpy(), days.to_numpy())[0]
    except ValueError:  # ex.: todos os instantes iguais
        return np.nan


def process_folder(folder: Path) -> dict:
    print(f"Processing: {folder.name}")
    system, software, version = extract_categorical_info(folder.name, unknown=UNKNOWN_SCENARIO)
    result = {"scenario_folder": folder.name, "scenario_label": f"{system} - {software} {version}",
              "system": system, "software": software, "version": version}
    frames = {}
    for column, (compute, metric) in SLOPES.items():
        if compute not in frames:
            frames[compute] = compute(folder)
        df = frames[compute]
        result[column] = np.nan if df is None else slope_per_day(df[metric])
    return result


def min_max(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    return (series - low) / (high - low) if high > low else series * 0.0


def bar_chart(series_or_frame, title, ylabel, filename, **kwargs):
    series_or_frame.plot(kind="bar", **kwargs)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Cenário de Teste")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico salvo em {filename}")


def main(logs_dir=DEFAULT_LOGS_DIR):
    folders = [path for path in Path(logs_dir).iterdir() if path.is_dir()]
    if not folders:
        print(f"Nenhuma subpasta encontrada em '{Path(logs_dir).resolve()}'.")
        return

    results = pd.DataFrame([process_folder(folder) for folder in folders])
    slope_cols = list(SLOPES)
    norm_cols = [f"{col}_norm" for col in slope_cols]
    for col, norm in zip(slope_cols, norm_cols):
        results[norm] = min_max(results[col])
    normalized = results[norm_cols].fillna(0)
    results["aging_index_euclidean"] = np.sqrt(normalized.pow(2).sum(axis=1))
    results["aging_index_sum"] = normalized.sum(axis=1)
    results = results.sort_values(by="aging_index_euclidean", ascending=False)

    print(results[["scenario_label", *slope_cols, *norm_cols, "aging_index_sum", "aging_index_euclidean"]].to_string())
    results.to_csv("aging_analysis_results.csv", index=False, sep=";", decimal=".")
    print("Resultados detalhados salvos em: aging_analysis_results.csv")

    plot_df = results.dropna(subset=slope_cols, how="all").set_index("scenario_label")
    if plot_df.empty:
        print("Nenhum dado válido para plotar.")
        return
    bar_chart(plot_df[slope_cols], "Slopes Theil-Sen por Cenário e Métrica (unidade/dia)", "Valor do Slope",
              "theil_sen_slopes_raw.png", figsize=(18, 9), width=0.8)
    bar_chart(plot_df[norm_cols], "Slopes Theil-Sen Normalizados (Min-Max) por Cenário e Métrica",
              "Valor do Slope Normalizado [0-1]", "theil_sen_slopes_normalized.png", figsize=(18, 9), width=0.8)
    bar_chart(plot_df["aging_index_euclidean"].sort_values(ascending=False),
              "Índice de Envelhecimento Euclidiano por Cenário", "Distância Euclidiana (Slopes Normalizados)",
              "aging_index_euclidean.png", figsize=(15, 7), color="teal")
    bar_chart(plot_df["aging_index_sum"].sort_values(ascending=False),
              "Índice de Envelhecimento (Soma dos Slopes Normalizados) por Cenário", "Soma dos Slopes Normalizados",
              "aging_index_sum_normalized.png", figsize=(15, 7), color="purple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", nargs="?", default=DEFAULT_LOGS_DIR,
                        help="pasta com uma subpasta por cenario (padrao: %(default)s)")
    main(parser.parse_args().logs_dir)
