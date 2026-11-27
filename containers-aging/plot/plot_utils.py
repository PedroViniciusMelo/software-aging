"""Grafico generico de series temporais: um SVG por coluna, com regressao linear e Mann-Kendall."""
from matplotlib import pyplot as plt
from pymannkendall import original_test
from sklearn.linear_model import LinearRegression

import style  # noqa: F401  (aplica o estilo global)


def plot(df, folder, title, ylabel=None, highlight=None) -> int:
    """
    Um grafico por coluna de `df` (indice = date_time), em <folder>/plots_img/<title>-<coluna>.svg.

    :param ylabel: rotulo do eixo y: dict {coluna: rotulo}, ou texto com "{col}" (padrao: o nome da coluna).
    :param highlight: faixas [(inicio, fim, cor, alpha)] com datetimes (fim None = ate' o ultimo dado),
                      ver phases.highlight_intervals.
    """
    if df is None or df.empty:
        return 0
    print(f"Plotting {title}")
    t0 = df.index[0]
    df = df.fillna(0)
    df.index = (df.index - t0).total_seconds() / 3600
    hours = lambda moment: df.index[-1] if moment is None else (moment - t0).total_seconds() / 3600

    for col in df.columns:
        label = ylabel.get(col, col) if isinstance(ylabel, dict) else (ylabel or "{col}").format(col=col)
        ax = df.plot(y=col, legend=False, ylabel=label, figsize=(10, 10), style="k", linewidth=5)
        ax.set_xlabel("Time(h)", labelpad=15)

        if highlight:  # as faixas nao mudam os limites do eixo x
            x_limits = ax.get_xlim()
            for start, end, color, alpha in highlight:
                ax.axvspan(hours(start), hours(end), color=color, alpha=alpha)
            ax.set_xlim(x_limits)

        x = df.index.to_numpy().reshape((-1, 1))
        ax.plot(x, LinearRegression().fit(x, df[col]).predict(x), color="red", linewidth=5)
        trend = original_test(df[col])
        ax.text(0.95, 0.05, f"trend: {trend.trend}\np-value: {trend.p:.4f}\nz: {trend.z:.4f}", transform=ax.transAxes,
                fontsize=25, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", alpha=0.8, color="white"))

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.get_figure().savefig(folder.joinpath("plots_img", f"{title}-{col}.svg"), bbox_inches="tight", dpi=300,
                                format="svg")
        plt.close("all")
    return 1
