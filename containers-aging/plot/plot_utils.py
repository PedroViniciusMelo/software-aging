"""Graficos genericos de series temporais dos CSVs do coletor."""
import pandas as pd
from matplotlib import pyplot as plt
from pymannkendall import original_test
from sklearn.linear_model import LinearRegression

import style  # noqa: F401  (aplica o estilo global)


def plot_time_series(save_folder, file_path, title, x_label, y_label, max_labels=20):
    # Carregar os dados do arquivo CSV
    df = pd.read_csv(file_path, sep=";")

    # Converter a coluna 'date_time' para o tipo datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Calcular o tempo decorrido em horas
    df['elapsed_hours'] = (df['date_time'] - df['date_time'].min()).dt.total_seconds() / 3600

    # Converter valores de nanosegundos para segundos (exceto 'date_time' e 'elapsed_hours')
    for col in df.columns:
        if col not in ['date_time', 'elapsed_hours']:
            df[col] = df[col] / 1e9  # Convertendo nanosegundos para segundos

    # Para cada coluna (exceto 'date_time' e 'elapsed_hours'), plota um gráfico separado
    for col in df.columns:
        if col not in ['date_time', 'elapsed_hours']:
            plt.figure(figsize=(10, 6))

            # Plotar a série temporal com o eixo X em horas
            plt.plot(df['elapsed_hours'], df[col], label=col)

            # Adicionar título e rótulos dos eixos
            plt.title(f"{title} - {col}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            # Adicionar legenda
            plt.legend()

            # Ajustar layout para evitar sobreposição e salvar o gráfico
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(save_folder.joinpath(f"{title} - {col}.png"))
            plt.close()

    return 1


def plot(
        folder,
        filename,
        ylabel,
        datetime="date_time",
        title=None,
        separator=';',
        decimal_separator=",",
        division=1,
        includeColYlabel=False,
        cols_to_divide=None,
        apply_mann_kendall=True,  # Variável para ativar/desativar Mann-Kendall e regressão
        highlight_intervals=None,
        # Parâmetro opcional para intervalos a destacar [(hora_inicial, hora_final, cor, alpha), ...]
        x_lim=None,  # Novo parâmetro opcional para limitar o eixo x (limite superior)
        y_lim=None,  # Novo parâmetro opcional para limitar o eixo y (limite superior)
        skip_columns=(),  # colunas do CSV que nao viram grafico
        legacy_swap_free=False,  # memory.csv antigo: a coluna "swap" e' o swap LIVRE; mostra 8192 MB - livre
):
    if cols_to_divide is None:
        cols_to_divide = []
    print(f"Plotting {filename}")
    df = pd.read_csv(
        folder.joinpath(filename),
        sep=separator,
        decimal=decimal_separator,
        dayfirst=False,
        parse_dates=[datetime]
    ).rename(columns={datetime: 'seconds'})

    t0 = df['seconds'][0]
    df['seconds'] = (df['seconds'] - t0).dt.total_seconds() / 3600
    df = df.set_index('seconds').replace(',', '.', regex=True).apply(lambda x: pd.to_numeric(x))
    df = df.drop(columns=[column for column in skip_columns if column in df.columns])
    cols_to_divide = cols_to_divide if len(cols_to_divide) != 0 else df.columns
    df[cols_to_divide] = df[cols_to_divide].div(division)

    for col in df.columns:
        col_mix = col + " " + ylabel if isinstance(ylabel, str) and includeColYlabel else ylabel

        df[col] = df[col].fillna(0)

        # So' no memory.csv antigo. Nos processos, "swap" ja' e' o swap usado (VmSwap): antes esta
        # conta valia para eles tambem e mostrava 8 GB de swap para um processo que usava 0.
        if col == "swap" and legacy_swap_free:
            df[col] = (df[col] - 8192) * -1

        # Configurar o gráfico
        ax = df.plot(
            y=col,
            legend=0,
            xlabel='Time(h)',
            ylabel=col_mix if isinstance(ylabel, str) else ylabel[col] if isinstance(ylabel,
                                                                                     dict) and col in ylabel else col,
            figsize=(10, 10),
            style='k',
            linewidth=5
        )

        ax.set_xlabel('Time(h)', labelpad=15)

        # Aplicar limites, se especificados
        if x_lim is not None:
            ax.set_xlim(right=x_lim)
        if y_lim is not None:
            ax.set_ylim(top=y_lim)

        # Destacar intervalos, se especificados (agora com alpha)
        # inicio/fim em horas, ou datetimes (fases do runs.csv, ver phases.py) convertidos com o t0
        # deste arquivo; fim None = ate' o ultimo dado. As faixas nao mudam os limites do eixo x.
        if highlight_intervals is not None:
            x_limits = ax.get_xlim()
            last_hour = df.index.max()
            for start, end, color, alpha in highlight_intervals:
                if not isinstance(start, (int, float)):
                    start = (start - t0).total_seconds() / 3600
                if end is None:
                    end = last_hour
                elif not isinstance(end, (int, float)):
                    end = (end - t0).total_seconds() / 3600
                ax.axvspan(start, end, color=color, alpha=alpha)
            ax.set_xlim(x_limits)

        # Análise Mann-Kendall e regressão
        if apply_mann_kendall:
            # Regressão linear
            x = df.index.to_numpy().reshape((-1, 1))
            y = df[col].to_numpy().reshape((-1, 1))
            model = LinearRegression()
            model.fit(x, y)
            Y_pred = model.predict(x)
            ax.plot(x, Y_pred, color='red', linewidth=5)

            # Teste de Mann-Kendall
            test_result = original_test(df[col])
            trend_text = (
                f"trend: {test_result.trend}\n"
                f"p-value: {test_result.p:.4f}\n"
                f"z: {test_result.z:.4f}"
            )

            # Adicionar o texto da análise Mann-Kendall no gráfico
            ax.text(
                0.95, 0.05, trend_text,
                transform=ax.transAxes,
                fontsize=25,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle="round", alpha=0.8, color='white')
            )

        # Ajustar a espessura da borda
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Salvar o gráfico
        fig = ax.get_figure()
        fig.savefig(folder.joinpath('plots_img').joinpath(f"{title}-{col}.svg"), bbox_inches='tight', dpi=300,
                    format="svg")
        plt.close('all')
    return 1
