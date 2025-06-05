import glob
import os
import re
import matplotlib.ticker as mticker # IMPORTANTE: Adicionar este import

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from pymannkendall import original_test  # Certifique-se de que este módulo está instalado

plt.rcParams.update({'font.size': 25})  # Aumenta o tamanho da fonte globalmente





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
        y_lim=None  # Novo parâmetro opcional para limitar o eixo y (limite superior)
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

    df['seconds'] = (df['seconds'] - df['seconds'][0]).dt.total_seconds() / 3600
    df = df.set_index('seconds').replace(',', '.', regex=True).apply(lambda x: pd.to_numeric(x))
    cols_to_divide = cols_to_divide if len(cols_to_divide) != 0 else df.columns
    df[cols_to_divide] = df[cols_to_divide].div(division)

    for col in df.columns:
        col_mix = col + " " + ylabel if isinstance(ylabel, str) and includeColYlabel else ylabel

        df[col] = df[col].fillna(0)

        if col == "swap":
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
        if highlight_intervals is not None:
            for interval in highlight_intervals:
                start, end, color, alpha = interval
                ax.axvspan(start, end, color=color, alpha=alpha)

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


def plot_jmeter(folder, file_name, ignore_chunck=False, chunk_size=10000000, throughput_interval=60):
    if not ignore_chunck:
        chunks = pd.read_csv(folder.joinpath(file_name), chunksize=chunk_size, low_memory=False, sep=",")

        # DataFrame acumulativo para todas as chunks
        df_total = pd.DataFrame()
        for i, chunk in enumerate(chunks):
            print(f'Processing chunk {i}')
            # Acumula os dados do chunk no dataframe total
            df_total = pd.concat([df_total, chunk], ignore_index=True)
    else:
        df_total = pd.read_csv(folder.joinpath(file_name), low_memory=False, sep=",")


    print(df_total)

    # Converter 'timeStamp' para segundos (relativos ao início)
    df_total['timeStamp'] = (df_total['timeStamp'] - df_total['timeStamp'].min()) / 1000  # Em segundos

    # Gerar as métricas gerais
    metricas_geral = calcular_metricas(df_total)
    print("General metrics (considering all chunks):")
    for key, value in metricas_geral.items():
        print(f"{key}: {value}")

    # Obter os labels únicos da coluna 'label'
    unique_labels = df_total['label'].unique()

    # Gerar gráficos de série temporal para cada label
    for label in unique_labels:
        df_label = df_total[df_total['label'] == label]

        # Gerar o gráfico de série temporal para o label específico
        plt.figure(figsize=(10, 6))
        plt.plot(df_label['timeStamp'] / 3600, df_label['elapsed'], label=f'Response Time (ms) - {label}', color='blue')

        # Configurações do gráfico
        plt.title(f'Time Series of Response Time - {label}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Response Time (seconds)')
        plt.grid(True)
        plt.legend()

        # Exibir o gráfico
        plt.tight_layout()
        plt.savefig(folder.joinpath('plots_img/jmeter').joinpath(f"Time Series of Response Time - {label}.png"))
        plt.show()

    # Gerar o gráfico de série temporal geral (todos os labels juntos)
    plt.figure(figsize=(10, 6))
    plt.plot(df_total['timeStamp'] / 3600, df_total['elapsed'], label='Overall Response Time (ms)', color='red')

    # Configurações do gráfico geral
    plt.title('Time Series of Overall Response Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    plt.legend()

    # Exibir o gráfico geral
    plt.tight_layout()
    plt.savefig(folder.joinpath('plots_img/jmeter').joinpath('Time Series of Overall Response Time.png'))
    plt.show()

    # Adiciona a coluna de intervalo de tempo em minutos
    df_total['timeBucket'] = (df_total['timeStamp'] // throughput_interval) * throughput_interval

    # Calcular a vazão em cada intervalo de tempo
    throughput_df = df_total.groupby('timeBucket').size().reset_index(name='Throughput')

    # Gerar o gráfico de série temporal da vazão
    plt.figure(figsize=(10, 6))
    plt.plot(throughput_df['timeBucket'] / 3600, throughput_df['Throughput'], label='Throughput (requests/interval)', color='green')

    # Configurações do gráfico de vazão
    plt.title('Time Series of Throughput')
    plt.xlabel('Time (hours)')
    plt.ylabel(f'Throughput (requests per {throughput_interval} seconds)')
    plt.grid(True)
    plt.legend()

    # Exibir o gráfico de vazão
    plt.tight_layout()
    plt.savefig(folder.joinpath('plots_img/jmeter').joinpath('Time Series of Throughput.png'))
    plt.show()
    return 1

def calcular_metricas(df):
    metricas = {}

    # Amostras (# Samples)
    metricas['# Samples'] = df.shape[0]

    # Média (Mean)
    metricas['Mean'] = df['elapsed'].mean() / 1000  # Converte para segundos

    # Min (Minimum)
    metricas['Min'] = df['elapsed'].min() / 1000  # Converte para segundos

    # Max (Maximum)
    metricas['Max'] = df['elapsed'].max() / 1000  # Converte para segundos

    # Desvio Padrão (Standard Deviation)
    metricas['Standard Deviation'] = df['elapsed'].std() / 1000  # Converte para segundos

    # Contagem de Erros (Number of Errors)
    metricas['Error Count'] = df[df['success'] == False].shape[0]

    # % de Erro (Percentage of Errors)
    metricas['Error Percentage'] = (metricas['Error Count'] / metricas['# Samples']) * 100

    # Vazão (Throughput)
    total_time = (df['timeStamp'].max() - df['timeStamp'].min())  # Em segundos
    metricas['Throughput'] = metricas['# Samples'] / total_time if total_time > 0 else 0

    return metricas


import glob
import re
from pathlib import Path
import functools  # Added for lru_cache


# Ensure plots_img directory exists (helper, can be placed elsewhere or ensured by user)
# Path("plots_img").mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=None)  # Cache results of this function
def extract_name_pid(process_str):
    """Extrai o nome do processo e o PID da string 'nome_do_processo(pid)'."""
    if not isinstance(process_str, str):  # Handle potential non-string inputs gracefully
        return str(process_str), None
    match = re.match(r"(.*)\((\d+)\)", process_str)
    if match:
        name, pid_str = match.groups()
        return name, pid_str  # Keep PID as string to match original logic if it implies type
        # If PID should be int, convert here: int(pid_str)
    return process_str, None


def load_and_split_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, sep=';', dtype={'UID': str, 'PID': str, 'PPID': str})  # Add dtype for IDs
    df['process_occurrences'] = pd.to_numeric(df['process_occurrences'], errors='coerce').fillna(0).astype(int)

    # Create a group identifier that increments each time 'process' is 'EMPTIED'
    df['group'] = (df['process'] == 'EMPTIED').cumsum()

    # Remove 'EMPTIED' rows
    df = df[df['process'] != 'EMPTIED']

    # Split the DataFrame into subdatasets based on the group identifier
    subdatasets = [group_df.reset_index(drop=True) for _, group_df in df.groupby('group')]

    return subdatasets


def process_subdatasets_optimized(subdatasets, process_counts):
    processed_list = []
    for subdataset_orig in subdatasets:
        subdataset = subdataset_orig.copy()

        subdataset['datetime'] = pd.to_datetime(subdataset['datetime'], format='%a %b %d %H:%M:%S %Y')

        p_info = pd.DataFrame(subdataset['process'].apply(extract_name_pid).tolist(), index=subdataset.index,
                              columns=['_p_name', '_p_pid'])
        pp_info = pd.DataFrame(subdataset['parent'].apply(extract_name_pid).tolist(), index=subdataset.index,
                               columns=['_pp_name', '_pp_pid'])

        subdataset_keyed = pd.concat([subdataset, p_info, pp_info], axis=1)

        subdataset_keyed['process_key'] = list(zip(
            subdataset_keyed['_p_name'], subdataset_keyed['_pp_name'],
            subdataset_keyed['UID'], subdataset_keyed['_p_pid'], subdataset_keyed['_pp_pid']
        ))

        # MODIFICATION START
        # Define a function to apply to each group, now accepting 'key' explicitly
        def cumulative_sum_with_prior(group_data, key):  # 'key' is now an argument
            # 'key' is the process_key from groupby, passed explicitly

            initial_offset = process_counts.get(key, 0)

            group_data['process_occurrences'] = group_data['process_occurrences'].cumsum() + initial_offset

            if not group_data.empty:
                process_counts[key] = group_data['process_occurrences'].iloc[-1]
            return group_data

        # Apply the function, passing g.name (the key) explicitly to cumulative_sum_with_prior
        subdataset_processed = subdataset_keyed.groupby('process_key', sort=False, group_keys=False).apply(
            lambda g: cumulative_sum_with_prior(g.copy(), g.name)  # Pass g.name as the 'key'
        )
        # MODIFICATION END

        subdataset_processed.drop(columns=['_p_name', '_p_pid', '_pp_name', '_pp_pid', 'process_key'], inplace=True,
                                  errors='ignore')
        processed_list.append(subdataset_processed)

    return processed_list, process_counts

def process_all_files(directory):
    # Encontrar todos os arquivos que começam com 'fragmentation_'
    pattern = os.path.join(directory, 'fragmentation_*.csv')
    files = sorted(glob.glob(pattern))  # Sort files to ensure chronological processing if names imply order

    all_datasets = []
    # process_counts stores the cumulative counts for process_keys across files
    process_counts = {}

    for file_path in files:
        print(f"Processing file: {file_path}")
        # Carregar e dividir o dataset em subdatasets
        subdatasets = load_and_split_dataset(file_path)

        # Processar cada subdataset e atualizar as contagens de processos
        # Uses the optimized version
        processed_subdatasets_list, process_counts = process_subdatasets_optimized(
            subdatasets,
            process_counts  # Pass and update the same dictionary
        )

        # Unir todos os subdatasets processados DO CURRENT FILE
        if processed_subdatasets_list:  # Check if list is not empty
            merged_dataset_for_file = pd.concat(processed_subdatasets_list, ignore_index=True)
            all_datasets.append(merged_dataset_for_file)

    # Concatenar todos os datasets de todos os arquivos em um só
    if not all_datasets:  # Handle case with no data
        print("Warning: No data found or processed.")
        # Return an empty DataFrame with expected columns to prevent downstream errors
        return pd.DataFrame(
            columns=['datetime', 'process', 'parent', 'UID', 'process_occurrences'])  # Adjust columns as needed

    final_dataset = pd.concat(all_datasets, ignore_index=True)
    return final_dataset


def plot_fragmentation(folder, merge_equals=True, process_per_plot=20, relevant_maximum=200,
                       time_threshold=None, name_filter=None, adjust_x_limits=False, highlight_processes=None,
                       top_n_processes=None, cache_file='processed_data_cache.csv'):
    if isinstance(folder, str):  # Ensure folder is a Path object
        folder = Path(folder)

    cache_path = folder.joinpath(cache_file)
    plots_img_path = folder.joinpath('plots_img')
    plots_img_path.mkdir(parents=True, exist_ok=True)  # Ensure plot directory exists

    # Verificar se o arquivo de cache existe
    if os.path.exists(cache_path):
        print("Carregando dados do cache...")
        final_dataset = pd.read_csv(cache_path, parse_dates=['datetime'])
    else:
        print("Processando os arquivos e salvando o cache...")
        # Processar todos os arquivos
        final_dataset = process_all_files(folder)  # Uses optimized processing chain

        # Converter 'process_occurrences' para numérico (already done in load_and_split, but good for safety)
        final_dataset['process_occurrences'] = pd.to_numeric(final_dataset['process_occurrences'],
                                                             errors='coerce').fillna(0).astype(int)

        # Salvar o dataset processado no arquivo de cache
        if not final_dataset.empty:
            final_dataset.to_csv(cache_path, index=False)
        else:
            print("Warning: final_dataset is empty, skipping cache save.")
            # If final_dataset is empty, we should probably return or handle this
            # as plotting will fail. For now, let it proceed to see errors if any.

    if final_dataset.empty:
        print("No data to plot after processing/loading from cache.")
        return 0  # Or some other indicator of no action

    print("Final dataset loaded/processed.")

    if merge_equals:
        print("Applying merge_equals logic...")
        # 1. Simplify 'process' column: apply extract_name_pid to get just the name part.
        #    The original loop updated final_dataset.loc[index, 'process'] = name
        final_dataset['process'] = final_dataset['process'].apply(lambda x: extract_name_pid(x)[0])

        # 2. Perform the cumulative sum logic.
        #    The original loop:
        #       if name in lastFounds: final_dataset.loc[index, 'process_occurrences'] += lastFounds[name]
        #       lastFounds[name] = final_dataset.loc[index, 'process_occurrences']
        #    This is equivalent to a groupby().cumsum() if 'process_occurrences' are the base values
        #    that need to be summed up chronologically for each simplified process name.
        #    `sort=False` is crucial to mimic the original row-by-row processing order.
        final_dataset['process_occurrences'] = final_dataset.groupby('process', sort=False)[
            'process_occurrences'].cumsum()
        print("merge_equals logic applied.")

    # 1. Agrupar os dados pelo processo (simplified name if merge_equals was true)
    grouped = final_dataset.groupby('process')

    # 2. Criar um subdataset para cada processo
    subdatasets = {process: group.copy() for process, group in
                   grouped}  # Use .copy() to avoid SettingWithCopyWarning later

    # 3. Filtrar os subdatasets onde o valor da última ocorrência é maior que o limite
    #    Ensure 'process_occurrences' is not empty before accessing iloc[-1]
    subdatasets_filtrados = {
        process: data for process, data in subdatasets.items()
        if not data['process_occurrences'].empty and data['process_occurrences'].iloc[-1] > relevant_maximum
    }

    # Filtro adicional por tempo, se 'time_threshold' for especificado
    if time_threshold is not None:
        subdatasets_filtrados = {
            process: data for process, data in subdatasets_filtrados.items()
            if not data['datetime'].empty and (
                        data['datetime'].max() - data['datetime'].min()).total_seconds() / 3600 <= time_threshold
        }

    # Filtro adicional por nome de processo, se 'name_filter' for especificado
    if name_filter is not None:
        name_filter_set = set(name_filter)  # Use a set for faster lookups if name_filter is large
        subdatasets_filtrados = {
            process: data for process, data in subdatasets_filtrados.items()
            if any(f in process for f in name_filter_set)
        }

    # Separar os processos destacados
    highlighted_subdatasets = {}
    if highlight_processes:
        for process_name in highlight_processes:
            if process_name in subdatasets_filtrados:
                highlighted_subdatasets[process_name] = subdatasets_filtrados.pop(process_name)

    y_formatter = mticker.ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-1, 1))

    # Plotar gráficos para os processos destacados (em um único gráfico)
    if highlighted_subdatasets:
        fig, ax = plt.subplots(figsize=(10, 10))
        max_time_highlighted = 0
        for process, subset in highlighted_subdatasets.items():
            if subset['datetime'].min() == subset['datetime'].max():  # Avoid division by zero for single point data
                subset.loc[:, 'time_in_hours'] = 0
            else:
                subset.loc[:, 'time_in_hours'] = (subset['datetime'] - subset[
                    'datetime'].min()).dt.total_seconds() / 3600

            ax.step(subset['time_in_hours'], subset['process_occurrences'], where='post', label=f'{process}', linewidth=5)
            if adjust_x_limits and not subset['time_in_hours'].empty:
                max_time_highlighted = max(max_time_highlighted, subset['time_in_hours'].max())

        ax.legend(loc='best', fontsize=15, title='Process')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Process occurrences', labelpad=15)

        ax.yaxis.set_major_formatter(y_formatter)

        if adjust_x_limits:
            ax.set_xlim([0, max_time_highlighted if max_time_highlighted > 0 else 1])  # ensure xlim is not [0,0]

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(plots_img_path.joinpath("Time Series of memory fragmentation - Highlighted Processes.svg"),
                    bbox_inches='tight', dpi=300, format="svg")
        plt.close(fig)

    # 4. Ordenar os subdatasets filtrados (remaining after highlight pop) pelo último valor de 'process_occurrences'
    # Ensure 'process_occurrences' is not empty before accessing iloc[-1]
    subdatasets_ordenados = dict(
        sorted(
            (item for item in subdatasets_filtrados.items() if not item[1]['process_occurrences'].empty),
            # Filter out empty occurrences series
            key=lambda item: item[1]['process_occurrences'].iloc[-1],
            reverse=True
        )
    )

    # Se 'top_n_processes' for especificado, limitar aos N processos com mais ocorrências
    if top_n_processes is not None:
        subdatasets_ordenados = dict(list(subdatasets_ordenados.items())[:top_n_processes])

    # 5. Definir o tempo máximo entre todos os subdatasets filtrados, se o ajuste do eixo X estiver ativado
    max_time_overall = 0
    if adjust_x_limits and subdatasets_ordenados:
        # Calculate time_in_hours for all relevant datasets first to find the true max_time
        temp_max_times = []
        for process_name_ord, data_ord in subdatasets_ordenados.items():
            if data_ord['datetime'].min() == data_ord['datetime'].max():
                data_ord.loc[:, 'time_in_hours'] = 0
                temp_max_times.append(0)
            else:
                data_ord.loc[:, 'time_in_hours'] = (data_ord['datetime'] - data_ord[
                    'datetime'].min()).dt.total_seconds() / 3600
                if not data_ord['time_in_hours'].empty:
                    temp_max_times.append(data_ord['time_in_hours'].max())
        if temp_max_times:  # Check if list is not empty
            max_time_overall = max(temp_max_times)

    # 6. Plotar os gráficos apenas com os processos que atendem ao critério e já estão ordenados
    process_list = list(subdatasets_ordenados.keys())
    if not process_list:  # If no processes left to plot
        print("No regular processes to plot after filtering and highlighting.")
        return 1  # Or 0 if highlighted plots were also not made.

    num_plots_to_generate = (len(process_list) - 1) // process_per_plot + 1 if process_per_plot > 0 else 0

    for i in range(num_plots_to_generate):
        fig, ax = plt.subplots(figsize=(10, 10))
        processes_to_plot_on_this_fig = process_list[i * process_per_plot:(i + 1) * process_per_plot]

        current_max_time_for_plot = 0  # For non-adjust_x_limits case per plot
        for process_name_iter in processes_to_plot_on_this_fig:
            subset = subdatasets_ordenados[process_name_iter]
            # 'time_in_hours' should already be calculated if adjust_x_limits was true.
            # If not, it needs to be calculated here.
            if 'time_in_hours' not in subset.columns:
                if subset['datetime'].min() == subset['datetime'].max():
                    subset.loc[:, 'time_in_hours'] = 0
                else:
                    subset.loc[:, 'time_in_hours'] = (subset['datetime'] - subset[
                        'datetime'].min()).dt.total_seconds() / 3600

            ax.step(subset['time_in_hours'], subset['process_occurrences'], where='post', label=f'{process_name_iter}', linewidth=5)
            if not adjust_x_limits and not subset['time_in_hours'].empty:  # find max time for this plot only
                current_max_time_for_plot = max(current_max_time_for_plot, subset['time_in_hours'].max())

        ax.legend(loc='best', fontsize=15, title='Process', ncol=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Process occurrences', labelpad=15)

        ax.yaxis.set_major_formatter(y_formatter)

        if adjust_x_limits:
            ax.set_xlim([0, max_time_overall if max_time_overall > 0 else 1])
        else:  # Set x_lim based on current plot's data if not adjusting globally
            ax.set_xlim([0, current_max_time_for_plot if current_max_time_for_plot > 0 else 1])

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(plots_img_path.joinpath(
            f"Time Series of memory fragmentation - Process {i * process_per_plot + 1} to {(i + 1) * process_per_plot}.svg"),
                    bbox_inches='tight', dpi=300, format="svg")
        plt.close(fig)

    return 1