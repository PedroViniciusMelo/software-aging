import pandas as pd
import numpy as np
from scipy.stats import theilslopes
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json  # Added for caching


# Helper functions for NaN conversion for JSON compatibility
def convert_nan_to_none(obj):
    """Recursively convert np.nan to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(i) for i in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def convert_none_to_nan(obj):
    """Recursively convert None back to np.nan after JSON deserialization."""
    if isinstance(obj, dict):
        return {k: convert_none_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_none_to_nan(i) for i in obj]
    elif obj is None:
        return np.nan
    return obj


# Função fornecida pelo usuário para extrair informações do nome da pasta
def extract_categorical_info(folder_name):
    """Extract System, Software, and Software Version based on folder name."""
    folder_name_lower = folder_name.lower()

    # Determine System
    if "debian" in folder_name_lower:
        system = "Debian 12"
    elif "ubuntu22" in folder_name_lower:
        system = "Ubuntu 22.04"
    elif "ubuntu24" in folder_name_lower:
        system = "Ubuntu 24.04"
    else:
        system = "Unknown OS"

    # Determine Software and Version
    if "docker" in folder_name_lower:
        software = "Docker"
        version = "23"  # Default Docker version
        if "novo" in folder_name_lower:  # Assuming "novo" implies the newer version
            version = "27.2"
    elif "podman" in folder_name_lower:
        software = "Podman"
        version = "5.1.3"  # Default Podman version if not specified otherwise
    else:
        software = "Unknown Software"
        version = "N/A"

    return system, software, version


# Robust Slope calculation
def calculate_slope(df, time_col_numeric, metric_col):
    """
    Calculates Theil-Sen slope for a given metric against time.
    Returns np.nan if data is insufficient or an error occurs.
    Ensures proper alignment and handling of NaNs.
    """
    # Ensure metric_col and time_col_numeric are present
    if metric_col not in df.columns or time_col_numeric not in df.columns:
        # print(f"Warning: Column {metric_col} or {time_col_numeric} not in DataFrame for slope calculation.")
        return np.nan

    # Convert to numeric, coercing errors. This might be redundant if already done, but safe.
    y_series = pd.to_numeric(df[metric_col], errors='coerce')
    x_series = pd.to_numeric(df[time_col_numeric], errors='coerce')

    # Create a temporary df with just these two series to drop NaNs row-wise for alignment
    temp_df = pd.DataFrame({'y': y_series, 'x': x_series}).dropna()

    y_aligned = temp_df['y']
    x_aligned = temp_df['x']

    # Theil-Sen requires at least 2 points
    if len(y_aligned) < 2:
        # print(f"Warning: Insufficient data points (<2) for {metric_col} after NaN removal.")
        return np.nan

    try:
        # Unpack all four values returned by theilslopes
        slope, intercept, low_slope, high_slope = theilslopes(y_aligned, x_aligned, alpha=0.95)
        return slope
    except ValueError:  # Handles errors from theilslopes, e.g., all x values are identical
        # print(f"Warning: ValueError during Theil-Sen calculation for {metric_col}.")
        return np.nan
    except Exception:  # Catch any other unexpected errors
        # print(f"Warning: Unexpected error during Theil-Sen calculation for {metric_col}.")
        return np.nan


# Processes a single scenario subfolder
def process_folder(subfolder_path):
    """
    Processes a single scenario subfolder. Reads CSVs, calculates slopes,
    and returns a dictionary with the results.
    Tries to load results from a cache file first.
    """
    folder_name = subfolder_path.name
    cache_file_path = subfolder_path / "_slope_cache.json"

    # Try to load from cache
    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'r') as f:
                cached_results_raw = json.load(f)
            cached_results = convert_none_to_nan(cached_results_raw)
            print(f"Cache HIT: Carregando resultados para {folder_name} do cache.")
            # Basic check for essential keys; more robust validation could be added
            if all(key in cached_results for key in ['scenario_folder', 'mem_used_slope']):
                return cached_results
            else:
                print(f"Cache WARN: Cache para {folder_name} com estrutura inválida. Recomputando.")
        except Exception as e:
            print(f"Cache WARN: Erro ao carregar cache para {folder_name}: {e}. Recomputando.")

    print(f"Cache MISS: Processando: {folder_name}")

    system, software, version = extract_categorical_info(folder_name)
    scenario_label = f"{system} - {software} {version}"

    current_folder_results = {
        'scenario_folder': folder_name,
        'scenario_label': scenario_label,
        'system': system,
        'software': software,
        'version': version,
        'mem_used_slope': np.nan,
        'mem_swap_slope': np.nan,
        'cpu_usr_slope': np.nan,
        'cpu_sys_slope': np.nan,
        'processing_log': []  # Initialize log
    }
    current_folder_results['processing_log'].append(f"Iniciando processamento para {folder_name}.")

    # Helper function to process a generic CSV file
    def process_csv_file(file_path, metrics_to_calculate, folder_log_list):
        try:
            df = pd.read_csv(file_path, sep=';')
            if df.empty:
                folder_log_list.append(f"  Arquivo {file_path.name} está vazio. Pulando.")
                return {metric: np.nan for metric in metrics_to_calculate}
        except FileNotFoundError:
            folder_log_list.append(f"  Arquivo {file_path.name} não encontrado. Pulando.")
            return {metric: np.nan for metric in metrics_to_calculate}
        except pd.errors.EmptyDataError:
            folder_log_list.append(f"  Arquivo {file_path.name} está vazio ou mal formatado. Pulando.")
            return {metric: np.nan for metric in metrics_to_calculate}

        try:
            df['date_time'] = pd.to_datetime(df['date_time'])
        except KeyError:
            folder_log_list.append(f"  Coluna 'date_time' não encontrada em {file_path.name}. Pulando.")
            return {metric: np.nan for metric in metrics_to_calculate}
        except Exception as e:
            folder_log_list.append(f"  Erro ao converter 'date_time' em {file_path.name}: {e}. Pulando.")
            return {metric: np.nan for metric in metrics_to_calculate}

        df = df.sort_values(by='date_time').reset_index(drop=True)

        if df.empty or df['date_time'].iloc[0] is pd.NaT:
            folder_log_list.append(
                f"  Dados insuficientes ou data inválida em {file_path.name} após carregar. Pulando.")
            return {metric: np.nan for metric in metrics_to_calculate}

        df['time_numeric_days'] = (df['date_time'] - df['date_time'].iloc[0]).dt.total_seconds() / (24 * 60 * 60)

        slopes = {}
        for col in metrics_to_calculate:
            if col in df.columns:
                # Replace comma with dot for decimal conversion if column is object type
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.', regex=False)

                # Ensure the column is numeric before any calculation
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Apply specific calculation for 'swap'
                if col == "swap":
                    # df[col] is already numeric here
                    df[col] = (df[col] - 8192) * -1

                if not df[col].dropna().empty:
                    slope_value = calculate_slope(df.copy(), 'time_numeric_days', col)  # Pass a copy to calculate_slope
                    slopes[col] = slope_value
                    if np.isnan(slope_value):
                        folder_log_list.append(f"  Slope para '{col}' em {file_path.name} resultou em NaN.")
                else:
                    slopes[col] = np.nan
                    folder_log_list.append(
                        f"  Aviso: Coluna '{col}' em {file_path.name} não possui dados numéricos válidos para cálculo do slope.")
            else:
                slopes[col] = np.nan
                folder_log_list.append(f"  Aviso: Coluna '{col}' não encontrada em {file_path.name}.")
        return slopes

    # Process memory.csv
    mem_file_path = subfolder_path / "memory.csv"
    mem_slopes = process_csv_file(mem_file_path, ['used', 'swap'], current_folder_results['processing_log'])
    current_folder_results['mem_used_slope'] = mem_slopes.get('used', np.nan)
    current_folder_results['mem_swap_slope'] = mem_slopes.get('swap', np.nan)

    # Process cpu.csv
    cpu_file_path = subfolder_path / "cpu.csv"
    cpu_slopes = process_csv_file(cpu_file_path, ['usr', 'sys'], current_folder_results['processing_log'])
    current_folder_results['cpu_usr_slope'] = cpu_slopes.get('usr', np.nan)
    current_folder_results['cpu_sys_slope'] = cpu_slopes.get('sys', np.nan)

    current_folder_results['processing_log'].append(f"Processamento para {folder_name} concluído.")

    # Save to cache
    try:
        with open(cache_file_path, 'w') as f:
            json.dump(convert_nan_to_none(current_folder_results), f, indent=4)
        print(f"Cache SAVE: Resultados para {folder_name} salvos no cache.")
    except Exception as e:
        print(f"Cache WARN: Erro ao salvar cache para {folder_name}: {e}")

    return current_folder_results


def main():
    root_folder_path = Path("./logs")
    all_results_data = []

    print(f"Analisando pastas em: {root_folder_path.resolve()}")
    if not root_folder_path.exists() or not root_folder_path.is_dir():
        print(f"Erro: Pasta raiz '{root_folder_path.resolve()}' não encontrada ou não é um diretório.")
        return

    subfolder_paths = [p for p in root_folder_path.iterdir() if p.is_dir()]
    if not subfolder_paths:
        print(f"Nenhuma subpasta encontrada em '{root_folder_path.resolve()}'.")
        return

    # Processamento sequencial
    for subfolder_path in subfolder_paths:
        result = process_folder(subfolder_path)
        all_results_data.append(result)

    if not all_results_data:
        print("Nenhum dado foi processado. Verifique os logs das pastas.")
        return

    print("Processamento concluído. Agregando resultados...")
    results_df = pd.DataFrame(all_results_data)

    if results_df.empty:
        print("DataFrame de resultados está vazio após o processamento.")
        return

    # Normalização Min-Max para os Slopes
    slope_cols = ['mem_used_slope', 'mem_swap_slope', 'cpu_usr_slope', 'cpu_sys_slope']
    normalized_slope_cols = []

    # Check if all slope columns exist before attempting normalization
    missing_slope_cols = [col for col in slope_cols if col not in results_df.columns]
    if missing_slope_cols:
        print(
            f"Aviso: As seguintes colunas de slope não foram encontradas no DataFrame e serão ignoradas na normalização: {missing_slope_cols}")
        # Filter out missing columns for normalization
        slope_cols = [col for col in slope_cols if col in results_df.columns]

    if not slope_cols:
        print("Nenhuma coluna de slope válida para normalização. Pulando cálculo do Aging Index e plotagem.")
        results_df['aging_index_euclidean'] = np.nan  # Ensure columns exist for printing
        results_df['aging_index_sum'] = np.nan
    else:
        scaler = MinMaxScaler()
        for col in slope_cols:
            # Ensure column is numeric and handle potential all-NaN columns
            if pd.api.types.is_numeric_dtype(results_df[col]):
                # Reshape for scaler if it's a Series (single column)
                data_to_scale = results_df[[col]].copy()  # Use .copy() to avoid SettingWithCopyWarning

                # Check if data_to_scale[col] has variance or is all NaN
                if data_to_scale[col].nunique(dropna=True) > 1:
                    scaled_values = scaler.fit_transform(data_to_scale)
                elif data_to_scale[col].nunique(dropna=True) == 1 and not data_to_scale[col].isnull().all():
                    # All same non-NaN value, scale to 0 (or 0.5 depending on desired behavior for single value)
                    # For min-max, if min==max, result is 0.
                    # To avoid DataConversionWarning if the single value is e.g. an integer
                    data_to_scale[col] = data_to_scale[col].astype(float)
                    scaled_values = scaler.fit_transform(data_to_scale)  # Should result in 0s
                else:  # All NaN or empty, or otherwise problematic for scaler
                    scaled_values = np.full_like(data_to_scale[col].values.reshape(-1, 1), np.nan, dtype=float)

                norm_col_name = f'{col}_norm'
                results_df[norm_col_name] = scaled_values
                normalized_slope_cols.append(norm_col_name)
            else:
                print(f"Aviso: Coluna de slope '{col}' não é numérica e será ignorada na normalização.")
                results_df[f'{col}_norm'] = np.nan  # Ensure normalized column exists as NaN

        if normalized_slope_cols:  # Proceed only if some columns were successfully normalized
            # Calcular o Aging Index (distância euclidiana)
            # Ensure we only use columns that were actually created
            valid_normalized_cols = [col for col in normalized_slope_cols if col in results_df.columns]
            if valid_normalized_cols:
                # Fill NaN with 0 for sum of squares calculation, or handle as per desired logic
                sum_of_squares = results_df[valid_normalized_cols].fillna(0).pow(2).sum(axis=1)
                results_df['aging_index_euclidean'] = np.sqrt(sum_of_squares)
                results_df['aging_index_sum'] = results_df[valid_normalized_cols].fillna(0).sum(axis=1)
            else:
                results_df['aging_index_euclidean'] = np.nan
                results_df['aging_index_sum'] = np.nan
                print("Nenhuma coluna normalizada válida para calcular o Aging Index.")
        else:
            print("Nenhuma coluna foi normalizada. Pulando cálculo do Aging Index.")
            results_df['aging_index_euclidean'] = np.nan  # Ensure columns exist even if not calculated
            results_df['aging_index_sum'] = np.nan

    # Sorting, ensure 'aging_index_euclidean' exists
    if 'aging_index_euclidean' in results_df.columns:
        results_df = results_df.sort_values(by='aging_index_euclidean', ascending=False, na_position='last')
    else:
        print("Coluna 'aging_index_euclidean' não encontrada para ordenação.")

    print("\n--- Resultados Finais ---")
    cols_to_print = ['scenario_label'] + \
                    [col for col in slope_cols if col in results_df.columns] + \
                    [col for col in normalized_slope_cols if col in results_df.columns] + \
                    [col for col in ['aging_index_sum', 'aging_index_euclidean'] if col in results_df.columns]

    print(results_df[cols_to_print].to_string())

    output_csv_file = "aging_analysis_results.csv"
    try:
        results_df.to_csv(output_csv_file, index=False, sep=';', decimal='.')
        print(f"\nResultados detalhados salvos em: {output_csv_file}")
    except Exception as e:
        print(f"\nErro ao salvar resultados em CSV: {e}")

    # Plotagem
    if not results_df.empty:
        # Ensure 'scenario_label' is present for indexing plots
        if 'scenario_label' not in results_df.columns:
            print("Aviso: Coluna 'scenario_label' não encontrada. Plotagem será limitada.")
            plot_df_valid = results_df.dropna(subset=slope_cols, how='all').copy()
            # If no scenario_label, plots might not be meaningful or might error
        else:
            # Use a copy to avoid SettingWithCopyWarning if we modify plot_df_valid later
            plot_df_valid = results_df.dropna(subset=slope_cols, how='all').copy()

        if not plot_df_valid.empty:
            # Set index if 'scenario_label' exists and is suitable for index
            if 'scenario_label' in plot_df_valid.columns and plot_df_valid['scenario_label'].nunique() == len(
                    plot_df_valid):
                plot_df = plot_df_valid.set_index('scenario_label')
            else:
                # If scenario_label is not unique or missing, use default index for plotting
                # This might make plots less interpretable but avoids errors.
                plot_df = plot_df_valid
                print(
                    "Aviso: 'scenario_label' não usado como índice para plotagem (não existe, não é único ou dados insuficientes).")

            try:
                # Gráfico dos slopes brutos
                # Ensure slope_cols that are actually in plot_df are used
                valid_slope_cols_for_plot = [col for col in slope_cols if col in plot_df.columns]
                if valid_slope_cols_for_plot:
                    plot_df[valid_slope_cols_for_plot].plot(kind='bar', figsize=(18, 9), width=0.8)
                    plt.title('Slopes Theil-Sen por Cenário e Métrica (unidade/dia ou %/dia)')
                    plt.ylabel('Valor do Slope')
                    plt.xlabel('Cenário de Teste')
                    plt.xticks(rotation=45, ha="right")
                    plt.legend(title="Métricas")
                    plt.tight_layout()
                    plt.savefig("theil_sen_slopes_raw.png")
                    plt.close()  # Close plot to free memory
                    print("Gráfico de slopes brutos salvo em theil_sen_slopes_raw.png")

                # Gráfico dos slopes normalizados
                # Ensure normalized_slope_cols that are actually in plot_df are used
                valid_normalized_cols_for_plot = [col for col in normalized_slope_cols if col in plot_df.columns]
                if valid_normalized_cols_for_plot:
                    plot_df[valid_normalized_cols_for_plot].plot(kind='bar', figsize=(18, 9), width=0.8)
                    plt.title('Slopes Theil-Sen Normalizados (Min-Max) por Cenário e Métrica')
                    plt.ylabel('Valor do Slope Normalizado [0-1]')
                    plt.xlabel('Cenário de Teste')
                    plt.xticks(rotation=45, ha="right")
                    plt.legend(title="Métricas Normalizadas")
                    plt.tight_layout()
                    plt.savefig("theil_sen_slopes_normalized.png")
                    plt.close()
                    print("Gráfico de slopes normalizados salvo em theil_sen_slopes_normalized.png")

                # Gráfico do Aging Index Euclidiano
                if 'aging_index_euclidean' in plot_df.columns and not plot_df['aging_index_euclidean'].isnull().all():
                    plot_df['aging_index_euclidean'].sort_values(ascending=False).plot(kind='bar', figsize=(15, 7),
                                                                                       color='teal')
                    plt.title('Índice de Envelhecimento Euclidiano por Cenário')
                    plt.ylabel('Distância Euclidiana (Slopes Normalizados)')
                    plt.xlabel('Cenário de Teste')
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig("aging_index_euclidean.png")
                    plt.close()
                    print("Gráfico do índice Euclidiano salvo em aging_index_euclidean.png")

                # Gráfico do Aging Index (Soma Simples dos Normalizados)
                if 'aging_index_sum' in plot_df.columns and not plot_df['aging_index_sum'].isnull().all():
                    plot_df['aging_index_sum'].sort_values(ascending=False).plot(kind='bar', figsize=(15, 7),
                                                                                 color='purple')  # Different color
                    plt.title('Índice de Envelhecimento (Soma dos Slopes Normalizados) por Cenário')
                    plt.ylabel('Soma dos Slopes Normalizados')
                    plt.xlabel('Cenário de Teste')
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig("aging_index_sum_normalized.png")  # New filename
                    plt.close()
                    print("Gráfico do índice de soma dos slopes normalizados salvo em aging_index_sum_normalized.png")

                print("\nAnálise concluída. Verifique os arquivos .png e .csv gerados.")
            except Exception as e:
                print(f"Erro durante a plotagem: {e}")
        else:
            print("Nenhum dado válido para plotar após filtrar cenários com todos os slopes NaN.")
    else:
        print("Nenhum resultado para plotar.")


if __name__ == "__main__":
    main()
