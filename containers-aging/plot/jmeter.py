"""Graficos e metricas dos resultados do JMeter (CSV com separador ",")."""
import pandas as pd
from matplotlib import pyplot as plt

import style  # noqa: F401  (aplica o estilo global)


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
