from pathlib import Path
import shutil

from concurrent.futures import ProcessPoolExecutor

from plot_utils import plot, plot_jmeter, plot_fragmentation, plot_time_series

base_dir = "D:/final"


def start(base_folder, qtd_item):
    plots_folder = base_folder.joinpath('plots_img')

    if plots_folder.exists():
        shutil.rmtree(plots_folder)
        plots_folder.mkdir()
    else:
        plots_folder.mkdir()

    fragmentation_folder = base_folder.joinpath('plots_img/fragmentation')
    fragmentation_folder.mkdir()

    jmeter_folder = base_folder.joinpath('plots_img/jmeter')
    jmeter_folder.mkdir()

    ploted = 0


    ploted += plot(
        title="CPU",
        folder=base_folder,
        filename='cpu.csv',
        ylabel='(percentage)',
        includeColYlabel=True
    )

    # ploted += plot(
    #     title="Disk",
    #     folder=base_folder,
    #     filename='disk.csv',
    #     ylabel='Disk usage (GB)',
    #     division=1048576
    # )
    #
    # ploted += plot(
    #     title="Zumbis",
    #     folder=base_folder,
    #     filename='process.csv',
    #     ylabel='Zumbis processes(qtt)'
    # )

    ploted += plot(
        title="Memory",
        folder=base_folder,
        filename='memory.csv',
        ylabel={
            "used": "Memory used(MB)",
            "cached": "Memory cached(MB)",
            "buffers": "Memory buffers(MB)",
            "swap": "Swap used(MB)"
        },
        division=1024,
        includeColYlabel=True
    )

    # ploted += plot(
    #     title="Server response time",
    #     folder=base_folder,
    #     filename=base_folder.name + '.csv',
    #     ylabel='Response time(s)',
    # )

    # ploted += plot(
    #     title="Read and Write",
    #     folder=base_folder,
    #     filename='disk_write_read.csv',
    #     ylabel={'tps': 'TPS', "kB_reads": "kB_reads", "kB_wrtns": "kB_wrtns", "kB_dscds": "kB_dscds"}
    # )
    #
    # if "docker" in base_folder.name:
    #     ploted += plot(
    #         title="Process - Docker",
    #         folder=base_folder,
    #         filename='docker.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - Dockerd",
    #         folder=base_folder,
    #         filename='dockerd.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - Containerd",
    #         folder=base_folder,
    #         filename='containerd.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - Containerd-shim",
    #         folder=base_folder,
    #         filename='containerd-shim.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - docker-proxy",
    #         folder=base_folder,
    #         filename='docker-proxy.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - runc",
    #         folder=base_folder,
    #         filename='runc.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    # else:
    #     ploted += plot(
    #         title="Process - Podman",
    #         folder=base_folder,
    #         filename='podman.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - Conmon",
    #         folder=base_folder,
    #         filename='conmon.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - Crun",
    #         folder=base_folder,
    #         filename='crun.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    #     ploted += plot(
    #         title="Process - systemd",
    #         folder=base_folder,
    #         filename='systemd.csv',
    #         ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #                 "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #         includeColYlabel=True,
    #         cols_to_divide=['rss', 'vsz', 'swap'],
    #         division=1024
    #     )
    #
    # # ------------------------------------------------- IMAGES --------------------------------------------
    # ploted += plot(
    #     title="Teastore - Java",
    #     folder=base_folder,
    #     filename='java.csv',
    #     ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #             "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #     includeColYlabel=True,
    #     cols_to_divide=['rss', 'vsz', 'swap'],
    #     division=1024
    # )
    #
    # ploted += plot(
    #     title="Rabbitmq - beam.smp",
    #     folder=base_folder,
    #     filename='beam.smp.csv',
    #     ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #             "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #     includeColYlabel=True,
    #     cols_to_divide=['rss', 'vsz', 'swap'],
    #     division=1024
    # )
    #
    # ploted += plot(
    #     title="Postgres - postgres",
    #     folder=base_folder,
    #     filename='postgres_process.csv',
    #     ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #             "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #     includeColYlabel=True,
    #     cols_to_divide=['rss', 'vsz', 'swap'],
    #     division=1024
    # )
    #
    # ploted += plot(
    #     title="Teastore - mysqld",
    #     folder=base_folder,
    #     filename='mysqld.csv',
    #     ylabel={'cpu': 'CPU usage (percentage)', "rss": "Physical memory usage(MB)",
    #             "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)", 'mem': 'Memory usage (percentage)'},
    #     includeColYlabel=True,
    #     cols_to_divide=['rss', 'vsz', 'swap'],
    #     division=1024
    # )

    # # --------------------------------------------- Container Metrics --------------------------------------------
    #
    # ploted += plot_time_series(
    #     save_folder=base_folder.joinpath('plots_img'),
    #     title="Container Metrics - Postgres",
    #     file_path=base_folder.joinpath('postgres.csv'),
    #     x_label='Time (hours)',
    #     y_label='Time taken (seconds)',
    # )
    #
    # ploted += plot_time_series(
    #     save_folder=base_folder.joinpath('plots_img'),
    #     title="Container Metrics - Redis",
    #     file_path=base_folder.joinpath('redis.csv'),
    #     x_label='Time (hours)',
    #     y_label='Time taken (seconds)',
    # )
    #
    # ploted += plot_time_series(
    #     save_folder=base_folder.joinpath('plots_img'),
    #     title="Container Metrics - RabbitMQ",
    #     file_path=base_folder.joinpath('rabbitmq.csv'),
    #     x_label='Time (hours)',
    #     y_label='Time taken (seconds)',
    # )
    #
    # ploted += plot_time_series(
    #     save_folder=base_folder.joinpath('plots_img'),
    #     title="Container Metrics - Nginx",
    #     file_path=base_folder.joinpath('nginx.csv'),
    #     x_label='Time (hours)',
    #     y_label='Time taken (seconds)',
    # )
    #
    # # ------------------------------------------------- Machine Metrics --------------------------------------------
    #
    # # ploted += plot_event_counts_per_hour(
    # #     save_folder=base_folder.joinpath('plots_img'),
    # #     title="Erros",
    # #     file_path=base_folder.joinpath('errors.csv'),
    # #     x_label='Time (hours)',
    # #     y_label='Occurrences(qtt)',
    # # )
    #
    # # ------------------------------------------------- PODMAN --------------------------------------------
    #
    #
    # ploted += plot_fragmentation(base_folder,
    #                    merge_equals=True,
    #                    adjust_x_limits=False,
    #                    process_per_plot=21,
    #                    top_n_processes=10,
    #                    name_filter=["mysqld", "http-nio", "nginx", "redis-server", "beam.smp", "rabbitmq-server", "python3", "systemd", "dockerd", "containerd", "containerd-shim", "docker-proxy", "runc", "conmon", "podman", "crun", "java", "postgres", "netavark"],
    #                    highlight_processes=["dockerd", "http-nio-8080-e", "podman"]
    #                    )  # name_filter=["dockerd", "containerd", "containerd-shim", "docker-proxy", "runc"])

    #ploted += plot_jmeter(base_folder, file_name="jmeter_" + base_folder.name + ".csv", ignore_chunck=False)
    #
    # plt.close('all')
    #
    # print(f"Ploted {ploted}/{qtd_item -4} ")


if __name__ == "__main__":
    # Obter todas as pastas no diretório base
    folders = [item for item in Path(base_dir).iterdir() if item.is_dir()]

    # Criar um pool de processos para executar as tarefas
    with ProcessPoolExecutor() as executor:
        # Submeter as tarefas para execução paralela
        futures = [
            executor.submit(start, folder, len([file for file in folder.iterdir() if file.suffix == '.csv']))
            for folder in folders
        ]

        # Aguardar a conclusão de todas as tarefas
        for future in futures:
            try:
                future.result()  # Captura possíveis exceções
            except Exception as e:
                print(f"Erro ao processar: {e}")
