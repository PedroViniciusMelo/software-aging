"""
Gera os graficos de cada cenario: uma subpasta de <logs_dir> por cenario (ex.:
results/debian12_docker_27.2.0, ver scenario.py), com as imagens salvas em <cenario>/plots_img.
Tambem aceita a pasta de um unico cenario (a que contem o cpu.csv).

    python3 plot/plot_data.py [logs_dir]      # padrao: ./logs/
"""
import argparse
import shutil
from pathlib import Path

from fragmentation import plot_fragmentation
from jmeter import plot_jmeter
from plot_utils import plot, plot_time_series

DEFAULT_LOGS_DIR = "./logs/"

# Periodos (em horas) destacados nos graficos: (inicio, fim, cor, alpha)
intervals = [
    (0, 12, "green", 0.3),
    (84, 96, "green", 0.3),
    (168, 180, "green", 0.3),
    (252, 264, "green", 0.3)
]

PROCESS_YLABEL = {
    'cpu': 'CPU usage (percentage)',
    "rss": "Physical memory usage(MB)",
    "vsz": "Virtual memory usage (MB)",
    "swap": "Swap used(MB)",
    'mem': 'Memory usage (percentage)'
}

# Processos monitorados (monitoring.processes no config.yaml): todo CSV da pasta com este
# cabecalho vira grafico, qualquer que seja o processo.
PROCESS_HEADER = "cpu;mem;rss;vsz;threads;swap;date_time"
# Titulos (e ordem) dos processos conhecidos; os demais saem como "Process - <nome>", em ordem alfabetica.
PROCESS_TITLES = {
    "docker.csv": "Process - Docker",
    "dockerd.csv": "Process - Dockerd",
    "containerd.csv": "Process - Containerd",
    "containerd-shim.csv": "Process - Containerd-shim",
    "docker-proxy.csv": "Process - docker-proxy",
    "runc.csv": "Process - runc",
    "podman.csv": "Process - Podman",
    "conmon.csv": "Process - Conmon",
    "crun.csv": "Process - Crun",
    "systemd.csv": "Process - systemd",
    "java.csv": "Teastore - Java",
    "beam.smp.csv": "Rabbitmq - beam.smp",
    "postgres_process.csv": "Postgres - postgres",
    "mysqld.csv": "Teastore - mysqld",
}
# process_<thread>.csv: custo por thread do coletor antigo (substituido por monitoring_cost.csv), mesmo cabecalho
OLD_COST_PREFIX = "process_"
# Tempos do ciclo de vida medidos pelo coletor (container_metrics)
CONTAINER_METRICS = [
    ("Postgres", "postgres.csv"),
    ("Redis", "redis.csv"),
    ("RabbitMQ", "rabbitmq.csv"),
    ("Nginx", "nginx.csv"),
]

FRAGMENTATION_NAME_FILTER = ["mysqld", "http-nio", "nginx", "redis-server", "beam.smp", "rabbitmq-server", "python3",
                             "systemd", "dockerd", "containerd", "containerd-shim", "docker-proxy", "runc", "conmon",
                             "podman", "crun", "java", "postgres", "netavark"]
FRAGMENTATION_HIGHLIGHT = ["dockerd", "http-nio-8080-e", "podman"]


def recreate_plot_folders(base_folder):
    plots_folder = base_folder.joinpath('plots_img')
    if plots_folder.exists():
        shutil.rmtree(plots_folder)
    plots_folder.mkdir()
    plots_folder.joinpath('fragmentation').mkdir()
    plots_folder.joinpath('jmeter').mkdir()


def _first_line(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            return file.readline().strip()
    except OSError:
        return ""


def find_process_files(base_folder):
    """(titulo, arquivo) de cada CSV de processo da pasta: os conhecidos primeiro, os outros em ordem alfabetica."""
    found = sorted(path.name for path in base_folder.glob("*.csv")
                   if not path.name.startswith(OLD_COST_PREFIX) and _first_line(path) == PROCESS_HEADER)
    known = [name for name in PROCESS_TITLES if name in found]
    others = [name for name in found if name not in PROCESS_TITLES]
    return [(PROCESS_TITLES[name], name) for name in known] +            [(f"Process - {name[:-len('.csv')]}", name) for name in others]


def plot_processes(base_folder, processes):
    plotted = 0
    for title, filename in processes:
        if base_folder.joinpath(filename).exists():
            plotted += plot(
                title=title,
                folder=base_folder,
                filename=filename,
                ylabel=PROCESS_YLABEL,
                includeColYlabel=True,
                cols_to_divide=['rss', 'vsz', 'swap'],
                division=1024
            )
    return plotted


def start(base_folder, qtd_item):
    recreate_plot_folders(base_folder)
    ploted = 0

    ploted += plot(
        title="CPU",
        folder=base_folder,
        filename='cpu.csv',
        ylabel='(percentage)',
        includeColYlabel=True,
        highlight_intervals=intervals
    )

    if base_folder.joinpath('disk.csv').exists():
        ploted += plot(title="Disk", folder=base_folder, filename='disk.csv', ylabel='Disk usage (GB)',
                       division=1048576)

    if base_folder.joinpath('process.csv').exists():
        ploted += plot(title="Zumbis", folder=base_folder, filename='process.csv', ylabel='Zumbis processes(qtt)')

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
        includeColYlabel=True,
        highlight_intervals=intervals
    )

    if base_folder.joinpath('energy_rapl.csv').exists():
        ploted += plot(
            title="Energy - RAPL",
            folder=base_folder,
            filename='energy_rapl.csv',
            ylabel={
                "package_j": "Package energy (J)",
                "core_j": "Core energy (J)",
                "uncore_j": "Uncore energy (J)",
                "dram_j": "DRAM energy (J)",
                "total_j": "Total energy (J)",
                "cpu_temp_c": "CPU package temperature (C)"
            },
            decimal_separator=".",
            highlight_intervals=intervals
        )

    if base_folder.joinpath('energy_smart_plug.csv').exists():
        ploted += plot(
            title="Energy - Smart plug",
            folder=base_folder,
            filename='energy_smart_plug.csv',
            ylabel={
                "power_w": "Power (W)",
                "kwh_interval": "Energy per interval (kWh)",
                "kwh_total": "Cumulative energy (kWh)",
                "state": "Plug state (1=ON, 0=OFF)"
            },
            decimal_separator=".",
            highlight_intervals=intervals
        )

    # tempo de resposta do servidor: <cenario>/<cenario>.csv
    if base_folder.joinpath(base_folder.name + '.csv').exists():
        ploted += plot(title="Server response time", folder=base_folder, filename=base_folder.name + '.csv',
                       ylabel='Response time(s)')

    if base_folder.joinpath('disk_write_read.csv').exists():
        ploted += plot(
            title="Read and Write",
            folder=base_folder,
            filename='disk_write_read.csv',
            ylabel={'tps': 'TPS', "kB_reads": "kB_reads", "kB_wrtns": "kB_wrtns", "kB_dscds": "kB_dscds"}
        )

    ploted += plot_processes(base_folder, find_process_files(base_folder))

    for title, filename in CONTAINER_METRICS:
        if base_folder.joinpath(filename).exists():
            ploted += plot_time_series(
                save_folder=base_folder.joinpath('plots_img'),
                title=f"Container Metrics - {title}",
                file_path=base_folder.joinpath(filename),
                x_label='Time (hours)',
                y_label='Time taken (seconds)',
            )

    ploted += plot_fragmentation(base_folder,
                                 merge_equals=True,
                                 adjust_x_limits=False,
                                 process_per_plot=21,
                                 top_n_processes=10,
                                 name_filter=FRAGMENTATION_NAME_FILTER,
                                 highlight_processes=FRAGMENTATION_HIGHLIGHT)

    jmeter_file = base_folder.joinpath("jmeter_" + base_folder.name + ".csv")
    if jmeter_file.exists():
        ploted += plot_jmeter(base_folder, file_name=jmeter_file.name, ignore_chunck=False)

    print(f"Ploted {ploted}/{qtd_item - 4} ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", nargs="?", default=DEFAULT_LOGS_DIR,
                        help="pasta com uma subpasta por cenario (padrao: %(default)s)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if logs_dir.joinpath("cpu.csv").exists():
        folders = [logs_dir]  # a propria pasta de um cenario
    else:
        folders = [item for item in logs_dir.iterdir() if item.is_dir()]

    for folder in folders:
        start(base_folder=folder, qtd_item=len([file for file in folder.iterdir() if file.suffix == '.csv']))


if __name__ == "__main__":
    main()
