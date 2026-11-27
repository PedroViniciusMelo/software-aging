"""
Gera os graficos de cada cenario: uma subpasta de <logs_dir> por cenario (ex.:
results/debian12_docker_27.2.0, ver scenario.py), com as imagens salvas em <cenario>/plots_img.
Tambem aceita a pasta de um unico cenario (a que contem o cpu.csv).

    python3 plot/plot_data.py [logs_dir]      # padrao: ./logs/
"""
import argparse
import shutil
from pathlib import Path

import metrics
from fragmentation import plot_fragmentation
from jmeter import plot_jmeter
from phases import highlight_intervals
from plot_utils import plot

DEFAULT_LOGS_DIR = "./logs/"

# (titulo, funcao de metrics.py, rotulo do eixo y, destaca as fases do runs.csv)
PLOTS = [
    ("CPU", metrics.cpu, "{col} (percentage)", True),
    ("Memory", metrics.memory, {"used": "Memory used(MB)", "cached": "Memory cached(MB)",
                                "buffers": "Memory buffers(MB)", "swap_used": "Swap used(MB)"}, True),
    ("Disk", metrics.disk, "Disk usage (GB)", False),
    ("Read and Write", metrics.disk_io, {"tps": "TPS", "kB_read_s": "kB read/s", "kB_wrtn_s": "kB written/s",
                                         "kB_dscd_s": "kB discarded/s"}, False),
    ("Zumbis", metrics.zombies, "Zumbis processes(qtt)", False),
    ("Energy - RAPL", metrics.energy_rapl, {"package_j": "Package energy (J)", "core_j": "Core energy (J)",
                                            "uncore_j": "Uncore energy (J)", "dram_j": "DRAM energy (J)",
                                            "total_j": "Total energy (J)",
                                            "cpu_temp_c": "CPU package temperature (C)"}, True),
    ("Energy - Smart plug", metrics.energy_smart_plug, {"power_w": "Power (W)",
                                                        "kwh_interval": "Energy per interval (kWh)",
                                                        "kwh_total": "Cumulative energy (kWh)",
                                                        "state": "Plug state (1=ON, 0=OFF)"}, True),
    ("Server response time", metrics.response_time, "Response time(s)", False),
]
PROCESS_YLABEL = {
    "cpu": "CPU usage (percentage of one core)", "mem": "Memory usage (percentage)",
    "rss": "Physical memory usage(MB)", "vsz": "Virtual memory usage (MB)", "swap": "Swap used(MB)",
    "threads": "Threads (all instances)", "processes": "Running instances",
}

FRAGMENTATION_NAME_FILTER = ["mysqld", "http-nio", "nginx", "redis-server", "beam.smp", "rabbitmq-server", "python3",
                             "systemd", "dockerd", "containerd", "containerd-shim", "docker-proxy", "runc", "conmon",
                             "podman", "crun", "java", "postgres", "netavark"]
FRAGMENTATION_HIGHLIGHT = ["dockerd", "http-nio-8080-e", "podman"]


def plot_scenario(folder: Path) -> None:
    plots_folder = folder.joinpath("plots_img")
    shutil.rmtree(plots_folder, ignore_errors=True)
    plots_folder.joinpath("fragmentation").mkdir(parents=True)
    plots_folder.joinpath("jmeter").mkdir()
    phases = highlight_intervals(folder)  # descanso (verde) e rejuvenescimento (laranja)

    plotted = 0
    for title, compute, ylabel, highlight in PLOTS:
        plotted += plot(compute(folder), folder, title, ylabel, phases if highlight else None)
    for name, df in metrics.processes(folder).items():
        plotted += plot(df, folder, f"Process - {name}", PROCESS_YLABEL)
    for name, df in metrics.lifecycle(folder).items():
        plotted += plot(df, folder, f"Container Metrics - {name.capitalize()}", "Time taken (seconds)")

    plotted += plot_fragmentation(folder, merge_equals=True, adjust_x_limits=False, process_per_plot=21,
                                  top_n_processes=10, name_filter=FRAGMENTATION_NAME_FILTER,
                                  highlight_processes=FRAGMENTATION_HIGHLIGHT)
    jmeter_file = folder.joinpath(f"jmeter_{folder.name}.csv")
    if jmeter_file.exists():
        plotted += plot_jmeter(folder, file_name=jmeter_file.name)
    print(f"{folder.name}: {plotted} groups of plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", nargs="?", default=DEFAULT_LOGS_DIR,
                        help="pasta com uma subpasta por cenario (padrao: %(default)s)")
    logs_dir = Path(parser.parse_args().logs_dir)
    folders = [logs_dir] if logs_dir.joinpath("cpu.csv").exists() else [item for item in logs_dir.iterdir()
                                                                         if item.is_dir()]
    for folder in folders:
        plot_scenario(folder)


if __name__ == "__main__":
    main()
