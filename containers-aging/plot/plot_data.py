from pathlib import Path
import shutil

from concurrent.futures import ProcessPoolExecutor

from plot_utils import plot, plot_jmeter, plot_fragmentation, plot_time_series

base_dir = "./logs/"

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
        includeColYlabel=True,
        highlight_intervals=intervals
    )

    if base_folder.joinpath('disk.csv').exists():
        ploted += plot(
            title="Disk",
            folder=base_folder,
            filename='disk.csv',
            ylabel='Disk usage (GB)',
            division=1048576
        )

    if base_folder.joinpath('process.csv').exists():
        ploted += plot(
            title="Zumbis",
            folder=base_folder,
            filename='process.csv',
            ylabel='Zumbis processes(qtt)'
        )

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
        highlight_intervals = intervals
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

    if base_folder.joinpath('energy_tomada.csv').exists():
        ploted += plot(
            title="Energy - Tomada",
            folder=base_folder,
            filename='energy_tomada.csv',
            ylabel={
                "power_w": "Power (W)",
                "kwh_interval": "Energy per interval (kWh)",
                "kwh_total": "Cumulative energy (kWh)",
                "state": "Plug state (1=ON, 0=OFF)"
            },
            decimal_separator=".",
            highlight_intervals=intervals
        )

    if base_folder.joinpath(base_folder.name + '.csv').exists():
        ploted += plot(
            title="Server response time",
            folder=base_folder,
            filename=base_folder.name + '.csv',
            ylabel='Response time(s)',
        )

    if base_folder.joinpath('disk_write_read.csv').exists():
        ploted += plot(
            title="Read and Write",
            folder=base_folder,
            filename='disk_write_read.csv',
            ylabel={'tps': 'TPS', "kB_reads": "kB_reads", "kB_wrtns": "kB_wrtns", "kB_dscds": "kB_dscds"}
        )

    if "docker" in base_folder.name:
        for process_title, process_file in [
            ("Docker", "docker.csv"),
            ("Dockerd", "dockerd.csv"),
            ("Containerd", "containerd.csv"),
            ("Containerd-shim", "containerd-shim.csv"),
            ("docker-proxy", "docker-proxy.csv"),
            ("runc", "runc.csv"),
        ]:
            if base_folder.joinpath(process_file).exists():
                ploted += plot(
                    title=f"Process - {process_title}",
                    folder=base_folder,
                    filename=process_file,
                    ylabel=PROCESS_YLABEL,
                    includeColYlabel=True,
                    cols_to_divide=['rss', 'vsz', 'swap'],
                    division=1024
                )
    else:
        for process_title, process_file in [
            ("Podman", "podman.csv"),
            ("Conmon", "conmon.csv"),
            ("Crun", "crun.csv"),
            ("systemd", "systemd.csv"),
        ]:
            if base_folder.joinpath(process_file).exists():
                ploted += plot(
                    title=f"Process - {process_title}",
                    folder=base_folder,
                    filename=process_file,
                    ylabel=PROCESS_YLABEL,
                    includeColYlabel=True,
                    cols_to_divide=['rss', 'vsz', 'swap'],
                    division=1024
                )

    for process_title, process_file in [
        ("Teastore - Java", "java.csv"),
        ("Rabbitmq - beam.smp", "beam.smp.csv"),
        ("Postgres - postgres", "postgres_process.csv"),
        ("Teastore - mysqld", "mysqld.csv"),
    ]:
        if base_folder.joinpath(process_file).exists():
            ploted += plot(
                title=process_title,
                folder=base_folder,
                filename=process_file,
                ylabel=PROCESS_YLABEL,
                includeColYlabel=True,
                cols_to_divide=['rss', 'vsz', 'swap'],
                division=1024
            )


    for process_title, process_file in [
        ("Postgres", "postgres.csv"),
        ("Redis", "redis.csv"),
        ("RabbitMQ", "rabbitmq.csv"),
        ("Nginx", "nginx.csv"),
    ]:
        if base_folder.joinpath(process_file).exists():
            ploted += plot_time_series(
                save_folder=base_folder.joinpath('plots_img'),
                title=f"Container Metrics - {process_title}",
                file_path=base_folder.joinpath(process_file),
                x_label='Time (hours)',
                y_label='Time taken (seconds)',
            )

    ploted += plot_fragmentation(base_folder,
                       merge_equals=True,
                       adjust_x_limits=False,
                       process_per_plot=21,
                       top_n_processes=10,
                       name_filter=["mysqld", "http-nio", "nginx", "redis-server", "beam.smp", "rabbitmq-server", "python3", "systemd", "dockerd", "containerd", "containerd-shim", "docker-proxy", "runc", "conmon", "podman", "crun", "java", "postgres", "netavark"],
                       highlight_processes=["dockerd", "http-nio-8080-e", "podman"]
                       )  # name_filter=["dockerd", "containerd", "containerd-shim", "docker-proxy", "runc"])

    jmeter_file = base_folder.joinpath("jmeter_" + base_folder.name + ".csv")

    if jmeter_file.exists():
        ploted += plot_jmeter(base_folder, file_name=jmeter_file.name, ignore_chunck=False)

    print(f"Ploted {ploted}/{qtd_item - 4} ")


if __name__ == "__main__":
    folders = [item for item in Path(base_dir).iterdir() if item.is_dir()]

    for folder in folders:
        start(base_folder=folder, qtd_item=len([file for file in folder.iterdir() if file.suffix == '.csv']))