"""
Golden master dos scripts de analise (plot/).

Gera logs sinteticos no formato do coletor para dois cenarios, roda cada script de
plot/ num diretorio temporario e grava: o stdout de cada script e o sha256 de todo
arquivo gerado (svg/png/csv/json). Saidas do matplotlib ficam deterministicas com
backend Agg, SOURCE_DATE_EPOCH e svg.hashsalt fixos.

Precisa de pandas/matplotlib/scipy/scikit-learn/pymannkendall no Python usado.

Uso: <python-com-deps> tests/plot_golden/harness.py <saida.json>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLOT = REPO / "plot"
SCENARIOS = ["debian12_docker_novo", "ubuntu22_podman"]
START = datetime(2024, 1, 1, 0, 0, 0)

# (comando, diretorio de trabalho relativo) -- os scripts antigos tinham caminhos fixos:
# plot_data.py/theilsen.py leem ./logs e mannkendall.py/temp.py leem "D:/final".
COMMANDS = [
    ("plot_data", ["plot_data.py"]),
    ("theilsen", ["theilsen.py"]),
    ("mannkendall", ["mannkendall.py"]),
    ("mannkendall_summary", ["temp.py"]),
]


def comma(value):
    return f"{value:.2f}".replace(".", ",")


def write(path, header, rows):
    path.write_text(header + "\n" + "".join(";".join(map(str, row)) + "\n" for row in rows))


def make_scenario(folder, seed, software):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    hours = 60
    times = [(START + timedelta(hours=h, minutes=int(rng.integers(0, 3)))).strftime("%Y-%m-%d %H:%M:%S")
             for h in range(hours)]

    write(folder / "cpu.csv", "usr;nice;sys;iowait;soft;date_time",
          [(comma(5 + h * 0.05 + rng.random()), comma(rng.random()), comma(2 + rng.random()),
            comma(rng.random()), comma(rng.random() / 2), t) for h, t in enumerate(times)])
    write(folder / "memory.csv", "used;cached;buffers;swap;date_time",
          [(800000 + h * 900 + int(rng.integers(0, 500)), 300000 + h * 10, 50000, 8000 - h, t)
           for h, t in enumerate(times)])
    write(folder / "disk.csv", "used;date_time", [(12345678 + h * 1000, t) for h, t in enumerate(times)])
    write(folder / "process.csv", "zombies;date_time", [(int(rng.integers(0, 3)), t) for t in times])
    write(folder / "disk_write_read.csv", "tps;kB_reads;kB_wrtns;kB_dscds;date_time",
          [(comma(1 + rng.random()), comma(10 + h), comma(20 + h), comma(0), t) for h, t in enumerate(times)])
    write(folder / "energy_rapl.csv", "package_j;core_j;uncore_j;dram_j;total_j;cpu_temp_c;date_time",
          [(round(25 + rng.random(), 4), round(12 + rng.random(), 4), round(3 + rng.random(), 4),
            round(4 + rng.random(), 4), round(29 + rng.random(), 4), round(45 + rng.random(), 1), t) for t in times])
    write(folder / "energy_tomada.csv", "power_w;kwh_interval;kwh_total;state;date_time",
          [(round(40 + rng.random(), 3), 0.003333, round(0.003333 * (h + 1), 6), 1, t)
           for h, t in enumerate(times)])
    write(folder / f"{folder.name}.csv", "response_time;date_time",
          [(comma(0.2 + rng.random() / 10), t) for t in times])

    processes = ["docker", "dockerd", "containerd", "containerd-shim", "runc"] if software == "docker" \
        else ["podman", "conmon", "crun", "systemd"]
    for name in processes + ["java", "beam.smp", "mysqld"]:
        write(folder / f"{name}.csv", "cpu;mem;rss;vsz;threads;swap;date_time",
              [(comma(rng.random() * 3), comma(rng.random()), 100000 + h * 50, 2000000 + h * 10, 12, 4, t)
               for h, t in enumerate(times)])
    write(folder / "postgres_process.csv", "cpu;mem;rss;vsz;threads;swap;date_time",
          [(comma(rng.random()), comma(rng.random()), 50000, 900000, 8, 0, t) for t in times])

    for name in ["postgres", "redis", "rabbitmq", "nginx"]:
        write(folder / f"{name}.csv", "load_image;start;stop;remove_container;remove_image;date_time",
              [tuple(int(v) for v in rng.integers(10**8, 10**9, 5)) + (t,) for t in times[::4]])

    frag_names = [("dockerd", 700), ("java", 900), ("http-nio-8080-e", 950), ("postgres", 600),
                  ("containerd", 650), ("podman", 720), ("python3", 30), ("kworker", 400)]
    for index in range(2):
        rows = []
        for block in range(5):
            stamp = START + timedelta(hours=index * 10 + block * 2)
            ctime = stamp.strftime("%a %b ") + f"{stamp.day:2d}" + stamp.strftime(" %H:%M:%S %Y")
            for pid, (name, base) in enumerate(frag_names):
                rows.append((f"{name}({1000 + pid})", f"systemd(1)", 0, base // 10 + int(rng.integers(0, 20)),
                             block * 3, ctime))
            if block == 2:
                rows.append(("EMPTIED",))
        write(folder / f"fragmentation_{index}.csv", "process;parent;UID;process_occurrences;fragmented;datetime",
              rows)

    jmeter_rows = ["timeStamp,elapsed,label,responseCode,success"]
    stamp = int(START.timestamp() * 1000)
    for i in range(400):
        label = ["home", "login", "category"][i % 3]
        jmeter_rows.append(f"{stamp + i * 9000},{int(rng.integers(20, 400))},{label},200,"
                           f"{'false' if i % 37 == 0 else 'true'}")
    (folder / f"jmeter_{folder.name}.csv").write_text("\n".join(jmeter_rows) + "\n")


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[1])
    result = {"stdout": {}, "returncode": {}, "files": {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for index, name in enumerate(SCENARIOS):
            make_scenario(tmp / "logs" / name, seed=index, software="docker" if "docker" in name else "podman")
        shutil.copytree(tmp / "logs", tmp / "D:" / "final")

        mpl_config = tmp / "mplconfig"
        mpl_config.mkdir()
        (mpl_config / "matplotlibrc").write_text("svg.hashsalt: golden\n")
        env = dict(os.environ, MPLBACKEND="Agg", SOURCE_DATE_EPOCH="0", MPLCONFIGDIR=str(mpl_config),
                   PYTHONHASHSEED="0")

        before = {p for p in tmp.rglob("*") if p.is_file()}
        for label, args in COMMANDS:
            proc = subprocess.run([sys.executable, str(PLOT / args[0])] + args[1:], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=600)
            result["stdout"][label] = proc.stdout.replace(str(tmp), "<TMP>").splitlines()
            result["returncode"][label] = proc.returncode
            if proc.returncode:
                result["stdout"][label] += ["STDERR:"] + proc.stderr.replace(str(tmp), "<TMP>").splitlines()[-15:]

        for path in sorted(p for p in tmp.rglob("*") if p.is_file() and p not in before):
            rel = str(path.relative_to(tmp))
            if rel.startswith("mplconfig"):
                continue
            result["files"][rel] = file_digest(path)

    output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
