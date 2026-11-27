"""
Contas sobre os CSVs brutos do coletor (src/monitoring.py). Cada funcao le' um arquivo de uma
pasta de resultados e devolve DataFrame(s) indexados pelo date_time, ja' nas unidades dos graficos
(ou None se o arquivo nao existe). Contadores acumulados viram a media do intervalo entre duas
amostras, usando a coluna uptime_s como relogio; a primeira amostra so' serve de referencia.

Nova metrica: uma funcao aqui e uma entrada em PLOTS (plot_data.py).
"""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

MB = 1024  # os arquivos brutos estao em kB


def read(folder, filename: str) -> Optional[pd.DataFrame]:
    path = Path(folder) / filename
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, sep=";", parse_dates=["date_time"])


def environment(folder) -> dict:
    """environment.txt (chave=valor) gravado pelo coletor."""
    path = Path(folder) / "environment.txt"
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def _by_time(frame: pd.DataFrame, date_time: pd.Series) -> pd.DataFrame:
    frame.index = pd.DatetimeIndex(date_time, name="date_time")
    return frame


def _per_second(df: pd.DataFrame) -> tuple:
    """(diferencas entre amostras consecutivas, segundos entre elas), sem a primeira amostra."""
    delta = df.drop(columns="date_time").diff().iloc[1:]
    return delta, delta.pop("uptime_s")


def cpu(folder) -> Optional[pd.DataFrame]:
    """% de CPU da maquina em cada intervalo, como o mpstat (usr e nice sem o tempo de guest)."""
    df = read(folder, "cpu.csv")
    if df is None:
        return None
    d, _ = _per_second(df)
    total = d[["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]].sum(axis=1)
    pct = lambda ticks: (ticks.clip(lower=0) / total * 100).where(total > 0, 0.0)
    out = pd.DataFrame({
        "usr": pct(d["user"] - d["guest"]), "nice": pct(d["nice"] - d["guest_nice"]), "sys": pct(d["system"]),
        "iowait": pct(d["iowait"]), "soft": pct(d["softirq"]), "irq": pct(d["irq"]), "steal": pct(d["steal"]),
        "idle": pct(d["idle"]),
    })
    return _by_time(out, df["date_time"].iloc[1:])


def memory(folder) -> Optional[pd.DataFrame]:
    """MB. used = MemTotal - MemAvailable (a definicao do "free" do procps 4, igual em todas as distros)."""
    df = read(folder, "memory.csv")
    if df is None:
        return None
    out = pd.DataFrame({
        "used": df["MemTotal"] - df["MemAvailable"], "cached": df["Cached"], "buffers": df["Buffers"],
        "swap_used": df["SwapTotal"] - df["SwapFree"],
    }) / MB
    return _by_time(out, df["date_time"])


def disk(folder) -> Optional[pd.DataFrame]:
    """GB usados em "/", como a coluna Used do df."""
    df = read(folder, "disk.csv")
    if df is None:
        return None
    used = (df["blocks"] - df["blocks_free"]) * df["block_size"] / 1024 ** 3
    return _by_time(pd.DataFrame({"used": used}), df["date_time"])


def disk_io(folder) -> Optional[pd.DataFrame]:
    """Taxas por segundo do disco (como o iostat -d); setores de 512 bytes."""
    df = read(folder, "disk_io.csv")
    if df is None:
        return None
    d, seconds = _per_second(df)
    out = pd.DataFrame({
        "tps": (d["rd_ios"] + d["wr_ios"] + d["dc_ios"]) / seconds, "kB_read_s": d["rd_sectors"] / 2 / seconds,
        "kB_wrtn_s": d["wr_sectors"] / 2 / seconds, "kB_dscd_s": d["dc_sectors"] / 2 / seconds,
    })
    return _by_time(out, df["date_time"].iloc[1:])


def zombies(folder) -> Optional[pd.DataFrame]:
    df = read(folder, "zombies.csv")
    return None if df is None else _by_time(df[["zombies"]].copy(), df["date_time"])


def processes(folder) -> dict:
    """
    {processo: DataFrame} com a soma de todas as instancias em cada amostra. cpu e' % de um nucleo
    (como o pidstat) no intervalo desde a amostra anterior: para cada instancia, o que ela gastou
    no intervalo (ou desde que nasceu, se nasceu dentro dele). Instancias que morreram no meio
    do intervalo nao entram (a CPU delas ja' nao pode ser lida).
    """
    df = read(folder, "processes.csv")
    if df is None:
        return {}
    clk_tck = int(environment(folder).get("CLK_TCK", 100))
    mem = read(folder, "memory.csv")
    mem_total_kb = mem["MemTotal"].iloc[0] if mem is not None else float("nan")
    df["ticks"] = df["utime_ticks"] + df["stime_ticks"]

    result = {}
    for name, rows in df.groupby("process", sort=False):
        records, previous, previous_uptime = [], {}, None
        for uptime, sample in rows.groupby("uptime_s", sort=True):
            current = {(pid, start): ticks for pid, start, ticks in
                       zip(sample["pid"], sample["start_ticks"], sample["ticks"])}
            cpu_pct = None
            if previous_uptime is not None:
                used = sum(ticks - previous[key] if key in previous else ticks if key[1] / clk_tck >= previous_uptime
                           else 0 for key, ticks in current.items())
                cpu_pct = used / clk_tck / (uptime - previous_uptime) * 100
            previous, previous_uptime = current, uptime
            records.append({
                "date_time": sample["date_time"].iloc[0], "cpu": cpu_pct,
                "mem": sample["rss_kb"].sum() / mem_total_kb * 100, "rss": sample["rss_kb"].sum() / MB,
                "vsz": sample["vsz_kb"].sum() / MB, "swap": sample["swap_kb"].sum() / MB,
                "threads": sample["threads"].sum(), "processes": len(sample),
            })
        out = pd.DataFrame(records).dropna(subset=["cpu"])
        if not out.empty:
            result[name] = out.set_index("date_time")
    return result


def lifecycle(folder) -> dict:
    """{container: DataFrame} com o tempo (s) de cada etapa do ciclo de vida."""
    df = read(folder, "lifecycle.csv")
    if df is None:
        return {}
    result = {}
    for name, rows in df.groupby("container", sort=False):
        steps = rows.drop(columns=["container", "date_time"]) / 1e9
        steps.columns = [column.removesuffix("_ns") for column in steps.columns]
        result[name] = _by_time(steps, rows["date_time"])
    return result


def energy_rapl(folder) -> Optional[pd.DataFrame]:
    """
    Joules de cada dominio RAPL ("package" soma package-0, package-1...) em cada intervalo, corrigindo
    a volta do contador; total_j = package + dram; cpu_temp_c da temperatura do pacote.
    """
    df = read(folder, "energy_rapl.csv")
    if df is None:
        return None
    df = df.sort_values(["zone", "uptime_s"])
    delta = df.groupby("zone")["energy_uj"].diff()
    df["joules"] = delta.where(delta >= 0, delta + df["max_energy_range_uj"]) / 1e6
    df["domain"] = df["zone"].map(lambda zone: re.sub(r"-\d+$", "", zone))
    out = df.dropna(subset=["joules"]).pivot_table(index="date_time", columns="domain", values="joules", aggfunc="sum")
    out.columns = [f"{domain}_j" for domain in out.columns]
    if {"package_j", "dram_j"} & set(out.columns):
        out["total_j"] = out.get("package_j", 0) + out.get("dram_j", 0)
    temp = read(folder, "cpu_temp.csv")
    if temp is not None:
        out = out.join(temp.groupby("date_time")["millicelsius"].mean().rename("cpu_temp_c") / 1000)
    return out


def energy_smart_plug(folder) -> Optional[pd.DataFrame]:
    """Potencia (W), energia em cada intervalo e acumulada (kWh) e estado da tomada."""
    df = read(folder, "energy_smart_plug.csv")
    if df is None:
        return None
    power_w = df["power_mw"] / 1000
    kwh = power_w * df["uptime_s"].diff().fillna(0) / 3_600_000
    out = pd.DataFrame({"power_w": power_w, "kwh_interval": kwh, "kwh_total": kwh.cumsum(), "state": df["state"]})
    return _by_time(out, df["date_time"])


def response_time(folder) -> Optional[pd.DataFrame]:
    """<cenario>/<cenario>.csv de start_server_response.sh (fora do coletor)."""
    df = read(folder, f"{Path(folder).name}.csv")
    if df is None:
        return None
    values = df.drop(columns="date_time").replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce")
    return _by_time(values, df["date_time"])
