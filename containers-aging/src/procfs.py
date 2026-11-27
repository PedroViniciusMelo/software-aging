"""
Leituras diretas de /proc, sem executar comandos: processos por nome, CPU, memoria e disco.

Substitui pgrep/pidstat/mpstat/iostat/free. Duas diferencas de proposito em relacao a eles:

- CPU e disco sao a media do intervalo inteiro entre duas amostras (diferenca dos contadores
  acumulados do kernel), e nao de 1 s amostrado. O "mpstat 1 1" media so' 1 s a cada
  sleep_time, e esse segundo caia junto com a rajada de comandos do proprio coletor.
- Processos sao encontrados pelo nome exato (/proc/<pid>/comm, como "pgrep -x"), e todas as
  instancias sao somadas. "pgrep -f" casava com qualquer parte da linha de comando e pegava so'
  o menor PID: "java" achava o "/bin/sh -c java ..." em vez das JVMs, "docker" achava o dockerd.
"""
import os
import threading
import time
from typing import Optional

from src.activity import CLK_TCK

try:
    PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
except (AttributeError, ValueError, OSError):  # fora do Linux
    PAGE_KB = 4
COMM_LENGTH = 15  # o kernel guarda so' os 15 primeiros caracteres do nome (comm)


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as file:
            return file.read()
    except OSError:
        return None


def uptime_seconds() -> float:
    return float(_read("/proc/uptime").split()[0])


def meminfo() -> dict:
    """Campos do /proc/meminfo, em kB."""
    info = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        if value:
            info[key] = int(value.split()[0])
    return info


# ---- processos --------------------------------------------------------------------------------
PIDS_CACHE_SECONDS = 2  # as threads de processo acordam quase juntas: uma varredura do /proc serve a todas
_pids_lock = threading.Lock()
_pids_cache = (float("-inf"), {})  # (instante monotonic, {comm: [pids]})


def pids_by_name(name: str) -> list:
    """PIDs cujo nome (comm) e' exatamente `name`, como "pgrep -x"."""
    global _pids_cache
    with _pids_lock:
        scanned_at, by_name = _pids_cache
        if time.monotonic() - scanned_at > PIDS_CACHE_SECONDS:
            by_name = {}
            for entry in os.listdir("/proc"):
                if entry.isdigit():
                    comm = (_read(f"/proc/{entry}/comm") or "").rstrip("\n")
                    by_name.setdefault(comm, []).append(int(entry))
            _pids_cache = (time.monotonic(), by_name)
    return list(by_name.get(name[:COMM_LENGTH], []))


def process_stat(pid: int) -> Optional[dict]:
    text = _read(f"/proc/{pid}/stat")
    if text is None:
        return None
    fields = text[text.rfind(")") + 2:].split()  # a partir do estado (campo 3)
    swap_kb = 0
    for line in (_read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("VmSwap:"):
            swap_kb = int(line.split()[1])
            break
    return {
        "ticks": int(fields[11]) + int(fields[12]),  # utime + stime
        "threads": int(fields[17]),
        "start": int(fields[19]),  # instante em que o processo comecou, em ticks desde o boot
        "vsz_kb": int(fields[20]) // 1024,
        "rss_kb": int(fields[21]) * PAGE_KB,
        "swap_kb": swap_kb,
    }


class ProcessSampler:
    """
    Soma de todas as instancias de um processo. A CPU (% de um nucleo, como no pidstat) e' a do
    intervalo desde a amostra anterior: para cada instancia, o que ela gastou nesse intervalo
    (ou desde que nasceu, se nasceu depois). Instancias que morreram no meio do intervalo nao
    entram (a CPU delas ja' nao pode ser lida).
    """

    def __init__(self, name: str):
        self.name = name
        self._previous = {}  # (pid, start) -> ticks
        self._previous_uptime = None
        self.last_raw = None  # (uptime, [(pid, stat)]) da ultima amostra, para logs/raw/

    def sample(self) -> Optional[dict]:
        """None se nenhuma instancia existe; cpu=None se ainda nao ha' amostra anterior."""
        stats = [(pid, stat) for pid in pids_by_name(self.name) for stat in [process_stat(pid)] if stat]
        if not stats:
            return None
        uptime = uptime_seconds()
        self.last_raw = (uptime, stats)

        cpu = None
        if self._previous_uptime is not None and uptime > self._previous_uptime:
            used_ticks = 0
            for pid, stat in stats:
                previous = self._previous.get((pid, stat["start"]))
                if previous is not None:
                    used_ticks += stat["ticks"] - previous
                elif stat["start"] / CLK_TCK >= self._previous_uptime:  # nasceu dentro do intervalo
                    used_ticks += stat["ticks"]
            cpu = used_ticks / CLK_TCK / (uptime - self._previous_uptime) * 100

        self._previous = {(pid, stat["start"]): stat["ticks"] for pid, stat in stats}
        self._previous_uptime = uptime
        mem_total = meminfo().get("MemTotal") or 1
        rss = sum(stat["rss_kb"] for _, stat in stats)
        return {
            "cpu": cpu,
            "mem": rss / mem_total * 100,
            "rss": rss,
            "vsz": sum(stat["vsz_kb"] for _, stat in stats),
            "threads": sum(stat["threads"] for _, stat in stats),
            "swap": sum(stat["swap_kb"] for _, stat in stats),
            "processes": len(stats),
        }


# ---- maquina ----------------------------------------------------------------------------------
class CpuSampler:
    """
    % de CPU da maquina na media do intervalo desde a amostra anterior, como o mpstat calcula
    (usr e nice sem o tempo de guest, que o kernel ja' soma neles).
    """
    COLUMNS = ["usr", "nice", "sys", "iowait", "soft", "irq", "steal", "idle"]
    RAW_COLUMNS = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"]

    def __init__(self):
        self._previous = self._read()
        self.last_raw = self._previous  # contadores acumulados (ticks) da linha "cpu" do /proc/stat

    @staticmethod
    def _read() -> list:
        for line in (_read("/proc/stat") or "").splitlines():
            if line.startswith("cpu "):
                values = [int(value) for value in line.split()[1:]]
                return values + [0] * (10 - len(values))
        raise RuntimeError("/proc/stat sem a linha 'cpu'")

    def sample(self) -> dict:
        current = self._read()
        delta = [now - before for now, before in zip(current, self._previous)]
        self._previous = self.last_raw = current
        user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice = delta
        total = user + nice + system + idle + iowait + irq + softirq + steal  # guest ja' esta' em user/nice
        if total <= 0:
            return {column: 0.0 for column in self.COLUMNS}
        pct = lambda value: max(value, 0) / total * 100
        return {"usr": pct(user - guest), "nice": pct(nice - guest_nice), "sys": pct(system),
                "iowait": pct(iowait), "soft": pct(softirq), "irq": pct(irq), "steal": pct(steal),
                "idle": pct(idle)}


class DiskSampler:
    """Taxas de um disco (/proc/diskstats) na media do intervalo, nas unidades do iostat -d."""
    COLUMNS = ["tps", "kB_reads", "kB_wrtns", "kB_dscds"]  # por segundo, como "iostat -d"
    # campos do /proc/diskstats (Documentation/admin-guide/iostats.rst); setores de 512 bytes, tempos em ms
    RAW_COLUMNS = ["rd_ios", "rd_merges", "rd_sectors", "rd_ms", "wr_ios", "wr_merges", "wr_sectors", "wr_ms",
                   "in_flight", "io_ms", "queue_ms", "dc_ios", "dc_merges", "dc_sectors", "dc_ms", "fl_ios", "fl_ms"]

    def __init__(self, device: str):
        self.device = device
        self._previous = self._read()
        self._previous_time = time.monotonic()

    def _read(self) -> list:
        """ios e setores de leitura, escrita e discard; guarda a linha inteira em last_raw."""
        for line in (_read("/proc/diskstats") or "").splitlines():
            fields = line.split()
            if len(fields) > 3 and fields[2] == self.device:
                values = [int(value) for value in fields[3:]]
                values += [0] * (len(self.RAW_COLUMNS) - len(values))  # kernels antigos tem menos campos
                self.last_raw = values[:len(self.RAW_COLUMNS)]
                return [values[0], values[2], values[4], values[6], values[11], values[13]]
        raise RuntimeError(f"disco {self.device} nao encontrado em /proc/diskstats")

    def sample(self) -> dict:
        current, now = self._read(), time.monotonic()
        elapsed = max(now - self._previous_time, 1e-9)
        reads, read_sectors, writes, write_sectors, discards, discard_sectors = (
            now_value - before for now_value, before in zip(current, self._previous))
        self._previous, self._previous_time = current, now
        per_second = lambda value: value / elapsed
        return {"tps": per_second(reads + writes + discards), "kB_reads": per_second(read_sectors / 2),
                "kB_wrtns": per_second(write_sectors / 2), "kB_dscds": per_second(discard_sectors / 2)}


MEMINFO_RAW = ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable", "Shmem",
               "SwapTotal", "SwapFree", "SwapCached"]


def memory() -> dict:
    """
    Memoria da maquina em kB. used = MemTotal - MemAvailable, a mesma definicao do "free" do
    procps 4 (Debian 12+): a formula fixa evita diferenca entre distros com procps 3 e 4.
    swap continua sendo o swap LIVRE (compatibilidade); swap_used e swap_total sao os novos.
    """
    info = meminfo()
    swap_total, swap_free = info.get("SwapTotal", 0), info.get("SwapFree", 0)
    return {"used": info.get("MemTotal", 0) - info.get("MemAvailable", 0), "cached": info.get("Cached", 0),
            "buffers": info.get("Buffers", 0), "swap": swap_free, "swap_total": swap_total,
            "swap_used": swap_total - swap_free, "raw": [info.get(key, "") for key in MEMINFO_RAW]}
