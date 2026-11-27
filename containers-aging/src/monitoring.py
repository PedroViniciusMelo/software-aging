"""
Threads de coleta: maquina e processos, ciclo de vida dos containers, fragmentacao (systemtap),
energia e o custo do proprio coletor.

Todos os CSVs guardam valores BRUTOS (contadores acumulados do kernel, kB, ns, uma linha por
instancia de processo), lidos direto de /proc e /sys, com o mesmo date_time por amostra. As contas
(%, taxas, somas, unidades) ficam em plot/metrics.py. Contadores acumulados vem com uptime_s, o
relogio para as taxas.

Nova metrica da maquina: um leitor (secao "leituras"), um metodo e uma entrada em
MachineMetrics.METRICS, e a conta correspondente em plot/metrics.py.
"""
import asyncio
import glob
import os
import shutil
import threading
import time
from typing import Optional

from src import activity
from src.config import Config
from src.runtime import ContainerLifecycle, ContainerRuntime, read_os_release
from src.utils import (
    CommandError,
    current_time,
    execute_command,
    execute_command_lines,
    get_time,
    log,
    log_error,
    write_csv_row,
)

# Gerado por install/install.sh: versoes/commits exatos do runtime e do systemtap compilados.
BUILD_MANIFEST = "/etc/containers-aging/build-manifest.txt"
ENVIRONMENT_FILE = "environment.txt"
DEFAULT_DISK_DEVICE = "vda"

LIFECYCLE_STEPS = ["load_image", "start", "stop", "remove_container", "remove_image"]
LIFECYCLE_HEADER = "container;" + ";".join(f"{step}_ns" for step in LIFECYCLE_STEPS) + ";date_time"
# Custo do monitoramento: CPU acumulada de cada parte (threads + comandos que elas executaram),
# separada pelo prefixo do nome da thread (src/activity.py). Por isso os nomes das threads importam.
COST_HEADER = "part;role;threads;threads_cpu_s;commands;commands_cpu_s;uptime_s;date_time"
COLLECTOR_HEADER = "self_cpu_s;commands_cpu_s;rss_kb;vsz_kb;threads;uptime_s;date_time"

# ---- leituras de /proc e /sys (valores brutos) -------------------------------------------------
try:
    CLK_TCK = os.sysconf("SC_CLK_TCK")
    PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
except (AttributeError, ValueError, OSError):  # fora do Linux
    CLK_TCK, PAGE_KB = 100, 4
COMM_LENGTH = 15  # o kernel guarda so' os 15 primeiros caracteres do nome do processo (comm)

# linha "cpu" do /proc/stat, em ticks acumulados desde o boot
CPU_COLUMNS = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"]
# campos do /proc/diskstats (Documentation/admin-guide/iostats.rst); setores de 512 bytes, tempos em ms
DISK_COLUMNS = ["rd_ios", "rd_merges", "rd_sectors", "rd_ms", "wr_ios", "wr_merges", "wr_sectors", "wr_ms",
                "in_flight", "io_ms", "queue_ms", "dc_ios", "dc_merges", "dc_sectors", "dc_ms", "fl_ios", "fl_ms"]
MEMINFO_COLUMNS = ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable", "Shmem",
                   "SwapTotal", "SwapFree", "SwapCached"]  # kB
PROCESS_COLUMNS = ["pid", "start_ticks", "utime_ticks", "stime_ticks", "threads", "vsz_kb", "rss_kb", "swap_kb"]

RAPL_BASE_PATH = os.environ.get("RAPL_BASE_PATH", "/sys/class/powercap")
THERMAL_BASE_PATH = os.environ.get("THERMAL_BASE_PATH", "/sys/class/thermal")
DEFAULT_RAPL_ZONES = ["package", "core", "uncore", "dram"]


def _read_file(path: Optional[str]) -> Optional[str]:
    try:
        with open(path) as file:
            return file.read()
    except (OSError, TypeError):
        return None


def _read_int(path: Optional[str]) -> Optional[int]:
    try:
        return int(_read_file(path).strip())
    except (AttributeError, ValueError):
        return None


def read_uptime() -> float:
    return float(_read_file("/proc/uptime").split()[0])


def read_meminfo() -> list:
    info = {}
    for line in (_read_file("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        if value:
            info[key] = int(value.split()[0])
    return [info.get(key) for key in MEMINFO_COLUMNS]


def read_cpu_ticks() -> list:
    for line in (_read_file("/proc/stat") or "").splitlines():
        if line.startswith("cpu "):
            values = [int(value) for value in line.split()[1:]]
            return (values + [0] * len(CPU_COLUMNS))[:len(CPU_COLUMNS)]
    raise RuntimeError("/proc/stat sem a linha 'cpu'")


def read_diskstats(device: str) -> list:
    for line in (_read_file("/proc/diskstats") or "").splitlines():
        fields = line.split()
        if len(fields) > 3 and fields[2] == device:
            values = [int(value) for value in fields[3:]]
            return (values + [0] * len(DISK_COLUMNS))[:len(DISK_COLUMNS)]  # kernels antigos tem menos campos
    raise RuntimeError(f"disco {device} nao encontrado em /proc/diskstats")


def read_processes() -> list:
    """(pid, comm, campos do /proc/<pid>/stat a partir do estado) de cada processo vivo."""
    result = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            text = _read_file(f"/proc/{entry}/stat")
            if text:
                close = text.rfind(")")
                result.append((int(entry), text[text.find("(") + 1:close], text[close + 2:].split()))
    return result


def read_process_row(pid: int, fields: list) -> list:
    """Valores de PROCESS_COLUMNS de uma instancia (fields = campos do stat a partir do estado)."""
    swap_kb = 0
    for line in (_read_file(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("VmSwap:"):
            swap_kb = int(line.split()[1])
            break
    return [pid, int(fields[19]), int(fields[11]), int(fields[12]), int(fields[17]), int(fields[20]) // 1024,
            int(fields[21]) * PAGE_KB, swap_kb]


def detect_root_disk() -> str:
    """Disco fisico (ex.: vda, sda, nvme0n1) que contem "/", atravessando LVM/particoes."""
    disk = execute_command('lsblk -nrso NAME,TYPE "$(findmnt -nvo SOURCE /)" | awk \'$2 == "disk" {print $1; exit}\'',
                           continue_if_error=True, warn_on_error=False)
    if not disk:
        log.warning("Could not detect the root disk, using %s", DEFAULT_DISK_DEVICE)
    return disk or DEFAULT_DISK_DEVICE


def _zone_matches(zone: str, keyword: str) -> bool:
    """Nome exato ou "keyword-N" ("package-0", "package-1"). Substring nao serve: "core" esta' em "uncore"."""
    zone, keyword = zone.lower(), keyword.lower()
    return zone == keyword or zone.startswith(f"{keyword}-")


def find_rapl_zones(keywords=None) -> dict:
    """{nome da zona: pasta sysfs} das zonas (e subzonas) RAPL que casam com `keywords`."""
    keywords = keywords or DEFAULT_RAPL_ZONES
    zones = {}
    for pattern in (f"{RAPL_BASE_PATH}/intel-rapl:*", f"{RAPL_BASE_PATH}/intel-rapl:*/intel-rapl:*"):
        for path in sorted(glob.glob(pattern)):
            name = (_read_file(f"{path}/name") or "").strip()
            if name and name not in zones and any(_zone_matches(name, keyword) for keyword in keywords):
                zones[name] = path
    return zones


def find_pkg_temp_path() -> Optional[str]:
    """Zona termica do pacote da CPU (tipo "x86_pkg_temp"); o indice thermal_zoneN muda entre maquinas."""
    for type_path in sorted(glob.glob(f"{THERMAL_BASE_PATH}/thermal_zone*/type")):
        if (_read_file(type_path) or "").strip() == "x86_pkg_temp":
            return type_path.replace("/type", "/temp")
    return None


async def _smart_plug_monitor(ip, username, password, interval, on_sample, on_error, should_stop):
    """
    Le' a potencia instantanea da tomada TP-Link Kasa/Tapo a cada `interval` s e chama
    on_sample(power_mw, ligada). Reconecta se a conexao cair. Precisa do pacote "python-kasa".
    """
    from kasa import Discover

    device = None
    while not should_stop():
        try:
            if device is None:
                device = await Discover.discover_single(ip, username=username, password=password)
            await device.update()
            if device.has_emeter:
                emeter = await device.get_emeter_realtime()
                on_sample(emeter.get("power_mw", 0), 1 if device.is_on else 0)
            else:
                on_error("device has no energy meter")
        except Exception as error:
            on_error(str(error))
            device = None  # redescobre na proxima volta
        await asyncio.sleep(interval)


# ---- metricas da maquina -------------------------------------------------------------------------
class MachineMetrics:
    """
    Metricas da maquina, lidas juntas a cada monitoring.sleep_time. Cada metodo devolve as linhas
    de uma amostra (listas de valores brutos); o date_time e' acrescentado no fim de cada linha.
    """
    METRICS = [  # (arquivo, colunas sem o date_time, metodo)
        ("cpu.csv", ["uptime_s", *CPU_COLUMNS], "cpu"),
        ("memory.csv", MEMINFO_COLUMNS, "memory"),
        ("disk.csv", ["block_size", "blocks", "blocks_free", "blocks_avail"], "disk"),
        ("disk_io.csv", ["uptime_s", *DISK_COLUMNS], "disk_io"),
        ("processes.csv", ["process", *PROCESS_COLUMNS, "uptime_s"], "processes"),
        ("zombies.csv", ["zombies"], "zombies"),
        ("kernel.csv", ["kind", "log"], "kernel"),
    ]

    def __init__(self, disk_device: str, process_names: list):
        self.disk_device = disk_device
        self.process_names = {name[:COMM_LENGTH]: name for name in process_names}
        self.metrics = list(self.METRICS)
        self._processes = []  # varredura do /proc da amostra atual (processes e zombies)
        self._kernel_seen = set()
        try:
            read_diskstats(disk_device)
        except RuntimeError as error:
            log.warning("Disk I/O will not be collected: %s", error)
            self.metrics = [metric for metric in self.metrics if metric[2] != "disk_io"]

    def sample(self):
        """(arquivo, cabecalho, linhas) de cada metrica; uma leitura que falha nao derruba as demais."""
        self._processes = read_processes()
        for filename, columns, method in self.metrics:
            try:
                yield filename, ";".join(columns) + ";date_time", getattr(self, method)()
            except (Exception, CommandError) as error:
                log.warning("Reading %s failed: %r", filename, error)

    def cpu(self):
        return [[read_uptime(), *read_cpu_ticks()]]

    def memory(self):
        return [read_meminfo()]

    def disk(self):
        stat = os.statvfs("/")
        return [[stat.f_frsize, stat.f_blocks, stat.f_bfree, stat.f_bavail]]

    def disk_io(self):
        return [[read_uptime(), *read_diskstats(self.disk_device)]]

    def processes(self):
        uptime = read_uptime()
        return [[self.process_names[comm], *read_process_row(pid, fields), uptime]
                for pid, comm, fields in self._processes if comm in self.process_names]

    def zombies(self):
        return [[sum(1 for _, _, fields in self._processes if fields[0] == "Z")]]

    def kernel(self):
        """Mensagens novas do kernel sobre processos mortos e falta de memoria desde a amostra anterior."""
        rows = []
        for line in execute_command_lines("dmesg | grep -i -e 'killed process' -e 'oom' || true"):
            if line not in self._kernel_seen:
                self._kernel_seen.add(line)
                log.warning("Kernel: %s", line)
                rows.append(["killed" if "killed process" in line.lower() else "oom", line])
        return rows


# ---- threads de coleta ---------------------------------------------------------------------------
class MonitoringEnvironment:
    def __init__(self, config: Config, runtime: ContainerRuntime):
        self.path = config.general.scripts_folder
        self.logs_path = config.logs_path
        self.containers = config.containers
        self.monitoring = config.monitoring
        self.energy = config.energy
        self.runtime = runtime
        self.lifecycle = ContainerLifecycle(runtime, config.lifecycle)
        self.machine = None
        self._stop = threading.Event()
        # o rejuvenescimento segura este lock: nao pega um container do monitor pela metade
        self.lifecycle_lock = threading.Lock()
        self.cost_tracker = activity.CostTracker()

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def _write(self, filename: str, header: str, *values) -> None:
        write_csv_row(f"{self.logs_path}/{filename}", header, *values)

    def _wait(self, seconds: float, reason: str = "waiting next sample") -> bool:
        """Espera `seconds` (ou ate' o stop). Devolve True se o monitoramento foi encerrado."""
        activity.waiting(seconds, reason)
        return self._stop.wait(max(seconds, 0))

    def _every(self, seconds: float, collect) -> None:
        """Chama collect() a cada `seconds` (descontado o tempo da coleta) ate' o stop."""
        while self.running:
            started = time.monotonic()
            try:
                collect()
            except (Exception, CommandError) as error:
                log.warning("%s failed: %r", collect.__name__, error)
            self._wait(seconds - (time.monotonic() - started))

    def start(self):
        disk_device = self.monitoring.disk_device or detect_root_disk()
        processes = self.monitoring.processes.for_runtime(self.runtime.name)
        self._write_environment(disk_device, processes)
        self.machine = MachineMetrics(disk_device, processes)

        threads = {  # nome da thread (define a parte no custo, ver src/activity.py) -> (alvo, argumentos...)
            "systemtap": (self.systemtap_loop,),
            "container_lifecycle": (self._every, self.monitoring.sleep_time_container_metrics, self.container_lifecycle),
            "machine": (self._every, self.monitoring.sleep_time, self.write_machine),
            "energy_rapl": (self.rapl_loop,) if self.energy.enable_rapl else None,
            "energy_smart_plug": (self.smart_plug_loop,) if self.energy.enable_smart_plug else None,
            "monitoring_cost": (self._every, self.monitoring.sleep_time, self.write_cost),
        }
        started = [name for name, target in threads.items() if target]
        for name in started:
            target, *args = threads[name]
            threading.Thread(target=target, name=name, args=args, daemon=True).start()
        log.info("Monitoring started: %s (disk %s; processes: %s)", ", ".join(started), disk_device,
                 ", ".join(processes) or "none")

    def _write_environment(self, disk_device: str, processes: list) -> None:
        """environment.txt (chave=valor): sistema, runtime e as constantes que plot/metrics.py usa."""
        os_info = read_os_release()
        info = {
            "Software": self.runtime.name,
            "Software Version": self.runtime.version_text(),
            "OS": os_info.get("ID", "").capitalize(),
            "OS Version": os_info.get("VERSION_ID", ""),
            "Kernel": os.uname().release,
            "CLK_TCK": CLK_TCK,
            "Disk": disk_device,
            "Processes": ",".join(processes),
        }
        with open(f"{self.logs_path}/{ENVIRONMENT_FILE}", "w") as file:
            file.writelines(f"{key}={value}\n" for key, value in info.items())
        log.info("Environment: %s", ", ".join(f"{key}={value}" for key, value in info.items()))
        if os.path.exists(BUILD_MANIFEST):
            shutil.copy(BUILD_MANIFEST, f"{self.logs_path}/build-manifest.txt")

    def stop(self):
        """Encerra a coleta: as threads saem na proxima espera e o systemtap e' interrompido."""
        log.info("Stopping monitoring")
        self._stop.set()
        # SIGINT deixa o stap descarregar o modulo do kernel e fechar o CSV corretamente.
        execute_command("pkill -INT -x stap", continue_if_error=True, warn_on_error=False)
        # nao deixa um container do monitor de ciclo de vida pela metade
        if self.lifecycle_lock.acquire(timeout=self.monitoring.lifecycle_stop_timeout):
            self.lifecycle_lock.release()
        else:
            log.warning("Container lifecycle still running after %ds; stopping anyway",
                        self.monitoring.lifecycle_stop_timeout)
        time.sleep(self.monitoring.stop_grace)

    def write_machine(self) -> None:
        date_time = current_time()
        for filename, header, rows in self.machine.sample():
            for row in rows:
                self._write(filename, header, *row, date_time)

    # ---- fragmentacao de memoria (systemtap) -------------------------------------------------
    def systemtap_loop(self):
        attempts = self.monitoring.stap_max_attempts
        for attempt in range(attempts):
            log.info("Starting systemtap (attempt %d/%d)", attempt + 1, attempts)
            execute_command(
                f"stap -o {self.logs_path}/fragmentation_{attempt}.csv "
                f"-D STP_OVERLOAD_THRESHOLD={self.monitoring.stap_overload_threshold}LL "
                f"-D STP_OVERLOAD_INTERVAL={self.monitoring.stap_overload_interval}LL "
                f"{self.path}/fragmentation.stp",
                continue_if_error=True, warn_on_error=False)
            if not self.running:
                return
            log.warning("systemtap exited; trying again in %ds", self.monitoring.stap_retry_delay)
            if self._wait(self.monitoring.stap_retry_delay):
                return
        log.warning("systemtap failed %d times; memory fragmentation will not be collected", attempts)

    # ---- ciclo de vida dos containers -----------------------------------------------------------
    def container_lifecycle(self):
        """Mede o tempo (ns) de cada etapa do ciclo de vida de cada container configurado."""
        try:
            for container in self.containers:
                if not self.running:
                    return
                # o rejuvenescimento espera o container atual terminar antes de limpar o runtime
                with self.lifecycle_lock:
                    if not self._container_lifecycle(container):
                        return
        except CommandError as error:
            # ex.: falha ao carregar ou remover a imagem; registra e tenta no proximo ciclo
            log.warning("Container lifecycle failed: %s", error.command)
            log_error(self.logs_path, "lifecycle", error.command, current_time())

    def _container_lifecycle(self, container) -> bool:
        """Um container: carrega a imagem, sobe, para e remove. False se ele nao subiu."""
        date_time = current_time()
        name = container.name

        load_image = get_time(self.runtime.load_image(f"{self.path}/{name}.tar"))
        start = self.lifecycle.start(name, name, container.host_port, container.port, container.env)
        if start is None:
            log_error(self.logs_path, "start", name, date_time)
            log.warning("Could not start container %s, skipping this lifecycle round", name)
            execute_command(self.runtime.force_remove(name), continue_if_error=True, warn_on_error=False)
            return False

        stop = self.lifecycle.stop(name, on_kill=lambda: log_error(self.logs_path, "kill", name, date_time))
        remove_container = self.lifecycle.remove(
            name, on_force=lambda: log_error(self.logs_path, "force-remove", name, date_time))
        remove_image = get_time(self.runtime.remove_image(name))

        self._write("lifecycle.csv", LIFECYCLE_HEADER, name, load_image, start, stop, remove_container, remove_image,
                    date_time)
        return True

    # ---- energia -----------------------------------------------------------------------------
    def rapl_loop(self):
        """Contadores RAPL acumulados (energy_uj, voltam a 0 no maximo) e a temperatura do pacote."""
        zones = find_rapl_zones(self.energy.rapl_zones)
        if not zones:
            log.warning("No configured RAPL zone (%s) was found, or it is not readable (run as root). "
                        "CPU energy will not be collected.", self.energy.rapl_zones)
            return
        temp_path = find_pkg_temp_path()

        def read_rapl():
            date_time, uptime = current_time(), read_uptime()
            for name, path in zones.items():
                self._write("energy_rapl.csv", "zone;energy_uj;max_energy_range_uj;uptime_s;date_time", name,
                            _read_int(f"{path}/energy_uj"), _read_int(f"{path}/max_energy_range_uj"), uptime, date_time)
            self._write("cpu_temp.csv", "millicelsius;date_time", _read_int(temp_path), date_time)

        self._every(self.energy.sleep_time_rapl, read_rapl)

    def smart_plug_loop(self):
        def on_sample(power_mw, state):
            self._write("energy_smart_plug.csv", "power_mw;state;uptime_s;date_time", power_mw, state,
                        read_uptime(), current_time())
            activity.waiting(self.energy.sleep_time_smart_plug, "waiting next sample")

        try:
            asyncio.run(_smart_plug_monitor(
                ip=self.energy.smart_plug_ip, username=self.energy.smart_plug_username,
                password=self.energy.smart_plug_password, interval=self.energy.sleep_time_smart_plug,
                on_sample=on_sample, should_stop=self._stop.is_set,
                on_error=lambda message: log_error(self.logs_path, "smart_plug", message, current_time()),
            ))
        except ImportError:
            log.warning("Package 'python-kasa' is not installed; smart plug disabled.")

    # ---- custo do proprio monitoramento ----------------------------------------------------------
    def write_cost(self):
        """
        CPU acumulada de cada parte do sistema (monitoring_cost.csv) e do processo coletor inteiro
        (collector.csv). So' le' /proc, sem executar comandos.
        """
        date_time, uptime = current_time(), read_uptime()
        parts = self.cost_tracker.sample()
        for part, row in parts.items():
            self._write("monitoring_cost.csv", COST_HEADER, part, row["role"], row["threads"], row["threads_cpu_s"],
                        row["children"], row["children_cpu_s"], uptime, date_time)
        process = activity.collector_process()
        self._write("collector.csv", COLLECTOR_HEADER, process.get("self_cpu_s"),
                    sum(row["children_cpu_s"] for row in parts.values()), process.get("VmRSS"),
                    process.get("VmSize"), process.get("Threads"), uptime, date_time)
