"""Threads de coleta: recursos da maquina, processos, ciclo de vida, fragmentacao, energia e custo."""
import asyncio
import os
import shutil
import threading
import time

from src import activity
from src.config import Config
from src.energy import RaplReader, find_pkg_temp_path, run_smart_plug_monitor, sample_power
from src.lifecycle import ContainerLifecycle
from src.runtime import ContainerRuntime, detect_root_disk
from src.utils import (
    CommandError,
    current_time,
    execute_command,
    execute_command_lines,
    get_time,
    log,
    log_error,
    write_csv_row,
    write_to_file,
)

# Custo do monitoramento: CPU de cada parte do sistema (threads + comandos que elas executam),
# separada pelo prefixo do nome da thread (src/activity.py). Por isso os nomes abaixo importam.
COST_FILE = "monitoring_cost.csv"
COST_HEADER = "part;role;threads;threads_cpu_s;commands;commands_cpu_s;cpu_s;cpu_pct;date_time"
COLLECTOR_FILE = "collector.csv"
COLLECTOR_HEADER = "cpu_s;commands_cpu_s;cpu_pct;monitoring_cpu_pct;rss_kb;vsz_kb;threads;date_time"

PROCESS_HEADER = 'cpu;mem;rss;vsz;threads;swap;date_time'
# postgres.csv ja e' usado pelo monitor de ciclo de vida do container postgres.
PROCESS_FILE_NAMES = {'postgres': 'postgres_process'}

# Gerado por install/install.sh: versoes/commits exatos do runtime e do systemtap compilados.
BUILD_MANIFEST = "/etc/containers-aging/build-manifest.txt"

# Intervalos, tentativas e limites da coleta: secao "monitoring" do config.yaml (MonitoringConfig).


def _fmt(value, ndigits=4):
    """Formata uma leitura numerica opcional para o CSV (vazio quando indisponivel)."""
    return "" if value is None else round(value, ndigits)


def _pgrep_pattern(process_name: str) -> str:
    """
    Padrao do "pgrep -f" que nao casa com o proprio shell que o executa: "[d]ocker" casa com
    "docker", mas nao com a linha de comando "pgrep -f [d]ocker".
    """
    return f"[{process_name[0]}]{process_name[1:]}"


class MonitoringEnvironment:
    def __init__(self, config: Config, runtime: ContainerRuntime, environment_description: str):
        self.path = config.general.scripts_folder
        self.logs_path = config.logs_path
        self.containers = config.containers
        self.monitoring = config.monitoring
        self.energy = config.energy
        self.runtime = runtime
        self.environment_description = environment_description
        self.lifecycle = ContainerLifecycle(runtime, config.lifecycle)
        self.disk_device = None
        self._seen_kernel_logs = {"killed.csv": set(), "omm.csv": set()}
        self._stop = threading.Event()
        # o rejuvenescimento segura este lock: nao pega um container do monitor pela metade
        self.lifecycle_lock = threading.Lock()
        self.cost_tracker = activity.CostTracker()

    def _log(self, filename: str, header: str, *values) -> None:
        write_csv_row(f"{self.logs_path}/{filename}", header, *values)

    def _wait(self, seconds: float) -> bool:
        """Espera `seconds` (ou ate o stop). Devolve True se o monitoramento foi encerrado."""
        activity.waiting(seconds, "waiting next sample")
        return self._stop.wait(max(seconds, 0))

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    @staticmethod
    def _start_thread(name: str, target, *args) -> None:
        threading.Thread(target=target, name=name, args=args, daemon=True).start()

    def start(self):
        log.info("Environment:%s", self.environment_description.rstrip())

        write_to_file(f"{self.logs_path}/environment.txt", "Environment", self.environment_description)
        if os.path.exists(BUILD_MANIFEST):
            shutil.copy(BUILD_MANIFEST, f"{self.logs_path}/build-manifest.txt")
        self.disk_device = self.monitoring.disk_device or detect_root_disk()

        threads = ["systemtap", "container_metrics"]
        self._start_thread("systemtap", self.systemtap_loop)
        self._start_thread("container_metrics", self.container_metrics_loop)
        processes = self.monitoring.processes.for_runtime(self.runtime.name)
        for process in processes:
            self._start_thread(f"processes_{process}", self.process_loop, process)
        threads.append(f"{len(processes)} process monitors ({', '.join(processes) or 'none'})")
        self._start_thread("machine_resources", self.machine_resources_loop)
        threads.append("machine_resources")
        if self.energy.enable_rapl:
            self._start_thread("energy_rapl", self.rapl_loop)
            threads.append("energy_rapl")
        if self.energy.enable_smart_plug:
            self._start_thread("energy_smart_plug", self.smart_plug_loop)
            threads.append("energy_smart_plug")
        self._start_thread("monitoring_cost", self.monitoring_cost_loop)
        threads.append("monitoring_cost")
        log.info("Monitoring started: %s (disk %s)", ", ".join(threads), self.disk_device)

    def stop(self):
        """Encerra a coleta: as threads saem na proxima espera e o systemtap e' interrompido."""
        log.info("Stopping monitoring")
        self._stop.set()
        # SIGINT deixa o stap descarregar o modulo do kernel e fechar o CSV corretamente.
        execute_command("pkill -INT -x stap", continue_if_error=True, error_informative=False)
        # nao deixa um container do monitor de ciclo de vida pela metade
        if self.lifecycle_lock.acquire(timeout=self.monitoring.lifecycle_stop_timeout):
            self.lifecycle_lock.release()
        else:
            log.warning("Container lifecycle still running after %ds; stopping anyway",
                        self.monitoring.lifecycle_stop_timeout)
        time.sleep(self.monitoring.stop_grace)

    # ---- fragmentacao de memoria (systemtap) -----------------------------------------
    def systemtap_loop(self):
        attempts = self.monitoring.stap_max_attempts
        for attempt in range(attempts):
            log.info("Starting systemtap (attempt %d/%d)", attempt + 1, attempts)
            execute_command(
                f"stap -o {self.logs_path}/fragmentation_{attempt}.csv "
                f"-D STP_OVERLOAD_THRESHOLD={self.monitoring.stap_overload_threshold}LL "
                f"-D STP_OVERLOAD_INTERVAL={self.monitoring.stap_overload_interval}LL "
                f"{self.path}/fragmentation.stp",
                continue_if_error=True, error_informative=False)
            if not self.running:
                return
            log.warning("systemtap exited; trying again in %ds", self.monitoring.stap_retry_delay)
            if self._wait(self.monitoring.stap_retry_delay):
                return
        log.warning("systemtap failed %d times; memory fragmentation will not be collected", attempts)

    # ---- processos -------------------------------------------------------------------
    def process_loop(self, process: str):
        while self.running:
            self.get_process_data(process)
            self._wait(self.monitoring.sleep_time - self.monitoring.pidstat_interval)

    def get_process_data(self, process_name: str):
        """
        Coleta cpu/mem/rss/vsz/threads/swap do processo. Se ele ainda nao existe, tenta de novo
        a cada process_retry_delay segundos ate encontra-lo.
        """
        date_time = current_time()

        while self.running:
            try:
                pid = execute_command(f"pgrep -f '{_pgrep_pattern(process_name)}' | head -n 1",
                                      error_informative=False)

                data = execute_command(f"pidstat -u -h -p {pid} -T ALL -r {self.monitoring.pidstat_interval} 1 "
                                       "| sed -n '4p'",
                                       error_informative=False).split()
                threads = execute_command(f"cat /proc/{pid}/status | grep Threads | awk '{{print $2}}'",
                                          continue_if_error=True)
                swap = execute_command(f"cat /proc/{pid}/status | grep Swap | awk '{{print $2}}'",
                                       continue_if_error=True)
                cpu, vsz, rss, mem = data[7], data[11], data[12], data[13]

                filename = PROCESS_FILE_NAMES.get(process_name, process_name)
                self._log(f"{filename}.csv", PROCESS_HEADER, cpu, mem, rss, vsz, threads, swap, date_time)
                return
            except (CommandError, Exception):
                self._wait(self.monitoring.process_retry_delay)

    # ---- ciclo de vida dos containers ---------------------------------------------------
    def container_metrics_loop(self):
        while self.running:
            started = time.time()
            try:
                self.container_lifecycle()
                log.debug("Container lifecycle metrics collected in %.1fs", time.time() - started)
            except CommandError as error:
                # ex.: falha ao carregar ou remover a imagem; registra e tenta no proximo ciclo
                log.warning("Container lifecycle failed: %s", error.command)
                log_error(self.logs_path, "lifecycle", error.command, current_time())
            self._wait(self.monitoring.sleep_time_container_metrics - (time.time() - started))

    def container_lifecycle(self):
        """Mede o tempo de cada etapa do ciclo de vida de cada container configurado."""
        for container in self.containers:
            if not self.running:
                return
            # o rejuvenescimento espera o container atual terminar antes de limpar o runtime
            with self.lifecycle_lock:
                if not self._container_lifecycle(container):
                    return

    def _container_lifecycle(self, container) -> bool:
        """Um container: carrega a imagem, sobe, para e remove. False se ele nao subiu."""
        date_time = current_time()
        name = container.name

        load_image_time = get_time(self.runtime.load_image(f"{self.path}/{name}.tar"))

        start_time = self.lifecycle.start(name, name, container.host_port, container.port, container.env)
        if start_time is None:
            log_error(self.logs_path, "start", name, date_time)
            log.warning("Could not start container %s, skipping this lifecycle round", name)
            execute_command(self.runtime.force_remove(name), continue_if_error=True, error_informative=False)
            return False

        stop_time = self.lifecycle.stop(name, on_kill=lambda: log_error(self.logs_path, "kill", name, date_time))
        remove_container_time = self.lifecycle.remove(
            name, on_force=lambda: log_error(self.logs_path, "force-remove", name, date_time))
        remove_image_time = get_time(self.runtime.remove_image(name))

        self._log(f"{name}.csv", "load_image;start;stop;remove_container;remove_image;date_time",
                  load_image_time, start_time, stop_time, remove_container_time, remove_image_time, date_time)
        return True

    # ---- recursos da maquina ---------------------------------------------------------
    def machine_resources_loop(self):
        collectors = [self.cpu_monitoring, self.disk_monitoring, self.memory_monitoring, self.process_monitoring,
                      self.disk_write_and_read_monitoring, self.kernel_killed_process_monitoring]
        while self.running:
            started = time.time()
            date_time = current_time()
            for collect in collectors:
                # uma leitura que falha nao derruba as demais nem a thread
                try:
                    collect(date_time)
                except (CommandError, Exception) as error:
                    log.warning("%s failed: %r", collect.__name__, error)
            # mpstat/iostat medem 1 s cada; desconta para manter o intervalo de sleep_time
            self._wait(self.monitoring.sleep_time - (time.time() - started))

    def cpu_monitoring(self, date_time):
        # "mpstat 1 1": uso no ultimo segundo ("mpstat" sozinho da' a media desde o boot)
        cpu_info = execute_command("mpstat 1 1 | grep all | head -n 1").split()
        index = cpu_info.index("all")  # o horario pode ocupar 1 ou 2 colunas (AM/PM)
        usr, nice, sys_used, iowait, soft = (cpu_info[index + 1], cpu_info[index + 2], cpu_info[index + 3],
                                             cpu_info[index + 4], cpu_info[index + 6])
        self._log("cpu.csv", "usr;nice;sys;iowait;soft;date_time", usr, nice, sys_used, iowait, soft, date_time)

    def disk_monitoring(self, date_time):
        used = execute_command("df | grep '/$' | awk '{print $3}'")
        self._log("disk.csv", "used;date_time", used, date_time)

    def memory_monitoring(self, date_time):
        used = execute_command("free | grep Mem | awk '{print $3}'")
        cached = execute_command("cat /proc/meminfo | grep -i Cached | sed -n '1p' | awk '{print $2}'")
        buffers = execute_command("cat /proc/meminfo | grep -i Buffers | sed -n '1p' | awk '{print $2}'")
        swap = execute_command("cat /proc/meminfo | grep -i Swap | grep -i Free | awk '{print $2}'")
        self._log("memory.csv", "used;cached;buffers;swap;date_time", used, cached, buffers, swap, date_time)

    def process_monitoring(self, date_time):
        zombies = execute_command("ps aux | awk '{if ($8 ~ \"Z\") {print $0}}' | wc -l")
        self._log("process.csv", "zombies;date_time", zombies, date_time)

    def disk_write_and_read_monitoring(self, date_time):
        # "-y 1 1": taxas do ultimo segundo ("iostat -d" sozinho da' a media desde o boot)
        data = execute_command(f"iostat -dy 1 1 | awk '$1 == \"{self.disk_device}\"'").split()
        tps, read, write, dscd = data[1], data[2], data[3], data[4]
        self._log("disk_write_read.csv", "tps;kB_reads;kB_wrtns;kB_dscds;date_time", tps, read, write, dscd,
                  date_time)

    def kernel_killed_process_monitoring(self, date_time):
        """Grava uma linha por mensagem nova do kernel (processos mortos / OOM) desde a ultima coleta."""
        self._log_new_kernel_lines("killed.csv", "dmesg | grep -i 'killed process' || true", date_time)
        self._log_new_kernel_lines("omm.csv", "dmesg | grep -i 'oom' || true", date_time)

    def _log_new_kernel_lines(self, filename, command, date_time):
        seen = self._seen_kernel_logs[filename]
        for line in execute_command_lines(command):
            if line not in seen:
                seen.add(line)
                log.warning("Kernel: %s", line)
                self._log(filename, "log;date_time", line, date_time)

    # ---- energia -----------------------------------------------------------------------
    def rapl_loop(self):
        reader = RaplReader(zone_filters=self.energy.rapl_zones)
        temp_path = find_pkg_temp_path()

        if not reader.available:
            log.warning("Nenhuma das zonas RAPL configuradas (%s) foi encontrada, ou o processo nao tem permissao "
                        "de leitura (rode como root). Monitoramento de energia da CPU desativado.",
                        reader.zone_filters)
            return

        columns = [f"{keyword}_j" for keyword in reader.selected_zones]
        if "package_j" in columns or "dram_j" in columns:
            columns.append("total_j")
        columns.append("cpu_temp_c")
        header = ";".join(columns) + ";date_time"

        while self.running:
            activity.waiting(self.energy.sleep_time_rapl, "sampling RAPL")
            sample = sample_power(reader, temp_path, self.energy.sleep_time_rapl)
            values = [_fmt(sample.get(column), 1 if column == "cpu_temp_c" else 4) for column in columns]
            self._log("energy_rapl.csv", header, *values, current_time())

    def smart_plug_loop(self):
        def on_sample(sample):
            self._log("energy_smart_plug.csv", "power_w;kwh_interval;kwh_total;state;date_time",
                      sample['power_w'], sample['kwh_interval'], sample['kwh_total'], sample['state'],
                      current_time())
            activity.waiting(self.energy.sleep_time_smart_plug, "waiting next sample")

        def on_error(message):
            log.debug("Smart plug: %s", message)
            log_error(self.logs_path, "smart_plug", message, current_time())

        try:
            asyncio.run(run_smart_plug_monitor(
                ip=self.energy.smart_plug_ip,
                username=self.energy.smart_plug_username,
                password=self.energy.smart_plug_password,
                interval=self.energy.sleep_time_smart_plug,
                on_sample=on_sample,
                on_error=on_error,
                should_stop=self._stop.is_set,
            ))
        except ImportError:
            log.warning("Pacote 'python-kasa' nao instalado; smart plug desativado.")

    # ---- custo do proprio monitoramento ----------------------------------------------------
    def monitoring_cost_loop(self):
        """
        A cada sleep_time grava a CPU de cada parte do sistema (monitoring_cost.csv) e do processo
        coletor inteiro (collector.csv). So' le' /proc, sem executar comandos: a propria medicao
        custa poucos milissegundos por amostra.
        """
        previous = None
        while self.running:
            try:
                previous = self._sample_cost(previous)
            except Exception as error:
                log.warning("Monitoring cost sample failed: %r", error)
            self._wait(self.monitoring.sleep_time)

    def _sample_cost(self, previous):
        date_time = current_time()
        parts = self.cost_tracker.sample()
        for part, row in parts.items():
            self._log(COST_FILE, COST_HEADER, part, row["role"], row["threads"], _fmt(row["threads_cpu_s"], 3),
                      row["children"], _fmt(row["children_cpu_s"], 3), _fmt(row["cpu_s"], 3),
                      _fmt(row["cpu_pct"], 3), date_time)

        # processo inteiro: CPU propria (todas as threads) + CPU de todos os comandos executados
        process = activity.collector_process()
        now = time.monotonic()
        total_s = process.get("self_cpu_s", 0.0) + sum(row["children_cpu_s"] for row in parts.values())
        cpu_pct = monitoring_pct = None
        if previous is not None and now > previous[0]:
            cpu_pct = max(total_s - previous[1], 0.0) / (now - previous[0]) * 100
            monitoring_pct = sum(row["cpu_pct"] or 0.0 for row in parts.values() if row["role"] == "monitoring")
            log.debug("Monitoring cost: %.2f%% of one core (whole collector %.2f%%, rss %s kB)",
                      monitoring_pct, cpu_pct, process.get("VmRSS", "?"))
        self._log(COLLECTOR_FILE, COLLECTOR_HEADER, _fmt(process.get("self_cpu_s"), 3),
                  _fmt(total_s - process.get("self_cpu_s", 0.0), 3), _fmt(cpu_pct, 3), _fmt(monitoring_pct, 3),
                  process.get("VmRSS", ""), process.get("VmSize", ""), process.get("Threads", ""), date_time)
        return now, total_s
