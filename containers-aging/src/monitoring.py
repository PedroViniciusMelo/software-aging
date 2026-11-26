"""Threads de coleta: recursos da maquina, processos, ciclo de vida, fragmentacao, energia e custo."""
import asyncio
import os
import shutil
import threading
import time

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
    log_error,
    write_csv_row,
    write_to_file,
)

# Os nomes das threads fazem parte da saida: o monitor de custo grava process_<thread>.csv
# para cada thread cujo nome comeca com um destes prefixos.
MONITORING_THREAD_PREFIXES = ('systemtap', 'container_metrics', 'docker_processes', 'podman_processes',
                              'monitoring', 'monitoring_smart_plug', 'monitoring_rapl')

PROCESS_HEADER = 'cpu;mem;rss;vsz;threads;swap;date_time'
# postgres.csv ja e' usado pelo monitor de ciclo de vida do container postgres.
PROCESS_FILE_NAMES = {'postgres': 'postgres_process'}

# Gerado por install/install.sh: versoes/commits exatos do runtime e do systemtap compilados.
BUILD_MANIFEST = "/etc/containers-aging/build-manifest.txt"

STAP_MAX_ATTEMPTS = 11
STAP_RETRY_DELAY = 120
PIDSTAT_DURATION = 1  # "pidstat ... 1 1" ja leva 1 s, descontado do intervalo de coleta
PROCESS_RETRY_DELAY = 1  # espera entre tentativas quando o processo ainda nao existe
STOP_GRACE = 3  # tempo para as leituras em andamento (pidstat/mpstat/iostat) terminarem no stop


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
        self.lifecycle = ContainerLifecycle(runtime)
        self.disk_device = None
        self._seen_kernel_logs = {"killed.csv": set(), "omm.csv": set()}
        self._stop = threading.Event()

    def _log(self, filename: str, header: str, *values) -> None:
        write_csv_row(f"{self.logs_path}/{filename}", header, *values)

    def _wait(self, seconds: float) -> bool:
        """Espera `seconds` (ou ate o stop). Devolve True se o monitoramento foi encerrado."""
        return self._stop.wait(max(seconds, 0))

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    @staticmethod
    def _start_thread(name: str, target, *args) -> None:
        threading.Thread(target=target, name=name, args=args, daemon=True).start()

    def start(self):
        print("Environment:")
        print(self.environment_description)
        print("Starting monitoring scripts")

        write_to_file(f"{self.logs_path}/environment.txt", "Environment", self.environment_description)
        if os.path.exists(BUILD_MANIFEST):
            shutil.copy(BUILD_MANIFEST, f"{self.logs_path}/build-manifest.txt")
        self.disk_device = self.monitoring.disk_device or detect_root_disk()

        self._start_thread("systemtap", self.systemtap_loop)
        self._start_thread("container_metrics", self.container_metrics_loop)
        for process in self.runtime.monitored_processes:
            self._start_thread(f"{self.runtime.name}_processes{process}", self.process_loop, process)
        self._start_thread("monitoring", self.machine_resources_loop)
        if self.energy.enable_rapl:
            self._start_thread("monitoring_rapl", self.rapl_loop)
        if self.energy.enable_smart_plug:
            self._start_thread("monitoring_smart_plug", self.smart_plug_loop)
        self._start_thread("monitoring_cost", self.monitoring_cost_loop)

    def stop(self):
        """Encerra a coleta: as threads saem na proxima espera e o systemtap e' interrompido."""
        print("Stopping monitoring scripts")
        self._stop.set()
        # SIGINT deixa o stap descarregar o modulo do kernel e fechar o CSV corretamente.
        execute_command("pkill -INT -x stap", continue_if_error=True, error_informative=False)
        time.sleep(STOP_GRACE)

    # ---- fragmentacao de memoria (systemtap) -----------------------------------------
    def systemtap_loop(self):
        for attempt in range(STAP_MAX_ATTEMPTS):
            execute_command(
                f"stap -o {self.logs_path}/fragmentation_{attempt}.csv "
                "-D STP_OVERLOAD_THRESHOLD=650000000LL "
                "-D STP_OVERLOAD_INTERVAL=1000000000LL "
                f"{self.path}/fragmentation.stp",
                continue_if_error=True, error_informative=False)
            if self._wait(STAP_RETRY_DELAY):
                return
        print("Tried to start systemtap more than 10 times, skipping....")

    # ---- processos -------------------------------------------------------------------
    def process_loop(self, process: str):
        while self.running:
            self.get_process_data(process)
            self._wait(self.monitoring.sleep_time - PIDSTAT_DURATION)

    def get_process_data(self, process_name: str, custom_pid=None):
        """
        Coleta cpu/mem/rss/vsz/threads/swap do processo. Se ele ainda nao existe, tenta de novo
        a cada PROCESS_RETRY_DELAY segundos ate encontra-lo. Com custom_pid (uma thread do
        proprio coletor) tenta uma vez so': a thread pode ter terminado.
        """
        date_time = current_time()

        while self.running:
            try:
                if custom_pid is not None:
                    pid = custom_pid
                else:
                    pid = execute_command(f"pgrep -f '{_pgrep_pattern(process_name)}' | head -n 1",
                                          error_informative=False)

                data = execute_command(f"pidstat -u -h -p {pid} -T ALL -r 1 1 | sed -n '4p'",
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
                if custom_pid is not None:
                    return
                self._wait(PROCESS_RETRY_DELAY)

    # ---- ciclo de vida dos containers ---------------------------------------------------
    def container_metrics_loop(self):
        while self.running:
            started = time.time()
            try:
                self.container_lifecycle()
            except CommandError as error:
                # ex.: falha ao carregar ou remover a imagem; registra e tenta no proximo ciclo
                log_error(self.logs_path, "lifecycle", error.command, current_time())
            self._wait(self.monitoring.sleep_time_container_metrics - (time.time() - started))

    def container_lifecycle(self):
        """Mede o tempo de cada etapa do ciclo de vida de cada container configurado."""
        for container in self.containers:
            if not self.running:
                return
            date_time = current_time()
            name = container.name

            load_image_time = get_time(self.runtime.load_image(f"{self.path}/{name}.tar"))

            start_time = self.lifecycle.start(name, name, container.host_port, container.port)
            if start_time is None:
                log_error(self.logs_path, "start", name, date_time)
                print(f"Could not start container, exiting {name} lifecycle")
                execute_command(self.runtime.force_remove(name), continue_if_error=True, error_informative=False)
                return

            up_time = self.lifecycle.wait_app_time(name) if self.monitoring.using_containers_app_time else ""
            stop_time = self.lifecycle.stop(name, on_kill=lambda: log_error(self.logs_path, "kill", name, date_time))
            remove_container_time = self.lifecycle.remove(
                name, on_force=lambda: log_error(self.logs_path, "force-remove", name, date_time))
            remove_image_time = get_time(self.runtime.remove_image(name))

            if self.monitoring.using_containers_app_time:
                self._log(f"{name}.csv", "load_image;start;up_time;stop;remove_container;remove_image;date_time",
                          load_image_time, start_time, up_time, stop_time, remove_container_time,
                          remove_image_time, date_time)
            else:
                self._log(f"{name}.csv", "load_image;start;stop;remove_container;remove_image;date_time",
                          load_image_time, start_time, stop_time, remove_container_time, remove_image_time,
                          date_time)

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
                    print(f"{date_time} - {collect.__name__} failed: {error!r}")
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
                self._log(filename, "log;date_time", line, date_time)

    # ---- energia -----------------------------------------------------------------------
    def rapl_loop(self):
        reader = RaplReader(zone_filters=self.energy.rapl_zones)
        temp_path = find_pkg_temp_path()

        if not reader.available:
            print(f"Nenhuma das zonas RAPL configuradas ({reader.zone_filters}) foi encontrada, "
                  "ou o processo nao tem permissao de leitura (rode como root). "
                  "Monitoramento de energia da CPU desativado.")
            return

        columns = [f"{keyword}_j" for keyword in reader.selected_zones]
        if "package_j" in columns or "dram_j" in columns:
            columns.append("total_j")
        columns.append("cpu_temp_c")
        header = ";".join(columns) + ";date_time"

        while self.running:
            sample = sample_power(reader, temp_path, self.energy.sleep_time_rapl)
            values = [_fmt(sample.get(column), 1 if column == "cpu_temp_c" else 4) for column in columns]
            self._log("energy_rapl.csv", header, *values, current_time())

    def smart_plug_loop(self):
        def on_sample(sample):
            self._log("energy_smart_plug.csv", "power_w;kwh_interval;kwh_total;state;date_time",
                      sample['power_w'], sample['kwh_interval'], sample['kwh_total'], sample['state'],
                      current_time())

        def on_error(message):
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
            print("Pacote 'python-kasa' nao instalado.")

    # ---- custo do proprio monitoramento ----------------------------------------------------
    def monitoring_cost_loop(self):
        while self.running:
            try:
                for thread in threading.enumerate():
                    if thread.name.startswith(MONITORING_THREAD_PREFIXES):
                        self.get_process_data(process_name="process_" + thread.name, custom_pid=thread.native_id)
            except Exception as e:
                print(f"Erro no monitoramento de custo: {e}")
            self._wait(self.monitoring.sleep_time)
