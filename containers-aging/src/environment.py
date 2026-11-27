"""
Orquestracao do experimento: preparacao, workload, monitoramento, ciclos de stress/descanso e
rejuvenescimento.

Rejuvenescimento (rejuvenation.enabled): aplicado ao fim de cada periodo de stress, na thread
principal, com o monitor de ciclo de vida pausado (lifecycle_lock), para nao encontrar containers
do monitoramento pela metade. O metodo e' escolhido por rejuvenation.method, entre as chaves de METHODS.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

from src import activity
from src.config import DEFAULT_CONFIG_PATH, Config, RejuvenationConfig, load_config
from src.console import Console, ExperimentStatus
from src.monitoring import MonitoringEnvironment
from src.runtime import ContainerRuntime, detect_runtime, scenario_name
from src.stressload import StressLoad
from src.utils import (
    attach_log_file,
    current_time,
    execute_command,
    execute_command_lines,
    log,
    setup_logging,
    write_csv_row,
)


REJUVENATION_FILE = "rejuvenation.csv"
REJUVENATION_HEADER = "method;duration_s;details;date_time"
TEASTORE_LABEL = re.compile(r"containers-aging\.role[=:]teastore")  # rotulo posto por start_teastore.sh


def _run(command: str) -> None:
    execute_command(command, continue_if_error=True, warn_on_error=False)


def cleanup_keep_teastore(runtime: ContainerRuntime, config: RejuvenationConfig, start_teastore) -> dict:
    """Remove containers, imagens e volumes sem uso, preservando tudo do TeaStore."""
    def protected(text: str) -> bool:
        return bool(TEASTORE_LABEL.search(text)) or any(pattern in text for pattern in config.protected_images)

    # containers: "<id> <imagem> <rotulos>"; protegido pela imagem ou pelo rotulo do TeaStore
    rows = execute_command_lines(runtime.command("ps -a --format '{{.ID}} {{.Image}} {{.Labels}}'"),
                                 continue_if_error=True) or []
    containers = [row.split()[0] for row in rows if not protected(row)]
    if containers:
        ids = " ".join(containers)
        _run(runtime.command(f"stop {ids}"))
        _run(runtime.command(f"rm -v -f {ids}"))

    # imagens: so' as que nao sao do TeaStore. Sem -f: o runtime recusa remover uma imagem que algum
    # container ainda usa, entao as do TeaStore ficam mesmo se o nome nao casar.
    rows = execute_command_lines(runtime.command("image ls --format '{{.ID}} {{.Repository}}:{{.Tag}}'"),
                                 continue_if_error=True) or []
    images = []
    for row in rows:
        image_id, _, reference = row.partition(" ")
        if not protected(reference):
            images.append(image_id if "<none>" in reference else reference)
    for image in images:
        _run(runtime.remove_image(image))

    _run(runtime.command("volume prune -f"))  # os do TeaStore estao em uso pelos containers preservados
    return {"containers": len(containers), "images": len(images)}


def full_reset(runtime: ContainerRuntime, config: RejuvenationConfig, start_teastore) -> dict:
    """Metodo antigo: apaga todos os containers, imagens e volumes (inclusive o TeaStore) e sobe o TeaStore de novo."""
    for command in runtime.cleanup_commands():
        _run(command)
    time.sleep(config.delay)
    start_teastore()
    return {}


METHODS = {"cleanup_keep_teastore": cleanup_keep_teastore, "full_reset": full_reset}


class Rejuvenation:
    """O metodo configurado; criado no inicio para que um nome errado falhe antes do experimento."""

    def __init__(self, runtime: ContainerRuntime, config: RejuvenationConfig, logs_path: str,
                 start_teastore: Callable[[], None], lifecycle_lock: threading.Lock):
        if config.method not in METHODS:
            raise SystemExit(f"Unknown rejuvenation method '{config.method}'. Available: {', '.join(METHODS)}")
        self.name = config.method
        self.method = METHODS[config.method]
        self.description = self.method.__doc__.strip().splitlines()[0]
        self.runtime, self.config, self.logs_path = runtime, config, logs_path
        self.start_teastore, self.lifecycle_lock = start_teastore, lifecycle_lock

    def __call__(self) -> None:
        """Aplica o metodo (com o monitor de ciclo de vida pausado), mede e registra."""
        log.info("Applying rejuvenation: %s", self.name)
        started = time.perf_counter()
        with self.lifecycle_lock:
            details = self.method(self.runtime, self.config, self.start_teastore)
        duration = time.perf_counter() - started
        summary = ",".join(f"{key}={value}" for key, value in details.items())
        log.info("Rejuvenation %s done in %.1fs %s", self.name, duration, summary)
        write_csv_row(f"{self.logs_path}/{REJUVENATION_FILE}", REJUVENATION_HEADER, self.name, round(duration, 3),
                      summary, current_time())
        time.sleep(self.config.delay)


class Environment:
    def __init__(self, config: Config, runtime: ContainerRuntime, config_path: str = None):
        self.config = config
        self.config_path = config_path
        self.stress = config.stressload
        self.runs = config.general.runs
        self.path = config.general.scripts_folder
        self.logs_path = config.logs_path
        self.runtime = runtime
        self.monitoring = MonitoringEnvironment(config, runtime)
        self.stressload = StressLoad(runtime, config.stressload, self.path, self.logs_path, config.lifecycle)
        self.status = ExperimentStatus(self.runs)
        self.rejuvenation = None
        if config.rejuvenation.enabled:
            self.rejuvenation = Rejuvenation(runtime, config.rejuvenation, self.logs_path, self.start_teastore,
                                             self.monitoring.lifecycle_lock)
            self.status.rejuvenation = f"{self.rejuvenation.name} - {self.rejuvenation.description}"

    # ---- preparacao ----------------------------------------------------------------------------
    def build_images(self) -> None:
        """
        Baixa cada imagem e salva duas copias na pasta do projeto:
          <name>.tar       -> monitor de ciclo de vida (imagem <name>:latest)
          temp_<name>.tar  -> carga de estresse (imagem temp_<name>:latest)
        """
        if not self.config.setup.rebuild_images:
            log.info("Rebuilding images is disabled. Skipping...")
            return
        for container in self.config.containers:
            name, version = container.name, container.version
            log.info("Building image %s:%s", name, version)
            for command in [
                f"pull docker.io/{name}:{version}",
                f"tag {name}:{version} {name}:latest",
                f"tag {name}:{version} temp_{name}:latest",
                f"save {name}:latest > {self.path}/{name}.tar",
                f"save temp_{name}:latest > {self.path}/temp_{name}.tar",
                f"rmi {name}:latest",
                f"rmi temp_{name}:latest",
                f"rmi {name}:{version}",
            ]:
                execute_command(self.runtime.command(command))

    def clear(self):
        """Apaga logs/ e todos os containers, imagens e volumes."""
        log.info("Cleaning old logs, containers, images and volumes")
        shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path)
        for command in self.runtime.cleanup_commands():
            execute_command(command, continue_if_error=True, warn_on_error=False)

    def start_teastore(self):
        interval = self.config.setup.teastore_start_interval
        log.info("Starting TeaStore (takes ~%d minutes)", round(6 * interval / 60))
        execute_command(f"{self.path}/start_teastore.sh {self.runtime} {interval}", informative=True)

    # ---- experimento -----------------------------------------------------------------------------
    def run(self):
        self.clear()
        if self.config.logging.to_file:
            attach_log_file(self.logs_path)
        if self.config_path:  # o config usado vai junto com os resultados (results/<cenario>/config.yaml)
            shutil.copy(self.config_path, f"{self.logs_path}/config.yaml")
        log.info("Rejuvenation: %s", self.status.rejuvenation)
        self.status.set_phase("starting TeaStore")
        self.start_teastore()
        self.monitoring.start()
        activity.enable_graceful_stop()
        if self.config.logging.console:
            Console(self.status, self.monitoring).start()

        cycle = self.stress.max_stress_time + self.stress.wait_after_stress
        self.status.expected_end = datetime.now() + timedelta(seconds=cycle * self.runs)
        log.info("Experiment: %d runs of %ds stress + %ds rest; should end at around %s", self.runs,
                 self.stress.max_stress_time, self.stress.wait_after_stress, f"{self.status.expected_end:%Y-%m-%d %H:%M:%S}")

        for current_run in range(1, self.runs + 1):
            log.info("Run %d/%d (%d%% done)", current_run, self.runs, 100 * (current_run - 1) // self.runs)
            if not self.stress.stressload_first:
                self._rest(current_run)
            if activity.stopping():
                break

            self._log_event("stress")
            self.status.set_phase("stress", current_run, self._until(self.stress.max_stress_time))
            log.info("Stress phase until %s", f"{self.status.phase_until:%Y-%m-%d %H:%M:%S}")
            self.stressload.run(self.config.containers, self.stress.max_stress_time)

            if self.rejuvenation and not activity.stopping():
                self._log_event("rejuvenation")
                self.status.set_phase("rejuvenation", current_run)
                self.rejuvenation()
            if self.stress.stressload_first and not activity.stopping():
                self._rest(current_run)
            if activity.stopping():
                break

        interrupted = activity.stopping()
        if interrupted:
            self._log_event("interrupted")
            log.warning("Experiment stopped by the user during run %d/%d", self.status.run, self.runs)
        else:
            log.info("Experiment ended")
        self.status.set_phase("interrupted" if interrupted else "finished")
        self.monitoring.stop()

        if self.config.general.plot_results or interrupted:
            folder = self.save_results(interrupted)
            if self.config.general.plot_results:
                self.plot(folder)

    def save_results(self, interrupted: bool) -> str:
        """Copia logs/ para results/<cenario>_<data>[_interrupted], no formato de nome lido por plot/."""
        folder = f"{self.path}/results/{scenario_name(self.runtime)}_{datetime.now():%Y%m%d-%H%M}"
        if interrupted:
            folder += "_interrupted"
        shutil.copytree(self.logs_path, folder)
        log.info("Results saved to %s", folder)
        return folder

    def plot(self, folder: str) -> None:
        """Gera os graficos num processo separado (o coletor nao importa as libs de plot/)."""
        log.info("Plotting results")
        result = subprocess.run([sys.executable, f"{self.path}/plot/plot_data.py", folder],
                                env={**os.environ, "MPLBACKEND": "Agg"})
        if result.returncode != 0:
            log.warning("Plotting failed. Run it again with: %s plot/plot_data.py %s", sys.executable, folder)

    @staticmethod
    def _until(seconds: int) -> datetime:
        return datetime.now() + timedelta(seconds=seconds)

    def _rest(self, current_run: int):
        self._log_event("sleep")
        self.status.set_phase("rest", current_run, self._until(self.stress.wait_after_stress))
        log.info("Rest phase until %s", f"{self.status.phase_until:%Y-%m-%d %H:%M:%S}")
        activity.waiting(self.stress.wait_after_stress, "rest phase")
        activity.wait(self.stress.wait_after_stress)

    def _log_event(self, event: str):
        write_csv_row(f"{self.logs_path}/runs.csv", "event;date_time", event, current_time())


def run_experiment(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    setup_logging(config.logging.informative)
    activity.install_signal_handlers()
    try:
        runtime = detect_runtime()
        environment = Environment(config, runtime, config_path)
        environment.build_images()
        environment.run()
    except KeyboardInterrupt:
        # parada forcada (2o Ctrl+C, ou o 1o antes de o monitoramento comecar)
        log.error("Forced stop: terminating running commands; results were not saved")
        activity.kill_children()
        execute_command("pkill -INT -x stap", continue_if_error=True, warn_on_error=False)
        raise SystemExit(130)
