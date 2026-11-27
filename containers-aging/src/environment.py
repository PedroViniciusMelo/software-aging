"""Orquestracao do experimento: preparacao, workload, monitoramento e ciclos de stress/descanso."""
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

from src import activity, shutdown
from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.console import Console, ExperimentStatus
from src.images import build_images
from src.logsetup import attach_log_file, setup_logging
from src.monitoring import MonitoringEnvironment
from src.rejuvenation import RejuvenationContext, create_method
from src.runtime import ContainerRuntime, describe_environment, detect_runtime, scenario_name
from src.stressload import StressLoad
from src.utils import current_time, execute_command, log, write_csv_row


class Environment:
    def __init__(self, config: Config, runtime: ContainerRuntime, monitoring_environment: MonitoringEnvironment,
                 config_path: str = None):
        self.config = config
        self.config_path = config_path
        self.general = config.general
        self.stress = config.stressload
        self.path = config.general.scripts_folder
        self.logs_path = config.logs_path
        self.runtime = runtime
        self.monitoring_environment = monitoring_environment
        self.stressload = StressLoad(runtime, config.stressload, self.path, self.logs_path, config.lifecycle)
        self.status = ExperimentStatus(self.general.runs)
        # criado ja' aqui para que um nome de metodo errado falhe antes de comecar o experimento
        self.rejuvenation = create_method(RejuvenationContext(
            runtime=runtime, config=config.rejuvenation, logs_path=self.logs_path,
            start_teastore=self.start_teastore, lifecycle_lock=monitoring_environment.lifecycle_lock,
        )) if config.rejuvenation.enabled else None
        if self.rejuvenation:
            self.status.rejuvenation = f"{self.rejuvenation.name} - {self.rejuvenation.description}"

    def clear(self):
        log.info("Cleaning old logs, containers, images and volumes")
        execute_command(f"rm -rf {self.logs_path}", continue_if_error=True, error_informative=False)
        execute_command(f"mkdir {self.logs_path}", continue_if_error=True, error_informative=False)
        self.clear_containers_and_images()

    def clear_containers_and_images(self):
        for command in self.runtime.cleanup_commands():
            execute_command(command, continue_if_error=True, error_informative=False)

    def start_teastore(self):
        interval = self.config.setup.teastore_start_interval
        log.info("Starting TeaStore (takes ~%d minutes)", round(6 * interval / 60))
        execute_command(f"{self.path}/start_teastore.sh {self.runtime} {interval}", informative=True,
                        error_informative=True)

    def run(self):
        self.clear()
        if self.config.logging.to_file:
            attach_log_file(self.logs_path)
        if self.config_path:  # o config usado vai junto com os resultados (results/<cenario>/config.yaml)
            shutil.copy(self.config_path, f"{self.logs_path}/config.yaml")
        log.info("Rejuvenation: %s", self.status.rejuvenation)
        self.status.set_phase("starting TeaStore")
        self.start_teastore()
        self.monitoring_environment.start()
        shutdown.enable_graceful()
        if self.config.logging.console:
            Console(self.status, self.monitoring_environment).start()

        cycle = self.stress.max_stress_time + self.stress.wait_after_stress
        self.status.expected_end = datetime.now() + timedelta(seconds=cycle * self.general.runs)
        log.info("Experiment: %d runs of %ds stress + %ds rest; should end at around %s",
                 self.general.runs, self.stress.max_stress_time, self.stress.wait_after_stress,
                 f"{self.status.expected_end:%Y-%m-%d %H:%M:%S}")

        for current_run in range(1, self.general.runs + 1):
            log.info("Run %d/%d (%d%% done)", current_run, self.general.runs,
                     100 * (current_run - 1) // self.general.runs)

            if not self.stress.stressload_first:
                self._rest(current_run)

            if shutdown.stopping():
                break
            self._log_event("stress")
            self.status.set_phase("stress", current_run, self._until(self.stress.max_stress_time))
            log.info("Stress phase until %s", f"{self.status.phase_until:%Y-%m-%d %H:%M:%S}")
            self.stressload.run(self.config.containers, self.stress.max_stress_time)

            if self.rejuvenation and not shutdown.stopping():
                self._log_event("rejuvenation")
                self.status.set_phase("rejuvenation", current_run)
                self.rejuvenation()

            if self.stress.stressload_first and not shutdown.stopping():
                self._rest(current_run)

            if shutdown.stopping():
                break

        interrupted = shutdown.stopping()
        if interrupted:
            self._log_event("interrupted")
            log.warning("Experiment stopped by the user during run %d/%d", self.status.run, self.general.runs)
        else:
            log.info("Experiment ended")
        self.status.set_phase("interrupted" if interrupted else "finished")
        self.monitoring_environment.stop()

        if self.general.plot_results:
            self.plot(self.save_results(interrupted))
        elif interrupted:
            self.save_results(interrupted)

    def save_results(self, interrupted: bool = False) -> str:
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
        shutdown.wait(self.stress.wait_after_stress)

    def _log_event(self, event: str):
        write_csv_row(f"{self.logs_path}/runs.csv", "event;date_time", event, current_time())


def run_experiment(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    setup_logging(config.logging.informative, config.logging.debug)
    shutdown.install()
    try:
        runtime = detect_runtime()
        monitoring_environment = MonitoringEnvironment(config, runtime, describe_environment(runtime))
        environment = Environment(config, runtime, monitoring_environment, config_path)

        build_images(runtime, config.containers, config.general.scripts_folder, config.setup.rebuild_images)
        environment.run()
    except KeyboardInterrupt:
        # parada forcada (2o Ctrl+C, ou o 1o antes de o monitoramento comecar)
        log.error("Forced stop: terminating running commands; results were not saved")
        shutdown.kill_children()
        execute_command("pkill -INT -x stap", continue_if_error=True, error_informative=False)
        raise SystemExit(130)
