"""Orquestracao do experimento: preparacao, workload, monitoramento e ciclos de stress/descanso."""
import sys
import time
from datetime import datetime, timedelta

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.images import build_images
from src.monitoring import MonitoringEnvironment
from src.runtime import ContainerRuntime, describe_environment, detect_runtime
from src.stressload import StressLoad
from src.utils import current_time, execute_command, write_csv_row

REJUVENATION_DELAY = 5
PROGRESS_BAR_SIZE = 50


class Environment:
    def __init__(self, config: Config, runtime: ContainerRuntime, monitoring_environment: MonitoringEnvironment):
        self.config = config
        self.general = config.general
        self.stress = config.stressload
        self.path = config.general.scripts_folder
        self.logs_path = config.logs_path
        self.runtime = runtime
        self.monitoring_environment = monitoring_environment
        self.stressload = StressLoad(runtime, config.stressload, self.path, self.logs_path,
                                     config.monitoring.using_containers_app_time)

    def clear(self):
        print("Cleaning old logs and containers")
        execute_command(f"rm -rf {self.logs_path}", continue_if_error=True, error_informative=False)
        execute_command(f"mkdir {self.logs_path}", continue_if_error=True, error_informative=False)
        self.clear_containers_and_images()

    def clear_containers_and_images(self):
        for command in self.runtime.cleanup_commands():
            execute_command(command, continue_if_error=True, error_informative=False)

    def rejuvenation_method(self):
        print("Applying rejuvenation")
        self.clear_containers_and_images()
        time.sleep(REJUVENATION_DELAY)
        self.start_teastore()

    def start_teastore(self):
        print("Starting teastore")
        execute_command(f"{self.path}/start_teastore.sh {self.runtime}", informative=True, error_informative=True)

    def run(self):
        self.clear()
        self.start_teastore()
        self.monitoring_environment.start()

        now = datetime.now()
        end = datetime.now() + timedelta(
            seconds=(self.stress.max_stress_time * self.general.runs + self.stress.wait_after_stress * self.general.runs))
        print(f"{now} - Script should end at around {end}")

        for current_run in range(self.general.runs):
            self._print_progress_bar(current_run)

            if not self.stress.stressload_first:
                self._rest()

            self._log_event("stress")
            self.stressload.run(self.config.containers, self.stress.max_stress_time)

            if self.general.apply_rejuvenation:
                self.rejuvenation_method()

            if self.stress.stressload_first:
                self._rest()

        self._print_progress_bar(self.general.runs)
        print(f"\nEnded at {datetime.now()}")

    def _rest(self):
        self._log_event("sleep")
        time.sleep(self.stress.wait_after_stress)

    def _log_event(self, event: str):
        write_csv_row(f"{self.logs_path}/runs.csv", "event;date_time", event, current_time())

    def _print_progress_bar(self, current_run: int, text: str = "Progress"):
        current_progress = current_run / self.general.runs
        sys.stdout.write(
            f"\r{text}: [{'=' * int(PROGRESS_BAR_SIZE * current_progress):{PROGRESS_BAR_SIZE}s}] "
            f"{round(current_progress, 2) * 100}%"
        )
        sys.stdout.flush()


def run_experiment(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    runtime = detect_runtime()

    build_images(runtime, config.containers, config.general.scripts_folder, config.setup.rebuild_images)

    monitoring_environment = MonitoringEnvironment(config, runtime, describe_environment(runtime))
    Environment(config, runtime, monitoring_environment).run()
