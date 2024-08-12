import sys
import threading
import time
from datetime import datetime, timedelta
from random import random
import yaml
import random

from src.monitoring import MonitoringEnvironment
from src.setup import Setup
from src.utils import execute_command, write_to_file, current_time, detect_used_software, check_environment


class Environment:
    def __init__(
            self,
            containers: list,
            sleep_time: int,
            software: str,
            max_stress_time: int,
            wait_after_stress: int,
            runs: int,
            scripts_folder: str,
            max_qtt_containers: int,
            min_qtt_containers: int,
            max_lifecycle_runs: int,
            min_lifecycle_runs: int,
            sleep_time_container_metrics: int,
            stressload_first: bool,
            using_containers_app_time: bool,
            monitoring_environment: MonitoringEnvironment
    ):
        self.logs_dir = "logs"
        self.containers = containers
        self.path = scripts_folder
        self.sleep_time = sleep_time
        self.software = software
        self.max_stress_time = max_stress_time
        self.wait_after_stress = wait_after_stress
        self.runs = runs
        self.max_qtt_containers = max_qtt_containers
        self.min_qtt_containers = min_qtt_containers
        self.max_lifecycle_runs = max_lifecycle_runs
        self.min_lifecycle_runs = min_lifecycle_runs
        self.sleep_time_container_metrics = sleep_time_container_metrics
        self.monitoring_environment = monitoring_environment
        self.stressload_first = stressload_first
        self.using_containers_app_time = using_containers_app_time

    def clear(self):
        print("Cleaning old logs and containers")
        execute_command(f"rm -rf {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
        execute_command(f"mkdir {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
        self.clear_containers_and_images()

    def clear_containers_and_images(self):
        execute_command(f"{self.software} stop $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} rm $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} rmi $({self.software} image ls -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} system prune -a -f)", continue_if_error=True, error_informative=False)

    def run(self):
        self.clear()
        self.start_teastore()
        self.monitoring_environment.start()

        now = datetime.now()
        end = datetime.now() + timedelta(
            seconds=(self.max_stress_time * self.runs + self.wait_after_stress * self.runs))

        print(f"{now} - Script should end at around {end}")

        for current_run in range(self.runs):
            self.__print_progress_bar(current_run, "Progress")

            if not self.stressload_first:
                write_to_file(
                    f"{self.path}/{self.logs_dir}/runs.csv",
                    "event;date_time",
                    f"sleep;{current_time()}"
                )
                time.sleep(self.wait_after_stress)

            write_to_file(
                f"{self.path}/{self.logs_dir}/runs.csv",
                "event;date_time",
                f"stress;{current_time()}"
            )
            self.init_containers_threads(self.max_stress_time)

            if self.stressload_first:
                write_to_file(
                    f"{self.path}/{self.logs_dir}/runs.csv",
                    "event;date_time",
                    f"sleep;{current_time()}"
                )
                time.sleep(self.wait_after_stress)

        self.__print_progress_bar(self.runs, "Progress")

        print(f"\nEnded at {datetime.now()}")
        # self.clear_containers_and_images()

    def start_teastore(self):
        print("Starting teastore")

        command = f"{self.path}/start_teastore.sh {self.software}"
        execute_command(command, informative=True, error_informative=True)

    def init_containers_threads(self, max_stress_time):
        threads = []
        for container in self.containers:
            thread = threading.Thread(
                target=self.container_thread,
                name=container,
                args=(container, max_stress_time)
            )

            thread.daemon = True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def container_stressload(self, image_name, host_port, container_port, min_container_wait_time,
                             max_container_wait_time, run):

        sleep_time = random.randint(min_container_wait_time, max_container_wait_time)
        qtt_containers = random.randint(self.min_qtt_containers, self.max_qtt_containers)

        image_name = "temp_" + image_name
        time.sleep(sleep_time)

        for i in range(qtt_containers):
            container_name = f"temp_{image_name}-{run}-{i}"

            execute_command(
                f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {image_name}")

            if self.using_containers_app_time:
                up_time = execute_command(
                    f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                    continue_if_error=True, error_informative=False)

                while up_time is None:
                    time.sleep(0.5)
                    up_time = execute_command(
                        f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                        continue_if_error=True, error_informative=False)

            execute_command(f"{self.software} stop {container_name}")
            execute_command(f"{self.software} rm {container_name}")
            execute_command(f"{self.software} volume prune -f", continue_if_error=True, error_informative=True)

    def container_thread(self, container, max_stress_time):
        now = datetime.now()
        max_date = now + timedelta(seconds=max_stress_time)
        execute_command(f"{self.software} load -i {self.path}/temp_{container['name']}.tar -q")

        while datetime.now() < max_date:
            exec_runs = random.randint(self.min_lifecycle_runs, self.max_lifecycle_runs)

            threads = []
            for index in range(exec_runs):
                thread = threading.Thread(
                    target=self.container_stressload,
                    name=container,
                    args=(
                        container["name"],
                        container["host_port"] + index + 100,
                        container["port"],
                        container["min_container_wait_time"],
                        container["max_container_wait_time"],
                        index
                    )
                )

                thread.daemon = True
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

    def __print_progress_bar(self, current_run, text):
        progress_bar_size = 50
        current_progress = current_run / self.runs
        sys.stdout.write(
            f"\r{text}: [{'=' * int(progress_bar_size * current_progress):{progress_bar_size}s}] "
            f"{round(current_progress, 2) * 100}%"
        )
        sys.stdout.flush()


class EnvironmentConfig:

    def __init__(self):
        with open("config.yaml", "r") as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)

        general_config = config["general"]
        monitoring_config = config["monitoring"]

        setup = Setup(
            containers=config["containers"],
            scripts_folder=general_config["scripts_folder"],
            software=detect_used_software()
        )

        setup.build_images()

        monitoring_enviroment = MonitoringEnvironment(
            path=general_config["scripts_folder"],
            sleep_time=monitoring_config["sleep_time"],
            containers=config["containers"],
            sleep_time_container_metrics=monitoring_config["sleep_time_container_metrics"],
            software=detect_used_software(),
            using_containers_app_time=monitoring_config["using_containers_app_time"],
            environment_description=check_environment()
        )

        framework = Environment(
            **config["general"],
            **config["monitoring"],
            **config["stressload"],
            containers=config["containers"],
            monitoring_environment=monitoring_enviroment,
            software=detect_used_software()
        )

        framework.run()
