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


class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self) -> any:
        threading.Thread.join(self)
        return self._return


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
            monitoring_environment: MonitoringEnvironment,
            apply_rejuvenation: bool,
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
        self.apply_rejuvenation = apply_rejuvenation

    def clear(self):
        print("Cleaning old logs and containers")
        execute_command(f"rm -rf {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
        execute_command(f"mkdir {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
        self.clear_containers_and_images()

    def clear_containers_and_images(self):
        execute_command(f"{self.software} stop $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} rm $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} rmi $({self.software} image ls -aq)", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} container prune -f", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} image prune -f", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} volume prune -f", continue_if_error=True, error_informative=False)
        execute_command(f"{self.software} system prune -a -f --volumes", continue_if_error=True, error_informative=False)

    def rejuvenation_method(self):
        print("Applying rejuvenation")
        self.clear_containers_and_images()
        time.sleep(5)
        self.start_teastore()


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

            if self.apply_rejuvenation:
                self.rejuvenation_method()

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
            thread = CustomThread(
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
                             max_container_wait_time, run) -> int:

        sleep_time = random.randint(min_container_wait_time, max_container_wait_time)
        qtt_containers = random.randint(self.min_qtt_containers, self.max_qtt_containers)

        image_name = "temp_" + image_name
        time.sleep(sleep_time)

        errors = 0
        for i in range(qtt_containers):
            container_name = f"temp_{image_name}-{run}-{i}"

            try:
                execute_command(
                    f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {image_name}")
            except:
                print("Could not start container, executing fallback method")
                has_container = execute_command(
                    f"{self.software} container inspect {container_name} > /dev/null 2>&1 && echo true",
                    continue_if_error=True, error_informative=False)
                if has_container is not None:
                    execute_command(f"{self.software} rm -v -f {container_name}", continue_if_error=False,
                                    error_informative=False)
                    has_container = None

                tries = 0
                while has_container is None:
                    if tries > 5:
                        print(f"Could not start container, exiting {image_name} lifecycle")
                        date_time = current_time()

                        write_to_file(
                            f'{self.path}/{self.logs_dir}/errors.csv',
                            'event;container;date_time',
                            f'start;{image_name};{date_time}'
                        )

                        execute_command(f"{self.software} rm -v -f {container_name}", continue_if_error=False,
                                        error_informative=False)
                        errors += 1
                        continue
                    time.sleep(2)
                    try:
                        execute_command(
                            f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {image_name}",
                            continue_if_error=False, error_informative=False)
                        has_container = execute_command(
                            f"{self.software} container inspect {container_name} > /dev/null 2>&1 && echo true",
                            continue_if_error=True, error_informative=False)
                    except:
                        tries += 1

            if self.using_containers_app_time:
                up_time = execute_command(
                    f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                    continue_if_error=True, error_informative=False)

                while up_time is None:
                    time.sleep(0.5)
                    up_time = execute_command(
                        f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                        continue_if_error=True, error_informative=False)

            try:
                execute_command(f"{self.software} stop {container_name}", continue_if_error=False,
                                error_informative=False)
            except:
                print("Killing container")
                date_time = current_time()

                write_to_file(
                    f'{self.path}/{self.logs_dir}/errors.csv',
                    'event;container;date_time',
                    f'kill;{image_name};{date_time}'
                )
                execute_command(f"{self.software} kill {container_name}", continue_if_error=True,
                                error_informative=False)

            try:
                execute_command(f"{self.software} rm -v {container_name}", continue_if_error=False, error_informative=False)
            except:
                print("Forced container removal")
                date_time = current_time()

                write_to_file(
                    f'{self.path}/{self.logs_dir}/errors.csv',
                    'event;container;date_time',
                    f'force-remove;{image_name};{date_time}'
                )
                execute_command(f"{self.software} rm -v -f {container_name}", continue_if_error=False, error_informative=True)
        return qtt_containers - errors

    def container_thread(self, container, max_stress_time):
        now = datetime.now()
        max_date = now + timedelta(seconds=max_stress_time)
        execute_command(f"{self.software} load -i {self.path}/temp_{container['name']}.tar -q")

        qtd_container = 0
        while datetime.now() < max_date:
            exec_runs = random.randint(self.min_lifecycle_runs, self.max_lifecycle_runs)

            threads = []
            for index in range(exec_runs):
                thread = CustomThread(
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
                result = thread.join()
                qtd_container += result

        write_to_file(
            f"{self.path}/{self.logs_dir}/qtd_containers.csv",
            "thread;qtd_containers;date",
            f"{container['name']};{qtd_container};{current_time()}"
        )


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
        setup_config = config["setup"]

        setup = Setup(
            containers=config["containers"],
            scripts_folder=general_config["scripts_folder"],
            rebuild_images=setup_config["rebuild_images"],
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
