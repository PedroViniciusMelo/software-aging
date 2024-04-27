import sys, threading, time, yaml
from datetime import datetime, timedelta
from random import random
from src.monitoring import MonitoringEnvironment

from src.utils import (
    execute_command,
    write_to_file,
    current_time
)


class Environment: 
    def __init__(self,
                 containers: list,
                 sleep_time: int,
                 software: str,
                 max_stress_time: int,
                 wait_after_stress: int,
                 runs: int,
                 old_software: bool,
                 system: str,
                 old_system: bool,
                 scripts_folder: str,
                 max_qtt_containers: int,
                 min_qtt_containers: int,
                 max_lifecycle_runs: int,
                 min_lifecycle_runs: int,
                 sleep_time_container_metrics: int,
                 monitoring_environment: MonitoringEnvironment
    ):
        log_dir = software
        if old_software:
            log_dir = f'{log_dir}_old_'
            
        else:
            log_dir = f'{log_dir}_new_'

        log_dir = log_dir + system
        if old_system:
            log_dir = f'{log_dir}_old'
            
        else:
            log_dir = f'{log_dir}_new'

        self.logs_dir = log_dir                                           # Logs path
        self.containers = containers                                      # Get list of all containers for monitoring
        self.sleep_time = sleep_time                                      # Sleep time the machine data will be collected in seconds
        self.software = software                                          # Use docker or podman
        self.max_stress_time = max_stress_time                            # Time the stress will be running in seconds (default 259200s = 3 days)
        self.wait_after_stress = wait_after_stress                        # Time the script will wait until next stress period in seconds (default 43200 = 12h)
        self.runs = runs                                                  # Changing this will affect the stress load, (max_stress_time + wait_after_stress) * (this value) will be total time of experiments
        self.path = scripts_folder                                        # Default path for start tests
        self.max_qtt_containers = max_qtt_containers                      # Max number of containers that will be created and deleted on each lifecycle run
        self.min_qtt_containers = min_qtt_containers                      # Min number of containers that will be created and deleted on each lifecycle run
        self.max_lifecycle_runs = max_lifecycle_runs                      # Max number of lifecycle runs for each container
        self.min_lifecycle_runs = min_lifecycle_runs                      # Min number of lifecycle runs for each container
        self.sleep_time_container_metrics = sleep_time_container_metrics  # Sleep time the machine data will be collected in seconds
        self.monitoring_environment = monitoring_environment              # Start monitoring metrics


    # INITIAL CALL
    def run(self):
        """
        Start class function
        """
        try:
            self.clear()
            self.start_teastore()
            self.monitoring_environment.start()

            now = datetime.now()
            end = datetime.now() + timedelta(
                seconds=(
                    self.max_stress_time * self.runs + self.wait_after_stress * self.runs
                )
            )

            print(f"{now} - Script should end at around {end}")

            for current_run in range(self.runs):
                self.__print_progress_bar(current_run, "Progress")
                
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

            self.__print_progress_bar(self.runs, "Progress")

            print(f"\nEnded at {datetime.now()}")
            # self.clear_containers_and_images()
            
        except Exception as e:
            print('Error in the enviroment.py file in the run() function: {e}')
            sys.exit(1)
        
        
    def clear_containers_and_images(self):
        try:
            execute_command(f"{self.software} stop $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
            execute_command(f"{self.software} rm $({self.software} ps -aq)", continue_if_error=True, error_informative=False)
            execute_command(f"{self.software} rmi $({self.software} image ls -aq)", continue_if_error=True, error_informative=False)
            
        except Exception as e:
            print('Error in the enviroment.py file in the clear_containers_and_images() function: {e}')
            sys.exit(1)
        
        
    def clear(self):
        """
        Cleaning old logs and old/current containers
        """
        try:
            print("Cleaning old logs and containers")
            
            execute_command(f"rm -rf {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
            execute_command(f"mkdir {self.path}/{self.logs_dir}", continue_if_error=True, error_informative=False)
            
            self.clear_containers_and_images()
            
        except Exception as e:
            print('Error in the enviroment.py file in the clear() function: {e}')
            sys.exit(1)


    def start_teastore(self):
        print("Starting teastore")

        try:
            if self.software == "docker":
                # command = f"docker compose -f {self.path}/docker-compose.yaml up -d --quiet-pull"
                return execute_command(f'docker compose -f {self.path}/docker-compose.yaml up -d --quiet-pull')
                
            else:
                # command = f"podman-compose -f {self.path}/docker-compose.yaml up -d --quiet-pull"
                return execute_command(f'podman-compose -f {self.path}/docker-compose.yaml up -d --quiet-pull')
                
            # execute_command(command, informative=True, error_informative=True)
            
        except Exception as e:
            print('Error in the enviroment.py file in the start_teastore() function: {e}')
            sys.exit(1)


    def init_containers_threads(self, max_stress_time):
        try:
            threads = []
            
            for container in self.containers:
                thread = threading.Thread(target=self.container_thread, name=container, args=(container, max_stress_time))
                thread.daemon = True
                thread.start()
                
                threads.append(thread)

            for thread in threads:
                thread.join()
                
        except Exception as e:
            print('Error in the enviroment.py file in the init_containers_threads() function: {e}')
            sys.exit(1)   


    def container_stressload(self, image_name, host_port, container_port, min_container_wait_time, max_container_wait_time, run):
        try:
            sleep_time = random.randint(min_container_wait_time, max_container_wait_time)
            qtt_containers = random.randint(self.min_qtt_containers, self.max_qtt_containers)

            image_name = f'temp_{image_name}'
            time.sleep(sleep_time)

            for i in range(qtt_containers):
                container_name = f"temp_{image_name}-{run}-{i}"

                execute_command(
                    f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {image_name}"
                )
                
                up_time = execute_command(
                    f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                    continue_if_error=True, error_informative=False
                )

                while up_time is None:
                    up_time = execute_command(
                        f"{self.software} exec -i {container_name} sh -c \"test -e /root/log.txt && cat /root/log.txt\"",
                        continue_if_error=True, error_informative=False
                    )

                execute_command(f"{self.software} stop {container_name}")
                execute_command(f"{self.software} rm {container_name}")
                
        except Exception as e:
            print('Error in the enviroment.py file in the container_stressload() function: {e}')
            sys.exit(1) 


    def container_thread(self, container, max_stress_time):
        try:
            # now = datetime.now()
            max_date = datetime.now() + timedelta(seconds=max_stress_time)
            execute_command(f"{self.software} load -i {self.path}/temp_{container['name']}.tar -q")

            while datetime.now() < max_date:
                exec_runs = random.randint(self.min_lifecycle_runs, self.max_lifecycle_runs)

                threads = []
                for index in range(exec_runs):
                    thread = threading.Thread(target=self.container_stressload, name=container,
                        args=(
                            container["name"],
                            container["host_port"] + index + 2,
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
                    
        except Exception as e:
            print('Error in the enviroment.py file in the container_stressload() function: {e}')
            sys.exit(1) 


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
        try:
            with open("config.yaml", "r") as yml_file:
                config = yaml.load(yml_file, Loader=yaml.FullLoader)

            general_config = config["general"]
            monitoring_config = config["monitoring"]

            monitoring_enviroment = MonitoringEnvironment(
                path=general_config["scripts_folder"],
                sleep_time=monitoring_config["sleep_time"],
                software=general_config["software"],
                containers=config["containers"],
                sleep_time_container_metrics=monitoring_config["sleep_time_container_metrics"],
                old_system=general_config["old_system"],
                old_software=general_config["old_software"],
                system=general_config["system"],
            )

            framework = Environment(
                **config["general"],
                **config["monitoring"],
                **config["stressload"],
                containers=config["containers"],
                monitoring_environment=monitoring_enviroment
            )

            framework.run()
        
        except Exception as e:
            print('Error in the enviroment.py file in the EnvironmentConfig class __init__ constructor: {e}')
            sys.exit(1) 