import threading
import time
import psutil

from src.utils import (
    execute_command,
    write_to_file,
    get_time,
    current_time,
)


class MonitoringEnvironment:
    def __init__(
            self,
            path: str,
            sleep_time: int,
            software: str,
            containers: list,
            sleep_time_container_metrics: int,
            environment_description: str,
            using_containers_app_time: bool
    ):
        self.path = path
        self.log_dir = "logs"
        self.sleep_time = sleep_time
        self.software = software
        self.containers = containers
        self.sleep_time_container_metrics = sleep_time_container_metrics
        self.environment_description = environment_description
        self.using_containers_app_time = using_containers_app_time

    def start(self):
        print("Environment:")
        print(self.environment_description)
        print("Starting monitoring scripts")

        write_to_file(
            f"{self.path}/{self.log_dir}/environment.txt",
            "Environment",
            self.environment_description
        )
        self.start_systemtap()
        self.start_container_lifecycle_monitoring()
        if self.software == "docker":
            self.start_docker_process_monitoring()
        elif self.software == "podman":
            self.start_podman_process_monitoring()
        self.start_machine_resources_monitoring()
        self.track_monitoring_cost()

    def start_systemtap(self):
        def systemtap():
            retry_count = 0
            while True:
                log_file = f"{self.path}/{self.log_dir}/fragmentation_{retry_count}.csv"
                if retry_count > 10:
                    print("Tried to start systemtap more than 10 times, skipping....")
                    break

                command = (
                    f"stap -o {log_file} "
                    "-D STP_OVERLOAD_THRESHOLD=650000000LL "
                    "-D STP_OVERLOAD_INTERVAL=1000000000LL "
                    f"{self.path}/fragmentation.stp"
                )

                execute_command(command, continue_if_error=True, error_informative=False)

                retry_count += 1
                time.sleep(120)

        monitoring_thread = threading.Thread(target=systemtap, name="systemtap")
        monitoring_thread.daemon = True
        monitoring_thread.start()

    def start_container_lifecycle_monitoring(self):
        container_metrics_thread = threading.Thread(target=self.container_metrics, name="container_metrics")
        container_metrics_thread.daemon = True
        container_metrics_thread.start()

    # process priority:
    #   ['docker', 'dockerd', 'containerd', 'containerd-shim', 'java', 'postgres', 'beam.smp', 'initdb', 'mysqld']
    def start_docker_process_monitoring(self):
        processes = ['docker', 'dockerd', 'containerd', 'containerd-shim', 'runc', 'docker-proxy', 'java', 'postgres',
                     'beam.smp', 'initdb', 'mysqld']

        for process in processes:
            process_thread = threading.Thread(target=self.process_monitoring_thread,
                                              name="docker_processes" + process, args=(process,))
            process_thread.daemon = True
            process_thread.start()

    def get_process_data(self, process_name: str, custom_pid: int = ''):
        date_time = current_time()

        data = []

        while len(data) == 0:
            try:
                if custom_pid != '':
                    pid = custom_pid
                else:
                    pid = execute_command(f'pgrep -f {process_name} | head -n 1')

                data = execute_command(f"pidstat -u -h -p {pid} -T ALL -r 1 1 | sed -n '4p'").split()

                threads = execute_command(f"cat /proc/{pid}/status | grep Threads | awk '{{print $2}}'",
                                          continue_if_error=True)
                swap = execute_command(f"cat /proc/{pid}/status | grep Swap | awk '{{print $2}}'",
                                       continue_if_error=True)

                cpu = data[7]
                mem = data[13]
                rss = data[12]
                vsz = data[11]

                if process_name == 'postgres':
                    write_to_file(
                        f'{self.path}/{self.log_dir}/postgres_process.csv',
                        'cpu;mem;rss;vsz;threads;swap;date_time',
                        f'{cpu};{mem};{rss};{vsz};{threads};{swap};{date_time}'
                    )
                else:
                    write_to_file(
                        f'{self.path}/{self.log_dir}/{process_name}.csv',
                        'cpu;mem;rss;vsz;threads;swap;date_time',
                        f'{cpu};{mem};{rss};{vsz};{threads};{swap};{date_time}'
                    )
            except:
                continue

    def process_monitoring_thread(self, process: str):
        while True:
            self.get_process_data(process)
            time.sleep(self.sleep_time - 1)

    # process priority:
    #   ['podman', 'conmon', 'java', 'postgres', 'beam.smp', 'initdb', 'mysqld']
    def start_podman_process_monitoring(self):
        processes = ['podman', 'conmon', 'cron', 'crun', 'systemd', 'java', 'postgres', 'beam.smp', 'initdb', 'mysqld']

        for process in processes:
            process_thread = threading.Thread(target=self.process_monitoring_thread,
                                              name="podman_processes" + process, args=(process,))
            process_thread.daemon = True
            process_thread.start()

    def start_machine_resources_monitoring(self):
        monitoring_thread = threading.Thread(target=self.machine_resources, name="monitoring")
        monitoring_thread.daemon = True
        monitoring_thread.start()

    def container_lifecycle(self):
        for container in self.containers:
            date_time = current_time()
            container_name = container["name"]
            host_port = container["host_port"]
            container_port = container["port"]

            load_image_time = get_time(f"{self.software} load -i {self.path}/{container_name}.tar -q")

            try:
                start_time = get_time(
                    f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {container_name}")
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
                        write_to_file(
                            f'{self.path}/{self.log_dir}/errors.csv',
                            'event;container;date_time',
                            f'start;{container_name};{date_time}'
                        )
                        print(f"Could not start container, exiting {container_name} lifecycle")
                        execute_command(f"{self.software} rm -v -f {container_name}", continue_if_error=False,
                                        error_informative=False)
                        return
                    time.sleep(2)
                    try:
                        start_time = get_time(
                            f"{self.software} run --name {container_name} -td -p {host_port}:{container_port} --init {container_name}",
                            continue_if_error=False, error_informative=False)
                        has_container = execute_command(
                            f"{self.software} container inspect {container_name} > /dev/null 2>&1 && echo true",
                            continue_if_error=True, error_informative=False)
                    except:
                        tries += 1

            up_time = ""
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
                stop_time = get_time(f"{self.software} stop {container_name}", continue_if_error=False,
                                     error_informative=False)
            except:
                write_to_file(
                    f'{self.path}/{self.log_dir}/errors.csv',
                    'event;container;date_time',
                    f'kill;{container_name};{date_time}'
                )
                stop_time = get_time(f"{self.software} kill {container_name}", continue_if_error=True,
                                     error_informative=False)

            try:
                remove_container_time = get_time(f"{self.software} rm -v {container_name}", continue_if_error=False,
                                                 error_informative=False)
            except:
                print("Forced container removal")
                write_to_file(
                    f'{self.path}/{self.log_dir}/errors.csv',
                    'event;container;date_time',
                    f'force-remove;{container_name};{date_time}'
                )
                remove_container_time = get_time(f"{self.software} rm -v -f {container_name}", continue_if_error=False,
                                                 error_informative=True)

            remove_image_time = get_time(f"{self.software} rmi {container_name}")

            if self.using_containers_app_time:
                write_to_file(
                    f"{self.path}/{self.log_dir}/{container_name}.csv",
                    "load_image;start;up_time;stop;remove_container;remove_image;date_time",
                    f"{load_image_time};{start_time};{up_time};{stop_time};{remove_container_time};{remove_image_time};{date_time}"
                )
            else:
                write_to_file(
                    f"{self.path}/{self.log_dir}/{container_name}.csv",
                    "load_image;start;stop;remove_container;remove_image;date_time",
                    f"{load_image_time};{start_time};{stop_time};{remove_container_time};{remove_image_time};{date_time}")

    def machine_resources(self):
        while True:
            date_time = current_time()
            self.cpu_monitoring(date_time)
            self.disk_monitoring(date_time)
            self.memory_monitoring(date_time)
            self.process_monitoring(date_time)
            self.disk_write_and_read_monitoring(date_time)
            self.kernel_killed_process_monitoring(date_time)
            time.sleep(self.sleep_time)

    def container_metrics(self):
        start_time = time.time()
        self.container_lifecycle()
        end_time = time.time()

        time_taken = end_time - start_time
        sleep_time = self.sleep_time_container_metrics - time_taken

        while True:
            if sleep_time > 0:
                time.sleep(sleep_time)
            start_time = time.time()
            self.container_lifecycle()
            end_time = time.time()

            time_taken = end_time - start_time
            sleep_time = self.sleep_time_container_metrics - time_taken

    def disk_monitoring(self, date_time):
        comando = "df | grep '/$' | awk '{print $3}'"
        mem = execute_command(comando)

        write_to_file(
            f"{self.path}/{self.log_dir}/disk.csv",
            "used;date_time",
            f"{mem};{date_time}"
        )

    def disk_write_and_read_monitoring(self, date_time):
        comando = "iostat -d | grep vda"
        data = execute_command(comando).split()
        tps = data[1]
        read = data[2]
        write = data[3]
        dscd = data[4]
        write_to_file(
            f"{self.path}/{self.log_dir}/disk_write_read.csv",
            "tps;kB_reads;kB_wrtns;kB_dscds;date_time",
            f"{tps};{read};{write};{dscd};{date_time}"
        )

    def cpu_monitoring(self, date_time):
        cpu_info = execute_command("mpstat | grep all").split()
        usr = cpu_info[2]
        nice = cpu_info[3]
        sys_used = cpu_info[4]
        iowait = cpu_info[5]
        soft = cpu_info[7]

        write_to_file(
            f"{self.path}/{self.log_dir}/cpu.csv",
            "usr;nice;sys;iowait;soft;date_time",
            f"{usr};{nice};{sys_used};{iowait};{soft};{date_time}"
        )

    def memory_monitoring(self, date_time):
        used = execute_command("free | grep Mem | awk '{print $3}'")
        cached = execute_command("cat /proc/meminfo | grep -i Cached | sed -n '1p' | awk '{print $2}'")
        buffers = execute_command("cat /proc/meminfo | grep -i Buffers | sed -n '1p' | awk '{print $2}'")
        swap = execute_command("cat /proc/meminfo | grep -i Swap | grep -i Free | awk '{print $2}'")

        write_to_file(
            f"{self.path}/{self.log_dir}/memory.csv",
            "used;cached;buffers;swap;date_time",
            f"{used};{cached};{buffers};{swap};{date_time}"
        )

    def process_monitoring(self, date_time):
        zombies = execute_command("ps aux | awk '{if ($8 ~ \"Z\") {print $0}}' | wc -l")

        write_to_file(
            f"{self.path}/{self.log_dir}/process.csv",
            "zombies;date_time",
            f"{zombies};{date_time}"
        )

    def kernel_killed_process_monitoring(self, date_time):
        killed = execute_command("dmesg | egrep -i 'killed process'")
        omm = execute_command("dmesg | grep -i 'oom'")

        write_to_file(
            f"{self.path}/{self.log_dir}/killed.csv",
            "log;date_time",
            f"{killed};{date_time}"
        )

        write_to_file(
            f"{self.path}/{self.log_dir}/omm.csv",
            "log;date_time",
            f"{omm};{date_time}"
        )

    def track_monitoring_cost(self):
        def monitoring_cost():
            while True:
                try:
                    for thread in threading.enumerate():
                        if thread.name.startswith(('systemtap', 'container_metrics', 'docker_processes',
                                                   'podman_processes', 'monitoring')):
                            tid = thread.native_id

                            self.get_process_data(process_name="process_" + thread.name, custom_pid=tid)
                    time.sleep(self.sleep_time)
                except Exception as e:
                    print(f"Erro no monitoramento de custo: {e}")
                    time.sleep(self.sleep_time)

        monitoring_thread = threading.Thread(target=monitoring_cost, name="monitoring_cost")
        monitoring_thread.daemon = True
        monitoring_thread.start()

