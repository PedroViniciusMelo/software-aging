"""Carga de estresse: cria e remove containers em paralelo durante o periodo de stress."""
import random
import threading
import time
from datetime import datetime, timedelta

from src.config import ContainerConfig, StressloadConfig
from src.lifecycle import ContainerLifecycle
from src.runtime import ContainerRuntime
from src.utils import current_time, execute_command, log_error, write_csv_row

# Cada lote paralelo usa portas a partir de host_port + PORT_OFFSET + indice do lote.
PORT_OFFSET = 100


class ResultThread(threading.Thread):
    """Thread daemon cujo join() devolve o valor retornado pelo target."""

    def __init__(self, target, name, args=()):
        super().__init__(target=target, name=name, args=args, daemon=True)
        self._result = None

    def run(self):
        self._result = self._target(*self._args)

    def join(self, timeout=None):
        super().join(timeout)
        return self._result


def run_in_threads(jobs: list) -> list:
    """Roda cada (name, target, args) numa ResultThread e devolve os resultados na ordem."""
    threads = [ResultThread(target=target, name=name, args=args) for name, target, args in jobs]
    for thread in threads:
        thread.start()
    return [thread.join() for thread in threads]


class StressLoad:
    def __init__(self, runtime: ContainerRuntime, config: StressloadConfig, scripts_folder: str,
                 logs_path: str, using_containers_app_time: bool):
        self.runtime = runtime
        self.config = config
        self.scripts_folder = scripts_folder
        self.logs_path = logs_path
        self.using_containers_app_time = using_containers_app_time
        self.lifecycle = ContainerLifecycle(runtime)

    def run(self, containers: list, max_stress_time: int) -> None:
        """Estressa todos os containers em paralelo por max_stress_time segundos."""
        run_in_threads([
            (f"stressload_{container.name}", self.container_thread, (container, max_stress_time))
            for container in containers
        ])

    def container_thread(self, container: ContainerConfig, max_stress_time: int) -> None:
        max_date = datetime.now() + timedelta(seconds=max_stress_time)
        execute_command(self.runtime.load_image(f"{self.scripts_folder}/temp_{container.name}.tar"))

        qtd_container = 0
        while datetime.now() < max_date:
            exec_runs = random.randint(self.config.min_lifecycle_runs, self.config.max_lifecycle_runs)
            results = run_in_threads([
                (f"stressload_{container.name}_{index}", self.container_stressload,
                 (container, container.host_port + index + PORT_OFFSET, index))
                for index in range(exec_runs)
            ])
            for result in results:
                qtd_container += result

        write_csv_row(f"{self.logs_path}/qtd_containers.csv", "thread;qtd_containers;date",
                      container.name, qtd_container, current_time())

    def container_stressload(self, container: ContainerConfig, host_port: int, run: int) -> int:
        """Cria/para/remove um lote de containers em sequencia; devolve quantos completaram o ciclo."""
        sleep_time = random.randint(container.min_container_wait_time, container.max_container_wait_time)
        qtt_containers = random.randint(self.config.min_qtt_containers, self.config.max_qtt_containers)

        image_name = f"temp_{container.name}"
        time.sleep(sleep_time)

        errors = 0
        for i in range(qtt_containers):
            container_name = f"temp_{image_name}-{run}-{i}"

            if self.lifecycle.start(container_name, image_name, host_port, container.port) is None:
                print(f"Could not start container, exiting {image_name} lifecycle")
                log_error(self.logs_path, "start", image_name, current_time())
                execute_command(self.runtime.force_remove(container_name), continue_if_error=False,
                                error_informative=False)
                errors += 1
                continue

            if self.using_containers_app_time:
                self.lifecycle.wait_app_time(container_name)

            self.lifecycle.stop(container_name, on_kill=lambda: self._on_kill(image_name))
            self.lifecycle.remove(
                container_name, on_force=lambda: log_error(self.logs_path, "force-remove", image_name, current_time()))
        return qtt_containers - errors

    def _on_kill(self, image_name: str) -> None:
        print("Killing container")
        log_error(self.logs_path, "kill", image_name, current_time())
