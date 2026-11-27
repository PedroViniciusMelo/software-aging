"""Carga de estresse: cria e remove containers em paralelo durante o periodo de stress."""
import random
import threading
from datetime import datetime, timedelta

from src import activity, shutdown
from src.config import ContainerConfig, LifecycleConfig, StressloadConfig
from src.lifecycle import ContainerLifecycle
from src.runtime import ContainerRuntime
from src.utils import current_time, execute_command, log, log_error, write_csv_row


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
    def __init__(self, runtime: ContainerRuntime, config: StressloadConfig, scripts_folder: str, logs_path: str,
                 lifecycle: LifecycleConfig = None):
        self.runtime = runtime
        self.config = config
        self.scripts_folder = scripts_folder
        self.logs_path = logs_path
        self.lifecycle = ContainerLifecycle(runtime, lifecycle)

    def run(self, containers: list, max_stress_time: int) -> None:
        """Estressa todos os containers em paralelo por max_stress_time segundos (ou ate' a parada)."""
        activity.waiting(max_stress_time, "waiting stressload threads")
        run_in_threads([
            (f"stressload_{container.name}", self.container_thread, (container, max_stress_time))
            for container in containers
        ])

    def container_thread(self, container: ContainerConfig, max_stress_time: int) -> None:
        max_date = datetime.now() + timedelta(seconds=max_stress_time)
        execute_command(self.runtime.load_image(f"{self.scripts_folder}/temp_{container.name}.tar"))

        qtd_container = 0
        while datetime.now() < max_date and not shutdown.stopping():
            exec_runs = random.randint(self.config.min_lifecycle_runs, self.config.max_lifecycle_runs)
            results = run_in_threads([
                (f"stressload_{container.name}_{index}", self.container_stressload,
                 (container, container.host_port + index + self.config.port_offset, index))
                for index in range(exec_runs)
            ])
            for result in results:
                qtd_container += result or 0  # None: o lote terminou com erro inesperado
            log.debug("Stressload %s: %d parallel batches done, %d containers so far",
                      container.name, exec_runs, qtd_container)
        log.info("Stressload %s finished%s: %d containers created and removed", container.name,
                 " (stop requested)" if shutdown.stopping() else "", qtd_container)

        write_csv_row(f"{self.logs_path}/qtd_containers.csv", "thread;qtd_containers;date",
                      container.name, qtd_container, current_time())

    def container_stressload(self, container: ContainerConfig, host_port: int, run: int) -> int:
        """Cria/para/remove um lote de containers em sequencia; devolve quantos completaram o ciclo."""
        sleep_time = random.randint(container.min_container_wait_time, container.max_container_wait_time)
        qtt_containers = random.randint(self.config.min_qtt_containers, self.config.max_qtt_containers)

        image_name = f"temp_{container.name}"
        activity.waiting(sleep_time, "waiting before batch")
        if shutdown.wait(sleep_time):
            return 0

        errors = 0
        done = 0
        for i in range(qtt_containers):
            if shutdown.stopping():  # termina o container atual, mas nao comeca outro
                break
            container_name = f"{image_name}-{run}-{i}"
            done += 1

            if self.lifecycle.start(container_name, image_name, host_port, container.port, container.env) is None:
                log.warning("Could not start container %s, skipping it", container_name)
                log_error(self.logs_path, "start", image_name, current_time())
                execute_command(self.runtime.force_remove(container_name), continue_if_error=True,
                                error_informative=False)
                errors += 1
                continue

            self.lifecycle.stop(container_name, on_kill=lambda: self._on_kill(image_name))
            self.lifecycle.remove(
                container_name, on_force=lambda: log_error(self.logs_path, "force-remove", image_name, current_time()))
        return done - errors

    def _on_kill(self, image_name: str) -> None:
        log.warning("Container of %s did not stop; killing it", image_name)
        log_error(self.logs_path, "kill", image_name, current_time())
