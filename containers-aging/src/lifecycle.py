"""Etapas do ciclo de vida de um container, usadas pelo monitoramento e pela carga de estresse."""
import time
from typing import Callable, Optional

from src.config import LifecycleConfig
from src.runtime import ContainerRuntime
from src.utils import CommandError, execute_command, get_time, log


class ContainerLifecycle:
    """Cada metodo executa uma etapa e devolve o tempo gasto (ns), com o fallback correspondente."""

    def __init__(self, runtime: ContainerRuntime, config: LifecycleConfig = None):
        self.runtime = runtime
        self.config = config or LifecycleConfig()

    def exists(self, container: str) -> bool:
        return execute_command(self.runtime.exists(container), continue_if_error=True,
                               error_informative=False) is not None

    def start(self, container: str, image: str, host_port: int, port: int, env: dict = None) -> Optional[int]:
        """
        Sobe o container. Se falhar, remove o que sobrou e tenta de novo a cada
        start_retry_delay segundos. Devolve None se desistir apos start_max_retries falhas.
        """
        command = self.runtime.run(container, image, host_port, port, env)
        try:
            return get_time(command)
        except CommandError:
            log.warning("Could not start container %s, retrying", container)

        if self.exists(container):
            execute_command(self.runtime.force_remove(container), continue_if_error=False, error_informative=False)

        failures = 0
        while failures <= self.config.start_max_retries:
            time.sleep(self.config.start_retry_delay)
            try:
                start_time = get_time(command, continue_if_error=False, error_informative=False)
                if self.exists(container):
                    return start_time
            except CommandError:
                failures += 1
        return None

    def stop(self, container: str, on_kill: Callable[[], None]) -> int:
        """Para o container; se nao parar, chama on_kill() e mata o container."""
        try:
            return get_time(self.runtime.stop(container, self.config.stop_timeout), continue_if_error=False,
                            error_informative=False)
        except CommandError:
            on_kill()
            return get_time(self.runtime.kill(container), continue_if_error=True, error_informative=False)

    def remove(self, container: str, on_force: Callable[[], None]) -> int:
        """Remove o container; se falhar, chama on_force() e remove a forca."""
        try:
            return get_time(self.runtime.remove(container), continue_if_error=False, error_informative=False)
        except CommandError:
            log.warning("Could not remove container %s, forcing removal", container)
            on_force()
            return get_time(self.runtime.force_remove(container), continue_if_error=False, error_informative=True)
