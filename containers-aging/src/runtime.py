"""
Runtime de containers (docker/podman): deteccao, montagem dos comandos, nome do cenario e as
etapas do ciclo de vida de um container (usadas pelo monitoramento e pela carga de estresse).
"""
import re
import shlex
import time
from typing import Callable, Optional

from src.config import LifecycleConfig
from src.utils import CommandError, execute_command, get_time, log


class ContainerRuntime:
    """Monta as linhas de comando do runtime; quem executa e' o chamador."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def command(self, args: str) -> str:
        return f"{self.name} {args}"

    def run(self, container: str, image: str, host_port: int, port: int, env: dict = None) -> str:
        env_args = "".join(f" -e {shlex.quote(f'{key}={value}')}" for key, value in (env or {}).items())
        return self.command(f"run --name {container} -td -p {host_port}:{port}{env_args} --init {image}")

    def exists(self, container: str) -> str:
        return self.command(f"container inspect {container} > /dev/null 2>&1 && echo true")

    def stop(self, container: str, timeout: int = None) -> str:
        return self.command(f"stop {container}" if timeout is None else f"stop -t {timeout} {container}")

    def kill(self, container: str) -> str:
        return self.command(f"kill {container}")

    def remove(self, container: str) -> str:
        return self.command(f"rm -v {container}")

    def force_remove(self, container: str) -> str:
        return self.command(f"rm -v -f {container}")

    def load_image(self, tar_path: str) -> str:
        return self.command(f"load -i {tar_path} -q")

    def remove_image(self, image: str) -> str:
        return self.command(f"rmi {image}")

    def cleanup_commands(self) -> list:
        """Para e apaga todos os containers, imagens e volumes."""
        sw = self.name
        return [
            f"{sw} stop $({sw} ps -aq)",
            f"{sw} rm $({sw} ps -aq)",
            f"{sw} rmi $({sw} image ls -aq)",
            f"{sw} container prune -f",
            f"{sw} image prune -f",
            f"{sw} volume prune -f",
            f"{sw} system prune -a -f --volumes",
        ]

    def version_text(self) -> str:
        """Ex.: "Docker version 27.2.0, build 3ab4256"."""
        return execute_command(self.command("-v"), continue_if_error=True, warn_on_error=False) or ""

    def version(self) -> str:
        """Versao numerica, ex.: "27.2.0"."""
        match = re.search(r"\d+(?:\.\d+)+", self.version_text())
        return match.group(0) if match else "unknown"


def detect_runtime() -> ContainerRuntime:
    """Exige exatamente um entre docker e podman instalado."""
    found = [name for name in ("docker", "podman")
             if execute_command(f"which {name}", continue_if_error=True, warn_on_error=False) is not None]
    if len(found) != 1:
        log.error("Both docker and podman are installed. Please, uninstall one of them." if found else
                  "Neither docker nor podman is installed. Please, install one of them.")
        raise SystemExit(1)
    return ContainerRuntime(found[0])


def read_os_release(path: str = "/etc/os-release") -> dict:
    info = {}
    try:
        with open(path) as file:
            for line in file:
                key, sep, value = line.strip().partition("=")
                if sep:
                    info[key] = value.strip('"')
    except FileNotFoundError:
        pass
    return info


def scenario_name(runtime: ContainerRuntime) -> str:
    """Nome do cenario no formato lido por plot/scenario.py, ex.: debian13_docker_27.2.0."""
    os_info = read_os_release()
    return f"{os_info.get('ID', 'linux')}{os_info.get('VERSION_ID', '')}_{runtime}_{runtime.version()}"


class ContainerLifecycle:
    """Cada metodo executa uma etapa e devolve o tempo gasto (ns), com o fallback correspondente."""

    def __init__(self, runtime: ContainerRuntime, config: LifecycleConfig = None):
        self.runtime = runtime
        self.config = config or LifecycleConfig()

    def exists(self, container: str) -> bool:
        return execute_command(self.runtime.exists(container), continue_if_error=True,
                               warn_on_error=False) is not None

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
            execute_command(self.runtime.force_remove(container), continue_if_error=False, warn_on_error=False)

        failures = 0
        while failures <= self.config.start_max_retries:
            time.sleep(self.config.start_retry_delay)
            try:
                start_time = get_time(command, continue_if_error=False, warn_on_error=False)
                if self.exists(container):
                    return start_time
            except CommandError:
                failures += 1
        return None

    def stop(self, container: str, on_kill: Callable[[], None]) -> int:
        """Para o container; se nao parar, chama on_kill() e mata o container."""
        try:
            return get_time(self.runtime.stop(container, self.config.stop_timeout), continue_if_error=False,
                            warn_on_error=False)
        except CommandError:
            on_kill()
            return get_time(self.runtime.kill(container), continue_if_error=True, warn_on_error=False)

    def remove(self, container: str, on_force: Callable[[], None]) -> int:
        """Remove o container; se falhar, chama on_force() e remove a forca."""
        try:
            return get_time(self.runtime.remove(container), continue_if_error=False, warn_on_error=False)
        except CommandError:
            log.warning("Could not remove container %s, forcing removal", container)
            on_force()
            return get_time(self.runtime.force_remove(container), continue_if_error=False, warn_on_error=True)
