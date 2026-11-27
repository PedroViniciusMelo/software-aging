"""Runtime de containers (docker/podman): deteccao e montagem dos comandos."""
import re
import shlex

from src.utils import execute_command, log

# Processos do host monitorados (pidstat) para cada runtime, na ordem em que as threads sobem.
MONITORED_PROCESSES = {
    "docker": ['docker', 'dockerd', 'containerd', 'containerd-shim', 'runc', 'docker-proxy', 'java', 'postgres',
               'beam.smp', 'initdb', 'mysqld'],
    "podman": ['podman', 'conmon', 'cron', 'crun', 'systemd', 'java', 'postgres', 'beam.smp', 'initdb', 'mysqld'],
}


class ContainerRuntime:
    """Monta as linhas de comando do runtime; quem executa e' o chamador."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    @property
    def monitored_processes(self) -> list:
        return MONITORED_PROCESSES.get(self.name, [])

    def command(self, args: str) -> str:
        return f"{self.name} {args}"

    def run(self, container: str, image: str, host_port: int, port: int, env: dict = None) -> str:
        env_args = "".join(f" -e {shlex.quote(f'{key}={value}')}" for key, value in (env or {}).items())
        return self.command(f"run --name {container} -td -p {host_port}:{port}{env_args} --init {image}")

    def exists(self, container: str) -> str:
        return self.command(f"container inspect {container} > /dev/null 2>&1 && echo true")

    def stop(self, container: str) -> str:
        return self.command(f"stop {container}")

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

    def version(self) -> str:
        return self.command("-v")

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


def detect_runtime() -> ContainerRuntime:
    """Exige exatamente um entre docker e podman instalado."""
    docker = execute_command("which docker", continue_if_error=True, error_informative=False)
    podman = execute_command("which podman", continue_if_error=True, error_informative=False)

    if docker is not None and podman is not None:
        log.error("Both docker and podman are installed. Please, uninstall one of them.")
        raise SystemExit(1)

    if docker is None and podman is None:
        log.error("Neither docker nor podman is installed. Please, install one of them.")
        raise SystemExit(1)

    return ContainerRuntime("podman" if podman is not None else "docker")


DEFAULT_DISK_DEVICE = "vda"


def detect_root_disk() -> str:
    """Nome do disco fisico (ex.: vda, sda, nvme0n1) que contem "/", atravessando LVM/particoes."""
    disk = execute_command(
        'lsblk -nrso NAME,TYPE "$(findmnt -nvo SOURCE /)" | awk \'$2 == "disk" {print $1; exit}\'',
        continue_if_error=True, error_informative=False)
    if not disk:
        log.warning("Could not detect the root disk, using %s", DEFAULT_DISK_DEVICE)
        return DEFAULT_DISK_DEVICE
    return disk


def read_os_release(path: str = "/etc/os-release") -> dict:
    """Campos do /etc/os-release (presente em toda distro; o lsb_release nem sempre esta' instalado)."""
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


def runtime_version(runtime: ContainerRuntime) -> str:
    """Versao numerica do runtime, ex.: "Docker version 27.2.0, build 3ab4256" -> "27.2.0"."""
    output = execute_command(runtime.version(), continue_if_error=True, error_informative=False) or ""
    match = re.search(r"\d+(?:\.\d+)+", output)
    return match.group(0) if match else "unknown"


def scenario_name(runtime: ContainerRuntime) -> str:
    """Nome do cenario no formato lido por plot/scenario.py, ex.: debian13_docker_27.2.0."""
    os_info = read_os_release()
    system = os_info.get("ID", "linux") + os_info.get("VERSION_ID", "")
    return f"{system}_{runtime}_{runtime_version(runtime)}"


def describe_environment(runtime: ContainerRuntime) -> str:
    os_info = read_os_release()
    software_version = execute_command(runtime.version(), continue_if_error=False, error_informative=True)

    return f"""
Software={runtime}
Software Version={software_version}
OS={os_info.get("ID", "").capitalize()}
OS Version={os_info.get("VERSION_ID", "")}
    """
