"""Runtime de containers (docker/podman): deteccao e montagem dos comandos."""
from src.utils import execute_command

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

    def run(self, container: str, image: str, host_port: int, port: int) -> str:
        return self.command(f"run --name {container} -td -p {host_port}:{port} --init {image}")

    def exists(self, container: str) -> str:
        return self.command(f"container inspect {container} > /dev/null 2>&1 && echo true")

    def read_app_time(self, container: str) -> str:
        return self.command(f"exec -i {container} sh -c \"test -e /root/log.txt && cat /root/log.txt\"")

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
        print("Both docker and podman are installed. Please, uninstall one of them.")
        raise SystemExit(1)

    if docker is None and podman is None:
        print("Neither docker nor podman is installed. Please, install one of them.")
        raise SystemExit(1)

    return ContainerRuntime("podman" if podman is not None else "docker")


DEFAULT_DISK_DEVICE = "vda"


def detect_root_disk() -> str:
    """Nome do disco fisico (ex.: vda, sda, nvme0n1) que contem "/", atravessando LVM/particoes."""
    disk = execute_command(
        'lsblk -nrso NAME,TYPE "$(findmnt -nvo SOURCE /)" | awk \'$2 == "disk" {print $1; exit}\'',
        continue_if_error=True, error_informative=False)
    if not disk:
        print(f"Could not detect the root disk, using {DEFAULT_DISK_DEVICE}")
        return DEFAULT_DISK_DEVICE
    return disk


def _lsb_release(flag: str, label: str) -> str:
    output = execute_command(f"lsb_release {flag}", continue_if_error=True, error_informative=False) or ""
    return output.replace("No LSB modules are available.", "").replace(label, "").strip()


def describe_environment(runtime: ContainerRuntime) -> str:
    system = _lsb_release("-i", "Distributor ID:")
    version = _lsb_release("-r", "Release:")
    software_version = execute_command(runtime.version(), continue_if_error=False, error_informative=True)

    return f""""
Software={runtime}
Software Version={software_version}
OS={system}
OS Version={version}
    """
