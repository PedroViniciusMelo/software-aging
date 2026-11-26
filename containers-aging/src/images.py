"""Preparacao das imagens (.tar) usadas pelo monitoramento e pela carga de estresse."""
from src.runtime import ContainerRuntime
from src.utils import execute_command, log


def build_images(runtime: ContainerRuntime, containers: list, scripts_folder: str, rebuild_images: bool) -> None:
    """
    Baixa cada imagem e salva duas copias em scripts_folder:
      <name>.tar       -> usada pelo monitor de ciclo de vida (imagem <name>:latest)
      temp_<name>.tar  -> usada pela carga de estresse (imagem temp_<name>:latest)
    """
    if not rebuild_images:
        log.info("Rebuilding images is disabled. Skipping...")
        return

    for container in containers:
        name, version = container.name, container.version
        log.info("Building image %s:%s", name, version)
        for command in [
            f"pull docker.io/{name}:{version}",
            f"tag {name}:{version} {name}:latest",
            f"tag {name}:{version} temp_{name}:latest",
            f"save {name}:latest > {scripts_folder}/{name}.tar",
            f"save temp_{name}:latest > {scripts_folder}/temp_{name}.tar",
            f"rmi {name}:latest",
            f"rmi temp_{name}:latest",
            f"rmi {name}:{version}",
        ]:
            execute_command(runtime.command(command))
