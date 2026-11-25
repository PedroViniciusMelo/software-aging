"""Informacoes do cenario a partir do nome da pasta de logs (ex.: debian12_docker_novo)."""


def extract_categorical_info(folder_name, unknown=(None, None, None)):
    """
    Devolve (sistema, software, versao). `unknown` define o valor usado para cada campo
    quando o nome da pasta nao identifica o sistema / o software (versao acompanha o software).
    """
    name = folder_name.lower()
    unknown_system, unknown_software, unknown_version = unknown

    if "debian" in name:
        system = "Debian 12"
    elif "ubuntu22" in name:
        system = "Ubuntu 22.04"
    elif "ubuntu24" in name:
        system = "Ubuntu 24.04"
    else:
        system = unknown_system

    if "docker" in name:
        software = "Docker"
        version = "27.2" if "novo" in name else "23"
    elif "podman" in name:
        software = "Podman"
        version = "5.1.3"
    else:
        software, version = unknown_software, unknown_version

    return system, software, version
