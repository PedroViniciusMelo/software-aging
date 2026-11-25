"""
Informacoes do cenario a partir do nome da pasta de logs, no formato

    <sistema><versao>_<runtime>_<versao do runtime>[_<sufixo livre>]

por exemplo: debian12_docker_27.2.0, ubuntu24.04_podman_5.1.2, fedora45_docker_27.2.0.
Nada e' deduzido alem do que esta' no nome: um nome fora do formato e' "desconhecido".
"""
import re

SYSTEM_NAMES = {"debian": "Debian", "ubuntu": "Ubuntu", "fedora": "Fedora"}
RUNTIME_NAMES = {"docker": "Docker", "podman": "Podman"}

_FOLDER = re.compile(
    r"^(?P<system>[a-z]+?)(?P<system_version>\d+(?:\.\d+)*)?"
    r"_(?P<runtime>" + "|".join(RUNTIME_NAMES) + r")_v?(?P<version>\d+(?:\.\d+)*)(?:_|$)"
)


def extract_categorical_info(folder_name, unknown=(None, None, None)):
    """
    Devolve (sistema, software, versao), ex.: ("Debian 12", "Docker", "27.2.0").
    `unknown` e' devolvido quando o nome nao segue o formato.
    """
    match = _FOLDER.match(folder_name.lower())
    if not match:
        return unknown

    system = SYSTEM_NAMES.get(match["system"], match["system"].capitalize())
    if match["system_version"]:
        system = f"{system} {match['system_version']}"
    return system, RUNTIME_NAMES[match["runtime"]], match["version"]
