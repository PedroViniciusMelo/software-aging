"""Leitura e validacao do config.yaml em objetos tipados."""
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

# Raiz do projeto (pasta do main.py): scripts, imagens .tar e logs ficam aqui.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DEFAULT_CONFIG_PATH = f"{PROJECT_ROOT}/config.yaml"


@dataclass
class GeneralConfig:
    runs: int
    apply_rejuvenation: bool
    plot_results: bool = True  # ao fim, copia logs/ para results/<cenario> e gera os graficos
    scripts_folder: str = PROJECT_ROOT  # detectado; so' defina no config.yaml para usar outra pasta


@dataclass
class SetupConfig:
    rebuild_images: bool


@dataclass
class MonitoringConfig:
    sleep_time: int
    sleep_time_container_metrics: int
    using_containers_app_time: bool
    disk_device: Optional[str] = None  # disco do iostat; None = detecta o disco que contem a raiz "/"


@dataclass
class EnergyConfig:
    enable_rapl: bool = False
    sleep_time_rapl: int = 1
    rapl_zones: Optional[list] = None
    enable_smart_plug: bool = False
    sleep_time_smart_plug: int = 300
    smart_plug_ip: str = ""
    smart_plug_username: str = ""
    smart_plug_password: str = ""


@dataclass
class StressloadConfig:
    stressload_first: bool
    max_stress_time: int
    wait_after_stress: int
    min_lifecycle_runs: int
    max_lifecycle_runs: int
    min_qtt_containers: int
    max_qtt_containers: int


@dataclass
class ContainerConfig:
    name: str
    version: str
    host_port: int
    port: int
    min_container_wait_time: int
    max_container_wait_time: int


@dataclass
class Config:
    general: GeneralConfig
    setup: SetupConfig
    monitoring: MonitoringConfig
    stressload: StressloadConfig
    containers: list
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    @property
    def logs_path(self) -> str:
        return f"{self.general.scripts_folder}/logs"


def _known_keys(cls, data: dict) -> dict:
    """Secoes opcionais (energy) ignoram chaves desconhecidas, avisando (ex.: as antigas tomada_*)."""
    names = {item.name for item in fields(cls)}
    unknown = sorted(key for key in data if key not in names)
    if unknown:
        print(f"Aviso: chaves desconhecidas ignoradas em {cls.__name__}: {', '.join(unknown)}")
    return {key: value for key, value in data.items() if key in names}


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, "r") as yml_file:
        data = yaml.safe_load(yml_file)

    return Config(
        general=GeneralConfig(**data["general"]),
        setup=SetupConfig(**data["setup"]),
        monitoring=MonitoringConfig(**data["monitoring"]),
        stressload=StressloadConfig(**data["stressload"]),
        containers=[ContainerConfig(**container) for container in data["containers"]],
        energy=EnergyConfig(**_known_keys(EnergyConfig, data.get("energy") or {})),
    )
