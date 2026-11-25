"""Leitura e validacao do config.yaml em objetos tipados."""
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass
class GeneralConfig:
    runs: int
    scripts_folder: str
    apply_rejuvenation: bool


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
    enable_tomada: bool = False
    sleep_time_tomada: int = 300
    tomada_ip: str = ""
    tomada_username: str = ""
    tomada_password: str = ""


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
    """Secoes opcionais (energy) ignoram chaves desconhecidas, como antes."""
    names = {item.name for item in fields(cls)}
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
