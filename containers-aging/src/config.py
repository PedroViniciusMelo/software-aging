"""Leitura do config.yaml em objetos tipados."""
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
    plot_results: bool = True  # ao fim, copia logs/ para results/<cenario> e gera os graficos
    scripts_folder: str = PROJECT_ROOT  # detectado; so' defina no config.yaml para usar outra pasta


@dataclass
class SetupConfig:
    rebuild_images: bool
    teastore_start_interval: int = 60  # espera (s) entre subir cada um dos 7 servicos do TeaStore


@dataclass
class ProcessesConfig:
    """
    Processos do host monitorados, pelo nome exato (/proc/<pid>/comm, como "pgrep -x"): a lista
    do runtime em uso seguida da comum, na ordem, sem repetir.
    """
    common: list = field(default_factory=lambda: ["java", "postgres", "beam.smp", "initdb", "mysqld"])
    docker: list = field(default_factory=lambda: ["docker", "dockerd", "containerd", "containerd-shim", "runc",
                                                  "docker-proxy"])
    podman: list = field(default_factory=lambda: ["podman", "conmon", "cron", "crun", "systemd"])

    def for_runtime(self, runtime: str) -> list:
        return list(dict.fromkeys((getattr(self, runtime, None) or []) + (self.common or [])))


@dataclass
class MonitoringConfig:
    sleep_time: int
    sleep_time_container_metrics: int
    disk_device: Optional[str] = None  # disco do disk_io.csv; None = detecta o disco que contem a raiz "/"
    stap_max_attempts: int = 11  # quantas vezes (re)iniciar o systemtap se ele sair
    stap_retry_delay: int = 120  # espera (s) antes de reiniciar o systemtap
    stap_overload_threshold: int = 650000000  # -D STP_OVERLOAD_THRESHOLD (ns de probe por intervalo)
    stap_overload_interval: int = 1000000000  # -D STP_OVERLOAD_INTERVAL (ns)
    stop_grace: int = 3  # espera (s) no fim para as leituras em andamento terminarem
    lifecycle_stop_timeout: int = 120  # espera maxima (s) no fim pelo container que o monitor de ciclo de vida mede
    processes: ProcessesConfig = field(default_factory=ProcessesConfig)


@dataclass
class LifecycleConfig:
    start_max_retries: int = 5  # novas tentativas de subir um container que falhou
    start_retry_delay: int = 2  # espera (s) entre essas tentativas
    stop_timeout: Optional[int] = None  # "stop -t" (s) antes de o runtime matar o container; None = padrao (10 s)


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
class RejuvenationConfig:
    enabled: bool = False  # aplica o metodo ao fim de cada periodo de stress
    method: str = "cleanup_keep_teastore"  # chave de METHODS em src/rejuvenation.py
    delay: int = 5  # espera (s) depois de aplicar o metodo
    # imagens (substring do nome) cujos containers, imagens e volumes o rejuvenescimento preserva
    protected_images: list = field(default_factory=lambda: ["descartesresearch/teastore"])


@dataclass
class LoggingConfig:
    informative: bool = True  # mensagens de progresso e eventos; false = so' avisos e erros
    to_file: bool = True  # tambem grava logs/collector.log
    console: bool = True  # terminal interativo (threads, status); ignorado se a entrada nao for um terminal


@dataclass
class StressloadConfig:
    stressload_first: bool
    max_stress_time: int
    wait_after_stress: int
    min_lifecycle_runs: int
    max_lifecycle_runs: int
    min_qtt_containers: int
    max_qtt_containers: int
    port_offset: int = 100  # cada lote paralelo usa a porta host_port + port_offset + indice do lote


@dataclass
class ContainerConfig:
    name: str
    version: str
    host_port: int
    port: int
    min_container_wait_time: int
    max_container_wait_time: int
    env: dict = field(default_factory=dict)  # variaveis de ambiente (docker run -e)


@dataclass
class Config:
    general: GeneralConfig
    setup: SetupConfig
    monitoring: MonitoringConfig
    stressload: StressloadConfig
    containers: list
    energy: EnergyConfig
    rejuvenation: RejuvenationConfig
    logging: LoggingConfig
    lifecycle: LifecycleConfig

    @property
    def logs_path(self) -> str:
        return f"{self.general.scripts_folder}/logs"


def _build(cls, data: Optional[dict]):
    """Instancia a secao; chaves desconhecidas (ex.: removidas em versoes novas) so' geram aviso."""
    data = dict(data or {})
    names = {item.name for item in fields(cls)}
    unknown = sorted(key for key in data if key not in names)
    if unknown:
        print(f"Aviso: chaves desconhecidas ignoradas em {cls.__name__}: {', '.join(unknown)}")
    return cls(**{key: value for key, value in data.items() if key in names})


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, "r") as yml_file:
        data = yaml.safe_load(yml_file)

    monitoring = dict(data["monitoring"])
    processes = _build(ProcessesConfig, monitoring.pop("processes", None))
    return Config(
        general=_build(GeneralConfig, data["general"]),
        setup=_build(SetupConfig, data["setup"]),
        monitoring=_build(MonitoringConfig, {**monitoring, "processes": processes}),
        stressload=_build(StressloadConfig, data["stressload"]),
        containers=[ContainerConfig(**container) for container in data["containers"]],
        energy=_build(EnergyConfig, data.get("energy")),
        rejuvenation=_build(RejuvenationConfig, data.get("rejuvenation")),
        logging=_build(LoggingConfig, data.get("logging")),
        lifecycle=_build(LifecycleConfig, data.get("lifecycle")),
    )
