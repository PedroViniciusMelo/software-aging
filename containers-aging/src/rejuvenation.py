"""
Metodos de rejuvenescimento, aplicados ao fim de cada periodo de stress (rejuvenation.enabled).

Para trocar o metodo, mude rejuvenation.method no config.yaml. Para criar um novo:

    @register("meu_metodo")
    class MeuMetodo(RejuvenationMethod):
        \"\"\"Uma linha dizendo o que ele faz (aparece no log e no console).\"\"\"

        def apply(self) -> dict:
            self.run("systemctl restart docker")   # self.runtime, self.config, self.context...
            return {"o_que_fez": 1}                 # vai para logs/rejuvenation.csv (opcional)

O metodo roda na thread principal, entre o stress e o descanso, com o monitor de ciclo de vida
pausado (context.lifecycle_lock): ele nao encontra containers do monitoramento pela metade.
"""
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

from src.config import RejuvenationConfig
from src.runtime import ContainerRuntime
from src.utils import current_time, execute_command, execute_command_lines, log, write_csv_row

REJUVENATION_FILE = "rejuvenation.csv"
REJUVENATION_HEADER = "method;duration_s;details;date_time"
# rotulo que start_teastore.sh poe nos containers do TeaStore
TEASTORE_LABEL = re.compile(r"containers-aging\.role[=:]teastore")


@dataclass
class RejuvenationContext:
    runtime: ContainerRuntime
    config: RejuvenationConfig
    logs_path: str
    start_teastore: Callable[[], None]
    lifecycle_lock: threading.Lock


METHODS = {}


def register(name: str):
    def decorator(cls):
        cls.name = name
        METHODS[name] = cls
        return cls
    return decorator


def create_method(context: RejuvenationContext) -> "RejuvenationMethod":
    """Instancia o metodo do config; falha logo no inicio se o nome nao existir."""
    cls = METHODS.get(context.config.method)
    if cls is None:
        raise SystemExit(f"Unknown rejuvenation method '{context.config.method}'. "
                         f"Available: {', '.join(sorted(METHODS))}")
    return cls(context)


class RejuvenationMethod:
    name = ""

    def __init__(self, context: RejuvenationContext):
        self.context = context
        self.runtime = context.runtime
        self.config = context.config

    @property
    def description(self) -> str:
        return (self.__doc__ or self.name).strip().splitlines()[0]

    def run(self, command: str):
        return execute_command(command, continue_if_error=True, error_informative=False)

    def apply(self) -> dict:
        raise NotImplementedError

    def __call__(self) -> None:
        """Aplica o metodo (com o monitor de ciclo de vida pausado), mede e registra."""
        log.info("Applying rejuvenation: %s", self.name)
        started = time.perf_counter()
        with self.context.lifecycle_lock:
            details = self.apply() or {}
        duration = time.perf_counter() - started
        summary = ",".join(f"{key}={value}" for key, value in details.items())
        log.info("Rejuvenation %s done in %.1fs %s", self.name, duration, summary)
        write_csv_row(f"{self.context.logs_path}/{REJUVENATION_FILE}", REJUVENATION_HEADER,
                      self.name, round(duration, 3), summary, current_time())
        time.sleep(self.config.delay)


@register("cleanup_keep_teastore")
class CleanupKeepTeastore(RejuvenationMethod):
    """Remove containers, imagens e volumes sem uso, preservando tudo do TeaStore."""

    def _protected(self, text: str) -> bool:
        return bool(TEASTORE_LABEL.search(text)) or any(pattern in text for pattern in self.config.protected_images)

    def apply(self) -> dict:
        runtime = self.runtime
        # containers: "<id> <imagem> <rotulos>"; protegido pela imagem ou pelo rotulo do TeaStore
        rows = execute_command_lines(runtime.command("ps -a --format '{{.ID}} {{.Image}} {{.Labels}}'"),
                                     continue_if_error=True) or []
        containers = [row.split()[0] for row in rows if not self._protected(row)]
        if containers:
            ids = " ".join(containers)
            self.run(runtime.command(f"stop {ids}"))
            self.run(runtime.command(f"rm -v -f {ids}"))

        # imagens: so' as que nao sao do TeaStore. Sem -f: o runtime recusa remover uma imagem
        # que algum container ainda usa, entao as do TeaStore ficam mesmo se o nome nao casar.
        rows = execute_command_lines(runtime.command("image ls --format '{{.ID}} {{.Repository}}:{{.Tag}}'"),
                                     continue_if_error=True) or []
        images = []
        for row in rows:
            image_id, _, reference = row.partition(" ")
            if self._protected(reference):
                continue
            images.append(image_id if "<none>" in reference else reference)
        for image in images:
            self.run(runtime.remove_image(image))

        # volumes sem nenhum container (os do TeaStore estao em uso pelos containers preservados)
        self.run(runtime.command("volume prune -f"))
        return {"containers": len(containers), "images": len(images)}


@register("full_reset")
class FullReset(RejuvenationMethod):
    """Metodo antigo: apaga todos os containers, imagens e volumes (inclusive o TeaStore) e sobe o TeaStore de novo."""

    def apply(self) -> dict:
        for command in self.runtime.cleanup_commands():
            self.run(command)
        time.sleep(self.config.delay)
        self.context.start_teastore()
        return {}
