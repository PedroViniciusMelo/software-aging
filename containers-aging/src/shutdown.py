"""
Parada graciosa: o primeiro Ctrl+C (ou SIGTERM) pede para o experimento parar; o segundo forca.

Na parada graciosa, as threads de stress terminam o container atual (param e removem o que
subiram) e nao comecam outro, o monitor de ciclo de vida termina o container atual, a coleta
para e os resultados parciais sao salvos em results/<cenario>_<data>_interrupted.

Antes de o monitoramento comecar (limpeza, imagens, TeaStore) nao ha' nada para salvar: o
primeiro Ctrl+C ja' encerra na hora.

Os comandos externos rodam numa sessao propria (start_new_session em src/utils.py): o Ctrl+C
do terminal chega so' ao coletor, que decide o que fazer, e nao interrompe um "docker rm" ou o
stap no meio. Na parada forcada, o coletor encerra esses comandos ele mesmo.
"""
import os
import signal
import threading

from src import activity
from src.utils import log

requested = threading.Event()
_graceful = False


def install() -> None:
    """Instala o tratamento de SIGINT/SIGTERM (precisa ser chamado na thread principal)."""
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def enable_graceful() -> None:
    """A partir daqui ha' dados coletados: o primeiro Ctrl+C para graciosamente."""
    global _graceful
    _graceful = True


def _handler(signum, _frame):
    name = signal.Signals(signum).name
    if _graceful and not requested.is_set():
        requested.set()
        log.warning("%s received: stopping gracefully (finishing the current containers and saving the results). "
                    "Press Ctrl+C again to force.", name)
        return
    raise KeyboardInterrupt


def stopping() -> bool:
    return requested.is_set()


def wait(seconds: float) -> bool:
    """Espera `seconds` ou ate' a parada ser pedida. Devolve True se foi pedida."""
    return requested.wait(max(seconds, 0))


def kill_children() -> None:
    """Parada forcada: encerra os comandos em execucao (cada um lider do proprio grupo)."""
    for child in activity.running_children():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (AttributeError, OSError):
            pass
