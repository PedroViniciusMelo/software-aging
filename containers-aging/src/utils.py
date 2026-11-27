"""Utilitarios compartilhados: log, execucao de comandos, horario e escrita de CSV."""
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from src import activity

log = logging.getLogger("aging")

LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
LOG_FILE = "collector.log"
LOG_LEVELS = {"info": logging.INFO, "quiet": logging.WARNING}  # quiet = so' avisos e erros

ERRORS_FILE = "errors.csv"
ERRORS_HEADER = "event;container;date_time"

_write_lock = threading.Lock()


# ---- log ------------------------------------------------------------------------------------
def setup_logging(informative: bool = True) -> None:
    """Terminal. Avisos e erros aparecem sempre; os informativos sao opcionais."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LOG_FORMAT)
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.INFO if informative else logging.WARNING)


def attach_log_file(logs_path: str) -> None:
    """Grava tambem em logs/collector.log (copiado para results/ com o resto)."""
    handler = logging.FileHandler(f"{logs_path}/{LOG_FILE}")
    handler.setFormatter(LOG_FORMAT)
    log.addHandler(handler)


def set_log_level(name: str) -> bool:
    if name not in LOG_LEVELS:
        return False
    log.setLevel(LOG_LEVELS[name])
    return True


def log_level_name() -> str:
    level = log.getEffectiveLevel()
    return next((name for name, value in LOG_LEVELS.items() if value == level), logging.getLevelName(level))


# ---- CSV --------------------------------------------------------------------------------------
def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv_row(filename: str, header: str, *values) -> None:
    """Acrescenta uma linha (valores separados por ";"), escrevendo o cabecalho se o arquivo estiver vazio."""
    with _write_lock, open(filename, "a+") as file:
        if file.tell() == 0:
            file.write(f"{header}\n")
        file.write(";".join("" if value is None else str(value) for value in values) + "\n")


def log_error(logs_path: str, event: str, subject: str, date_time: str) -> None:
    write_csv_row(f"{logs_path}/{ERRORS_FILE}", ERRORS_HEADER, event, subject, date_time)


# ---- comandos ---------------------------------------------------------------------------------
class CommandError(SystemExit):
    """
    Um comando obrigatorio (continue_if_error=False) falhou. Herda de SystemExit: sem tratamento,
    encerra o programa na thread principal e so' a thread atual nas demais.
    """

    def __init__(self, command: str, return_code: int):
        super().__init__(return_code)
        self.command = command
        self.return_code = return_code


class _AccountedPopen(subprocess.Popen):
    """
    Popen que coleta o filho com wait4 em vez de waitpid, guardando o rusage dele: a CPU gasta
    pelo comando e por todos os processos que ele esperou (os de um pipeline do sh, por exemplo).
    _try_wait e' interno ao subprocess (POSIX); se nao existir, so' a contabilizacao fica desligada.
    """
    rusage = None

    def _try_wait(self, wait_flags):
        try:
            pid, status, rusage = os.wait4(self.pid, wait_flags)
        except ChildProcessError:  # mesmo tratamento do original
            return self.pid, 0
        if pid == self.pid:
            self.rusage = rusage
        return pid, status


_POPEN = _AccountedPopen if hasattr(os, "wait4") and hasattr(subprocess.Popen, "_try_wait") else subprocess.Popen


def _run(command: str, continue_if_error: bool, warn_on_error: bool) -> Optional[str]:
    # stdin=DEVNULL: o terminal fica livre para o console interativo.
    # start_new_session: o Ctrl+C do terminal chega so' ao coletor (src/activity.py), que para
    # graciosamente sem interromper um "docker rm" ou o stap no meio.
    process = _POPEN(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     start_new_session=True)
    activity.child_started(process.pid, command)
    try:
        output, error = process.communicate()
        return_code = process.wait()
    finally:
        activity.child_finished(process.pid, getattr(process, "rusage", None))

    if return_code != 0:
        if warn_on_error:
            log.warning("command failed (rc=%s): %s\n    $ %s", return_code,
                        error.decode("utf-8", "replace").strip(), command)
        if not continue_if_error:
            raise CommandError(command, return_code)
        return None
    return output.decode("utf-8", "replace").strip()


def execute_command(command: str, informative=False, continue_if_error=False, warn_on_error=True) -> Optional[str]:
    """
    Executa `command` no shell e devolve a saida (sem quebras de linha).

    :param informative: mostra a saida do comando no log.
    :param continue_if_error: se False, uma falha levanta CommandError; se True, devolve None.
    :param warn_on_error: registra a falha como aviso.
    """
    output = _run(command, continue_if_error, warn_on_error)
    if output is None:
        return None
    if informative and output:
        log.info("%s", output)
    return output.replace("\n", "")


def execute_command_lines(command: str, continue_if_error=False, warn_on_error=True) -> Optional[list]:
    """Como execute_command, mas devolve a saida como lista de linhas nao vazias."""
    output = _run(command, continue_if_error, warn_on_error)
    return None if output is None else [line for line in output.splitlines() if line.strip()]


def get_time(command: str, continue_if_error=False, warn_on_error=True) -> int:
    """Executa `command` (como execute_command) e devolve o tempo gasto em nanossegundos."""
    start_time = time.perf_counter_ns()
    execute_command(command, continue_if_error=continue_if_error, warn_on_error=warn_on_error)
    return time.perf_counter_ns() - start_time
