"""Utilitarios compartilhados: execucao de comandos, horario e escrita de CSV."""
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

from src import activity

ERRORS_FILE = "errors.csv"
ERRORS_HEADER = "event;container;date_time"

log = logging.getLogger("aging")

_write_lock = threading.Lock()


class CommandError(SystemExit):
    """
    Um comando obrigatorio (continue_if_error=False) falhou.

    Herda de SystemExit de proposito, preservando o comportamento original (que chamava
    exit()): sem tratamento, encerra o programa quando acontece na thread principal e
    encerra so' a thread atual nas demais. Quem quiser se recuperar captura CommandError.
    """

    def __init__(self, command: str, return_code: int):
        super().__init__(return_code)
        self.command = command
        self.return_code = return_code


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_to_file(filename: str, header: str, content: str) -> None:
    """Acrescenta `content` ao arquivo, escrevendo `header` antes se o arquivo estiver vazio."""
    with _write_lock, open(filename, "a+") as file:
        if file.tell() == 0:
            file.write(f"{header}\n")
        file.write(f"{content}\n")


def write_csv_row(filename: str, header: str, *values) -> None:
    write_to_file(filename, header, ";".join(str(value) for value in values))


def log_error(logs_path: str, event: str, subject: str, date_time: str) -> None:
    write_csv_row(f"{logs_path}/{ERRORS_FILE}", ERRORS_HEADER, event, subject, date_time)


def execute_command(command: str, informative=False, continue_if_error=False, error_informative=True) -> Optional[str]:
    """
    Executa `command` no shell e devolve a saida (sem quebras de linha).

    :param informative: mostra a saida do comando no log informativo.
    :param continue_if_error: se False, uma falha levanta CommandError; se True, devolve None.
    :param error_informative: registra a falha como aviso (senao, so' no log de depuracao).
    """
    output = _run(command, continue_if_error, error_informative)
    if output is None:
        return None
    if informative and output:
        log.info("%s", output)
    return output.replace("\n", "")


def execute_command_lines(command: str, continue_if_error=False, error_informative=True) -> Optional[list]:
    """Como execute_command, mas devolve a saida como lista de linhas nao vazias."""
    output = _run(command, continue_if_error, error_informative)
    return None if output is None else [line for line in output.splitlines() if line.strip()]


class _AccountedPopen(subprocess.Popen):
    """
    Popen que coleta o filho com wait4 em vez de waitpid, guardando o rusage dele: a CPU gasta
    pelo comando e por todos os processos que ele esperou (os de um pipeline do sh, por exemplo).
    Custa o mesmo que o waitpid. _try_wait e' interno ao subprocess (POSIX, Python 3.3+); se nao
    existir, a contabilizacao por comando fica desligada e o resto funciona igual.
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


def _run(command: str, continue_if_error: bool, error_informative: bool) -> Optional[str]:
    # stdin=DEVNULL: nenhum comando le' o terminal, que fica livre para o console interativo.
    # start_new_session: o Ctrl+C do terminal nao chega aos comandos, so' ao coletor (src/shutdown.py),
    # que para graciosamente sem interromper um "docker rm" ou o stap no meio.
    started = time.perf_counter()
    process = _POPEN(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     start_new_session=True)
    activity.child_started(process.pid, command)
    try:
        output, error = process.communicate()
        return_code = process.wait()
    finally:
        cpu = activity.child_finished(process.pid, getattr(process, "rusage", None))

    if log.isEnabledFor(logging.DEBUG):
        log.debug("rc=%s wall=%.3fs cpu=%s $ %s", return_code, time.perf_counter() - started,
                  "?" if cpu is None else f"{cpu:.3f}s", command)

    if return_code != 0:
        level = logging.WARNING if error_informative else logging.DEBUG
        if log.isEnabledFor(level):
            log.log(level, "command failed (rc=%s): %s\n    $ %s", return_code,
                    error.decode("utf-8", "replace").strip(), command)
        if not continue_if_error:
            raise CommandError(command, return_code)
        return None
    return output.decode("utf-8", "replace").strip()


def get_time(command: str, informative=False, continue_if_error=False, error_informative=True) -> int:
    """Executa `command` (como execute_command) e devolve o tempo gasto em nanossegundos."""
    start_time = time.perf_counter_ns()
    execute_command(command, informative=informative, continue_if_error=continue_if_error,
                    error_informative=error_informative)
    return time.perf_counter_ns() - start_time
