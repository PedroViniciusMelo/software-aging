"""Utilitarios compartilhados: execucao de comandos, horario e escrita de CSV."""
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

ERRORS_FILE = "errors.csv"
ERRORS_HEADER = "event;container;date_time"

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

    :param informative: imprime a saida do comando.
    :param continue_if_error: se False, uma falha levanta CommandError; se True, devolve None.
    :param error_informative: imprime a mensagem de erro quando o comando falha.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return_code = process.wait()

    if return_code != 0:
        if error_informative:
            print(f'\nERROR {current_time()}: {error.decode("utf-8").strip()}\nCOMMAND: ${command}')
        if not continue_if_error:
            raise CommandError(command, return_code)
        return None

    if informative:
        print(output.decode("utf-8").strip())
    return output.decode("utf-8").strip().replace("\n", "")


def get_time(command: str, informative=False, continue_if_error=False, error_informative=True) -> int:
    """Executa `command` (como execute_command) e devolve o tempo gasto em nanossegundos."""
    start_time = time.perf_counter_ns()
    execute_command(command, informative=informative, continue_if_error=continue_if_error,
                    error_informative=error_informative)
    return time.perf_counter_ns() - start_time
