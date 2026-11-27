"""
Quem esta' fazendo o que: a qual parte do sistema cada thread pertence, os comandos externos em
andamento, o que cada thread esta' esperando e quanto de CPU cada parte gastou.

Manter isso e' barato: algumas operacoes de dicionario por comando executado (que ja' custa um
fork/exec) e por espera. As leituras de /proc so' acontecem quando alguem pede: o console
interativo, sob demanda, ou o monitor de custo, uma vez por intervalo.

Custo de CPU de uma parte = CPU das suas threads Python (/proc/self/task/<tid>/stat)
                          + CPU dos comandos que elas executaram (rusage do wait4, que inclui
                            os netos: pipelines do sh, staprun/stapio do stap...)
                          + CPU, ate' agora, dos comandos ainda em execucao (ex.: o stap).

Parada graciosa: o primeiro Ctrl+C (ou SIGTERM) pede para o experimento parar; o segundo forca.
Na parada graciosa, as threads de stress terminam o container atual (param e removem o que
subiram) e nao comecam outro, o monitor de ciclo de vida termina o container atual, a coleta
para e os resultados parciais sao salvos em results/<cenario>_<data>_interrupted. Antes de o
monitoramento comecar nao ha' nada para salvar: o primeiro Ctrl+C ja' encerra na hora.
Os comandos externos rodam numa sessao propria (src/utils.py): o Ctrl+C do terminal chega so'
ao coletor, que nao interrompe um "docker rm" ou o stap no meio; na parada forcada, ele mesmo
encerra esses comandos.
"""
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Optional

# (prefixo do nome da thread, parte, papel). O papel separa o custo do monitoramento
# ("monitoring") da carga do experimento ("workload") e do controle ("control").
PARTS = [
    ("systemtap", "systemtap", "monitoring"),
    ("container_lifecycle", "container_lifecycle", "monitoring"),
    ("machine", "machine", "monitoring"),  # recursos da maquina e processos
    ("energy_rapl", "energy_rapl", "monitoring"),
    ("energy_smart_plug", "energy_smart_plug", "monitoring"),
    ("monitoring_cost", "monitoring_cost", "monitoring"),
    ("stressload", "stressload", "workload"),
    ("MainThread", "experiment", "control"),
    ("console", "console", "control"),
]
ROLES = {part: role for _, part, role in PARTS}
PART_ORDER = [part for _, part, _ in PARTS] + ["other"]

try:
    CLK_TCK = os.sysconf("SC_CLK_TCK")
except (AttributeError, ValueError, OSError):  # fora do Linux
    CLK_TCK = 100


def part_of(thread_name: str) -> str:
    for prefix, part, _ in PARTS:
        if thread_name.startswith(prefix):
            return part
    return "other"


def role_of(part: str) -> str:
    return ROLES.get(part, "control")


# ---- comandos externos -----------------------------------------------------------------
class Child:
    __slots__ = ("pid", "command", "part", "thread_name", "started")

    def __init__(self, pid: int, command: str, thread_name: str):
        self.pid = pid
        self.command = command
        self.thread_name = thread_name
        self.part = part_of(thread_name)
        self.started = time.monotonic()


_lock = threading.Lock()
_running = {}  # pid -> Child
_by_thread = {}  # ident da thread -> Child em execucao (cada thread roda um comando por vez)
_finished_cpu = defaultdict(float)  # parte -> CPU (s) dos comandos que ja' terminaram
_finished_count = defaultdict(int)  # parte -> quantidade de comandos que ja' terminaram
_waiting = {}  # ident da thread -> (motivo, instante monotonic em que a espera acaba)


def child_started(pid: int, command: str) -> None:
    thread = threading.current_thread()
    child = Child(pid, command, thread.name)
    with _lock:
        _running[pid] = child
        _by_thread[thread.ident] = child


def child_finished(pid: int, rusage) -> Optional[float]:
    """Contabiliza o comando que terminou; devolve a CPU (s) gasta por ele, se conhecida."""
    cpu = None if rusage is None else rusage.ru_utime + rusage.ru_stime
    with _lock:
        child = _running.pop(pid, None)
        if child is None:
            return cpu
        _by_thread.pop(threading.current_thread().ident, None)
        _finished_count[child.part] += 1
        if cpu is not None:
            _finished_cpu[child.part] += cpu
    return cpu


def running_children() -> list:
    with _lock:
        return list(_running.values())


def child_of_thread(ident: int) -> Optional[Child]:
    return _by_thread.get(ident)


# ---- esperas -----------------------------------------------------------------------------
def waiting(seconds: float, reason: str = "waiting") -> None:
    """Marca que a thread atual vai esperar `seconds` (so' para o console mostrar)."""
    _waiting[threading.get_ident()] = (reason, time.monotonic() + max(seconds, 0))


def wait_state(ident: int) -> Optional[tuple]:
    """(motivo, segundos restantes) se a thread esta' numa espera marcada que ainda nao acabou."""
    state = _waiting.get(ident)
    if state is None:
        return None
    remaining = state[1] - time.monotonic()
    return (state[0], remaining) if remaining > 0 else None


def forget_thread(ident: int) -> None:
    _waiting.pop(ident, None)


# ---- leituras de /proc (so' sob demanda) ---------------------------------------------------
def _stat_fields(path: str) -> Optional[list]:
    """Campos do /proc/.../stat a partir do estado (campo 3), pulando o nome entre parenteses."""
    try:
        with open(path) as file:
            text = file.read()
    except OSError:
        return None
    return text[text.rfind(")") + 2:].split()


def thread_cpu_seconds(native_id: int) -> Optional[float]:
    """CPU (user+sys) de UMA thread do coletor. /proc/<tid>/stat daria a do processo inteiro."""
    fields = _stat_fields(f"/proc/self/task/{native_id}/stat")
    if fields is None:
        return None
    return (int(fields[11]) + int(fields[12])) / CLK_TCK


def process_tree_cpu_seconds(pid: int, _depth: int = 0) -> float:
    """CPU ate' agora de um processo em execucao: ele, os filhos ja' coletados e os vivos."""
    fields = _stat_fields(f"/proc/{pid}/stat")
    if fields is None:
        return 0.0
    total = sum(int(value) for value in fields[11:15]) / CLK_TCK  # utime, stime, cutime, cstime
    if _depth < 8:
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as file:
                children = file.read().split()
        except OSError:
            children = []
        total += sum(process_tree_cpu_seconds(int(child), _depth + 1) for child in children)
    return total


def children_cpu() -> dict:
    """parte -> (CPU acumulada dos comandos, terminados + em execucao; quantidade de comandos)."""
    with _lock:
        finished_cpu = dict(_finished_cpu)
        finished_count = dict(_finished_count)
        running = list(_running.values())
    live = defaultdict(float)
    for child in running:
        live[child.part] += process_tree_cpu_seconds(child.pid)
    parts = set(finished_cpu) | set(finished_count) | set(live)
    return {part: (finished_cpu.get(part, 0.0) + live.get(part, 0.0),
                   finished_count.get(part, 0) + sum(1 for child in running if child.part == part))
            for part in parts}


def collector_process() -> dict:
    """Processo do coletor inteiro: CPU propria, CPU dos filhos coletados, RSS, VSZ e threads."""
    info = {}
    fields = _stat_fields("/proc/self/stat")
    if fields is not None:
        info["self_cpu_s"] = (int(fields[11]) + int(fields[12])) / CLK_TCK
        info["reaped_children_cpu_s"] = (int(fields[13]) + int(fields[14])) / CLK_TCK
    try:
        with open("/proc/self/status") as file:
            for line in file:
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmSize", "Threads"):
                    info[key] = int(value.split()[0])
    except OSError:
        pass
    return info


class CostTracker:
    """Acumula a CPU de cada parte entre amostras, sem perder a das threads que ja' terminaram."""

    def __init__(self):
        self._threads = {}  # native_id -> (parte, CPU na ultima amostra)
        self._dead = defaultdict(float)  # parte -> CPU das threads que ja' terminaram
        self._previous = {}  # parte -> CPU acumulada na amostra anterior
        self._previous_time = None
        self.last = {}  # ultima amostra, para o console

    def sample(self) -> dict:
        """
        parte -> dict(role, threads, threads_cpu_s, children, children_cpu_s, cpu_s, cpu_pct).
        cpu_pct e' o uso no intervalo desde a amostra anterior, em % de um nucleo.
        """
        now = time.monotonic()
        alive = {}
        for thread in threading.enumerate():
            cpu = thread_cpu_seconds(thread.native_id) if thread.native_id else None
            if cpu is not None:
                alive[thread.native_id] = (part_of(thread.name), cpu)

        for tid, (part, cpu) in self._threads.items():
            current = alive.get(tid)
            # thread terminou, ou o tid foi reaproveitado por outra thread
            if current is None or current[0] != part or current[1] < cpu:
                self._dead[part] += cpu
        self._threads = alive

        threads_cpu = defaultdict(float, self._dead)
        threads_count = defaultdict(int)
        for part, cpu in alive.values():
            threads_cpu[part] += cpu
            threads_count[part] += 1
        commands = children_cpu()

        elapsed = None if self._previous_time is None else now - self._previous_time
        result = {}
        for part in PART_ORDER:
            if part not in threads_cpu and part not in commands:
                continue
            children_cpu_s, children = commands.get(part, (0.0, 0))
            cpu_s = threads_cpu[part] + children_cpu_s
            cpu_pct = None
            if elapsed:
                cpu_pct = max(cpu_s - self._previous.get(part, 0.0), 0.0) / elapsed * 100
            result[part] = {
                "role": role_of(part), "threads": threads_count[part], "threads_cpu_s": threads_cpu[part],
                "children": children, "children_cpu_s": children_cpu_s, "cpu_s": cpu_s, "cpu_pct": cpu_pct,
            }
            self._previous[part] = cpu_s
        self._previous_time = now
        self.last = result
        return result


# ---- parada (Ctrl+C / SIGTERM) -------------------------------------------------------------------
_stop_requested = threading.Event()
_graceful = False


def install_signal_handlers() -> None:
    """Instala o tratamento de SIGINT/SIGTERM (precisa ser chamado na thread principal)."""
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)


def enable_graceful_stop() -> None:
    """A partir daqui ha' dados coletados: o primeiro Ctrl+C para graciosamente."""
    global _graceful
    _graceful = True


def _signal_handler(signum, _frame):
    log = logging.getLogger("aging")  # nao importa src.utils (que importa este modulo)
    name = signal.Signals(signum).name
    if _graceful and not _stop_requested.is_set():
        _stop_requested.set()
        log.warning("%s received: stopping gracefully. Each thread finishes its current container (a 'docker stop' "
                    "can take ~10 s) and the results are saved; %d commands running now. "
                    "Press Ctrl+C again to force.", name, len(running_children()))
        return
    log.error("%s received %s: forced stop", name,
              "again" if _stop_requested.is_set() else "before the monitoring started (nothing to save yet)")
    raise KeyboardInterrupt


def stopping() -> bool:
    return _stop_requested.is_set()


def wait(seconds: float) -> bool:
    """Espera `seconds` ou ate' a parada ser pedida. Devolve True se foi pedida."""
    return _stop_requested.wait(max(seconds, 0))


def kill_children() -> None:
    """Parada forcada: encerra os comandos em execucao (cada um lider do proprio grupo)."""
    for child in running_children():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (AttributeError, OSError):
            pass
