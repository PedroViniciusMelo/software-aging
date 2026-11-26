"""
Console interativo: digite comandos no terminal enquanto o experimento roda.

Fica bloqueado lendo a entrada (sem consumir CPU) e so' le' /proc quando um comando pede.
Liga com console.enabled no config.yaml; desliga sozinho se a entrada nao for um terminal
(nohup, systemd, redirecionamento). Ctrl+C continua encerrando o experimento.
"""
import sys
import threading
import time
from datetime import datetime

from src import activity
from src.logsetup import level_name, set_level
from src.utils import log

HELP = """Commands:
  threads  (t)   threads of each part of the system: TID, CPU, what each one is doing now
  children (c)   external commands running right now (pidstat, docker, stap...)
  status   (s)   experiment phase, run, rejuvenation and logging
  cost           last sample of the monitoring cost (CPU per part)
  log <level>    change the logging level: debug | info | quiet
  help     (h)   this help
  exit           close the console (the experiment keeps running)"""


class ExperimentStatus:
    """Fase atual do experimento, mantida pela thread principal e lida pelo console."""

    def __init__(self, runs: int):
        self.runs = runs
        self.run = 0
        self.phase = "starting"
        self.phase_started = datetime.now()
        self.phase_until = None
        self.expected_end = None
        self.rejuvenation = "disabled"

    def set_phase(self, phase: str, run: int = None, until: datetime = None) -> None:
        self.phase = phase
        if run is not None:
            self.run = run
        self.phase_started = datetime.now()
        self.phase_until = until


def _duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def _short(text: str, size: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= size else text[:size - 3] + "..."


class Console:
    def __init__(self, status: ExperimentStatus, monitoring):
        self.status = status
        self.monitoring = monitoring
        self.commands = {
            "threads": self.threads, "t": self.threads,
            "children": self.children, "c": self.children,
            "status": self.show_status, "s": self.show_status,
            "cost": self.cost,
            "log": self.log_level,
            "help": self.help, "h": self.help, "?": self.help,
        }

    @staticmethod
    def available() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def start(self) -> bool:
        if not self.available():
            log.info("Interactive console disabled: stdin is not a terminal")
            return False
        threading.Thread(target=self._loop, name="console", daemon=True).start()
        log.info("Interactive console ready: type 'help' and Enter")
        return True

    def _loop(self):
        for line in sys.stdin:
            words = line.split()
            if not words:
                continue
            if words[0] in ("exit", "quit", "q"):
                print("Console closed; the experiment keeps running.")
                return
            handler = self.commands.get(words[0])
            try:
                print(handler(*words[1:]) if handler else f"Unknown command '{words[0]}'. Type 'help'.", flush=True)
            except Exception as error:  # o console nunca derruba o experimento
                print(f"Console error: {error!r}", flush=True)

    # ---- comandos --------------------------------------------------------------------------
    @staticmethod
    def help(*_):
        return HELP

    def threads(self, *_):
        now = time.monotonic()
        groups = {}
        for thread in threading.enumerate():
            groups.setdefault(activity.part_of(thread.name), []).append(thread)

        lines = [f"{'PART':18} {'THREAD':34} {'TID':>7} {'CPU':>9}  STATE"]
        for part in activity.PART_ORDER:
            for thread in sorted(groups.get(part, []), key=lambda item: item.name):
                cpu = activity.thread_cpu_seconds(thread.native_id) if thread.native_id else None
                lines.append(f"{part:18} {_short(thread.name, 34):34} {thread.native_id or '-':>7} "
                             f"{'-' if cpu is None else f'{cpu:.2f}s':>9}  {self._state(thread, now)}")
        total = sum(len(threads) for threads in groups.values())
        summary = ", ".join(f"{part}={len(groups[part])}" for part in activity.PART_ORDER if part in groups)
        lines.append(f"{total} threads: {summary}")
        return "\n".join(lines)

    @staticmethod
    def _state(thread: threading.Thread, now: float) -> str:
        child = activity.child_of_thread(thread.ident)
        if child is not None:
            return f"running {_duration(now - child.started)}: {_short(child.command, 60)}"
        wait = activity.wait_state(thread.ident)
        if wait is not None:
            return f"{wait[0]} ({_duration(wait[1])} left)"
        if thread is threading.current_thread():
            return "reading the terminal"
        return "in python (no command running)" if thread.is_alive() else "finished"

    @staticmethod
    def children(*_):
        children = activity.running_children()
        if not children:
            return "No external command running."
        now = time.monotonic()
        lines = [f"{'PID':>7} {'PART':18} {'ELAPSED':>10} {'CPU':>9}  COMMAND"]
        for child in sorted(children, key=lambda item: item.started):
            cpu = activity.process_tree_cpu_seconds(child.pid)
            lines.append(f"{child.pid:>7} {child.part:18} {_duration(now - child.started):>10} {cpu:>8.2f}s  "
                         f"{_short(child.command, 80)}")
        return "\n".join(lines)

    def show_status(self, *_):
        status = self.status
        now = datetime.now()
        phase = f"{status.phase} (run {status.run}/{status.runs}, for {_duration((now - status.phase_started).total_seconds())}"
        if status.phase_until:
            phase += f", until {status.phase_until:%Y-%m-%d %H:%M:%S}"
        lines = [
            f"Phase:        {phase})",
            f"Expected end: {status.expected_end:%Y-%m-%d %H:%M:%S}" if status.expected_end else "Expected end: -",
            f"Rejuvenation: {status.rejuvenation}",
            f"Monitoring:   {'running' if self.monitoring.running else 'stopped'}",
            f"Logging:      {level_name()}",
        ]
        return "\n".join(lines)

    def cost(self, *_):
        sample = self.monitoring.cost_tracker.last
        if not sample:
            return "No monitoring cost sample yet (one per monitoring.sleep_time)."
        lines = [f"{'PART':18} {'ROLE':10} {'THREADS':>7} {'CMDS':>8} {'CPU TOTAL':>10} {'CPU %':>7}"]
        for part, row in sample.items():
            pct = "-" if row["cpu_pct"] is None else f"{row['cpu_pct']:.2f}"
            lines.append(f"{part:18} {row['role']:10} {row['threads']:>7} {row['children']:>8} "
                         f"{row['cpu_s']:>9.1f}s {pct:>7}")
        lines.append("CPU % = share of one core since the previous sample (threads + their commands)")
        return "\n".join(lines)

    @staticmethod
    def log_level(*args):
        if args and set_level(args[0]):
            return f"Logging level: {args[0]}"
        return f"Logging level: {level_name()}. Use: log debug | info | quiet"
