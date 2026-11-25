"""
Harness de "golden master" do framework de coleta.

Roda o main.py de verdade (o mesmo ponto de entrada usado em producao) num processo
isolado, mas com o mundo externo simulado de forma deterministica:

  * subprocess.Popen  -> respostas fixas para docker/podman/pidstat/mpstat/... e falhas
                         injetadas de proposito para exercitar os caminhos de fallback;
  * time.sleep / asyncio.sleep / datetime.now / time.time / perf_counter_ns
                      -> relogio virtual POR THREAD (cada thread avanca o proprio
                         relogio), entao a saida nao depende do escalonamento;
  * random            -> valores fixos;
  * /sys (RAPL/thermal) e python-kasa -> arvores/modulos falsos.

Cada thread e' identificada pelo "caminho" de criacao (ex.: 0/3/1 = 2a thread criada
pela 4a thread criada pela 1a thread criada pela principal), que e' estavel entre
execucoes. O resultado (comandos executados por thread, saida no stdout por thread,
excecoes que mataram threads e o conteudo de todos os arquivos em logs/) vai para um
JSON que o tests/test_golden.py compara com o snapshot gravado.

As threads de monitoramento (loops infinitos) "estacionam" quando o relogio virtual
delas passa de BUDGET_SECONDS e so' terminam quando o fluxo principal acaba, para que
todas estejam vivas durante o experimento inteiro (o monitor de custo enumera threads).

Uso: python3 tests/golden/harness.py <cenario> <saida.json>
"""
import asyncio
import datetime as _dt
import importlib
import json
import os
import random
import re
import runpy
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_DATETIME = _dt.datetime(2024, 1, 1, 0, 0, 0)
BASE_EPOCH = 1_700_000_000.0
BUDGET_SECONDS = 1200
MONITOR_PREFIXES = ("systemtap", "container_metrics", "docker_processes", "podman_processes", "monitoring")

CONTAINERS = [
    {"name": "postgres", "version": "16.4-bookworm", "host_port": 5432, "port": 5432,
     "min_container_wait_time": 1, "max_container_wait_time": 10},
    {"name": "redis", "version": "7.4.0-bookworm", "host_port": 6380, "port": 6379,
     "min_container_wait_time": 1, "max_container_wait_time": 10},
    {"name": "rabbitmq", "version": "4.0-rc-management", "host_port": 5672, "port": 5672,
     "min_container_wait_time": 1, "max_container_wait_time": 10},
    {"name": "nginx", "version": "1.27.1-bookworm", "host_port": 8085, "port": 80,
     "min_container_wait_time": 1, "max_container_wait_time": 10},
]


def scenario_config(name, folder):
    stressload = {
        "stressload_first": False, "max_stress_time": 6, "wait_after_stress": 100,
        "min_lifecycle_runs": 1, "max_lifecycle_runs": 2,
        "min_qtt_containers": 1, "max_qtt_containers": 2,
    }
    if name == "docker":
        return "docker", {
            "general": {"runs": 2, "scripts_folder": folder, "apply_rejuvenation": False},
            "setup": {"rebuild_images": True},
            "monitoring": {"sleep_time": 60, "sleep_time_container_metrics": 240,
                           "using_containers_app_time": False},
            "energy": {"enable_rapl": True, "sleep_time_rapl": 30,
                       "rapl_zones": ["package", "core", "uncore", "dram"],
                       "enable_tomada": True, "sleep_time_tomada": 300, "tomada_ip": "127.0.0.1",
                       "tomada_username": "user", "tomada_password": "pass"},
            "stressload": stressload,
            "containers": CONTAINERS,
        }
    if name == "podman":
        return "podman", {
            "general": {"runs": 1, "scripts_folder": folder, "apply_rejuvenation": True},
            "setup": {"rebuild_images": False},
            "monitoring": {"sleep_time": 60, "sleep_time_container_metrics": 240,
                           "using_containers_app_time": True},
            # sem secao "energy": exercita os valores padrao
            "stressload": dict(stressload, stressload_first=True),
            "containers": CONTAINERS,
        }
    if name == "rapl_only_package":
        return "docker", {
            "general": {"runs": 1, "scripts_folder": folder, "apply_rejuvenation": False},
            "setup": {"rebuild_images": False},
            "monitoring": {"sleep_time": 60, "sleep_time_container_metrics": 240,
                           "using_containers_app_time": False},
            "energy": {"enable_rapl": True, "sleep_time_rapl": 10, "rapl_zones": ["package", "core"]},
            "stressload": stressload,
            "containers": CONTAINERS[:1],
        }
    if name == "giveup":
        # "badimg" nunca sobe no monitor de ciclo de vida: exercita a desistencia apos 6
        # tentativas (que encerra a rodada inteira, pulando os containers seguintes).
        bad = dict(CONTAINERS[0], name="badimg", host_port=7000)
        return "docker", {
            "general": {"runs": 1, "scripts_folder": folder, "apply_rejuvenation": False},
            "setup": {"rebuild_images": True},
            "monitoring": {"sleep_time": 60, "sleep_time_container_metrics": 240,
                           "using_containers_app_time": False},
            "energy": {"enable_rapl": False},
            "stressload": stressload,
            "containers": [CONTAINERS[0], bad, CONTAINERS[1]],
        }
    raise SystemExit(f"cenario desconhecido: {name}")


class StopSim(BaseException):
    """Encerra uma thread de monitoramento no fim da simulacao (nao e' Exception de proposito)."""


class Sim:
    def __init__(self, software, sysfs):
        self.software = software
        self.sysfs = sysfs
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.threads = {}  # key -> {"name", "commands", "stdout", "died_with"}
        self.failed_once = set()
        self.parked = set()

    # ---- identificacao das threads ----------------------------------------------
    @staticmethod
    def state(thread=None):
        thread = thread or threading.current_thread()
        if not hasattr(thread, "_sim_path"):
            thread._sim_path = ("?" + thread.name,)
            thread._sim_clock = 0.0
            thread._sim_start = 0.0
            thread._sim_children = 0
            thread._sim_perf = 0
        return thread

    def entry(self, thread=None):
        thread = self.state(thread)
        key = "/".join(str(part) for part in thread._sim_path) or "main"
        with self.lock:
            if key not in self.threads:
                self.threads[key] = {"name": thread.name, "commands": [], "stdout": "", "died_with": None}
            return self.threads[key]

    # ---- relogio virtual -------------------------------------------------------
    def sleep(self, seconds):
        thread = self.state()
        thread._sim_clock += seconds
        if thread.name == "monitoring_rapl":
            self.advance_rapl(seconds)
        if thread.name.startswith(MONITOR_PREFIXES) and thread._sim_clock - thread._sim_start > BUDGET_SECONDS:
            with self.lock:
                self.parked.add(thread)
            self.shutdown.wait()
            raise StopSim()

    def wait_monitors_parked(self, timeout=120):
        """Espera todas as threads de monitoramento vivas estacionarem (saida completa e estavel)."""
        for _ in range(int(timeout / 0.05)):
            with self.lock:
                pending = [t for t in threading.enumerate()
                           if t.name.startswith(MONITOR_PREFIXES) and t.is_alive() and t not in self.parked]
            if not pending:
                return
            threading.Event().wait(0.05)
        raise RuntimeError(f"threads de monitoramento nao estacionaram: {[t.name for t in pending]}")

    def now(self):
        thread = self.state()
        thread._sim_clock += 1
        return BASE_DATETIME + _dt.timedelta(seconds=thread._sim_clock)

    def time(self):
        return BASE_EPOCH + self.state()._sim_clock

    def perf_counter_ns(self):
        thread = self.state()
        thread._sim_perf += 1_234_567
        return thread._sim_perf

    # ---- RAPL falso: contadores avancam conforme o relogio da thread de RAPL ---------
    def advance_rapl(self, seconds):
        for zone_dir, watts in self.rapl_zones:
            energy_path = zone_dir / "energy_uj"
            max_range = int((zone_dir / "max_energy_range_uj").read_text())
            value = int(energy_path.read_text()) + int(watts * seconds * 1_000_000)
            energy_path.write_text(str(value % max_range))

    def build_sysfs(self):
        powercap = self.sysfs / "powercap"
        layout = [
            ("intel-rapl:0", "package-0", 25.5, 10**12),
            ("intel-rapl:0/intel-rapl:0:0", "core", 12.25, 10**12),
            ("intel-rapl:0/intel-rapl:0:1", "uncore", 3.125, 10**12),
            ("intel-rapl:1", "dram", 4.0, 500_000_000),  # range pequeno: forca wrap-around
        ]
        self.rapl_zones = []
        for rel, name, watts, max_range in layout:
            zone = powercap / rel
            zone.mkdir(parents=True, exist_ok=True)
            (zone / "name").write_text(name + "\n")
            (zone / "energy_uj").write_text("1000")
            (zone / "max_energy_range_uj").write_text(str(max_range))
            self.rapl_zones.append((zone, watts))
        thermal = self.sysfs / "thermal"
        for index, (kind, temp) in enumerate([("acpitz", "30000"), ("x86_pkg_temp", "47500")]):
            zone = thermal / f"thermal_zone{index}"
            zone.mkdir(parents=True, exist_ok=True)
            (zone / "type").write_text(kind + "\n")
            (zone / "temp").write_text(temp + "\n")

    # ---- shell falso -------------------------------------------------------------
    def normalize(self, command):
        tids = {t.native_id: t.name for t in threading.enumerate() if t.native_id}

        def repl(match):
            number = int(match.group(2))
            return match.group(1) + (f"<tid:{tids[number]}>" if number in tids else match.group(2))

        command = re.sub(r"(-p |/proc/)(\d+)", repl, command)
        return command.replace(str(self.sysfs.parent), "<TMP>")

    def respond(self, command):
        sw = self.software
        rules = [
            (r"^which (docker|podman)$", lambda m: (0, f"/usr/bin/{m.group(1)}") if m.group(1) == sw else (1, "")),
            (r"^lsb_release -i$", lambda m: (0, "No LSB modules are available.\nDistributor ID:\tDebian")),
            (r"^lsb_release -r$", lambda m: (0, "No LSB modules are available.\nRelease:\t12")),
            (r"^(docker|podman) -v$", lambda m: (0, f"{m.group(1)} version 1.2.3, build abc")),
            (r"^lsblk ", lambda m: (0, "vda")),
            (r"^pgrep -f ", lambda m: (0, "1234")),
            (r"^pidstat ", lambda m: (0, "1700000000 0 1234 1.00 2.00 0.00 0.00 3.00 1 10.00 0.00 20000 10000 0.50 cmd")),
            (r"grep Threads", lambda m: (0, "12")),
            (r"grep Swap", lambda m: (0, "4")),
            (r"^df ", lambda m: (0, "1234567")),
            (r"^iostat ", lambda m: (0, "vda 1.50 2.50 3.50 4.50 10 20 30 40")),
            (r"^mpstat ", lambda m: (0, "12:00:00 all 1.10 0.20 2.20 0.30 0.00 0.40 0.00 0.00 0.00 95.80")),
            (r"^free ", lambda m: (0, "800000")),
            (r"grep -i Cached", lambda m: (0, "300000")),
            (r"grep -i Buffers", lambda m: (0, "50000")),
            (r"grep -i Swap \| grep -i Free", lambda m: (0, "8000000")),
            (r"^ps aux", lambda m: (0, "2")),
            (r"killed process", lambda m: (0, "[1.0] Killed process 11\n[2.0] Killed process 22")),
            (r"'oom'", lambda m: (0, "[3.0] oom-kill:constraint")),
            (r" container inspect ", lambda m: (0, "true")),
            (r" run --name (\S+) ", self.run_container),
            (r" stop (\S+)$", lambda m: (1, "") if "rabbitmq" in m.group(1) else (0, m.group(1))),
            (r" rm -v (\S+)$", lambda m: (1, "") if "nginx" in m.group(1) else (0, m.group(1))),
            (r" exec -i (\S+) ", self.exec_container),
        ]
        for pattern, handler in rules:
            match = re.search(pattern, command)
            if match:
                return handler(match)
        return 0, ""

    def run_container(self, match):
        name = match.group(1)
        if name == "badimg":
            return 125, ""
        with self.lock:
            if "redis" in name and name not in self.failed_once:
                self.failed_once.add(name)
                return 125, ""
        return 0, "c0ffee"

    def exec_container(self, match):
        key = "exec:" + match.group(1)
        with self.lock:
            if key not in self.failed_once:
                self.failed_once.add(key)
                return 1, ""
        return 0, "app started in 42 ms"


SIM = None


class FakePopen:
    def __init__(self, command, shell=False, stdout=None, stderr=None, **kwargs):
        SIM.entry()["commands"].append(SIM.normalize(command))
        self.returncode, output = SIM.respond(command)
        self._out = output.encode()
        self._err = b"simulated failure" if self.returncode else b""

    def communicate(self, *args, **kwargs):
        return self._out, self._err

    def wait(self, *args, **kwargs):
        return self.returncode


class ThreadStdout:
    def write(self, text):
        SIM.entry()["stdout"] += text
        return len(text)

    def flush(self):
        pass


class FakeDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return SIM.now()


def install_patches():
    original_start = threading.Thread.start

    def start(self):
        parent = SIM.state()
        self._sim_path = parent._sim_path + (parent._sim_children,)
        parent._sim_children += 1
        self._sim_clock = parent._sim_clock
        self._sim_start = parent._sim_clock
        self._sim_children = 0
        self._sim_perf = 0
        return original_start(self)

    def excepthook(args):
        if args.exc_type is StopSim:
            return
        kind = "SystemExit" if issubclass(args.exc_type, SystemExit) else args.exc_type.__name__
        SIM.entry(args.thread)["died_with"] = kind

    async def fake_async_sleep(delay, result=None):
        SIM.sleep(delay)
        return result

    threading.Thread.start = start
    threading.excepthook = excepthook
    subprocess.Popen = FakePopen
    time.sleep = SIM.sleep
    time.time = SIM.time
    time.perf_counter_ns = SIM.perf_counter_ns
    asyncio.sleep = fake_async_sleep
    random.randint = lambda a, b: (a + b) // 2
    random.random = lambda: 0.5
    random.uniform = lambda a, b: (a + b) / 2


def import_project_modules():
    """Importa todos os modulos do projeto antes de rodar o main.py para trocar o `datetime`."""
    skip = {"tests", "plot", "tools", "env", "venv", ".venv", "logs"}
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if rel.parts[0] in skip or rel.parts[0].startswith(".") or len(rel.parts) == 1:
            continue
        module = ".".join(rel.with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        importlib.import_module(module)

    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None) or ""
        if file.startswith(str(REPO)) and getattr(module, "datetime", None) is _dt.datetime:
            module.datetime = FakeDatetime


def collect_logs(logs_dir):
    files = {}
    for path in sorted(logs_dir.rglob("*")):
        if path.is_file():
            lines = path.read_text().split("\n")
            # write_to_file original tem uma corrida (checa tamanho e escreve sem lock), entao duas
            # threads podem repetir o cabecalho; isso e' ignorado aqui para o snapshot ser estavel.
            body = [line for line in lines[1:] if line != lines[0]]
            files[str(path.relative_to(logs_dir))] = {"first_line": lines[0], "lines": sorted(body)}
    return files


def main():
    global SIM
    scenario, output = sys.argv[1], sys.argv[2]

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "work"
        (workdir / "logs").mkdir(parents=True)  # o "mkdir" real e' simulado pelo FakePopen
        software, config = scenario_config(scenario, str(workdir))

        import yaml
        (workdir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

        SIM = Sim(software, Path(tmp) / "sysfs")
        SIM.build_sysfs()
        os.environ["RAPL_BASE_PATH"] = str(SIM.sysfs / "powercap")
        os.environ["THERMAL_BASE_PATH"] = str(SIM.sysfs / "thermal")
        sys.path.insert(0, str(REPO / "tools" / "fake_kasa"))
        sys.path.insert(0, str(REPO))
        os.chdir(workdir)

        main_thread = threading.current_thread()
        main_thread._sim_path = ()
        main_thread._sim_clock = 0.0
        main_thread._sim_start = 0.0
        main_thread._sim_children = 0
        main_thread._sim_perf = 0

        install_patches()
        import_project_modules()

        real_stdout = sys.stdout
        sys.stdout = ThreadStdout()
        exit_code = 0
        try:
            sys.argv = ["main.py"]
            runpy.run_path(str(REPO / "main.py"), run_name="__main__")
        except SystemExit as error:
            exit_code = error.code
        finally:
            SIM.wait_monitors_parked()
            SIM.shutdown.set()
            for thread in threading.enumerate():
                if thread is not main_thread:
                    thread.join(timeout=30)
            sys.stdout = real_stdout

        for key, data in SIM.threads.items():
            if data["name"] == "monitoring_cost":
                data["commands"].sort()  # a ordem de threading.enumerate() nao e' deterministica

        result = {
            "exit_code": exit_code,
            "threads": dict(sorted(SIM.threads.items())),
            "files": collect_logs(workdir / "logs"),
        }

    Path(output).write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
