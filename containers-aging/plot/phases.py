"""
Fases reais do experimento (descanso, stress, rejuvenescimento), lidas do runs.csv do cenario.

O coletor grava em runs.csv o inicio de cada fase ("sleep", "stress", "rejuvenation" e, se
parado com Ctrl+C, "interrupted"). Cada fase vai ate' o evento seguinte; a ultima, ate' o fim
dos dados.
"""
import csv
from datetime import datetime
from pathlib import Path

RUNS_FILE = "runs.csv"
PHASE_NAMES = {"sleep": "rest", "stress": "stress", "rejuvenation": "rejuvenation"}
HIGHLIGHT = {"rest": ("green", 0.3), "rejuvenation": ("orange", 0.4)}  # fases destacadas nos graficos


def read_phases(folder) -> list:
    """[{name, run, start, end}] na ordem; end=None na ultima fase (vai ate' o fim dos dados)."""
    path = Path(folder) / RUNS_FILE
    if not path.exists():
        return []
    with open(path, newline="") as file:
        events = [(row["event"], datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S"))
                  for row in csv.DictReader(file, delimiter=";")]

    phases = []
    first_event = next((event for event, _ in events if event in PHASE_NAMES), None)
    run = 0
    for index, (event, start) in enumerate(events):
        if event not in PHASE_NAMES:  # "interrupted" so' marca o fim da fase anterior
            continue
        if event == first_event:  # cada run comeca pelo mesmo tipo de fase (depende de stressload_first)
            run += 1
        end = events[index + 1][1] if index + 1 < len(events) else None
        phases.append({"name": PHASE_NAMES[event], "run": run, "start": start, "end": end})
    return phases


def highlight_intervals(folder) -> list:
    """Faixas (inicio, fim, cor, alpha) dos graficos, em datetimes."""
    return [(phase["start"], phase["end"], *HIGHLIGHT[phase["name"]]) for phase in read_phases(folder)
            if phase["name"] in HIGHLIGHT]


def phase_windows(folder, t0) -> dict:
    """Janelas {rotulo: (inicio_h, fim_h)} em horas desde t0 (primeira amostra da serie analisada)."""
    to_hours = lambda moment: (moment - t0).total_seconds() / 3600
    return {f"Run {phase['run']} - {phase['name']}":
            (to_hours(phase["start"]), float("inf") if phase["end"] is None else to_hours(phase["end"]))
            for phase in read_phases(folder)}
