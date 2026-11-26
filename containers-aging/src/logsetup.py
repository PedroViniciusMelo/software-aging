"""Logs do coletor: informativos (INFO), depuracao (DEBUG) e so' avisos/erros quando desligados."""
import logging
import sys

from src.utils import log

FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "quiet": logging.WARNING}
LOG_FILE = "collector.log"


def level_for(informative: bool, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    return logging.INFO if informative else logging.WARNING


def setup_logging(informative: bool = True, debug: bool = False) -> None:
    """Terminal. Avisos e erros aparecem sempre; os informativos e os de depuracao sao opcionais."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(level_for(informative, debug))


def attach_log_file(logs_path: str) -> None:
    """Grava tambem em logs/<LOG_FILE> (copiado para results/ com o resto)."""
    handler = logging.FileHandler(f"{logs_path}/{LOG_FILE}")
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    log.addHandler(handler)


def set_level(name: str) -> bool:
    if name not in LEVELS:
        return False
    log.setLevel(LEVELS[name])
    return True


def level_name() -> str:
    level = log.getEffectiveLevel()
    return next((name for name, value in LEVELS.items() if value == level), logging.getLevelName(level))
