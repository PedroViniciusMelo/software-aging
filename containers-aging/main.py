import sys

from src.config import DEFAULT_CONFIG_PATH
from src.environment import run_experiment

if __name__ == "__main__":
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
