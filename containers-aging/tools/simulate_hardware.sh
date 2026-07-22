#!/usr/bin/env bash
# Sobe o hardware "falso" (RAPL + tomada) para testar o framework de monitoramento de
# energia em uma maquina/VM sem Intel RAPL e sem a tomada real na rede.
#
# IMPORTANTE: rode com "source" (nao "./"), para as variaveis de ambiente valerem no seu
# shell atual, onde voce vai rodar o main.py em seguida:
#
#   source tools/simulate_hardware.sh
#
# O que isso faz:
#   1. Sobe tools/fake_rapl.py em background, escrevendo uma arvore de sysfs falsa em
#      ./fake_sysfs (contadores de energia e temperatura atualizados continuamente).
#   2. Exporta RAPL_BASE_PATH / THERMAL_BASE_PATH apontando pra essa pasta falsa, para
#      que src/energy.py leia de la em vez do /sys real (ver o topo do proprio energy.py).
#   3. Exporta PYTHONPATH incluindo tools/fake_kasa, para que "from kasa import Discover"
#      (usado pelo monitoramento da tomada) resolva para o modulo falso -- nem precisa
#      ter o pacote "python-kasa" real instalado para testar.
#
# Depois disso, rode o framework normalmente NO MESMO SHELL (ex.: python3 main.py), com
# energy.enable_rapl: true e energy.enable_tomada: true no config.yaml. O valor de
# energy.tomada_ip pode ser qualquer coisa (ex.: "127.0.0.1"): o modulo falso nao usa rede.
#
# Para parar o simulador de RAPL depois: kill "$FAKE_RAPL_PID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAKE_SYSFS_DIR="${FAKE_SYSFS_DIR:-$SCRIPT_DIR/../fake_sysfs}"

python3 "$SCRIPT_DIR/fake_rapl.py" --base-dir "$FAKE_SYSFS_DIR" &
export FAKE_RAPL_PID=$!

export RAPL_BASE_PATH="$FAKE_SYSFS_DIR/powercap"
export THERMAL_BASE_PATH="$FAKE_SYSFS_DIR/thermal"
export PYTHONPATH="$SCRIPT_DIR/fake_kasa:$PYTHONPATH"

echo "Simulador de RAPL rodando (PID $FAKE_RAPL_PID), escrevendo em $FAKE_SYSFS_DIR"
echo "RAPL_BASE_PATH=$RAPL_BASE_PATH"
echo "THERMAL_BASE_PATH=$THERMAL_BASE_PATH"
echo "PYTHONPATH inclui o kasa falso: $SCRIPT_DIR/fake_kasa"
echo
echo "Agora rode o framework normalmente neste mesmo shell (ex.: python3 main.py)."
echo "Para parar o simulador de RAPL depois: kill \$FAKE_RAPL_PID"