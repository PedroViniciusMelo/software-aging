#!/usr/bin/env python3
"""
Simulador de hardware para testar o monitoramento de energia (src/energy.py) em uma
maquina/VM que nao tem Intel RAPL nem os thermal zones esperados.

So usa a biblioteca padrao (os, random, time, argparse) -- nao precisa instalar nada.

O que ele faz:
  1. Cria uma arvore de arquivos parecida com /sys/class/powercap e /sys/class/thermal
     (mesma estrutura de pastas, mesmos nomes de arquivo: name, energy_uj,
     max_energy_range_uj, type, temp).
  2. Fica rodando em loop incrementando os contadores de energia com uma potencia
     simulada (com variacao aleatoria), igual um pacote Intel real faria, inclusive
     estourando (wrap-around) o contador quando ele passa de max_energy_range_uj --
     isso deixa testar de verdade o tratamento de overflow que existe no RaplReader.
  3. Atualiza a temperatura simulada da CPU.

Uso:
  python3 tools/fake_rapl.py --base-dir ./fake_sysfs

Depois, para o framework (main.py / src/energy.py) ler daqui em vez do /sys real, exporte:
  export RAPL_BASE_PATH=./fake_sysfs/powercap
  export THERMAL_BASE_PATH=./fake_sysfs/thermal

--fast-wrap deixa o max_energy_range_uj bem pequeno de propósito, pra forcar o
contador estourar em poucos segundos (util pra validar rapidamente o tratamento de
overflow, em vez de esperar horas como levaria com os ranges reais).
"""
import argparse
import os
import random
import time

# Potencia simulada (W) para cada dominio -- soma dessas variacoes fica numa faixa
# plausivel de CPU desktop/servidor real.
DOMAINS = {
    "package-0": {"base_w": 25.0, "jitter_w": 6.0, "max_range_uj": 262143328850},
    "core": {"base_w": 12.0, "jitter_w": 4.0, "max_range_uj": 65712999613},
    "uncore": {"base_w": 3.0, "jitter_w": 1.5, "max_range_uj": 65712999613},
    "dram": {"base_w": 4.0, "jitter_w": 1.0, "max_range_uj": 65712999613},
}

FAST_WRAP_MAX_RANGE_UJ = 5_000_000  # ~5 J, estoura em poucos segundos com a potencia simulada acima


def write(path, content):
    with open(path, "w") as f:
        f.write(str(content))


def setup_tree(base_dir, fast_wrap):
    powercap_dir = os.path.join(base_dir, "powercap")
    thermal_dir = os.path.join(base_dir, "thermal")

    package_dir = os.path.join(powercap_dir, "intel-rapl:0")
    children = {
        "core": os.path.join(package_dir, "intel-rapl:0:0"),
        "uncore": os.path.join(package_dir, "intel-rapl:0:1"),
        "dram": os.path.join(package_dir, "intel-rapl:0:2"),
    }

    os.makedirs(package_dir, exist_ok=True)
    for child_dir in children.values():
        os.makedirs(child_dir, exist_ok=True)

    zone_dirs = {"package-0": package_dir, **children}

    for name, zone_dir in zone_dirs.items():
        max_range = FAST_WRAP_MAX_RANGE_UJ if fast_wrap else DOMAINS[name]["max_range_uj"]
        write(os.path.join(zone_dir, "name"), name)
        write(os.path.join(zone_dir, "energy_uj"), 0)
        write(os.path.join(zone_dir, "max_energy_range_uj"), max_range)

    # thermal_zone0 e' um "acpitz" de mentira, so pra provar que o codigo real
    # procura pelo type "x86_pkg_temp" em vez de assumir sempre o zone0
    decoy_zone = os.path.join(thermal_dir, "thermal_zone0")
    pkg_zone = os.path.join(thermal_dir, "thermal_zone2")
    os.makedirs(decoy_zone, exist_ok=True)
    os.makedirs(pkg_zone, exist_ok=True)
    write(os.path.join(decoy_zone, "type"), "acpitz")
    write(os.path.join(decoy_zone, "temp"), 35000)
    write(os.path.join(pkg_zone, "type"), "x86_pkg_temp")
    write(os.path.join(pkg_zone, "temp"), 45000)

    return zone_dirs, pkg_zone


def run(base_dir, fast_wrap, tick_seconds):
    zone_dirs, pkg_thermal_zone = setup_tree(base_dir, fast_wrap)

    energy_state = {name: 0 for name in zone_dirs}
    temp_c = 45.0

    print(f"Simulando RAPL em {base_dir} (Ctrl+C para parar)")
    print(f"  export RAPL_BASE_PATH={os.path.join(base_dir, 'powercap')}")
    print(f"  export THERMAL_BASE_PATH={os.path.join(base_dir, 'thermal')}")
    if fast_wrap:
        print("  --fast-wrap ativo: max_energy_range_uj propositalmente pequeno, "
              "os contadores devem estourar (wrap-around) em poucos segundos.")

    try:
        while True:
            for name, zone_dir in zone_dirs.items():
                domain = DOMAINS[name]
                watts = max(0.0, domain["base_w"] + random.uniform(-domain["jitter_w"], domain["jitter_w"]))
                energy_uj_increment = int(watts * tick_seconds * 1_000_000)

                max_range = FAST_WRAP_MAX_RANGE_UJ if fast_wrap else domain["max_range_uj"]
                energy_state[name] = (energy_state[name] + energy_uj_increment) % max_range

                write(os.path.join(zone_dir, "energy_uj"), energy_state[name])

            # temperatura passeando devagar entre ~40 e ~70 C
            temp_c = min(70.0, max(40.0, temp_c + random.uniform(-1.0, 1.0)))
            write(os.path.join(pkg_thermal_zone, "temp"), int(temp_c * 1000))

            time.sleep(tick_seconds)
    except KeyboardInterrupt:
        print("\nSimulador de RAPL encerrado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default="./fake_sysfs", help="Pasta onde a arvore falsa sera criada")
    parser.add_argument("--tick-seconds", type=float, default=0.5, help="Intervalo (s) entre atualizacoes dos contadores")
    parser.add_argument("--fast-wrap", action="store_true",
                         help="Usa max_energy_range_uj bem pequeno para forcar overflow rapido (teste do wrap-around)")
    args = parser.parse_args()

    run(base_dir=os.path.abspath(args.base_dir), fast_wrap=args.fast_wrap, tick_seconds=args.tick_seconds)