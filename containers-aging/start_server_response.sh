#!/usr/bin/env bash

server_response="../rejuvenation/machine_resources_monitoring/server-response-time.sh"

# ------------------------------------- GET IPS ADDRESSES AND PORTS ------------------------------------------ #
# shellcheck disable=SC2034
ONE_IP_AND_PORT="192.168.1.105:8080"
# shellcheck disable=SC2034
TWO_IP_AND_PORT="192.168.1.106:8080"
# shellcheck disable=SC2034
THREE_IP_AND_PORT="192.168.1.107:8080"

# ------------------------------------- GENERATE FILE OLD DOCKER VERSION ------------------------------------- #
# shellcheck disable=SC2034
DOCKER_OLD_VERSION_ONE_FILE="docker_old_version_debian12.csv"
# shellcheck disable=SC2034
DOCKER_OLD_VERSION_TWO_FILE="docker_old_version_ubuntu22_04_4.csv"
# shellcheck disable=SC2034
DOCKER_OLD_VERSION_THREE_FILE="docker_old_version_ubuntu24_04.csv"

# ------------------------------------- GENERATE DOCKER CURRENT VERSION FILE --------------------------------- #
# shellcheck disable=SC2034
DOCKER_CURRENT_VERSION_ONE_FILE="docker_current_version_debian12.csv"
# shellcheck disable=SC2034
DOCKER_CURRENT_VERSION_TWO_FILE="docker_current_version_ubuntu22_04_4.csv"
# shellcheck disable=SC2034
DOCKER_CURRENT_VERSION_THREE_FILE="docker_current_version_ubuntu24_04.csv"

# ------------------------------------- GENERATE PODMAN CURRENT VERSION FILE --------------------------------- #
# shellcheck disable=SC2034
PODMAN_CURRENT_VERSION_ONE_FILE="podman_current_version_debian12.csv"
# shellcheck disable=SC2034
PODMAN_CURRENT_VERSION_TWO_FILE="podman_current_version_ubuntu22_04_4.csv"
# shellcheck disable=SC2034
PODMAN_CURRENT_VERSION_THREE_FILE="podman_current_version_ubuntu24_04.csv"


pids=() # Array para armazenar os PIDs dos processos

trap "EXIT" SIGINT SIGTERM

EXIT() {
    echo "Exiting..."

    for pid in "${pids[@]}"; do
        kill "$pid"
    done
    exit 0
}

run_server_response() {
    local ip_address="$1"
    local output_file="$2"

    "$server_response" "$ip_address" "$output_file" &
    pids+=($!)  # Adiciona o PID do processo à lista
}

# ADD MORE ADDRESSES AND OTHERS CSV FILES
# ------------------------------------------------- DOCKER OLD --------------------------------------------------- #
# run_server_response "$THREE_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_OLD_VERSION_ONE_FILE"
# run_server_response "$TWO_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_OLD_VERSION_TWO_FILE"
run_server_response "$ONE_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_OLD_VERSION_THREE_FILE"

# ----------------------------------------------- CURRENT DOCKER ------------------------------------------------- #
# run_server_response "$ONE_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_CURRENT_VERSION_ONE_FILE"
# run_server_response "$TWO_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_CURRENT_VERSION_TWO_FILE"
# run_server_response "$THREE_IP_AND_PORT/tools.descartes.teastore.webui/" "$DOCKER_CURRENT_VERSION_THREE_FILE"

# ------------------------------------------------- PODMAN ------------------------------------------------------- #
# run_server_response "$ONE_IP_AND_PORT/tools.descartes.teastore.webui/" "$PODMAN_CURRENT_VERSION_ONE_FILE"
# run_server_response "$TWO_IP_AND_PORT/tools.descartes.teastore.webui/" "$PODMAN_CURRENT_VERSION_TWO_FILE"
# run_server_response "$THREE_IP_AND_PORT/tools.descartes.teastore.webui/" "$PODMAN_CURRENT_VERSION_THREE_FILE"

sleep 31680     # equivale a 11 dias em segundos; favor verificar isso direito
EXIT