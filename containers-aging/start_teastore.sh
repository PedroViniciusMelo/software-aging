#!/usr/bin/env bash

ip_address=$(hostname -I | awk '{print $1}')

software=$1
sleep_time=${2:-60}  # espera entre os servicos (teastore.start_interval no config.yaml)
# rotulo usado pelo rejuvenescimento (src/rejuvenation.py) para preservar os containers do TeaStore
label="--label containers-aging.role=teastore"

$software run $label -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=10000" -p 10000:8080 -d --restart=always docker.io/descartesresearch/teastore-registry

sleep $sleep_time

$software run $label -p 3306:3306 -d --restart=always docker.io/descartesresearch/teastore-db

sleep $sleep_time

$software run $label -e "REGISTRY_HOST=$ip_address" -e "REGISTRY_PORT=10000" -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=1111" -e "DB_HOST=$ip_address" -e "DB_PORT=3306" -p 1111:8080 -d --restart=always docker.io/descartesresearch/teastore-persistence

sleep $sleep_time

$software run $label -e "REGISTRY_HOST=$ip_address" -e "REGISTRY_PORT=10000" -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=2222" -p 2222:8080 -d --restart=always docker.io/descartesresearch/teastore-auth

sleep $sleep_time

$software run $label -e "REGISTRY_HOST=$ip_address" -e "REGISTRY_PORT=10000" -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=3333" -p 3333:8080 -d --restart=always docker.io/descartesresearch/teastore-recommender

sleep $sleep_time

$software run $label -e "REGISTRY_HOST=$ip_address" -e "REGISTRY_PORT=10000" -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=4444" -p 4444:8080 -d --restart=always docker.io/descartesresearch/teastore-image

sleep $sleep_time

$software run $label -e "REGISTRY_HOST=$ip_address" -e "REGISTRY_PORT=10000" -e "HOST_NAME=$ip_address" -e "SERVICE_PORT=8080" -p 8080:8080 -d --restart=always docker.io/descartesresearch/teastore-webui
