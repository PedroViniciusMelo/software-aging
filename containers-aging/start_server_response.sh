#!/usr/bin/env bash

server_response="../rejuvenation/machine_resources_monitoring/server-response-time.sh"
jmeter="/home/jojema/Transferências/apache-jmeter-5.6.3/bin"

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
    "$server_response" "$ip_address/tools.descartes.teastore.webui/" "$output_file" &
    pids+=($!)  # Adiciona o PID do processo à lista
}

printf "Choose the host to run the tests: \n[1] Host 1 - Vms: 105, 106, 107\n[2] Host 2 - Vms: 108, 109, 110\n[3] Host3 - Vms: 111, 112, 113\n"
read -r -p "choice: " choice

if [ "$choice" -eq 1 ]; then
  run_server_response "192.168.0.105:8080" "debian12_docker_novo.csv"
  run_server_response "192.168.0.106:8080" "ubuntu22_docker_velho.csv"
  run_server_response "192.168.0.107:8080" "ubuntu24_podman.csv"

  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.105 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_debian12_docker_novo.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.106 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu22_docker_velho.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.107 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu24_podman.csv -n &
  pids+=($!)

elif [ "$choice" -eq 2 ]; then
  run_server_response "192.168.0.108:8080" "debian12_docker_velho.csv"
  run_server_response "192.168.0.109:8080" "ubuntu22_podman.csv"
  run_server_response "192.168.0.110:8080" "ubuntu24_docker_novo.csv"

  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.108 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_debian12_docker_velho.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.109 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu22_podman.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.110 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu24_docker_novo.csv -n &
  pids+=($!)
elif [ "$choice" -eq 3 ]; then
  run_server_response "192.168.0.111:8080" "debian12_podman.csv"
  run_server_response "192.168.0.112:8080" "ubuntu22_docker_novo.csv"
  run_server_response "192.168.0.113:8080" "ubuntu24_docker_velho.csv"

  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.111 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_debian12_podman.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.112 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu22_docker_novo.csv -n &
  pids+=($!)
  java -jar "$jmeter"/ApacheJMeter.jar -t teastore_browse_nogui.jmx -Jhostname=192.168.0.113 -Jport=8080 -JnumUser=100 -JrampUp=60 -Jusername=user2 -Jpassword=password -Jfilename=jmeter_ubuntu24_docker_velho.csv -n &
  pids+=($!)
else
  echo "Invalid choice"
  exit 1
fi

sleep infinity
EXIT