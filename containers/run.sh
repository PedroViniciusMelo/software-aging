#!/bin/bash

# Change the container functions to the one you want to test
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$DIR/containers_functions/docker.sh"

# Define a trap function
trap cleanup SIGINT
trap "error $LINENO" ERR

error() {
  echo "Error: $1"
  cleanup
  exit 1
}

# The command to execute on script end or Ctrl + C
cleanup() {
  #rm -f image.tar || echo "No file, skipping"
  #rm -f image.tar.gz || echo "No file, skipping"
  stop_command || echo "No container, skipping"
  remove_container_command || echo "No container, skipping"
  remove_image_command || echo "No Image, skipping"
  exit 0
}

function get_date_time() {
  local date_time
  date_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "$date_time"
}

function progress {
  clear
  local _done _left _fill _empty _progress current
  current=$(($1))
  _progress=$((($current * 10000 / $2) / 100))
  _done=$(($_progress * 6 / 10))
  _left=$((60 - $_done))
  _fill=$(printf "%${_done}s")
  _empty=$(printf "%${_left}s")
  local NC='\033[0m'
  printf "\r$current / $2 : [${NC}${_fill// /#}${_empty// /-}] ${_progress}%%${NC}"
}

# Function used to get the time execution of a function
function get_command_time() {
  local start end total
  start=$(date +%s%N)

  "$1" >/dev/null

  end=$(date +%s%N)
  total=$((end - start))
  echo $total
}

count=0

while [[ $count -lt $max_runs ]]; do
  progress $count "$max_runs"
  #download_time=$(get_command_time download_command)
  load_time=$(get_command_time load_command)
  image_available=$(is_image_available)

  if [ -n "$image_available" ]; then
    instantiate_time=$(get_command_time start_command)
    service_up_time=""

    until [ -n "$service_up_time" ]; do
        service_up_time=$(get_up_time | tr -d '\r')
    done

    stop_time=$(get_command_time stop_command)
    container_removal_time=$(get_command_time remove_container_command)
    image_removal_time=$(get_command_time remove_image_command)
    display_date=$(get_date_time)

    echo "$load_time;$instantiate_time;$service_up_time;$stop_time;$container_removal_time;$image_removal_time;$image_size;$application;$service;$display_date" >> "logs/$log_file"
    count=$((count + 1))
    image_available=""
  fi
done

progress $count "$max_runs"

printf "\n"
