#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$DIR/../config.sh"

function download_command() {
  scp root@"$server_link":/root/$service/$image_name.tar image.tar
}

function load_command() {
  docker load -q -i image.tar
  #rm -f image.tar
}

function start_command() {
  docker run --name "$image_name" -td -p "$mapping_port:$mapping_port" --init "localhost/$image_name" || exit 1
}

function stop_command() {
  docker container stop "$image_name" || exit 1
}

function remove_image_command() {
  docker rmi "localhost/$image_name" || exit 1
}

function remove_container_command() {
  docker rm "$image_name" || exit 1
}

function get_up_time() {
  docker exec -it "$image_name" sh -c "test -e /root/log.txt && cat /root/log.txt"
}

function is_image_available() {
  docker image ls -a | grep "$image_name" | awk '{print $3}'
}
