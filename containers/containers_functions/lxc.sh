#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$DIR/../config.sh"

function download_command() {
  scp root@$server_link:"/root/$service/$image_name.tar.gz" image.tar.gz
}

function load_command() {
  lxc image import image.tar.gz --alias "$image_name"
  #rm -f image.tar.gz
}

function start_command() {
  lxc launch "$image_name" "$image_name" || exit 1
  lxc config device add "$image_name" myport$mapping_port proxy listen=tcp:0.0.0.0:$mapping_port connect=tcp:127.0.0.1:$mapping_port || exit 1
}

function stop_command() {
  lxc stop "$image_name" || exit 1
}

function remove_image_command() {
  lxc image delete "$image_name" || exit 1
}

function remove_container_command() {
  lxc delete --force "$image_name" || exit 1

}

function get_up_time() {
  lxc exec "$image_name" -- cat /root/log.txt || exit 1
}

function is_image_available() {
  lxc image list | grep "$image_name" | awk '{print $4}'
}
