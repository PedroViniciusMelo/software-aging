#!/bin/bash

max_runs=1000 #Number of runs default 1000
server_link="172.20.101.13" #The IP of the server

#Test scenarios
application="nginx"         #redis, nginx, rabbitmq or postgres
service="docker"               #docker, podman, lxc
image_size="500"            #500, 1024, 2048, 4096

#Dont change this
image_name="$application-${image_size}mb"

#Port to map the container
if [ "$application" == "nginx" ]; then
  mapping_port="80"
elif [ "$application" == "redis" ]; then
  mapping_port="6379"
elif [ "$application" == "rabbitmq" ]; then
  mapping_port="5672"
elif [ "$application" == "postgres" ]; then
  mapping_port="5432"
fi

mkdir -p "logs"

log_file="$service-$image_name.csv"

#echo "download_image;load_container;start_container;start_application;stop_container;remove_container;remove_image;image_size;application;service;date_time" >"logs/$log_file"
echo "load_container;start_container;start_application;stop_container;remove_container;remove_image;image_size;application;service;date_time" >"logs/$log_file"
