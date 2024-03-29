#!/bin/bash

# Script to continuously monitor the VirtualBox service process (VBoxSVC) and log resource usage to a CSV file 
# VBoxSVC keeps track of all virtual machines that are running on the host and is started automatically when the first guest boots

#echo $pid
while true; do
  :

  pidSVC=$(pidof -s VBoxSVC)
  date_time=$(date +%d-%m-%Y-%H:%M:%S)

  if [ -n "$pidSVC" ]; then
    data=$(pidstat -u -h -p $pidSVC -T ALL -r 1 1 | sed -n '4p')
    thread=$(cat /proc/"$pidSVC"/status | grep Threads | awk '{print $2}')
    cpu=$(echo "$data" | awk '{print $8}')
    mem=$(echo "$data" | awk '{print $14}')
    vmrss=$(echo "$data" | awk '{print $13}')
    vsz=$(echo "$data" | awk '{print $12}')
    swap=$(cat /proc/"$pidSVC"/status | grep Swap | awk '{print $2}')

    echo "$cpu;$mem;$vmrss;$vsz;$thread;$swap;$date_time" >> logs/vbox_monitoring-VBoxSVC.csv
  else
    sleep 1
    echo "0;0;0;0;0;0;0" >> logs/vbox_monitoring-VBoxSVC.csv
  fi

done