#!/bin/bash

# Script to continuously monitor the VirtualBox headless process (VBoxHeadless) and log resource usage to a CSV file
# VBoxHeadless is responsible for running virtual machines without a graphical user interface

#echo $pid
while true; do
  :

  pidHead=$(pidof -s VBoxHeadless)
  date_time=$(date +%d-%m-%Y-%H:%M:%S)

  if [ -n "$pidHead" ]; then
    data=$(pidstat -u -h -p $pidHead -T ALL -r 1 1 | sed -n '4p')
    thread=$(cat /proc/"$pidHead"/status | grep Threads | awk '{print $2}')
    cpu=$(echo "$data" | awk '{print $8}')
    mem=$(echo "$data" | awk '{print $14}')
    vmrss=$(echo "$data" | awk '{print $13}')
    vsz=$(echo "$data" | awk '{print $12}')
    swap=$(cat /proc/"$pidHead"/status | grep Swap | awk '{print $2}')

    echo "$cpu;$mem;$vmrss;$vsz;$thread;$swap;$date_time" >> logs/vbox_monitoring-VBoxHeadless.csv
  else
    sleep 1
    echo "0;0;0;0;0;0;0" >> logs/vbox_monitoring-VBoxHeadless.csv
  fi
done