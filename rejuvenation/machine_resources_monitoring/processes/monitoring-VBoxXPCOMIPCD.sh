#!/bin/bash

# Script to continuously monitor the resource usage of the VirtualBox XPCOM process (VBoxXPCOMIPCD) and to log the relevant metrics to a CSV file
# VBoxXPCOMIPCD is responsible for interprocess communication between guests and management applications on non-Windows hosts.

while true; do
  :

  pidXPCO=$(pidof -s VBoxXPCOMIPCD)
  date_time=$(date +%d-%m-%Y-%H:%M:%S)

  if [ -n "$pidXPCO" ]; then
    data=$(pidstat -u -h -p $pidXPCO -T ALL -r 1 1 | sed -n '4p')
    thread=$(cat /proc/"$pidXPCO"/status | grep Threads | awk '{print $2}')
    cpu=$(echo "$data" | awk '{print $8}')
    mem=$(echo "$data" | awk '{print $14}')
    vmrss=$(echo "$data" | awk '{print $13}')
    vsz=$(echo "$data" | awk '{print $12}')
    swap=$(cat /proc/"$pidXPCO"/status | grep Swap | awk '{print $2}')

    echo "$cpu;$mem;$vmrss;$vsz;$thread;$swap;$date_time" >> logs/vbox_monitoring-VBoxXPCOMIPCD.csv
  else
    sleep 1
    echo "0;0;0;0;0;0;0" >> logs/vbox_monitoring-VBoxXPCOMIPCD.csv
  fi

done
