#!/usr/bin/env bash

# REFERENCES:
# https://wiki.xenproject.org/wiki/Xen_Project_Beginners_Guide
# https://xen-tools.org/software/xen-tools/
# https://wiki.debian.org/LVM#List_of_VG_commands

# ############################## IMPORTS #############################
source ../../machine_resources_monitoring/general_dependencies.sh
# ####################################################################

# FUNCTION=SYSTEM_UPDATE()
# DESCRIPTION:
# Attempts to update the host's repositories and system apps
SYSTEM_UPDATE() {
  apt-get update && apt-get upgrade
} 

# FUNCTION=INSTALL_XEN_AND_DEPENDENCIES()
# DESCRIPTION:
# Installs Xen dependencies if not already installed
INSTALL_XEN_DEPENDENCIES() {
  if ! which xen-system >/dev/null; then
    apt-get install xen-system -y
  fi
}

# FUNCTION=INSTALL_UTILS()
# DESCRIPTION:
# Installs recommended tools for the setup of the Xen hypervisor in a Debian host
#
# xen-tools: This package will allow the creation of new guest Xen domains on a Debian host
# lvm2: Allows the management of storage devices in a more abstract manner using LVM or 'Linux Logical Volume Manager'
# net-tools: Includes the important tools for controlling the Linux kernel's networking subsystem
# bridge-utils: Acts as a virtual switch, enabling the attachment of VMs to the external network
INSTALL_UTILS(){
  apt install xen-tools lvm2 net-tools bridge-utils
}

# FUNCTION=CONFIGURE_GRUB_FOR_XEN()
# DESCRIPTION:
# Configures GRUB to set up boot priority for Xen, modifying the default Linux GRUB script
# Ensures that Xen is initialized along with the system and that it has access to the hardware components
CONFIGURE_GRUB_FOR_XEN(){
  dpkg-divert --divert /etc/grub.d/08_linux_xen --rename /etc/grub.d/20_linux_xen
  update-grub
}

# FUNCTION=NETWORK_CONFIG()
# DESCRIPTION:
# Creates a bridge interface (xenbr0) in the dom0, connects it to the default network interface of the host by altering the '/etc/network/interfaces' file
NETWORK_CONFIG(){
  local config_file="/etc/network/interfaces"
  local default_interface=$(ip -o -4 route show to default | awk '{print $5}' | grep -v '^lo$' | grep -v '^vir' | head -n 1)

    if [ -z "$default_interface" ]; then
        echo "Error: No proper network interface found."
        exit 1
    fi

    echo "Updating network configuration file..."
    cat > "$config_file" <<EOL

# This file describes the network interfaces available on your system 
# and how to activate them. For more information, see interfaces (5).

source /etc/network/interfaces.d/*

# The loopback network interface
auto lo
iface lo inet loopback

# The primary network interface
allow-hotplug enp0s3
iface enp0s3 inet manual

auto xenbr0
iface xenbr0 inet dhcp
    bridge_ports $default_interface
EOL

  service networking restart
}

# REDIRECT_PORTS()
# DESCRIPTION:
# Redirect SSH traffic from port 2222 on the host to port 22 on the Xen domU
# Redirect HTTP traffic from port 8080 on the host to port 80 on the Xen domU
# Check if the redirection rules are correctly applied
REDIRECT_PORTS(){
  sudo iptables -t nat -A PREROUTING -p tcp --dport 2222 -j DNAT --to <domU_IP>:22

  sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to <domU_IP>:80

  if sudo iptables -t nat -L | grep -qE '(to:10.0.2.17:22|to:10.0.2.17:80)'; then
    echo "Port redirection rules have been successfully applied."
  else
    echo "Failed to apply port redirection rules. Please check iptables configuration."
  fi
}

# FUNCTION=STORAGE_SETUP()
# Configures /dev/sda4 to be the physical volume of LVM or 'Linux Logical Volume Manager' in order to 
# set up foundation for creating disks for future VMs
# 
# Useful definitions:
# PV - Physical Volumes. This means the hard disk, hard disk partitions, RAID or LUNs from a SAN which form "Physical Volumes" (or PVs)
# VG - Volume Groups. This is a collection of one or more Physical Volumes
# LV - Logical Volumes. LVs sit inside a Volume Group and form, in effect, a virtual partition
#
# LVM COMMANDS:
# pvcreate - declares /dev/sda4 as a physical volume available for the LVM
# vgcreate - creates a volume group called 'vg0'
#
# REMINDER: Before using this function, ensure that /dev/sda4 is a dedicated partition you created for LVM use
STORAGE_SETUP() { 
    pvcreate /dev/sda4
    vgcreate vg0 /dev/sda4
}

DEPENDENCIES_MAIN(){
  SYSTEM_UPDATE
  INSTALL_GENERAL_DEPENDENCIES
  INSTALL_XEN_DEPENDENCIES
  INSTALL_UTILS
  CONFIGURE_GRUB_FOR_XEN
  NETWORK_CONFIG
  STORAGE_SETUP
  reboot now
}

DEPENDENCIES_MAIN

