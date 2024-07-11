#!/usr/bin/env bash

MARIADB_VERSION=1:10.4.12+maria~bionic
MARIADB_MAJOR=10.4

GPG_KEYS=177F4010FE56CA3336300305F1656F24C74CD1D8

sed -i "s/max_connections.*/max_connections = 2048/g" /etc/mysql/my.cnf

cp setup.sql /docker-entrypoint-initdb.d/setup_0.sql

set -ex
{
    echo "mariadb-server-$MARIADB_MAJOR" mysql-server/root_password password 'unused'
    echo "mariadb-server-$MARIADB_MAJOR" mysql-server/root_password_again password 'unused'

} | debconf-set-selections

apt update;  apt install -y "mariadb-server=$MARIADB_VERSION" mariadb-backup socat 
rm -rf /var/lib/apt/lists/*
sed -ri 's/^user\s/#&/' /etc/mysql/my.cnf /etc/mysql/conf.d/*
rm -rf /var/lib/mysql
mkdir -p /var/lib/mysql /var/run/mysqld
chown -R mysql:mysql /var/lib/mysql /var/run/mysqld
chmod 777 /var/run/mysqld
find /etc/mysql/ -name '*.cnf' -print0   | xargs -0 grep -lZE '^(bind-address|log)' | xargs -rt -0 sed -Ei 's/^(bind-address|log)/#&/'
echo '[mysqld]\nskip-host-cache\nskip-name-resolve' > /etc/mysql/conf.d/docker.cnf                                                                                                                                                                                                                                                                                                                                                       

set -e
echo "deb http://ftp.osuosl.org/pub/mariadb/repo/$MARIADB_MAJOR/ubuntu bionic main" > /etc/apt/sources.list.d/mariadb.list
{   
    echo 'Package: *'
    echo 'Pin: release o=MariaDB'
    echo 'Pin-Priority: 999'
    
} > "/etc/apt/preferences.d/mariadb"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

set -ex
GNUPGHOME="$(mktemp -d)"
export GNUPGHOME

for key in $GPG_KEYS; do
    gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"

done
gpg --batch --export $GPG_KEYS > /etc/apt/trusted.gpg.d/mariadb.gpg

command -v gpgconf > /dev/null && gpgconf --kill all || :
rm -r "$GNUPGHOME"
apt-key list                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       4.95kB    
