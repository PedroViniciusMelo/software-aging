#!/usr/bin/env bash

# ------------------------------------------------- DEPENDENCIES -----------------------------------------
wget "http://ftp.us.debian.org/debian/pool/main/liba/libaio/libaio1_0.3.113-4_amd64.deb" && {
    dpkg -i libaio1_0.3.113-4_amd64.deb

    apt install libncurses5
}

wget "https://archive.mariadb.org/mariadb-10.4.12/bintar-linux-systemd-x86_64/mariadb-10.4.12-linux-systemd-x86_64.tar.gz" && {
    mv mariadb-10.4.12-linux-systemd-x86_64.tar.gz /usr/local/mariadb-10.4.12-linux-systemd-x86_64.tar.gz
}
# --------------------------------------------------------------------------------------------------------

cd /usr/local || echo "erro ao entrar em /usr/local"

tar -zxvpf mariadb-10.4.12-linux-systemd-x86_64.tar.gz

rm -rf mariadb-10.4.12-linux-systemd-x86_64.tar.gz

ln -s mariadb-10.4.12-linux-systemd-x86_64.tar.gz mysql

cd mysql || echo "erro ao entrar em /usr/local/mysql"

groupadd mysql
useradd -g mysql mysql

./scripts/mariadb-install-db --user=mysql

chown -R root .
chown -R mysql data

export PATH=$PATH:/usr/local/mysql/bin/

echo "export PATH=\$PATH:/usr/local/mysql/bin/" >> /root/.bashrc; source /root/.bashrc

cp support-files/mysql.server /etc/init.d/mysql.server

cp support-files/systemd/mariadb.service /usr/lib/systemd/system/mariadb.service

# ------------------------------------------------------------------
mkdir /etc/systemd/system/mariadb.service.d/

cat > /etc/systemd/system/mariadb.service.d/datadir.conf <<EOF
[Service]
ReadWritePaths=/usr/local/mysql/data
EOF

systemctl daemon-reload
# ------------------------------------------------------------------

systemctl start mariadb.service
systemctl enable mariadb.service

mysql < ./setup.sql

mkdir -p /etc/mysql
ln -s /usr/local/mariadb-10.4.12-linux-systemd-x86_64/mysql-test/suite/storage_engine/my.cnf /etc/mysql/my.cnf
cp /etc/mysql/my.cnf /etc/mysql/my.cnf.backup

echo "[mysqld]" >> /etc/mysql/my.cnf
echo "max_connections = 2048" >> /etc/mysql/my.cnf

systemctl restart mariadb.service