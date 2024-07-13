#!/usr/bin/env bash

rm -r /usr/local/tomcat*

[[ "$(pwd)" != "/root/teastore-no-containers" ]] && {
    echo -e "\nverifique o pwd, execute diretamente no diretorio ./teastore-no-containers: pwd = $(pwd)"; exit 1
}

PWD_PATH="$(pwd)/Tea"


GET_DEPENDENCIES() {
    local pwd_tea_base; pwd_tea_base=$(pwd)

    # shellcheck disable=SC1091
    bash "$pwd_tea_base/Tea/tools.descartes.teastore.dockerbase/tea_base.sh"

    cp -r "/usr/local/tomcat" "/root/tomcat-cp"
}


CONFIG_ALL_SERVICES() {
    echo 'export JAVA_HOME="/usr/local/openjdk-11"' >> /root/.bashrc
    echo "export PATH=\$PATH:$JAVA_HOME/bin" >> /root/.bashrc

    # shellcheck disable=SC1091
    source /root/.bashrc

    # -------------------------- REGISTRY SERVICE
    reset; echo -e "\nconfigure and start service: registry\n"
    bash "$PWD_PATH/registry.sh"
    cp -r "/root/tomcat-cp" "/usr/local/tomcat"

    # -------------------------- WEBUI SERVICE
    echo -e "\nconfigure and start service: webui\n"
    bash "$PWD_PATH/webui.sh"
    cp -r "/root/tomcat-cp" "/usr/local/tomcat"

    # -------------------------- PERSISTENCE SERVICE
    echo -e "\nconfigure and start service: persistence\n"
    bash "$PWD_PATH/persistence.sh"
    cp -r "/root/tomcat-cp" "/usr/local/tomcat"

    # -------------------------- AUTH SERVICE
    echo -e "\nconfigure and start service: auth\n"
    bash "$PWD_PATH/auth.sh"
    cp -r "/root/tomcat-cp" "/usr/local/tomcat"

    # -------------------------- IMAGE SERVICE
    echo -e "\nconfigure and start service: image\n"
    bash "$PWD_PATH/image.sh"
    cp -r "/root/tomcat-cp" "/usr/local/tomcat"

    # -------------------------- RECOMMENDER SERVICE
    echo -e "\nconfigure and start service: recommender\n"
    bash "$PWD_PATH/recommender.sh"
}


MAIN() {
    GET_DEPENDENCIES
    CONFIG_ALL_SERVICES
}

MAIN
rm -r /root/tomcat-cp