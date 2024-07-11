#!/usr/bin/env bash

PID_SERVICES=()

GET_DEPENDENCIES() {
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/tea_base.sh

    cp -r /usr/local/tomcat /root/tomcat-cp
}

START_SERVICES() {
    local service=$1

    # shellcheck disable=SC1090
    source ./Tea/tools.descartes.teastore.dockerbase/"$service"
}

CONFIG_ALL_SERVICES() {
    echo 'export JAVA_HOME="/usr/local/openjdk-11"' >> /root/.bashrc
    echo "export PATH=\$PATH:$JAVA_HOME/bin" >> /root/.bashrc

    echo "export CATALINA_HOME=/usr/local/tomcat" >> /root/.bashrc
    echo "export TOMCAT_HEAP_MEM_PERCENTAGE=50" >> /root/.bashrc

    # shellcheck disable=SC1091
    source /root/.bashrc

    # -------------------------- REGISTRY SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/registry.sh && {
        START_SERVICES start-registry.sh &
        PID_SERVICES+=($!)
        cp -r /root/tomcat-cp /usr/local/tomcat
    }

    # -------------------------- WEBUI SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/webui.sh && {
        START_SERVICES start-webui.sh &
        PID_SERVICES+=($!)
        cp -r /root/tomcat-cp /usr/local/tomcat
    }

    # -------------------------- PERSISTENCE SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/persistence.sh && {
        START_SERVICES start-persistence.sh &
        PID_SERVICES+=($!)
        cp -r /root/tomcat-cp /usr/local/tomcat
    }

    # -------------------------- AUTH SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/auth.sh && {
        START_SERVICES start-auth.sh &
        PID_SERVICES+=($!)
        cp -r /root/tomcat-cp /usr/local/tomcat
    }

    # -------------------------- IMAGE SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/image.sh && {
        START_SERVICES start-image.sh &
        PID_SERVICES+=($!)
        cp -r /root/tomcat-cp /usr/local/tomcat
    }

    # -------------------------- RECOMMENDER SERVICE
    # shellcheck disable=SC1091
    source ./Tea/tools.descartes.teastore.dockerbase/recommender.sh && {
        START_SERVICES start-recommender.sh &
        PID_SERVICES+=($!)
    }

    # REMOVING TOMCAT-BASE
    rm -r /root/tomcat-cp
}


MAIN() {
    GET_DEPENDENCIES
    CONFIG_ALL_SERVICES

    echo -e "\nMostrando pids em segundo plano dos serviços executados\n"
    local i=1
    export PID_SERVICES && {
        for pid in "${PID_SERVICES[@]}"; do
            echo "serviço $i: $pid"
            ((i+=1))
        done
    }
    echo -e "\nArray pids dos serviços, use kill -9 (pid) em cada pid para matar serviço: $PID_SERVICES\n"
}

MAIN