#!/usr/bin/env bash

CATALINA_HOME=$( pwd )
TOMCATS_ARRAY=("tomcat-registry" "tomcat-webui" "tomcat-persistence" "tomcat-image" "tomcat-auth" "tomcat-recommender")
TOMCAT_HEAP_MEM_PERCENTAGE=50

reset

for index in ${TOMCATS_ARRAY[@]}; do

    echo "iniciando $index"; sleep 1.5

    rm -r $CATALINA_HOME/$index/kieker
    mkdir -p $CATALINA_HOME/$index/kieker
    mkdir -p $CATALINA_HOME/$index/kieker/config
    mkdir -p $CATALINA_HOME/$index/kieker/agent
    mkdir -p $CATALINA_HOME/$index/kieker/logs

    cp $CATALINA_HOME/kieker.monitoring.properties $CATALINA_HOME/$index/kieker/config/kieker.monitoring.properties
    cp $CATALINA_HOME/aop.xml $CATALINA_HOME/$index/lib/aop.xml
    cp $CATALINA_HOME/kieker-1.15-SNAPSHOT-aspectj.jar $CATALINA_HOME/$index/kieker/agent/agent.jar

    rm $CATALINA_HOME/$index/lib/websocket-api.jar
    rm $CATALINA_HOME/$index/lib/tomcat-websocket.jar

    cp $CATALINA_HOME/baseContext.xml $CATALINA_HOME/$index/conf/context.xml
    cp $CATALINA_HOME/mysql-connector-java-5.1.44-bin.jar $CATALINA_HOME/$index/lib/mysql-connector-java-5.1.44-bin.jar
    cp $CATALINA_HOME/dockermemoryconfigurator-1.4.2.jar $CATALINA_HOME/$index/bin/dockermemoryconfigurator.jar

    rm -r $CATALINA_HOME/$index/conf/server.xml
    cp $CATALINA_HOME/server.xml $CATALINA_HOME/$index/conf/server.xml

    cp -r $CATALINA_HOME/ssl $CATALINA_HOME/$index

    # Copy script to replace placeholders in context.xml with the environment variables
    cp $CATALINA_HOME/start.sh $CATALINA_HOME/$index/bin/start.sh
    chmod +x $CATALINA_HOME/$index/bin/start.sh

    # Import self-signed certificate for HTTPS teastore into keystore. NOTE: to adapt certificate hostnames, configure and run ssl/generate_cert.sh and rebuild the container!
    keytool -import -noprompt -trustcacerts -alias teastoressl -file $CATALINA_HOME/$index/ssl/cert.pem -keystore "/usr/local/jdk-17.0.11/lib/security/cacerts" -storepass changeit

    sed -i "/JDK_JAVA_OPTIONS.*/d" $CATALINA_HOME/$index/bin/catalina.sh

    if [[ "$index" == "registry" ]]; then
        cp $CATALINA_HOME/teastores_war/*.registry.war $CATALINA_HOME/$index/webapps
    
    elif [[ "$index" == "webui" ]]; then
        cp $CATALINA_HOME/teastores_war/*.webui.war $CATALINA_HOME/$index/webapps

    elif [[ "$index" == "persistence" ]]; then
        cp $CATALINA_HOME/teastores_war/*.persistence.war $CATALINA_HOME/$index/webapps

    elif [[ "$index" == "image" ]]; then
        cp $CATALINA_HOME/teastores_war/*.image.war $CATALINA_HOME/$index/webapps

    elif [[ "$index" == "auth" ]]; then
        cp $CATALINA_HOME/teastores_war/*.auth.war $CATALINA_HOME/$index/webapps

    elif [[ "$index" == "recommender" ]]; then
        cp $CATALINA_HOME/teastores_war/*.recommender.war $CATALINA_HOME/$index/webapps

    fi

    # java -jar $CATALINA_HOME/$index/bin/dockermemoryconfigurator.jar ${TOMCAT_HEAP_MEM_PERCENTAGE};
    # bash $CATALINA_HOME/$index/bin/start.sh $CATALINA_HOME/$index
    # bash $CATALINA_HOME/$index/bin/catalina.sh run

done