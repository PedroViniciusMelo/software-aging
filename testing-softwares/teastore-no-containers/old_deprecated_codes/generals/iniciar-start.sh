#!/bin/bash

reset

# export CATALINA_HOME="$CATALINA_HOME/tomcat-registry"
# export CATALINA_HOME="$CATALINA_HOME/tomcat-webui"
# export CATALINA_HOME="$CATALINA_HOME/tomcat-persistence"
# export CATALINA_HOME="$CATALINA_HOME/tomcat-image"
# export CATALINA_HOME="$CATALINA_HOME/tomcat-auth"
# export CATALINA_HOME="$CATALINA_HOME/tomcat-recommender"

export CATALINA_HOME=$( pwd )/$1

echo "aqui o echo $CATALINA_HOME"

java -jar $CATALINA_HOME/bin/dockermemoryconfigurator.jar 50;
bash $CATALINA_HOME/bin/start.sh $CATALINA_HOME && bash $CATALINA_HOME/bin/catalina.sh run

sleep 2