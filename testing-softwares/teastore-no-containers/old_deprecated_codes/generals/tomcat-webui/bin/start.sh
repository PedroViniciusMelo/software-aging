#!/bin/bash

CONFIG_PATH=$1
CATALINA_HOME=$( pwd )

# Service configuration. SERVICE_PORT is the host port, mapped to the exposed docker port. HOST_NAME
# is the IP address or hostname of the host running the docker container
SERVICE_PORT=8181
HOST_NAME=unset
USE_POD_IP=false
REGISTRY_HOST=127.0.0.1
REGISTRY_PORT=8080
DB_HOST=127.0.0.1
DB_PORT=3306
RECOMMENDER_RETRAIN_LOOP_TIME=0
RECOMMENDER_ALGORITHM=SlopeOne
PROXY_NAME=unset
PROXY_PORT=unset
TOMCAT_HEAP_MEM_PERCENTAGE=50
LOG_TO_FILE=false
RABBITMQ_HOST=unset
USE_HTTPS=false

echo "${CONFIG_PATH}/conf/context.xml"

# DANGER! MAKE SURE THIS FILE HAS UNIX-STYLE LINE-ENDINGS OR THE DOCKER CONTAINER WILL NOT START!
sed -i "s/<Environment name=\"servicePort\" value=.*/<Environment name=\"servicePort\" value=\"${SERVICE_PORT}\"/g" "${CONFIG_PATH}/conf/context.xml"

if [ "$USE_HTTPS" == "true" ]
then
  sed -i "s|<Environment name=\"registryURL\" value=.*|<Environment name=\"registryURL\" value=\"https://${REGISTRY_HOST}:${REGISTRY_PORT}/tools.descartes.teastore.registry/rest/services/\"|g" "${CONFIG_PATH}/conf/context.xml"
  sed -i 's|<Connector port="8009" protocol="AJP/1.3" redirectPort="8443" secretRequired="false" />|<Connector port="8009" protocol="AJP/1.3" redirectPort="8443" secretRequired="false" />\n\n    <Connector port="8443" protocol="org.apache.coyote.http11.Http11AprProtocol" maxThreads="150" SSLEnabled="true" defaultSSLHostConfigName="hostname.unset">\n        <UpgradeProtocol className="org.apache.coyote.http2.Http2Protocol" />\n        <SSLHostConfig hostName="hostname.unset">\n            <Certificate certificateKeyFile="ssl/key.pem"\n                         certificateFile="ssl/cert.pem"/>\n        </SSLHostConfig>\n    </Connector>|g' ${CONFIG_PATH}/conf/server.xml
else
  sed -i "s|<Environment name=\"registryURL\" value=.*|<Environment name=\"registryURL\" value=\"http://${REGISTRY_HOST}:${REGISTRY_PORT}/tools.descartes.teastore.registry/rest/services/\"|g" "${CONFIG_PATH}/conf/context.xml"
fi
if [ "$HOST_NAME" != "unset" ]
then
	sed -i "s|<Environment name=\"hostName\" value=.*|<Environment name=\"hostName\" value=\"${HOST_NAME}\" type=\"java.lang.String\" override=\"false\"/>|g" "${CONFIG_PATH}/conf/context.xml"
	if [ "$USE_HTTPS" == "true" ]
  then
	  sed -i "s|<Connector port=\"8443\" protocol=\"org.apache.coyote.http11.Http11AprProtocol\".*|<Connector port=\"8443\" protocol=\"org.apache.coyote.http11.Http11AprProtocol\" maxThreads=\"150\" SSLEnabled=\"true\" defaultSSLHostConfigName=\"${HOST_NAME}\">|g" ${CONFIG_PATH}/conf/server.xml
	  sed -i "s|<SSLHostConfig hostName=.*|<SSLHostConfig hostName=\"${HOST_NAME}\">|g" ${CONFIG_PATH}/conf/server.xml
	fi
else
	sed -i "/Environment name=\"hostName\"/d" "${CONFIG_PATH}/conf/context.xml"
fi
sed -i "s|<Environment name=\"useHostIP\" value=.*|<Environment name=\"useHostIP\" value=\"${USE_POD_IP}\"|g" "${CONFIG_PATH}/conf/context.xml"
sed -i "s/<Environment name=\"databaseHost\" value=.*/<Environment name=\"databaseHost\" value=\"${DB_HOST}\"/g" "${CONFIG_PATH}/conf/context.xml"
sed -i "s/<Environment name=\"databasePort\" value=.*/<Environment name=\"databasePort\" value=\"${DB_PORT}\"/g" "${CONFIG_PATH}/conf/context.xml"
sed -i "s/<Environment name=\"recommenderLoopTime\" value=.*/<Environment name=\"recommenderLoopTime\" value=\"${RECOMMENDER_RETRAIN_LOOP_TIME}\"/g" "${CONFIG_PATH}/conf/context.xml"
sed -i "s/<Environment name=\"recommenderAlgorithm\" value=.*/<Environment name=\"recommenderAlgorithm\" value=\"${RECOMMENDER_ALGORITHM}\"/g" "${CONFIG_PATH}/conf/context.xml"
if [ "$PROXY_NAME" != "unset" ] && [ "$PROXY_PORT" != "unset" ]
then
	sed -i "s/<Connector port=\"8080\" protocol=\"HTTP\/1.1\".*/<Connector port=\"8080\" protocol=\"HTTP\/1.1\" proxyName=\"${PROXY_NAME}\" proxyPort=\"${PROXY_PORT}\"/g" ${CONFIG_PATH}/conf/server.xml
fi
sed -i 's/securerandom.source=file:\/dev.*/securerandom.source=file:\/dev\/urandom/g'  ${JAVA_HOME}/conf/security/java.security

if [ "$RABBITMQ_HOST" != "unset" ]
then
  sed -i "s/kieker.monitoring.writer=kieker.monitoring.writer.filesystem.FileWriter/kieker.monitoring.writer=kieker.monitoring.writer.collector.ChunkingCollector/g" "${CONFIG_PATH}/kieker/config/kieker.monitoring.properties"
  sed -i "s/kieker.monitoring.writer.amqp.ChunkingAmqpWriter.uri=amqp:\/\/admin:nimda@RABBITMQ_PORT_PLACEHOLDER/kieker.monitoring.writer.amqp.ChunkingAmqpWriter.uri=amqp:\/\/admin:nimda@${RABBITMQ_HOST}/g" "${CONFIG_PATH}/kieker/config/kieker.monitoring.properties"
fi

if [ "$LOG_TO_FILE" != "true" ] && [ "$RABBITMQ_HOST" == "unset" ]
then
  sed -i 's/kieker.monitoring.enabled=true/kieker.monitoring.enabled=false/g' "${CONFIG_PATH}/kieker/config/kieker.monitoring.properties"
fi

touch ${CONFIG_PATH}/bin/setenv.sh
chmod +x ${CONFIG_PATH}/bin/setenv.sh
echo "export JAVA_OPTS='-javaagent:${CONFIG_PATH}/kieker/agent/agent.jar --add-opens=java.base/java.lang=ALL-UNNAMED -Dkieker.monitoring.configuration="${CONFIG_PATH}/kieker/config/kieker.monitoring.properties" -Daj.weaving.verbose=false -Dorg.aspectj.weaver.loadtime.configuration=aop.xml -Dkieker.monitoring.skipDefaultAOPConfiguration=true -Daj.weaving.loadersToSkip=java.net.URLClassLoader -Dcom.sun.jndi.ldap.object.disableEndpointIdentification=true'" > $CONFIG_PATH/bin/setenv.sh
echo "export CLASSCONFIG_PATH=$CLASSCONFIG_PATH:$CATALINA_HOME/slf4j-simple-1.7.21.jar" >> ${CONFIG_PATH}/bin/setenv.sh

