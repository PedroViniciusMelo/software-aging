#!/bin/bash

# Verifica se o parâmetro foi passado
if [ -z "$1" ]; then
  echo "Erro: O caminho para o arquivo de configuração deve ser fornecido como argumento."
  exit 1
fi

CONFIG_PATH=$1
CATALINA_HOME=$(pwd)

# Configuração do serviço
SERVICE_PORT=8080
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

# Debug: Verifica se a variável CONFIG_PATH está correta
echo "CONFIG_PATH é: $CONFIG_PATH"
echo "Arquivo alvo: ${CONFIG_PATH}/conf/context.xml"

# Certifique-se de que o arquivo tenha terminações de linha no estilo UNIX
sed -i "s/<Environment name=\"servicePort\" value=.*/<Environment name=\"servicePort\" value=\"${SERVICE_PORT}\"/g" "${CONFIG_PATH}/conf/context.xml"

# Verifica se o comando sed foi executado com sucesso
if [ $? -eq 0 ]; then
  echo "O comando sed foi executado com sucesso."
else
  echo "Erro: O comando sed falhou."
fi
