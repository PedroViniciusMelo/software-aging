#!/usr/bin/env bash

# tomcat-registry
# tomcat-webui
# tomcat-auth
# tomcat-persistence
# tomcat-recommender
# tomcat-image
# tomcat-db
TOMCAT_INSTANCE="tomcat-$1"

TARGET_WAR="TeaStore/services/tools.descartes.teastore.auth/target/*.war"
TOMCAT_WEBAPPS="tomcat/webapps"

TOMCAT_START="/usr/local/tomcat/bin/start.sh"
CATALINA_RUN="/usr/local/tomcat/bin/catalina.sh run"

# /usr/local/tomcat/bin/dockermemoryconfigurator.jar
PATH_DOCKER_MEM_CONFIG="path_to_dockermemoryconfigurator.jar"

TOMCAT_SHA512=aee019d70807b6fe25e6f4e2a36a495337c8010992af7bdeedce1bc7f6affa12899257265ecb71a46e86c20f4bd6635d4dd4c014deb3e80c518c041b3e53376b
TOMCAT_VERSION=10.0.7
TOMCAT_MAJOR=10
GPG_KEYS=A9C5DF4D22E99998D9875A5110C01C5A2F6059E7
LD_LIBRARY_PATH=/usr/local/tomcat/native-jni-lib
TOMCAT_NATIVE_LIBDIR=/usr/local/tomcat/native-jni-lib
WORKDIR=/usr/local/tomcat
TOMCAT_PATH=/usr/local/tomcat/bin:/usr/local/openjdk-11/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CATALINA_HOME=/usr/local/tomcat

JAVA_VERSION=11.0.11+9
LANG=C.UTF-8
JAVA_PATH=/usr/local/openjdk-11/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
JAVA_HOME=/usr/local/openjdk-11

GNUPGHOME="$(mktemp -d)"
export GNUPGHOME

CP_TARGET_WAR() {
    local war=$1
    local webapps=$2
    # shellcheck disable=SC2086
    cp $war $webapps
}

CP_DOCKER_MEM_CONFIG() {
    local tomcat_heap=$1
    local docker_mem_config="$PATH_DOCKER_MEM_CONFIG"

    cp target/jars/dockermemoryconfigurator* /usr/local/tomcat/bin/dockermemoryconfigurator.jar

    java -jar $docker_mem_config "$tomcat_heap"
}

START_TOMCAT_CATALINA() {
    bash "$TOMCAT_START"
    bash "$CATALINA_RUN"
}

JDK_JAVA_OPTIONS() {
    sed -i "/JDK_JAVA_OPTIONS.*/d" /usr/local/tomcat/bin/catalina.sh
}

CP_SLF4J() {
    cp slf4j-simple-1.7.21.jar /usr/local/slf4j-simple-1.7.21.jar
}

CP_KIEKER() {
    cp kieker-1.15-SNAPSHOT-aspectj.jar /kieker/agent/agent.jar
}

CP_AOP() {
    cp aop.xml /usr/local/tomcat/lib/aop.xml
}

CP_KIEKER_PROPERTIES() {
    cp kieker.monitoring.properties /kieker/config/kieker.monitoring.properties
}

KEYTOOL() {
    bash keytool -import -noprompt -trustcacerts -alias teastoressl -file /usr/local/tomcat/ssl/cert.pem -keystore "/usr/local/openjdk-11/lib/security/cacerts" -storepass changeit
}

CP_START_SH() {
    cp start.sh /usr/local/tomcat/bin/start.sh
    chmod +x /usr/local/tomcat/bin/start.sh
}

CP_SSL() {
    cp ssl/ /usr/local/tomcat/ssl
}

CP_SERVER_XML() {
    rm -r /usr/local/tomcat/conf/server.xml
    cp server.xml /usr/local/tomcat/conf/
}

CP_MYSQL() {
    cp mysql-connector-java-5.1.44-bin.jar /usr/local/tomcat/lib/mysql-connector-java-5.1.44-bin.jar
}

CP_CONTEXT_XML() {
    rm -r /usr/local/tomcat/conf/context.xml
    cp baseContext.xml /usr/local/tomcat/conf/context.xml
}

RM_WEBSOCKETS() {
    rm -r /usr/local/tomcat/lib/tomcat-websocket.jar
    rm -r /usr/local/tomcat/lib/websocket-api.jar
}

NATIVE_LINES() {
    set -eux

    nativeLines="$(catalina.sh configtest 2>&1)"
    nativeLines="$(echo "$nativeLines" | grep 'Apache Tomcat Native')"
    nativeLines="$(echo "$nativeLines" | sort -u)"

    if ! echo "$nativeLines" | grep -E 'INFO: Loaded( APR based)? Apache Tomcat Native library' >&2; then
        echo >&2 "$nativeLines"
        exit 1

    fi
}

GENERAL_DEPENDENCIES() {
    set -eux

    savedAptMark="$(apt-mark showmanual)"
    apt update
    apt install -y --no-install-recommends gnupg dirmngr wget ca-certificates

    ddist() {
        local f="$1"
        shift
        local distFile="$1"
        shift
        local mvnFile="${1:-}"
        local success=
        local distUrl=

        for distUrl in "https://www.apache.org/dyn/closer.cgi?action=download&filename=$distFile" "https://www-us.apache.org/dist/$distFile" "https://www.apache.org/dist/$distFile" "https://archive.apache.org/dist/$distFile" ${mvnFile:+"https://repo1.maven.org/maven2/org/apache/tomcat/tomcat/$mvnFile"}; do
            if wget -O "$f" "$distUrl" --progress=dot:giga && [ -s "$f" ]; then
                success=1
                break
            fi
        done

        [ -n "$success" ]
    }

    ddist 'tomcat.tar.gz' "tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz" "$TOMCAT_VERSION/tomcat-$TOMCAT_VERSION.tar.gz"
    echo "$TOMCAT_SHA512 *tomcat.tar.gz" | sha512sum --strict --check -
    ddist 'tomcat.tar.gz.asc' "tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz.asc" "$TOMCAT_VERSION/tomcat-$TOMCAT_VERSION.tar.gz.asc"

    for key in $GPG_KEYS; do
        gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"
    done

    gpg --batch --verify tomcat.tar.gz.asc tomcat.tar.gz
    tar -xf tomcat.tar.gz --strip-components=1
    rm bin/*.bat
    rm tomcat.tar.gz*

    command -v gpgconf && gpgconf --kill all || :
    rm -rf "$GNUPGHOME"

    mv webapps webapps.dist
    mkdir -p webapps

    nativeBuildDir="$(mktemp -d)"

    tar -xf bin/tomcat-native.tar.gz -C "$nativeBuildDir" --strip-components=1

    apt-get install -y --no-install-recommends dpkg-dev gcc libapr1-dev libssl-dev make

    export CATALINA_HOME="$PWD"
    cd "$nativeBuildDir/native"

    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"
    aprConfig="$(command -v apr-1-config)"

    ./configure --build="$gnuArch" --libdir="$TOMCAT_NATIVE_LIBDIR" --prefix="$CATALINA_HOME" --with-apr="$aprConfig" --with-java-home="$JAVA_HOME" --with-ssl=yes
    make -j "$(nproc)"
    make install
    rm -rf "$nativeBuildDir"
    rm bin/tomcat-native.tar.gz
    apt-mark auto '.*' >/dev/null

    [ -z "$savedAptMark" ] || apt-mark manual "$savedAptMark" >/dev/null

    find "$TOMCAT_NATIVE_LIBDIR" -type f -executable -exec ldd '{}' ';' | awk '/=>/ { print $(NF-1) }' | xargs -rt readlink -e | sort -u | xargs -rt dpkg-query --search | cut -d: -f1 | sort -u | xargs -r apt-mark manual

    apt purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false

    rm -rf /var/lib/apt/lists/*

    find ./bin/ -name '*.sh' -exec sed -ri 's|^#!/bin/sh$|#!/usr/bin/env bash|' '{}' +

    chmod -R +rX .
    chmod 777 logs temp work
    catalina.sh version
}

DOCKER_JAVA_HOME() {
    {
        echo '#/bin/sh'
        echo 'echo "$JAVA_HOME"'
        
    } >"/usr/local/bin/docker-java-home"

    chmod +x /usr/local/bin/docker-java-home
    [ "$JAVA_HOME" = "$(docker-java-home)" ]
}

set -eux
apt update
apt install -y --no-install-recommends bzip2 unzip xz-utils fontconfig libfreetype6 ca-certificates p11-kit
rm -rf /var/lib/apt/lists/*

apt update && apt install -y --no-install-recommends git mercurial openssh-client subversion procps
rm -rf /var/lib/apt/lists/*

set -ex
if ! command -v gpg >/dev/null; then
    apt update
    apt install -y --no-install-recommends gnupg dirmngr
    rm -rf /var/lib/apt/lists/*
fi

set -eux
apt update
apt install -y --no-install-recommends ca-certificates curl netbase wget
rm -rf /var/lib/apt/lists/*
