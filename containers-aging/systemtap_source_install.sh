#!/usr/bin/env bash

reset
[[ -d "/root/systemtap" ]] && {
    echo "A pasta do systemtap já existe, talvés você já o tenha configurado"
    echo "Caso não, apague a pasta e execute novamente"
    exit 0
}

echo "Verifique se você está no diretorio /root: "; echo "Seu dir atual é: $(pwd)"; echo "Caso não, mude para ele!"

GIT() {
    apt install git -y
    git clone "git://sourceware.org/git/systemtap.git" || {
        echo "erro ao tentar baixar o codigo fonte do systemtap a partir do repositório git"; echo "saindo..."; exit 1
    }

}

TAR() {
    # Será baixado a versao 5.1 do systemtap a partir dos arquivos do site do systemtap o sourceware
    # Qualquer coisa acesse a lista de releases com: https://sourceware.org/ftp/systemtap/releases
    apt install wget tar -y
    wget "https://sourceware.org/ftp/systemtap/releases/systemtap-5.1.tar.gz"
    tar -xvf "systemtap-5.1.tar.gz"
    mv "systemtap-5.1" "systemtap"
    rm -r "systemtap-5.1.tar.gz"

}

GET_SYSTEMTAP() {
    apt install gcc g++ build-essential zlib1g-dev elfutils libdw-dev gettext -y || {
        echo "erro ao instalar dependencias para o systemtap"; echo "saindo"; exit 1
    }


    mv systemtap /root || {
        echo "erro ao mover systemtap para o dir /root"
    }

    cd "/root/systemtap" || {
        echo "erro ao tentar entrar no diretorio do systemtap"; echo "saindo..."; exit 1
    }

    configurar=$(./configure | grep -o "./configure.*prefix=/root/systemtap-[0-9]\+\.[0-9]\+\-[0-9]\+")

    eval "$configurar" || {
        echo "Error em fazer ./configure "; echo "saindo..."; exit 1
    }

    make all || {
        echo "erro ao tentar compilar o systemtap com make all"; echo "saindo"; exit 1
    }

    make install || {
        echo "erro ao tentar instalar compilação do systemtap com make install"; echo "saindo"; exit 1
    }

    echo "adicionando path do systemtap em /root/.bashrc"; echo "export PATH=\$PATH:/root/systemtap" >> "/root/.bashrc"
    source /root/.bashrc
    echo "$PATH"

    printf "\n%s\n" "Caso o systemtap não seja caregado, faca: source /root/.bashrc"; echo "verifique se a path foi adicionada com: echo \$PATH"

    echo "Caso deseje executar como user sudo ou comum, faça: ( echo export PATH=\$PATH:/home/$(logname)/systemtap >> /home/$(logname)/.bashrc ) e faça ( source /home/$(logname)/.bashrc )"
}

RUNNING_SMALL_SYSTEMTAP_TEST() {
    print "\n\n\n"
    echo "Rodando teste de instalação do systemtap"
    # shellcheck disable=SC2317
    stap -ve 'probe begin { log("hello world") exit () }'

    # shellcheck disable=SC2317
    stap -c df -e 'probe syscall.* { if (target()==pid()) log(name." ".argstr) }'
}

MAIN() {
    echo "Deseja instalar via git ou .tar?"
    echo "Git - [1]"
    echo ".tar - [2]"
    echo "none - qualquer tecla"

    read -r -p "escolha: " escolha

    if [ "$escolha" -eq 1 ]; then
        GIT
        GET_SYSTEMTAP

    elif [ "$escolha" -eq 2 ]; then
        TAR
        GET_SYSTEMTAP

    else
        echo "saindo...."
        exit 0
    fi

    # shellcheck disable=SC2317
    echo "Deseja rodar um pequeno teste de verificação de instalação? [s]/[n]"
    # shellcheck disable=SC2317
    read -r -p "escolha: " escolha

    # shellcheck disable=SC2317
    [[ "$escolha" == "s" ]] && RUNNING_SMALL_SYSTEMTAP_TEST
}

MAIN
