#!/usr/bin/env bash
# HOW TO USE:
#   bash configAll.sh
# 
# DESCRIPTION:
#   USADO PARA DEFINIR AUTOCONFIGURAÇÃO DE VARIAVEIS DE AMBIENTE E TESTES PADRÕES

# CREATE A FILE WITH ALL PRE CONFIGS
GeneratedFileConfig() {
    echo "file configs >>>>>>>>>>>>>>" > generatedConfig.cfg
}

Lxc() {
    echo "1"
}

Xen() {
    echo "1"
}

Kvm() {
    echo "1"
}

Virtualbox() {
    echo "1"
}