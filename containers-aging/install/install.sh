#!/usr/bin/env bash
#
# Instala as dependencias do framework de envelhecimento de software em containers.
#
# Docker, Podman e systemtap sao COMPILADOS do fonte, em versoes fixas.
#
# Uso:
#   install/install.sh                       # menu interativo
#   install/install.sh --runtime docker --monitoring yes
#
# Opcoes:
#   --runtime docker|podman|none       runtime de containers a instalar
#   --docker-version V                 versao do Docker (padrao: a unica em versions/)
#   --podman-version V                 versao do Podman
#   --monitoring yes|no                ferramentas de monitoramento + systemtap
#   --no-kernel-debug                  nao instala simbolos de debug do kernel
#   --no-python                        nao cria o venv do coletor
#   --keep-build-dir                   nao apaga o BUILD_ROOT (para depurar a compilacao)
#
# Distros: Debian, Ubuntu e Fedora. Para outra distro, crie install/lib/distro/<familia>.sh.
set -euo pipefail
export LC_ALL=C

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$INSTALL_DIR")"

# shellcheck source=lib/common.sh
. "$INSTALL_DIR/lib/common.sh"
# shellcheck source=lib/sandbox.sh
. "$INSTALL_DIR/lib/sandbox.sh"
for component in "$INSTALL_DIR"/components/*.sh; do
  # shellcheck disable=SC1090
  . "$component"
done
# shellcheck source=versions/toolchains.env
. "$INSTALL_DIR/versions/toolchains.env"

RUNTIME=""
MONITORING=""
WITH_PYTHON=1
WITH_KERNEL_DEBUG=1
KEEP_BUILD_ROOT=0
DOCKER_VERSION=27.2.0
PODMAN_VERSION=5.1.2

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
  exit "${1:-0}"
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --runtime) RUNTIME="$2"; shift ;;
      --docker-version) DOCKER_VERSION="$2"; shift ;;
      --podman-version) PODMAN_VERSION="$2"; shift ;;
      --monitoring) MONITORING="$2"; shift ;;
      --no-kernel-debug) WITH_KERNEL_DEBUG=0 ;;
      --no-python) WITH_PYTHON=0 ;;
      --keep-build-dir) KEEP_BUILD_ROOT=1 ;;
      -h | --help) usage ;;
      *) warn "opcao desconhecida: $1"; usage 1 ;;
    esac
    shift
  done
}

ask_missing() {
  if [ -z "$MONITORING" ]; then
    read -r -p "Instalar as ferramentas de monitoramento (sysstat, systemtap)? [s/N] " answer
    case "$answer" in [sSyY]*) MONITORING=yes ;; *) MONITORING=no ;; esac
  fi
  if [ -z "$RUNTIME" ]; then
    printf '%s\n' "Runtime de containers:" "  [1] Docker $DOCKER_VERSION" "  [2] Podman $PODMAN_VERSION" \
      "  [outra tecla] nenhum"
    read -r -p "escolha: " answer
    case "$answer" in 1) RUNTIME=docker ;; 2) RUNTIME=podman ;; *) RUNTIME=none ;; esac
  fi
}

main() {
  parse_args "$@"
  require_root
  ask_missing
  case "$RUNTIME" in docker | podman | none) ;; *) die "--runtime invalido: $RUNTIME" ;; esac
  case "$MONITORING" in yes | no) ;; *) die "--monitoring deve ser yes ou no" ;; esac

  detect_distro
  setup_build_root
  pkg_update

  # 1) pacotes que ficam no sistema
  pkg_install "${PKGS_BASE[@]}"
  [ "$WITH_PYTHON" = 1 ] && install_python
  [ "$MONITORING" = yes ] && install_monitoring_runtime
  [ "$RUNTIME" = docker ] && pkg_install "${PKGS_DOCKER[@]}"
  [ "$RUNTIME" = podman ] && pkg_install "${PKGS_PODMAN[@]}"

  local build_deps=()
  [ "$MONITORING" = yes ] && read -ra build_deps <<< "$(monitoring_build_deps)"
  case "$RUNTIME" in
    docker) read -ra extra <<< "$(docker_build_deps)"; build_deps+=("${extra[@]}") ;;
    podman) read -ra extra <<< "$(podman_build_deps)"; build_deps+=("${extra[@]}") ;;
  esac
  if [ ${#build_deps[@]} -gt 0 ]; then
    build_deps_begin "${BUILD_COMMON[@]}" "${build_deps[@]}"
    manifest_reset
    [ "$MONITORING" = yes ] && install_systemtap
    [ "$RUNTIME" = docker ] && install_docker
    [ "$RUNTIME" = podman ] && install_podman
    build_deps_end
  fi

  chmod a+rx "$REPO_DIR/start_teastore.sh" "$REPO_DIR/start_server_response.sh"
  log "Concluido"
  [ -f "$MANIFEST" ] && cat "$MANIFEST"
  return 0
}

main "$@"
