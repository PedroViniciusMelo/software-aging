# shellcheck shell=bash
# Debian e Ubuntu (apt).

export DEBIAN_FRONTEND=noninteractive

pkg_update() { apt-get update -q; }
pkg_install() { apt-get install -y -q --no-install-recommends "$@"; }
pkg_remove() { apt-get purge -y -q "$@"; }
pkg_list_installed() { dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' | awk '$1 == "ii" {print $2}'; }
pkg_mark_manual() { apt-mark manual "$@" > /dev/null; }
pkg_mark_auto() { apt-mark auto "$@" > /dev/null; }
pkg_list_unneeded() { apt-get -s autoremove | awk '/^Remv /{print $2}'; }

# Pacote dono de um arquivo (tenta o caminho como esta' e o real, por causa do /usr merge).
pkg_owner_of() {
  local path
  for path in "$1" "$(realpath "$1")" "/usr$1" "${1#/usr}"; do
    dpkg -S "$path" 2> /dev/null | head -n1 | cut -d: -f1 && return 0
  done
  return 1
}

# ---- pacotes que ficam no sistema (runtime) -----------------------------------------------
PKGS_BASE=(ca-certificates)
PKGS_PYTHON=(python3 python3-venv)
PKGS_MONITORING=(sysstat procps util-linux gcc make)
PKGS_KERNEL_HEADERS=("linux-headers-$KERNEL_VERSION")
PKGS_DOCKER=(iptables)
PKGS_PODMAN=(iptables)
PKGS_NFTABLES=(nftables)

# ---- pacotes so' de compilacao (removidos no final) ---------------------------------------
BUILD_COMMON=(git curl xz-utils make gcc g++ libc6-dev pkg-config autoconf automake libtool)
BUILD_DOCKER=(libseccomp-dev cmake)
BUILD_PODMAN=(libseccomp-dev libsystemd-dev libgpgme-dev libassuan-dev libgpg-error-dev libglib2.0-dev
  libcap-dev libyajl-dev protobuf-compiler python3)
BUILD_SYSTEMTAP=(libdw-dev libelf-dev libboost-dev gettext)

# Simbolos de debug do kernel (opcional: o systemtap funciona sem, so' resolve menos detalhes).
install_kernel_debug_symbols() {
  if [ "$DISTRO_ID" = ubuntu ]; then
    pkg_install ubuntu-dbgsym-keyring
    printf 'deb http://ddebs.ubuntu.com %s main restricted universe multiverse\n' \
      "$DISTRO_CODENAME" "$DISTRO_CODENAME-updates" "$DISTRO_CODENAME-proposed" > /etc/apt/sources.list.d/ddebs.list
    pkg_update
    pkg_install "linux-image-$KERNEL_VERSION-dbgsym"
  else
    echo "deb http://deb.debian.org/debian-debug/ $DISTRO_CODENAME-debug main" > /etc/apt/sources.list.d/debian-debug.list
    pkg_update
    pkg_install "linux-image-$KERNEL_VERSION-dbg"
  fi
}

# Runtimes de container da distro, removidos antes de instalar os compilados.
DISTRO_RUNTIME_PKGS=(podman podman-docker crun conmon netavark aardvark-dns catatonit buildah skopeo
  golang-github-containers-common golang-github-containers-image containernetworking-plugins
  docker.io docker-cli docker-ce docker-ce-cli docker-buildx docker-compose containerd containerd.io runc)
