# shellcheck shell=bash
# Fedora (dnf5).

pkg_update() { dnf -y -q makecache; }
pkg_install() { dnf -y -q install --setopt=install_weak_deps=False "$@"; }
pkg_remove() { dnf -y -q remove --setopt=clean_requirements_on_remove=False "$@"; }
pkg_list_installed() { rpm -qa --qf '%{NAME}\n' | grep -v '^gpg-pubkey$'; }
pkg_owner_of() { rpm -qf --qf '%{NAME}\n' "$1" 2> /dev/null || rpm -qf --qf '%{NAME}\n' "$(realpath "$1")" 2> /dev/null; }
pkg_mark_manual() { dnf -y -q mark user "$@" > /dev/null; }
pkg_mark_auto() { dnf -y -q mark dependency "$@" > /dev/null; }
pkg_list_unneeded() { dnf -q repoquery --unneeded --qf '%{name}\n'; }

# ---- pacotes que ficam no sistema (runtime) -----------------------------------------------
PKGS_BASE=(ca-certificates)
PKGS_PYTHON=(python3)
PKGS_MONITORING=(sysstat procps-ng util-linux gcc make)
PKGS_KERNEL_HEADERS=("kernel-devel-$KERNEL_VERSION")
PKGS_DOCKER=(iptables-nft)
PKGS_PODMAN=(iptables-nft)

# ---- pacotes so de compilacao (removidos no final) ---------------------------------------
BUILD_COMMON=(git-core curl xz make gcc gcc-c++ glibc-static pkgconf-pkg-config autoconf automake libtool)
BUILD_DOCKER=(libseccomp-devel cmake)
BUILD_PODMAN=(libseccomp-devel systemd-devel gpgme-devel libassuan-devel libgpg-error-devel glib2-devel
  libcap-devel yajl-devel protobuf-compiler python3)
BUILD_SYSTEMTAP=(elfutils-devel boost-devel gettext)

install_kernel_debug_symbols() {
  pkg_install dnf-plugins-core
  dnf -y -q debuginfo-install "kernel-core-$KERNEL_VERSION"
}
