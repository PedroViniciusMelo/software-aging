# shellcheck shell=bash
# Podman compilado do fonte com crun, conmon, netavark, aardvark-dns e catatonit (--init),
# todos em versoes fixas e em /usr/local (helpers em /usr/local/libexec/podman).

PODMAN_LIBEXEC=/usr/local/libexec/podman

# Mesmo conjunto de recursos em todas as distros (sem deteccao automatica do Makefile).
# Padrao das versoes 5.x; o arquivo versions/podman-<versao>.env pode redefinir.
PODMAN_BUILDTAGS="apparmor selinux seccomp systemd exclude_graphdriver_btrfs exclude_graphdriver_devicemapper"

podman_build_deps() { echo "${BUILD_PODMAN[@]}"; }

install_podman() {
  load_versions "podman-$PODMAN_VERSION"
  use_go "$GO_VERSION"
  use_rust "$RUST_VERSION"
  local src="$BUILD_ROOT/src"
  mkdir -p "$PODMAN_LIBEXEC"

  log "crun $CRUN_VERSION"
  fetch_tarball "$CRUN_URL" "$CRUN_SHA256" "$src/crun"
  # So' o binario (os testes do crun 1.15 nao compilam com gcc >= 14). Fontes gerados e
  # libocispec primeiro: o alvo crun nao declara dependencia dos headers gerados (corrida no -j).
  (cd "$src/crun" && ./configure --prefix=/usr/local --disable-shared > /dev/null &&
    make .version git-version.h && make -C libocispec && make crun)
  install -m 755 "$src/crun/crun" /usr/local/bin/crun

  log "conmon $CONMON_REF"
  fetch_git "$CONMON_REPO" "$CONMON_REF" "$CONMON_COMMIT" "$src/conmon"
  make -C "$src/conmon" PREFIX=/usr/local GIT_COMMIT="$CONMON_COMMIT"
  install -m 755 "$src/conmon/bin/conmon" "$PODMAN_LIBEXEC/conmon"

  log "catatonit $CATATONIT_REF"
  fetch_git "$CATATONIT_REPO" "$CATATONIT_REF" "$CATATONIT_COMMIT" "$src/catatonit"
  (cd "$src/catatonit" && ./autogen.sh > /dev/null && ./configure --prefix=/usr/local > /dev/null && make)
  install -m 755 "$src/catatonit/catatonit" "$PODMAN_LIBEXEC/catatonit"

  log "netavark $NETAVARK_REF e aardvark-dns $AARDVARK_REF"
  fetch_git "$NETAVARK_REPO" "$NETAVARK_REF" "$NETAVARK_COMMIT" "$src/netavark"
  (cd "$src/netavark" && cargo build --release --locked --quiet)
  install -m 755 "$src/netavark/target/release/netavark" "$PODMAN_LIBEXEC/netavark"
  fetch_git "$AARDVARK_REPO" "$AARDVARK_REF" "$AARDVARK_COMMIT" "$src/aardvark-dns"
  (cd "$src/aardvark-dns" && cargo build --release --locked --quiet)
  install -m 755 "$src/aardvark-dns/target/release/aardvark-dns" "$PODMAN_LIBEXEC/aardvark-dns"

  log "podman $PODMAN_REF"
  fetch_git "$PODMAN_REPO" "$PODMAN_REF" "$PODMAN_COMMIT" "$src/podman"
  make -C "$src/podman" BUILDTAGS="$PODMAN_BUILDTAGS" PREFIX=/usr/local GIT_COMMIT="$PODMAN_COMMIT" \
    EPOCH_TEST_COMMIT="$PODMAN_COMMIT" binaries
  make -C "$src/podman" BUILDTAGS="$PODMAN_BUILDTAGS" PREFIX=/usr/local install.bin install.systemd

  install_podman_config
  keep_runtime_libs_of /usr/local/bin/{podman,crun} "$PODMAN_LIBEXEC"/{conmon,netavark,aardvark-dns}

  manifest_add "podman=$PODMAN_REF@$PODMAN_COMMIT buildtags=\"$PODMAN_BUILDTAGS\""
  manifest_add "crun=$CRUN_VERSION conmon=$CONMON_REF@$CONMON_COMMIT catatonit=$CATATONIT_REF@$CATATONIT_COMMIT"
  manifest_add "netavark=$NETAVARK_REF@$NETAVARK_COMMIT aardvark-dns=$AARDVARK_REF@$AARDVARK_COMMIT firewall=${NETAVARK_FIREWALL_DRIVER:-auto}"
  manifest_add "podman_go=$GO_VERSION podman_rust=$RUST_VERSION"

  has_systemd && systemctl daemon-reload
  /usr/local/bin/podman --version
}

# Configuracao minima e versionada aqui (antes vinha do branch main do containers/image,
# que muda com o tempo).
install_podman_config() {
  mkdir -p /etc/containers
  [ -f /etc/containers/policy.json ] || cat > /etc/containers/policy.json << 'EOF'
{
  "default": [{"type": "insecureAcceptAnything"}]
}
EOF
  [ -f /etc/containers/registries.conf ] || cat > /etc/containers/registries.conf << 'EOF'
unqualified-search-registries = ["docker.io"]
EOF
  [ -f /etc/containers/containers.conf ] || cat > /etc/containers/containers.conf << EOF
[containers]
init_path = "$PODMAN_LIBEXEC/catatonit"

[engine]
runtime = "crun"
helper_binaries_dir = ["$PODMAN_LIBEXEC"]
EOF
  # netavark >= 2 so' tem o driver nftables; fixa-lo evita depender da deteccao automatica.
  if [ -n "${NETAVARK_FIREWALL_DRIVER:-}" ] && ! grep -q '^\[network\]' /etc/containers/containers.conf; then
    printf '\n[network]\nfirewall_driver = "%s"\n' "$NETAVARK_FIREWALL_DRIVER" >> /etc/containers/containers.conf
  fi
  [ -f /etc/containers/storage.conf ] || cat > /etc/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
runroot = "/run/containers/storage"
graphroot = "/var/lib/containers/storage"
EOF
}
