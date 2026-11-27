# shellcheck shell=bash
# Docker Engine compilado do fonte: dockerd + docker-proxy (moby), docker (cli), containerd,
# runc e docker-init (tini). Tudo em /usr/local, sem misturar com pacotes da distro.

docker_build_deps() { echo "${BUILD_DOCKER[@]}"; }

install_docker() {
  load_versions "docker-$DOCKER_VERSION"
  use_go "$GO_VERSION"
  local src="$BUILD_ROOT/src"

  log "containerd $CONTAINERD_REF"
  fetch_git "$CONTAINERD_REPO" "$CONTAINERD_REF" "$CONTAINERD_COMMIT" "$src/containerd"
  make -C "$src/containerd" BUILDTAGS="no_btrfs" VERSION="$CONTAINERD_REF" REVISION="$CONTAINERD_COMMIT" binaries
  make -C "$src/containerd" PREFIX=/usr/local install
  install -m 644 "$src/containerd/containerd.service" /etc/systemd/system/containerd.service

  log "runc $RUNC_REF"
  fetch_git "$RUNC_REPO" "$RUNC_REF" "$RUNC_COMMIT" "$src/runc"
  make -C "$src/runc" COMMIT="$RUNC_COMMIT" RUNC_BUILDTAGS="${RUNC_BUILDTAGS:-}"
  make -C "$src/runc" PREFIX=/usr/local install

  log "tini $TINI_REF (docker-init)"
  fetch_git "$TINI_REPO" "$TINI_REF" "$TINI_COMMIT" "$src/tini"
  # CMAKE_POLICY_VERSION_MINIMUM: o tini 0.19 declara cmake 2.8, recusado pelo CMake >= 4
  (cd "$src/tini" && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 . > /dev/null && make tini-static)
  install -m 755 "$src/tini/tini-static" /usr/local/bin/docker-init

  # Ate a v28 o moby usa vendor.mod + GOPATH (sem go.mod); a partir da v29 e' um modulo Go.
  # O cli usa vendor.mod + GOPATH em todas as versoes suportadas (o script dele ja' faz isso).
  log "moby $MOBY_REF (dockerd, docker-proxy)"
  local moby="$GOPATH/src/github.com/docker/docker" moby_gomod=off
  fetch_git "$MOBY_REPO" "$MOBY_REF" "$MOBY_COMMIT" "$moby"
  [ -f "$moby/go.mod" ] && moby_gomod=on
  (cd "$moby" && GO111MODULE="$moby_gomod" VERSION="$DOCKER_VERSION" DOCKER_GITCOMMIT="${MOBY_COMMIT:0:7}" \
    DOCKER_BUILDTAGS="exclude_graphdriver_btrfs exclude_graphdriver_devicemapper" hack/make.sh dynbinary)
  install -m 755 "$moby/bundles/dynbinary-daemon/dockerd" /usr/local/bin/dockerd
  install -m 755 "$moby/bundles/dynbinary-daemon/docker-proxy" /usr/local/bin/docker-proxy
  sed 's#/usr/bin/dockerd#/usr/local/bin/dockerd#' "$moby/contrib/init/systemd/docker.service" \
    > /etc/systemd/system/docker.service
  install -m 644 "$moby/contrib/init/systemd/docker.socket" /etc/systemd/system/docker.socket

  log "docker cli $CLI_REF"
  local cli="$GOPATH/src/github.com/docker/cli"
  fetch_git "$CLI_REPO" "$CLI_REF" "$CLI_COMMIT" "$cli"
  (cd "$cli" && GO111MODULE=off VERSION="$DOCKER_VERSION" GITCOMMIT="${CLI_COMMIT:0:7}" ./scripts/build/binary)
  install -m 755 "$cli/build/docker" /usr/local/bin/docker

  getent group docker > /dev/null || groupadd --system docker
  keep_runtime_libs_of /usr/local/bin/{containerd,containerd-shim-runc-v2,dockerd,docker-proxy,docker} /usr/local/sbin/runc

  manifest_add "docker=$DOCKER_VERSION moby=$MOBY_REF@$MOBY_COMMIT cli=$CLI_REF@$CLI_COMMIT"
  manifest_add "containerd=$CONTAINERD_REF@$CONTAINERD_COMMIT runc=$RUNC_REF@$RUNC_COMMIT runc_buildtags=\"${RUNC_BUILDTAGS:-}\" tini=$TINI_REF@$TINI_COMMIT"
  manifest_add "docker_go=$GO_VERSION"

  if has_systemd; then
    systemctl daemon-reload
    systemctl enable --now containerd.service docker.socket docker.service
  else
    warn "systemd nao esta rodando: servicos instalados, mas nao habilitados"
  fi
  /usr/local/bin/docker --version
}
