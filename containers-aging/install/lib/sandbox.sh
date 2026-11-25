# shellcheck shell=bash
# "Sandbox" de compilacao: tudo que so' serve para compilar (fontes, toolchains Go/Rust,
# caches) fica em $BUILD_ROOT, que e' apagado no final. Nada vai para ~/go, ~/.cargo,
# ~/.cache ou ~/.profile, e o sistema medido recebe apenas os binarios finais.

setup_build_root() {
  BUILD_ROOT="${BUILD_ROOT:-/var/tmp/containers-aging-build}"
  rm -rf "$BUILD_ROOT"
  mkdir -p "$BUILD_ROOT"/{src,toolchains,cache}

  export GOPATH="$BUILD_ROOT/gopath"
  export GOCACHE="$BUILD_ROOT/cache/go-build"
  export GOMODCACHE="$BUILD_ROOT/cache/go-mod"
  export GOTOOLCHAIN=local GOFLAGS="-buildvcs=false"
  export CARGO_HOME="$BUILD_ROOT/toolchains/cargo"
  export RUSTUP_HOME="$BUILD_ROOT/toolchains/rustup"
  export MAKEFLAGS="-j$(nproc_build)"
  trap cleanup_build_root EXIT
}

cleanup_build_root() {
  local status=$?
  build_deps_end || true
  if [ "${KEEP_BUILD_ROOT:-0}" = 1 ]; then
    warn "mantendo $BUILD_ROOT (--keep-build-dir)"
  else
    rm -rf "$BUILD_ROOT"
  fi
  exit "$status"
}

# fetch_git <repo> <tag> <commit> <destino>: clone raso da tag, conferindo o commit.
fetch_git() {
  local repo="$1" ref="$2" commit="$3" dest="$4"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$ref" "$repo" "$dest" ||
    die "falha ao baixar $repo ($ref)"
  local actual
  actual="$(git -C "$dest" rev-parse HEAD)"
  [ "$actual" = "$commit" ] || die "$repo $ref aponta para $actual, esperado $commit"
  # data de build = data do commit (builds reproduziveis; ex.: moby grava isso no binario)
  SOURCE_DATE_EPOCH="$(git -C "$dest" log -1 --format=%ct)"
  export SOURCE_DATE_EPOCH
}

# fetch_tarball <url> <sha256> <destino>: baixa, confere o checksum e extrai em <destino>.
fetch_tarball() {
  local url="$1" sha256="$2" dest="$3" file
  file="$BUILD_ROOT/cache/$(basename "$url")"
  curl -fsSL --retry 3 -o "$file" "$url" || die "falha ao baixar $url"
  echo "$sha256  $file" | sha256sum -c --quiet - || die "checksum invalido: $url"
  mkdir -p "$dest"
  tar -xf "$file" -C "$dest" --strip-components=1
}

# use_go <versao>: instala o toolchain Go oficial (checksum fixo) no sandbox e o coloca no PATH.
use_go() {
  local version="$1" var sha root="$BUILD_ROOT/toolchains/go-$1"
  var="GO_SHA256_${version//./_}"
  sha="${!var:-}"
  [ -n "$sha" ] || die "checksum do Go $version ausente em versions/toolchains.env"
  if [ ! -x "$root/bin/go" ]; then
    log "Toolchain Go $version"
    fetch_tarball "https://go.dev/dl/go${version}.linux-amd64.tar.gz" "$sha" "$root"
  fi
  export GOROOT="$root"
  export PATH="$root/bin:$PATH"
  go version
}

# use_rust <versao>: toolchain Rust fixo via rustup, so' no sandbox.
use_rust() {
  local version="$1"
  if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    log "Toolchain Rust $version"
    curl -fsSL --retry 3 https://sh.rustup.rs |
      sh -s -- -y --quiet --no-modify-path --profile minimal --default-toolchain "$version"
  fi
  export PATH="$CARGO_HOME/bin:$PATH"
  cargo --version
}

# ---------------------------------------------------------------------------------------
# Manifesto: registra exatamente o que foi compilado (copiado para logs/ pelo coletor).
# ---------------------------------------------------------------------------------------
MANIFEST=/etc/containers-aging/build-manifest.txt

manifest_reset() {
  mkdir -p "$(dirname "$MANIFEST")"
  {
    echo "# Gerado por install/install.sh em $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "system=${PRETTY_NAME:-$DISTRO_ID $DISTRO_VERSION}"
    echo "kernel=$KERNEL_VERSION"
    echo "gcc=$(gcc -dumpfullversion 2>/dev/null || echo n/a)"
  } > "$MANIFEST"
}

manifest_add() {
  echo "$*" >> "$MANIFEST"
}
