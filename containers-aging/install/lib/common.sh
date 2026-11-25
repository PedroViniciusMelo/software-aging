# shellcheck shell=bash
# Funcoes compartilhadas pelo instalador: log, deteccao da distro e pacotes de build.

log() { printf '\n==> %s\n' "$*"; }
warn() { printf 'AVISO: %s\n' "$*" >&2; }
die() { printf 'ERRO: %s\n' "$*" >&2; exit 1; }

require_root() {
  [ "$(id -u)" -eq 0 ] || die "rode como root"
}

has_systemd() {
  [ -d /run/systemd/system ]
}

nproc_build() {
  nproc 2>/dev/null || echo 2
}

# Carrega /etc/os-release e o arquivo da familia de pacotes (lib/distro/<familia>.sh),
# que define as funcoes pkg_* e as listas de pacotes.
detect_distro() {
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO_ID="$ID"
  DISTRO_VERSION="${VERSION_ID:-}"
  DISTRO_CODENAME="${VERSION_CODENAME:-}"
  KERNEL_VERSION="$(uname -r)"

  case "$DISTRO_ID" in
    debian | ubuntu) PKG_FAMILY=debian ;;
    fedora) PKG_FAMILY=fedora ;;
    *) die "distro '$DISTRO_ID' nao suportada (suportadas: debian, ubuntu, fedora)" ;;
  esac

  # shellcheck source=distro/debian.sh
  . "$INSTALL_DIR/lib/distro/$PKG_FAMILY.sh"
  log "Sistema: ${PRETTY_NAME:-$DISTRO_ID $DISTRO_VERSION}, kernel $KERNEL_VERSION"
}

load_versions() {
  local file="$INSTALL_DIR/versions/$1.env"
  [ -f "$file" ] || die "versao nao suportada: $1 (veja install/versions/)"
  # shellcheck disable=SC1090
  . "$file"
}

# ---------------------------------------------------------------------------------------
# Pacotes de build temporarios
#
# build_deps_begin <pacotes...> guarda a lista de pacotes instalados e instala os de build;
# build_deps_end remove so' o que entrou depois do begin. Antes de remover, os pacotes que
# fornecem bibliotecas usadas pelos binarios instalados (keep_runtime_libs_of) sao mantidos,
# entao nenhuma lista de "pacote de runtime" por distro precisa ser mantida a mao.
# ---------------------------------------------------------------------------------------
BUILD_DEPS_SNAPSHOT=""
RUNTIME_KEEP=""

build_deps_begin() {
  [ -z "$BUILD_DEPS_SNAPSHOT" ] || die "build_deps_begin chamado duas vezes"
  BUILD_DEPS_SNAPSHOT="$BUILD_ROOT/packages-before.txt"
  pkg_list_installed | sort -u > "$BUILD_DEPS_SNAPSHOT"
  log "Instalando dependencias de compilacao (serao removidas no final)"
  pkg_install "$@"
}

keep_runtime_libs_of() {
  local binary lib owner
  for binary in "$@"; do
    [ -e "$binary" ] || die "binario esperado nao foi instalado: $binary"
    while read -r lib; do
      owner="$(pkg_owner_of "$lib")" || continue
      RUNTIME_KEEP="$RUNTIME_KEEP $owner"
    done < <(ldd "$binary" 2>/dev/null | awk '/=> \//{print $3}')
  done
}

# Marca tudo que entrou depois do begin como "dependencia" e os donos das bibliotecas de
# runtime como "manual"; remove so' o que o gerenciador de pacotes passa a considerar
# desnecessario E que entrou durante o build (orfaos antigos do sistema nao sao tocados).
build_deps_end() {
  [ -n "$BUILD_DEPS_SNAPSHOT" ] || return 0
  local new keep remove
  new="$(comm -13 "$BUILD_DEPS_SNAPSHOT" <(pkg_list_installed | sort -u))"
  keep="$(printf '%s\n' $RUNTIME_KEEP | sort -u | grep -xF -f <(echo "$new"))" || true

  if [ -n "$new" ]; then
    # shellcheck disable=SC2086
    pkg_mark_auto $new
  fi
  if [ -n "$keep" ]; then
    log "Mantendo bibliotecas de runtime: $(echo $keep)"
    # shellcheck disable=SC2086
    pkg_mark_manual $keep
  fi
  remove="$(comm -12 <(pkg_list_unneeded | sort -u) <(echo "$new"))" || true
  if [ -n "$remove" ]; then
    log "Removendo $(echo "$remove" | wc -l) pacotes usados so' na compilacao"
    # shellcheck disable=SC2086
    pkg_remove $remove
  fi
  BUILD_DEPS_SNAPSHOT=""
}
