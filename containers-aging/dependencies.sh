#!/usr/bin/env bash
#
# Instala as dependencias do framework de envelhecimento de software em containers.
#
# Distros suportadas: Debian, Ubuntu, Fedora, Arch Linux, openSUSE (Leap/Tumbleweed).
#
# FreeBSD nao esta na lista de proposito: Docker, Podman e o systemtap (usado pela sonda
# de fragmentacao de memoria, fragmentation.stp) dependem de namespaces/cgroups e de um
# tracepoint especifico do kernel Linux (mm_page_alloc_extfrag) -- nada disso existe no
# FreeBSD, entao nao daria pra portar so trocando o instalador de pacotes. Ver o
# openSUSE incluido aqui como o "quinto sistema, de familia diferente" no lugar dele:
# zypper/rpm com filosofia propria (YaST, Btrfs+snapper por padrao), e um dos ambientes
# com MELHOR suporte a systemtap (SUSE e' um dos patrocinadores upstream do projeto).
#
# shellcheck disable=SC1091
source /etc/os-release

readonly DISTRO_ID="$ID"
readonly DISTRO_ID_LIKE="${ID_LIKE:-}"
readonly DISTRO_CODENAME="${VERSION_CODENAME:-}"
readonly SYSTEM_VERSION="${VERSION_ID:-}"

# Normaliza a distro detectada numa "familia" de pacotes, pra o resto do script nao
# precisar ficar checando $DISTRO_ID toda hora.
case "$DISTRO_ID" in
  debian|ubuntu)
    PKG_FAMILY="debian"
    ;;
  fedora)
    PKG_FAMILY="fedora"
    ;;
  arch)
    PKG_FAMILY="arch"
    ;;
  opensuse-leap|opensuse-tumbleweed|opensuse|sles)
    PKG_FAMILY="suse"
    ;;
  *)
    echo "Distro '$DISTRO_ID' nao suportada por este script."
    echo "Suportadas: debian, ubuntu, fedora, arch, opensuse-leap/opensuse-tumbleweed."
    exit 1
    ;;
esac

export PATH=$PATH:/usr/local/go/bin
export PKG_CONFIG_PATH=/usr/lib/pkgconfig

# Versao minima de Go exigida para compilar o Podman atual (ver
# https://podman.io/docs/installation#building-from-source) e' 1.23+; a versao abaixo e'
# a que sera baixada/instalada quando o Go do sistema estiver ausente ou for mais antigo
# que isso. Ajuste se uma versao mais nova for necessaria no futuro.
readonly GO_MIN_MINOR=23
readonly GO_VERSION="1.26.5"

KERNEL_VERSION=$(uname -r)
BASE_DIR=$(pwd)

RESET_TO_ORIGINAL_DIR() {
  cd "$BASE_DIR" || exit 1
}

# --------------------------------------------------------------------------------------
# Instalacao de pacotes: um unico ponto de entrada, com o comando/flags certos por familia.
# As LISTAS de pacotes continuam sendo escolhidas em cada funcao especifica (Python,
# systemtap, etc.), porque os nomes dos pacotes mudam por distro -- so' o "como instalar"
# e' unificado aqui.
# --------------------------------------------------------------------------------------
pkg_update() {
  case "$PKG_FAMILY" in
    debian) apt-get update ;;
    fedora) dnf makecache ;;
    arch) pacman -Sy ;;
    suse) zypper --non-interactive refresh ;;
  esac
}

pkg_install() {
  case "$PKG_FAMILY" in
    debian) apt-get install -y "$@" ;;
    fedora) dnf install -y "$@" ;;
    arch) pacman -S --noconfirm --needed "$@" ;;
    suse) zypper --non-interactive install --no-confirm "$@" ;;
  esac
}

ADD_UBUNTU_DOCKER_REPOSITORY() {
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu" \
  "jammy stable" > /etc/apt/sources.list.d/docker.list

  apt-get update
}

ADD_DEBIAN_DOCKER_REPOSITORY() {
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  bullseye stable" >> /etc/apt/sources.list.d/docker.list
  apt-get update
}

ADD_FEDORA_DOCKER_REPOSITORY() {
  dnf -y install dnf-plugins-core
  dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
}

INSTALL_PYTHON_DEPENDENCIES() {
  RESET_TO_ORIGINAL_DIR
  echo -e "installing python dependencies...."

  case "$PKG_FAMILY" in
    debian)
      pkg_install python3 python3-venv python3-pip
      ;;
    fedora)
      # o python3 do Fedora ja vem com o modulo venv embutido
      pkg_install python3 python3-pip
      ;;
    arch)
      # o pacote "python" do Arch ja inclui venv/pip
      pkg_install python python-pip
      ;;
    suse)
      pkg_install python3 python3-pip python3-venv
      ;;
  esac

  python3 -m venv env

  #echo "export PATH=\$PATH:/root/podman/bin" >> env/bin/activate

  source env/bin/activate

  pip install PyYAML python-kasa
}

INSTALL_LIBRARIES_FOR_MONITORING() {
  reset
  printf "\n%s\n" "Would you like to install monitoring libraries?"
  printf "%s\n" "Yes - [1]"
  printf "%s\n" "No - [2]"

  read -r -p "choice: " choice
  if [ "$choice" -eq 1 ]; then
    # sysstat (mpstat/iostat/pidstat) tem o mesmo nome de pacote nas 5 distros
    case "$PKG_FAMILY" in
      debian)
        pkg_install gnupg curl wget sysstat

        if [ "$DISTRO_ID" == "ubuntu" ]; then
          pkg_install ubuntu-dbgsym-keyring && {
            echo "deb http://ddebs.ubuntu.com $DISTRO_CODENAME main restricted universe multiverse
            deb http://ddebs.ubuntu.com $DISTRO_CODENAME-updates main restricted universe multiverse
            deb http://ddebs.ubuntu.com $DISTRO_CODENAME-proposed main restricted universe multiverse" > "/etc/apt/sources.list.d/ddebs.list"

            apt-get update
          }

          pkg_install linux-headers-"$KERNEL_VERSION" linux-image-"$KERNEL_VERSION"-dbgsym gcc
        else
          # Debian: os simbolos de debug do kernel vem de um repositorio separado
          # (debian-debug); se ele nao estiver configurado, isso falha -- nesse caso
          # o systemtap ainda tenta compilar, so' com menos capacidade de resolver
          # detalhes internos do kernel.
          pkg_install linux-headers-"$KERNEL_VERSION" gcc
          pkg_install linux-image-"$KERNEL_VERSION"-dbg || {
            echo "Aviso: nao consegui instalar os simbolos de debug do kernel (linux-image-*-dbg)."
            echo "No Debian eles vem do repositorio 'debian-debug' -- configure-o manualmente se precisar"
            echo "do systemtap resolvendo detalhes internos do kernel com mais precisao."
          }
        fi
        ;;

      fedora)
        pkg_install gnupg2 curl wget sysstat kernel-devel-"$KERNEL_VERSION" gcc
        # mecanismo padrao do Fedora pra puxar os simbolos de debug certos pro kernel rodando
        pkg_install dnf-utils
        dnf debuginfo-install -y kernel || {
          echo "Aviso: nao consegui instalar o kernel-debuginfo (dnf debuginfo-install kernel)."
          echo "Confira se os repositorios *-debuginfo estao habilitados."
        }
        ;;

      arch)
        pkg_install gnupg curl wget sysstat linux-headers gcc
        echo "Aviso: o Arch nao tem um repositorio oficial de simbolos de debug do kernel"
        echo "(equivalente aos ddebs do Ubuntu / debuginfo do Fedora). O systemtap vai"
        echo "compilar e rodar, mas com menos capacidade de resolver detalhes internos."
        ;;

      suse)
        pkg_install gnupg2 curl wget sysstat kernel-default-devel gcc
        pkg_install kernel-default-debuginfo || {
          echo "Aviso: nao consegui instalar kernel-default-debuginfo."
          echo "No openSUSE ele vem do repositorio de debug oficial -- habilite-o com"
          echo "'zypper ar' se precisar do systemtap com resolucao completa de simbolos."
        }
        ;;
    esac

    cp /proc/kallsyms /boot/System.map-"$KERNEL_VERSION"

    COMPILE_SYSTEMTAP

  else
    echo -e "not installing library monitoring dependencies!"
  fi
}

INSTALL_DOCKER() {
  VERSION_STRING=$1
  case "$PKG_FAMILY" in
    debian)
      pkg_install docker-ce="$VERSION_STRING" docker-ce-cli="$VERSION_STRING" containerd.io docker-buildx-plugin docker-compose-plugin
      ;;
    fedora)
      dnf install -y docker-ce-"$VERSION_STRING" docker-ce-cli-"$VERSION_STRING" containerd.io docker-buildx-plugin docker-compose-plugin
      ;;
  esac
}

DOCKER() {
  reset

  printf "\n%s\n" "Do you want to install the new version or old version of docker?"
  printf "%s\n" "new version - [1]"
  printf "%s\n" "old version - [2]"
  printf "%s\n" "None - [Another Keyboard]"

  read -r -p "choice: " version_choice

  case "$PKG_FAMILY" in
    debian)
      pkg_install ca-certificates curl
      install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
      chmod a+r /etc/apt/keyrings/docker.asc

      if [ "$version_choice" -eq 1 ]; then
        case "$DISTRO_ID" in
        "ubuntu")
          ADD_UBUNTU_DOCKER_REPOSITORY
          INSTALL_DOCKER "5:27.2.0-1~ubuntu.22.04~jammy"
          ;;
        "debian")
          ADD_DEBIAN_DOCKER_REPOSITORY
          INSTALL_DOCKER "5:27.2.0-1~debian.11~bullseye"
          ;;
        esac
      elif [ "$version_choice" -eq 2 ]; then
        case "$DISTRO_ID" in
        "ubuntu")
          ADD_UBUNTU_DOCKER_REPOSITORY
          INSTALL_DOCKER "5:23.0.0-1~ubuntu.22.04~jammy"
          ;;
        "debian")
          ADD_DEBIAN_DOCKER_REPOSITORY
          INSTALL_DOCKER "5:23.0.0-1~debian.11~bullseye"
          ;;
        esac
      else
        printf "%s\n" "Error - error docker install"
      fi
      ;;

    fedora)
      # Fedora tem repositorio oficial do Docker (igual debian/ubuntu), entao o pin de
      # versao funciona do mesmo jeito, so' com nomes de pacote/versao no formato do dnf.
      ADD_FEDORA_DOCKER_REPOSITORY
      if [ "$version_choice" -eq 1 ]; then
        INSTALL_DOCKER "27.2.0-1.fc$(rpm -E %fedora)"
      elif [ "$version_choice" -eq 2 ]; then
        INSTALL_DOCKER "23.0.0-1.fc$(rpm -E %fedora)"
      else
        printf "%s\n" "Error - error docker install"
      fi
      ;;

    arch)
      # Arch nao tem repositorio oficial do Docker Inc. nem mantem versoes antigas no
      # repo normal (rolling release). Pin de versao exato so' e' possivel via o Arch
      # Linux Archive (um mirror com snapshots datados de todos os repos). Isso fixa
      # "a versao que estava disponivel numa certa data", nao um numero de versao exato
      # do Docker -- ajuste ARCH_SNAPSHOT_DATE se precisar de outra data/versao.
      echo "Arch: pin de versao exata do Docker nao existe nos repos normais (rolling release)."
      ARCH_SNAPSHOT_DATE="${ARCH_SNAPSHOT_DATE:-2024/09/01}"  # ~docker 27.2, ajuste conforme necessario
      if [ "$version_choice" -eq 2 ]; then
        ARCH_SNAPSHOT_DATE="${ARCH_SNAPSHOT_DATE_OLD:-2023/02/01}"  # ~docker 23.0
      fi
      echo "Usando snapshot do Arch Linux Archive de $ARCH_SNAPSHOT_DATE"
      pacman -Sy --noconfirm --config <(sed "s#^Include.*#Server = https://archive.archlinux.org/repos/$ARCH_SNAPSHOT_DATE/\$repo/os/\$arch#" /etc/pacman.conf) \
        docker || echo "Falhou; confira se a data do snapshot tem o pacote docker disponivel."
      ;;

    suse)
      # openSUSE empacota o Docker nos proprios repositorios (nao no repo oficial do
      # Docker Inc.), entao o pin de versao depende do que a distro tiver disponivel --
      # pode nao bater exatamente com a versao usada em debian/ubuntu/fedora.
      echo "openSUSE: versoes disponiveis do pacote docker:"
      ZYPPER_DOCKER_SEARCH=$(zypper search -s docker)
      echo "$ZYPPER_DOCKER_SEARCH" | grep -i '^docker ' || true
      if [ "$version_choice" -eq 1 ]; then
        DOCKER_VERSION_HINT="27.2"
      elif [ "$version_choice" -eq 2 ]; then
        DOCKER_VERSION_HINT="23.0"
      else
        printf "%s\n" "Error - error docker install"
        return
      fi
      MATCHED_VERSION=$(echo "$ZYPPER_DOCKER_SEARCH" | awk -v hint="$DOCKER_VERSION_HINT" '$3 ~ hint {print $3; exit}')
      if [ -z "$MATCHED_VERSION" ]; then
        echo "Nao achei uma versao do docker batendo com $DOCKER_VERSION_HINT no repositorio."
        echo "Rode 'zypper search -s docker' e instale manualmente com 'zypper install docker=<versao>'."
        return
      fi
      zypper --non-interactive install "docker=$MATCHED_VERSION"
      ;;
  esac

  docker --version
}

#
# conmon/crun/netavark sao sempre compilados do source (nunca via pacote da distro) de
# proposito: a ideia do experimento e' manter o MESMO ambiente (mesmas versoes, builds
# equivalentes) em todas as distros, e a versao empacotada de cada uma varia bastante
# distro a distro. So o Go (ENSURE_RECENT_GO) e' condicional, porque nao faz sentido
# reinstalar um toolchain de build que ja atende ao minimo exigido.
#
ENSURE_RECENT_GO() {
  if command -v go >/dev/null 2>&1; then
    current=$(go version | awk '{print $3}' | sed 's/^go//')
    cur_minor=$(echo "$current" | cut -d. -f2)
    if [ "${cur_minor:-0}" -ge "$GO_MIN_MINOR" ] 2>/dev/null; then
      echo "Go $current ja atende ao minimo exigido (1.$GO_MIN_MINOR+), nao vou reinstalar."
      return
    fi
  fi

  echo "Instalando Go $GO_VERSION (o Go do sistema esta ausente ou e' mais antigo que 1.$GO_MIN_MINOR)..."
  cd ~ || exit 1
  wget "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
  tar -xzf "go${GO_VERSION}.linux-amd64.tar.gz" -C /usr/local
  rm "go${GO_VERSION}.linux-amd64.tar.gz"
  echo "export PATH=\$PATH:/usr/local/go/bin" >> "$HOME/.profile"
  export PATH=$PATH:/usr/local/go/bin
}

ENSURE_CRUN() {
  echo "Compilando crun do source..."
  cd ~ || exit 1
  git clone https://github.com/containers/crun.git
  cd crun || exit 1
  ./autogen.sh
  ./configure CFLAGS='-I/usr/include/libseccomp'
  make
  make install
  cd ~ || exit 1
}

ENSURE_CONMON() {
  echo "Compilando conmon do source..."
  cd ~ || exit 1
  git clone https://github.com/containers/conmon
  cd conmon || exit 1
  export GOCACHE="$(mktemp -d)"
  make
  make podman
  cd ~ || exit 1
}

ENSURE_NETAVARK() {
  echo "Compilando netavark do source (precisa de Rust/cargo, instalado via rustup)..."
  case "$PKG_FAMILY" in
    debian) pkg_install protobuf-compiler ;;
    fedora) pkg_install protobuf-compiler ;;
    arch) pkg_install protobuf ;;
    suse) pkg_install protobuf-devel ;;
  esac

  curl https://sh.rustup.rs -sSf | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"

  cd ~ || exit 1
  git clone https://github.com/containers/netavark.git
  cd netavark || exit 1
  make
  mkdir -p /usr/local/libexec/podman
  cp bin/* /usr/local/libexec/podman/
  cd ~ || exit 1
}

SETUP_PODMAN_CONFIGURATION() {
  mkdir -p /etc/containers

  if [ ! -f /etc/containers/policy.json ]; then
    cat <<EOF >/etc/containers/policy.json
{
  "default": [
    {
      "type": "insecureAcceptAnything"
    }
  ]
}
EOF
  fi

  if [ ! -f /etc/containers/registries.conf ]; then
    curl -fsSL -o /etc/containers/registries.conf \
      https://raw.githubusercontent.com/containers/image/main/registries.conf || true
  fi
}

COMPILE_PODMAN() {
  cd ~ || exit 1

  git clone https://github.com/containers/podman/
  cd podman || exit 1
  git checkout v5.1

  case "$PKG_FAMILY" in
    debian)
      pkg_install \
        btrfs-progs \
        gcc \
        git \
        golang-go \
        go-md2man \
        iptables \
        libapparmor-dev \
        libassuan-dev \
        libbtrfs-dev \
        libc6-dev \
        libdevmapper-dev \
        libglib2.0-dev \
        libgpgme-dev \
        libgpg-error-dev \
        libprotobuf-dev \
        libprotobuf-c-dev \
        libseccomp-dev \
        libselinux1-dev \
        libsystemd-dev \
        make \
        pkg-config \
        uidmap \
        fuse-overlayfs
      pkg_install passt || true
      ;;

    fedora)
      pkg_install dnf-plugins-core
      dnf builddep -y rpm/podman.spec
      pkg_install catatonit containers-common-extra iptables nftables slirp4netns fuse-overlayfs
      ;;

    arch)
      pkg_install base-devel go git libseccomp gpgme libassuan systemd-libs glib2 \
        btrfs-progs go-md2man pkgconf iptables uidmap
      ;;

    suse)
      pkg_install libseccomp-devel libgpgme-devel libbtrfs-devel make man iptables catatonit
      ;;
  esac

  ENSURE_CRUN
  ENSURE_CONMON
  ENSURE_NETAVARK
  ENSURE_RECENT_GO
  cd ~/podman || exit 1

  make BUILDTAGS="selinux seccomp exclude_graphdriver_devicemapper systemd" PREFIX=/usr
  make install PREFIX=/usr

  SETUP_PODMAN_CONFIGURATION

  podman --version
  RESET_TO_ORIGINAL_DIR
}

COMPILE_SYSTEMTAP() {
  cd /root || exit 1

  # A HEAD atual do systemtap (clonada "as-is", sem fixar tag de release) passou a exigir
  # Boost (boost::asio::thread_pool, usado em parse.cxx para paralelizar o parsing) --
  # sem o pacote de dev do boost o build quebra com "need boost thread_pool.hpp".
  case "$PKG_FAMILY" in
    debian)
      pkg_install git gcc g++ build-essential zlib1g-dev elfutils libdw-dev gettext libboost-all-dev
      ;;
    fedora)
      pkg_install git gcc gcc-c++ make zlib-devel elfutils elfutils-libs elfutils-devel gettext boost-devel
      ;;
    arch)
      pkg_install git base-devel zlib gettext elfutils boost
      ;;
    suse)
      pkg_install git gcc gcc-c++ make zlib-devel libelf-devel libdw-devel gettext-tools \
        libboost_headers-devel libboost_thread-devel libboost_system-devel
      ;;
  esac

  git clone "git://sourceware.org/git/systemtap.git"
  cd "systemtap" || exit 1
  ./configure
  make clean
  make all
  make install
  RESET_TO_ORIGINAL_DIR
}

INSTALL_LXD() {
  case "$PKG_FAMILY" in
    debian)
      pkg_install snapd
      snap install lxd
      lxd init --minimal
      ;;
    *)
      # LXD virou projeto da Canonical com licenciamento proprio; a maioria das distros
      # (incluindo Fedora, Arch e openSUSE) migrou para o Incus, o fork comunitario, e
      # parou de empacotar o LXD original. Por isso so' tentamos snap (Ubuntu/Debian)
      # e sugerimos Incus nas demais em vez de forcar um pacote que provavelmente nao existe mais.
      echo "LXD nao e' mais empacotado oficialmente nesta distro (a maioria migrou para o Incus,"
      echo "o fork comunitario do LXD). Instale o Incus no lugar, por exemplo:"
      case "$PKG_FAMILY" in
        fedora) echo "  dnf install incus" ;;
        arch) echo "  pacman -S incus" ;;
        suse) echo "  zypper install incus" ;;
      esac
      ;;
  esac
}

pkg_update
INSTALL_PYTHON_DEPENDENCIES
INSTALL_LIBRARIES_FOR_MONITORING

printf "%s\n" "Which service are you using?"
printf "%s\n" "Docker - [1]"
printf "%s\n" "Podman - [2]"
printf "%s\n" "LXD/LXC - [3]"
printf "%s\n" "None - [Another Keyboard]"

read -r -p "choice: " service
if [ "$service" -eq 1 ]; then
  DOCKER
elif [ "$service" -eq 2 ]; then
  COMPILE_PODMAN
elif [ "$service" -eq 3 ]; then
  INSTALL_LXD
fi

case "$PKG_FAMILY" in
  debian) apt-get autoremove -y ;;
  fedora) dnf autoremove -y ;;
  arch)
    # pacman nao tem um "autoremove" de uma linha so'; isso lista os pacotes orfaos
    # (instalados como dependencia e sem nada mais precisando deles) e remove.
    orphans=$(pacman -Qtdq || true)
    if [ -n "$orphans" ]; then
      echo "$orphans" | xargs -r pacman -Rns --noconfirm
    fi
    ;;
  suse)
    echo "openSUSE: para remover pacotes sem uso, rode manualmente:"
    echo "  zypper packages --unneeded"
    echo "  zypper remove <pacotes listados acima>"
    ;;
esac

chmod a+wrx start_teastore.sh
chmod a+wrx start_server_response.sh