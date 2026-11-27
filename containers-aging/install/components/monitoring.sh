# shellcheck shell=bash
# Ferramentas usadas pelo coletor: sysstat (mpstat/iostat/pidstat), headers e simbolos de
# debug do kernel e o systemtap (sonda de fragmentacao), compilado numa versao fixa.
# gcc/make/headers ficam instalados: o systemtap compila um modulo de kernel a cada execucao.

monitoring_build_deps() { echo "${BUILD_SYSTEMTAP[@]}"; }

install_monitoring_runtime() {
  log "Ferramentas de monitoramento"
  pkg_install "${PKGS_MONITORING[@]}"
  pkg_install "${PKGS_KERNEL_HEADERS[@]}" ||
    warn "headers do kernel $KERNEL_VERSION nao encontrados no repositorio: o systemtap precisa deles para rodar"

  if [ "${WITH_KERNEL_DEBUG:-1}" = 1 ]; then
    install_kernel_debug_symbols ||
      warn "simbolos de debug do kernel nao instalados; o systemtap roda, mas com menos detalhes"
  fi

  # O systemtap procura o System.map em /boot. O Debian instala ali um placeholder (uma linha
  # "The real System.map is in the linux-image-<versao>-dbg package"): sem simbolos de texto,
  # ele e' trocado pelo mapa real do pacote -dbg ou, sem ele, pelo /proc/kallsyms.
  local map="/boot/System.map-$KERNEL_VERSION" real="/usr/lib/debug/boot/System.map-$KERNEL_VERSION"
  if ! grep -q ' [Tt] ' "$map" 2> /dev/null; then
    if [ -s "$real" ]; then
      cp "$real" "$map" || true
    elif [ -r /proc/kallsyms ]; then
      cp /proc/kallsyms "$map" || true
    fi
  fi
}

install_systemtap() {
  load_versions monitoring
  local src="$BUILD_ROOT/src/systemtap"

  log "systemtap $SYSTEMTAP_VERSION"
  fetch_tarball "$SYSTEMTAP_URL" "$SYSTEMTAP_SHA256" "$src"
  # So' o backend de modulo de kernel (o usado pela fragmentation.stp): sem servidor de
  # compilacao, dyninst, sondas python, documentacao ou NLS.
  (cd "$src" && ./configure --prefix=/usr/local --disable-docs --disable-publican --disable-refdocs \
    --disable-htmldocs --disable-server --disable-nls --without-nss --without-avahi --without-dyninst \
    --without-python2-probes --without-python3-probes --without-rpm --without-debuginfod --without-bpf \
    > /dev/null && make && make install)

  keep_runtime_libs_of /usr/local/bin/stap
  manifest_add "systemtap=$SYSTEMTAP_VERSION sha256=$SYSTEMTAP_SHA256"
  /usr/local/bin/stap -V 2>&1 | head -n1
}
