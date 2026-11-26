# shellcheck shell=bash
# Ambiente Python do coletor e dos graficos (venv em <repo>/env). O coletor nao importa as libs
# de plot/: elas so' rodam, em outro processo, depois que o monitoramento termina.

install_python() {
  log "Ambiente Python"
  pkg_install "${PKGS_PYTHON[@]}"
  python3 -m venv "$REPO_DIR/env"
  "$REPO_DIR/env/bin/pip" install --quiet --disable-pip-version-check -r "$REPO_DIR/requirements.txt" -r "$REPO_DIR/plot/requirements.txt"
}
