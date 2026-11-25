# shellcheck shell=bash
# Ambiente Python do coletor (venv em <repo>/env).

install_python() {
  log "Ambiente Python"
  pkg_install "${PKGS_PYTHON[@]}"
  python3 -m venv "$REPO_DIR/env"
  "$REPO_DIR/env/bin/pip" install --quiet --disable-pip-version-check -r "$REPO_DIR/requirements.txt"
}
