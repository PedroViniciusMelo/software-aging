#!/usr/bin/env bash
# Mantido por compatibilidade: o instalador agora fica em install/install.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/install/install.sh" "$@"
