#!/usr/bin/env bash
# Mini teste rapido numa imagem ja' instalada pelo run-in-container.sh (nao recompila nada):
#   tests/install/smoke.sh                          # todas as imagens containers-aging-test:*
#   tests/install/smoke.sh debian-12-docker         # uma imagem
set -uo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
tags=("$@")
[ ${#tags[@]} -gt 0 ] || mapfile -t tags < <(docker images --format '{{.Tag}}' containers-aging-test)

proxy_env=()
[ "${TEST_USE_PROXY:-0}" = 1 ] && proxy_env=(-e http_proxy -e https_proxy -e no_proxy -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY)

failed=0
for tag in "${tags[@]}"; do
  runtime="${tag##*-}"
  printf '\n######## %s\n' "$tag"
  if docker run --rm --privileged --cgroupns=private --network host "${proxy_env[@]}" \
    -v /var/lib/docker -v /var/lib/containers \
    -v "$repo:/src:ro" "containers-aging-test:$tag" bash /src/tests/install/verify.sh "$runtime"; then
    echo "PASSOU: $tag"
  else
    echo "FALHOU: $tag"; failed=1
  fi
done
exit "$failed"
