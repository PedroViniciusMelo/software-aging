#!/usr/bin/env bash
# Testa o instalador num container descartavel de uma distro e verifica o resultado:
#   tests/install/run-in-container.sh <imagem> <docker|podman|none> [opcoes extras do install.sh]
#
# Se passar, o container ja' instalado vira a imagem containers-aging-test:<distro>-<runtime>,
# usada pelo smoke.sh para repetir a verificacao sem recompilar nada.
#
# O container e' privilegiado para que o runtime compilado consiga subir um container de
# verdade (docker-in-docker / podman-in-docker). /var/lib/docker e /var/lib/containers sao
# volumes (no disco do host), porque o overlay nao funciona sobre o overlayfs do container. O host nao e' alterado: o repositorio e'
# copiado para dentro do container. KEEP_FAILED=1 mantem o container se o teste falhar.
set -euo pipefail
image="$1" runtime="$2"; shift 2
repo="$(cd "$(dirname "$0")/../.." && pwd)"
tag="containers-aging-test:$(echo "${image%%:*}-${image#*:}" | tr -c 'a-z0-9.\n' '-')-$runtime"
name="aging-install-test-$(echo "$image" | tr ':/.' '---')-$runtime"

# Downloads diretos por padrao (mirrors de pacotes ficaram ~10x mais lentos pelo proxy);
# TEST_USE_PROXY=1 repassa http(s)_proxy do host para o container.
proxy_env=()
[ "${TEST_USE_PROXY:-0}" = 1 ] && proxy_env=(-e http_proxy -e https_proxy -e no_proxy -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY)

docker rm -fv "$name" > /dev/null 2>&1 || true
status=0
docker run --name "$name" --privileged --cgroupns=private --network host \
  --cpus "${TEST_CPUS:-3}" --memory "${TEST_MEMORY:-5g}" "${proxy_env[@]}" \
  -v /var/lib/docker -v /var/lib/containers \
  -v "$repo:/src:ro" "$image" bash -c "
    set -e
    cp -a /src /repo && rm -rf /repo/env
    # pipelining HTTP do apt fica muito lento atras de alguns proxies
    [ -d /etc/apt/apt.conf.d ] && echo 'Acquire::http::Pipeline-Depth \"0\";' > /etc/apt/apt.conf.d/99-no-pipelining
    [ -f /etc/dnf/dnf.conf ] && echo 'max_parallel_downloads=10' >> /etc/dnf/dnf.conf
    /repo/install/install.sh --runtime $runtime --monitoring yes --no-kernel-debug $*
    bash /src/tests/install/verify.sh $runtime
  " || status=$?

if [ "$status" -eq 0 ]; then
  # o estado do runtime "dentro" (imagens/containers de teste) nao entra na imagem salva
  docker commit --change 'CMD ["bash"]' "$name" "$tag" > /dev/null
  echo "imagem salva: $tag"
  docker rm -v "$name" > /dev/null
elif [ "${KEEP_FAILED:-0}" != 1 ]; then
  docker rm -v "$name" > /dev/null
fi
exit "$status"
