#!/usr/bin/env bash
# Roda dentro do container de teste, depois do install.sh.
set -euo pipefail
runtime="$1"
echo "===== VERIFICACAO ($runtime)"

/repo/env/bin/python -c "import yaml; print('venv ok, PyYAML', yaml.__version__)"
/usr/local/bin/stap -V 2>&1 | head -n1
for tool in mpstat iostat pidstat gcc make; do command -v "$tool" > /dev/null || { echo "faltando: $tool"; exit 1; }; done
cat /etc/containers-aging/build-manifest.txt

for tool in git cmake g++ cargo go protoc; do
  if command -v "$tool" > /dev/null; then echo "ferramenta de build ficou instalada: $tool"; exit 1; fi
done
[ ! -e /var/tmp/containers-aging-build ] || { echo "BUILD_ROOT nao foi apagado"; exit 1; }

# Runtime aninhado com cgroup v2: a raiz do cgroup do container de teste tem processos, entao
# os controllers nao podem ser delegados. Move os processos para um filho e habilita os
# controllers (mesmo truque do entrypoint oficial do docker:dind). So' no ambiente de teste.
if [ -f /sys/fs/cgroup/cgroup.controllers ] && [ "$runtime" != none ]; then
  mkdir -p /sys/fs/cgroup/init
  xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs 2> /dev/null || :
  sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers > /sys/fs/cgroup/cgroup.subtree_control
fi

case "$runtime" in
  docker)
    containerd > /tmp/containerd.log 2>&1 &
    sleep 3
    dockerd > /tmp/dockerd.log 2>&1 &
    for _ in $(seq 30); do docker info > /dev/null 2>&1 && break; sleep 1; done
    docker version
    docker run --rm --init -p 18080:80 docker.io/library/busybox sh -c 'ps -o pid,comm | head -3'
    ;;
  podman)
    podman version
    podman info --format '{{.Host.OCIRuntime.Name}} {{.Host.OCIRuntime.Version}} | conmon {{.Host.Conmon.Version}} | net {{.Host.NetworkBackend}}'
    podman run --rm --init -p 18080:80 docker.io/library/busybox sh -c 'ps -o pid,comm | head -3'
    ;;
esac
echo "===== OK"
