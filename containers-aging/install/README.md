# Instalador

```bash
sudo install/install.sh                                   # menu interativo
sudo install/install.sh --runtime docker --monitoring yes # sem perguntas
sudo install/install.sh --help
```

`dependencies.sh` na raiz continua funcionando e chama este script.

Distros suportadas: Debian, Ubuntu e Fedora.

## Por que compilar tudo

Em estudos de envelhecimento e benchmark, a diferença entre os cenários deve ser só o fator
estudado (sistema, runtime, versão). Os pacotes de cada distro trazem versões, patches e flags
de compilação diferentes. Por isso o instalador:

- **Fixa versões e confere a origem.** Docker, Podman, os componentes de cada um e o systemtap
  são baixados pela tag, e o commit é conferido. Tarballs e toolchains Go são conferidos por
  sha256. Tudo fica em `versions/`.
- **Usa o mesmo compilador em todas as distros.** Go 1.21.13 (o do release oficial do
  Docker 27.2.0) para Docker e Podman, e Rust 1.79.0 para netavark e aardvark-dns. As build
  tags também são as mesmas em todas as distros.
- **Instala os componentes oficiais.** O Docker usa containerd, runc e tini nas versões do
  release 27.2.0. O Podman usa crun, conmon, netavark, aardvark-dns e catatonit nas versões
  publicadas até o Podman 5.1.2.
- **Isola a compilação.** Fontes, toolchains e caches ficam em `BUILD_ROOT`
  (`/var/tmp/containers-aging-build`), que é apagado no final. Os pacotes `-dev`/`-devel`
  instalados só para compilar também são removidos. Só ficam as bibliotecas que os binários
  instalados usam, detectadas com `ldd`. Nada é escrito em `~/go`, `~/.cargo` ou `~/.profile`.
- **Registra o que foi compilado.** `/etc/containers-aging/build-manifest.txt` guarda versões,
  commits, toolchains, gcc, distro e kernel. O coletor copia esse arquivo para `logs/` em cada
  experimento.

Tudo é instalado em `/usr/local`, sem misturar com pacotes da distro. O systemtap mantém
`gcc`, `make` e os headers do kernel instalados, porque compila um módulo a cada execução.

## Estrutura

```
install.sh              entrada: opções, ordem das etapas
versions/               versões fixas (um arquivo por versão de runtime)
lib/common.sh           log, detecção da distro, remoção dos pacotes de build
lib/sandbox.sh          BUILD_ROOT, download conferido, toolchains Go/Rust, manifesto
lib/distro/<familia>.sh comandos do gerenciador de pacotes e nomes dos pacotes
components/*.sh         python, monitoring (systemtap), docker, podman
```

## Como estender

- **Nova versão de runtime:** crie `versions/docker-<versão>.env` (ou `podman-`) com as tags
  e os commits. Use `--docker-version <versão>`. Se precisar de outro Go, adicione o sha256 em
  `versions/toolchains.env`.
- **Nova distro:** crie `lib/distro/<familia>.sh` com as mesmas funções `pkg_*` e listas de
  pacotes, e adicione o `ID` dela em `detect_distro` (`lib/common.sh`).

Se uma versão antiga não compilar num sistema novo (por exemplo, por um gcc mais recente), o
primeiro recurso é ajustar flags no componente. O "malabarismo" antigo (usar o repositório de
outra distro) só vale se o binário resultante for comprovadamente o mesmo nos sistemas.

## Testes

`tests/install/run-in-container.sh <imagem> <docker|podman>` roda o instalador num container
descartável e privilegiado. Depois verifica o venv, o systemtap, a remoção das ferramentas de
build e sobe um container de verdade com o runtime compilado (`--init` e porta publicada). O
host não é alterado.

```bash
tests/install/run-in-container.sh debian:12 docker
tests/install/run-in-container.sh fedora:latest podman
```

Quando passa, o container instalado vira a imagem `containers-aging-test:<distro>-<runtime>`
(ex.: `containers-aging-test:debian-12-docker`). Para repetir a verificação nessas imagens sem
recompilar nada:

```bash
tests/install/smoke.sh                    # todas as imagens
tests/install/smoke.sh fedora-latest-podman
```

Esse teste não exercita o systemtap contra o kernel: o kernel do container é o do host, e os
headers e símbolos dele normalmente não existem no repositório da imagem.

## Nomes das pastas de resultados

Os scripts de `plot/` identificam o cenário pelo nome da pasta, no formato
`<sistema><versão>_<runtime>_<versão>`, por exemplo `debian12_docker_27.2.0` ou
`fedora45_podman_5.1.2`. Nomes antigos (`docker_novo`, `docker_velho`) não são mais reconhecidos
(aparecem como `unknown`): renomeie as pastas de resultados antigas para o formato com versão.
