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
- **Usa o mesmo compilador em todas as distros.** Cada par de versões usa o Go do release
  oficial do Docker correspondente para Docker e Podman, e um Rust fixo para netavark e
  aardvark-dns. As build tags também são as mesmas em todas as distros.
- **Instala os componentes oficiais.** O Docker usa containerd, runc e tini nas versões do
  release oficial. O Podman usa crun, conmon, netavark, aardvark-dns e catatonit nas versões
  publicadas até o release do Podman.

Versões disponíveis (`--docker-version` / `--podman-version`):

| Runtime | Versão | Go | Componentes |
|---|---|---|---|
| Docker | 27.2.0 (padrão) | 1.21.13 | containerd 1.7.21, runc 1.1.13, tini 0.19.0 |
| Docker | 29.10.1 | 1.27.4 | containerd 2.4.2, runc 1.5.3, tini 0.19.0 |
| Podman | 5.1.2 (padrão) | 1.21.13, Rust 1.79.0 | crun 1.15, conmon 2.1.12, netavark/aardvark-dns 1.11.0, catatonit 0.2.0 |
| Podman | 6.2.1 | 1.27.4, Rust 1.100.1 | crun 1.31, conmon 2.2.1, netavark/aardvark-dns 2.1.0, catatonit 0.2.1 |

O netavark 2.x só tem firewall nftables: com o Podman 6 o instalador instala o `nftables` e
fixa `firewall_driver = "nftables"` em `/etc/containers/containers.conf`.
- **Isola a compilação.** Fontes, toolchains e caches ficam em `BUILD_ROOT`
  (`/var/tmp/containers-aging-build`), que é apagado no final. Os pacotes `-dev`/`-devel`
  instalados só para compilar também são removidos. Só ficam as bibliotecas que os binários
  instalados usam, detectadas com `ldd`. Nada é escrito em `~/go`, `~/.cargo` ou `~/.profile`.
- **Registra o que foi compilado.** `/etc/containers-aging/build-manifest.txt` guarda versões,
  commits, toolchains, gcc, distro e kernel. O coletor copia esse arquivo para `logs/` em cada
  experimento.

Tudo é instalado em `/usr/local`, sem misturar com pacotes da distro. Runtimes de container que
venham com a distro (o Fedora Server traz o Podman) são **removidos** antes de instalar Docker ou
Podman: os compilados sempre têm prioridade e o sistema base fica igual nos dois cenários (a lista
é `DISTRO_RUNTIME_PKGS` em `lib/distro/<familia>.sh`; no Fedora o `container-selinux` fica). O systemtap mantém
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

## Nomes das pastas de resultados

Os scripts de `plot/` identificam o cenário pelo nome da pasta, no formato
`<sistema><versão>_<runtime>_<versão>`, por exemplo `debian12_docker_27.2.0` ou
`fedora45_podman_5.1.2`. Nomes antigos (`docker_novo`, `docker_velho`) não são mais reconhecidos
(aparecem como `unknown`): renomeie as pastas de resultados antigas para o formato com versão.
