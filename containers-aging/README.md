# containers-aging

Experimento de envelhecimento de software em runtimes de containers (Docker e Podman).
O coletor sobe o TeaStore, cria e destrói containers em ciclos de estresse e mede
continuamente o uso de recursos da máquina, dos processos do runtime e, se configurado,
o consumo de energia.

## Como rodar

> **Atenção:** rode só na máquina dedicada ao experimento. No início, o coletor apaga
> `logs/` e **remove todos os containers, imagens e volumes** do runtime
> (`system prune -a`).

1. **Instale as dependências** (como root). O instalador compila a versão fixa do runtime
   e o systemtap, e cria o ambiente Python em `env/`:

   ```bash
   sudo install/install.sh --runtime docker --monitoring yes   # ou --runtime podman
   ```

   Distros suportadas: Debian, Ubuntu e Fedora. Detalhes em [`install/README.md`](install/README.md).

2. **Ajuste o `config.yaml`.** Os campos principais são:
   - `general.scripts_folder`: caminho deste diretório;
   - `general.runs`, `stressload.max_stress_time` e `stressload.wait_after_stress`:
     a duração total é `runs × (max_stress_time + wait_after_stress)`;
   - `containers`: as imagens usadas nos ciclos de estresse;
   - `energy`: RAPL e tomada inteligente (opcionais).

3. **Rode o experimento** (como root):

   ```bash
   env/bin/python main.py              # usa ./config.yaml
   env/bin/python main.py outro.yaml   # outro arquivo de configuração
   ```

   O runtime (Docker ou Podman) é detectado automaticamente. Os resultados vão para
   `logs/`.

4. **Gere os gráficos** em outra máquina, com as pastas de resultados renomeadas para o
   cenário (por exemplo `logs/debian12_docker_27.2.0`, `logs/ubuntu24.04_podman_5.1.2`):

   ```bash
   pip install -r plot/requirements.txt
   python3 plot/plot_data.py logs/       # gráficos de cada cenário
   python3 plot/theilsen.py logs/        # tendência (Theil-Sen) e índice de envelhecimento
   python3 plot/mannkendall.py logs/     # teste de Mann-Kendall
   python3 plot/mannkendall_summary.py logs/
   ```

A carga do JMeter (`start_server_response.sh` e `teastore_browse_nogui.jmx`) roda em
outro host, apontando para a máquina do experimento.

## O que é gerado em `logs/`

| Arquivo | Conteúdo |
|---|---|
| `cpu.csv`, `memory.csv`, `disk.csv`, `disk_write_read.csv`, `process.csv` | recursos da máquina (mpstat, free, df, iostat, zumbis) |
| `<processo>.csv` | CPU, memória, threads e swap de cada processo do runtime (dockerd, containerd, conmon, …) |
| `<container>.csv` | tempos de cada etapa do ciclo de vida (carregar imagem, subir, parar, remover) |
| `fragmentation_*.csv` | fragmentação de memória (systemtap, `fragmentation.stp`) |
| `killed.csv`, `omm.csv` | mensagens novas do kernel sobre processos mortos e OOM |
| `energy_rapl.csv`, `energy_tomada.csv` | energia, quando habilitada |
| `errors.csv` | falhas ao subir, matar ou remover containers |
| `environment.txt`, `build-manifest.txt` | sistema, kernel e versões exatas do que foi compilado |

## Organização

```
main.py                 ponto de entrada: lê o config e roda o experimento
config.yaml             parâmetros do experimento
src/
  config.py             leitura e validação do config.yaml
  environment.py        fluxo principal: limpa, sobe o TeaStore, monitora, aplica o estresse
  runtime.py            comandos do Docker/Podman e detecção do runtime e do disco
  lifecycle.py          subir, parar e remover um container (com tentativas)
  stressload.py         ciclos de estresse: vários containers em paralelo
  monitoring.py         threads de coleta (máquina, processos, kernel, fragmentação, energia)
  energy.py             leitura do RAPL e da tomada inteligente
  images.py             baixa as imagens e salva os .tar usados no monitoramento e no estresse
  utils.py              execução de comandos e escrita dos CSVs
start_teastore.sh       sobe a aplicação TeaStore
fragmentation.stp       script systemtap de fragmentação de memória
install/                instalador (versões fixas, compilação do fonte)
plot/                   análise e gráficos dos resultados
tools/                  hardware falso (RAPL e tomada) para testar a coleta de energia
```
