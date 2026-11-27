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
   - `general.runs`, `stressload.max_stress_time` e `stressload.wait_after_stress`:
     a duração total é `runs × (max_stress_time + wait_after_stress)`;
   - `containers`: as imagens usadas nos ciclos de estresse;
   - `energy`: RAPL e smart plug (opcionais, desligados por padrão).

   A pasta do projeto é detectada sozinha (a do `main.py`); não precisa configurar caminho.

3. **Rode o experimento** (como root):

   ```bash
   env/bin/python main.py              # usa o config.yaml do projeto
   env/bin/python main.py outro.yaml   # outro arquivo de configuração
   ```

   O runtime (Docker ou Podman) é detectado automaticamente. Durante o experimento os dados
   vão para `logs/`. Ao terminar, o coletor para o monitoramento, copia `logs/` para
   `results/<cenário>_<data>` (por exemplo `results/debian13_docker_27.2.0_20261125-1830`) e
   gera os gráficos em `plots_img/` dentro dessa pasta. Para desligar, use
   `general.plot_results: false`.

   Enquanto roda, digite comandos no terminal (`logging.console`): `threads` mostra as
   threads de cada parte (monitores, stressload, principal) com TID, CPU e o que cada uma
   está fazendo; `children` mostra os comandos externos em execução; `status`, a fase e o
   run; `cost`, o custo do monitoramento; `log info|quiet` troca o nível de log. Os
   logs informativos (`logging.informative`) também vão para `logs/collector.log`; com
   `false`, só aparecem avisos e erros.

   **Para parar antes do fim**, aperte Ctrl+C (ou envie SIGTERM): o coletor termina os
   containers que estão no meio do ciclo (para e remove o que subiu), encerra a coleta e
   salva os resultados parciais em `results/<cenário>_<data>_interrupted`. Um segundo
   Ctrl+C força a saída sem salvar. Antes de o monitoramento começar (limpeza, imagens,
   TeaStore), o primeiro Ctrl+C já encerra na hora.

## Rejuvenescimento

Com `rejuvenation.enabled: true`, o método `rejuvenation.method` roda ao fim de cada
período de stress, com o monitor de ciclo de vida pausado. Métodos disponíveis
(`METHODS` em `src/environment.py`):

- `cleanup_keep_teastore` (padrão): remove os containers, as imagens e os volumes sem uso,
  **preservando tudo do TeaStore** (containers com o rótulo `containers-aging.role=teastore`,
  posto pelo `start_teastore.sh`, ou com imagem em `rejuvenation.protected_images`);
- `full_reset`: o método antigo; apaga tudo, inclusive o TeaStore, e o sobe de novo.

Para criar um método, escreva uma função `(runtime, config, start_teastore) -> dict` em
`src/environment.py`, adicione-a em `METHODS` e coloque o nome em `rejuvenation.method`. Cada
aplicação vai para `logs/rejuvenation.csv` (método, duração e o que foi removido).

4. **Compare os cenários** juntando as pastas de `results/` das várias máquinas numa só:

   ```bash
   env/bin/python plot/plot_data.py results/      # refaz os gráficos de cada cenário
   env/bin/python plot/theilsen.py results/       # tendência (Theil-Sen) e índice de envelhecimento
   env/bin/python plot/mannkendall.py results/    # teste de Mann-Kendall
   env/bin/python plot/mannkendall_summary.py results/
   ```

   Os arquivos do JMeter entram na pasta do cenário como `<cenário>.csv` e
   `jmeter_<cenário>.csv`.

A carga do JMeter (`start_server_response.sh` e `teastore_browse_nogui.jmx`) roda em
outro host, apontando para a máquina do experimento.

## O que é gerado em `logs/`

O coletor grava só **valores brutos** (contadores acumulados do kernel, kB, ns, uma linha por
instância de processo), um CSV por métrica, todos com `date_time`. As contas (CPU em %, taxas de
disco, soma das instâncias de cada processo, MB/GB, energia por intervalo) são feitas em
`plot/metrics.py`, que `plot_data.py`, `mannkendall.py` e `theilsen.py` usam. Contadores
acumulados vêm com `uptime_s`, o relógio usado nas taxas.

| Arquivo | Conteúdo |
|---|---|
| `cpu.csv` | linha `cpu` do `/proc/stat` (ticks acumulados: user, nice, system, idle, iowait, ...) |
| `memory.csv` | campos do `/proc/meminfo` (kB) |
| `disk.csv` | `statvfs("/")`: tamanho do bloco, blocos totais, livres e disponíveis |
| `disk_io.csv` | os 17 campos do `/proc/diskstats` do disco monitorado |
| `processes.csv` | uma linha por instância de cada processo monitorado (`monitoring.processes`): pid, início, utime/stime em ticks, threads, VSZ, RSS e swap (kB) |
| `zombies.csv` | quantidade de processos zumbis |
| `kernel.csv` | mensagens novas do kernel sobre processos mortos (`killed`) e OOM (`oom`) |
| `lifecycle.csv` | tempo (ns) de cada etapa do ciclo de vida de cada container (carregar imagem, subir, parar, remover) |
| `fragmentation_*.csv` | fragmentação de memória (systemtap, `fragmentation.stp`) |
| `energy_rapl.csv`, `cpu_temp.csv`, `energy_smart_plug.csv` | contadores RAPL (`energy_uj` e o máximo, para a volta do contador), temperatura do pacote (m°C) e potência da tomada (mW), quando habilitados |
| `errors.csv` | falhas ao subir, matar ou remover containers |
| `monitoring_cost.csv` | CPU acumulada de cada parte do coletor (threads + comandos que elas executam) |
| `collector.csv` | processo coletor inteiro: CPU acumulada própria e dos comandos, RSS, VSZ e threads |
| `qtd_containers.csv` | containers criados e removidos por cada thread de estresse |
| `rejuvenation.csv`, `runs.csv` | aplicações do rejuvenescimento; início de cada fase (stress, sleep, rejuvenation) |
| `collector.log` | logs do coletor (com `logging.to_file`) |
| `config.yaml` | cópia do config usado na execução |
| `environment.txt`, `build-manifest.txt` | sistema, kernel, `CLK_TCK` (usado nas contas de CPU dos processos) e versões exatas do que foi compilado |

Para uma **métrica nova**: um leitor e um método em `MachineMetrics` (`src/monitoring.py`), com
uma entrada em `MachineMetrics.METRICS`; a conta em `plot/metrics.py`; e uma linha em `PLOTS`
(`plot/plot_data.py`).

## Organização

```
main.py                 ponto de entrada: lê o config, roda o experimento e gera os gráficos
config.yaml             parâmetros do experimento
src/
  config.py             leitura do config.yaml
  environment.py        fluxo principal (limpa, imagens, TeaStore, monitora, estresse) e rejuvenescimento
  runtime.py            comandos do Docker/Podman, nome do cenário e ciclo de vida de um container
  stressload.py         ciclos de estresse: vários containers em paralelo
  monitoring.py         leituras de /proc e /sys e as threads de coleta (máquina, processos, kernel,
                        ciclo de vida, fragmentação, energia, custo)
  activity.py           partes do sistema, comandos em andamento, CPU de cada parte e parada (Ctrl+C)
  console.py            terminal interativo (threads, comandos, status, custo)
  utils.py              logs, execução de comandos e escrita dos CSVs
start_teastore.sh       sobe a aplicação TeaStore
fragmentation.stp       script systemtap de fragmentação de memória
install/                instalador (versões fixas, compilação do fonte)
plot/                   contas sobre os dados brutos (metrics.py), gráficos e análises
```
