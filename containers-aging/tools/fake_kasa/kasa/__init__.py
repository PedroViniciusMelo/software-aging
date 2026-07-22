"""
Substituto falso da biblioteca "python-kasa", so pra testar o monitoramento da tomada
(src/energy.py -> run_tomada_monitor) em uma maquina/VM sem a tomada real na rede.

Nao implementa o protocolo de rede de verdade dos dispositivos TP-Link Kasa/Tapo (esse
protocolo e' proprietario e, na Tapo, criptografado -- reimplementa-lo so pra teste seria
complexo e fragil). Em vez disso, imita a MESMA interface que o kasa de verdade expoe
(Discover.discover_single, device.update(), device.has_emeter, device.get_emeter_realtime(),
device.is_on), com potencia simulada oscilando de forma realista.

Uso: coloque esta pasta na frente do PYTHONPATH antes de rodar o framework, para que
"from kasa import Discover" resolva para este modulo falso em vez do pacote real (que nem
precisa estar instalado):

  export PYTHONPATH=tools/fake_kasa:$PYTHONPATH

O "ip"/"username"/"password" passados para discover_single sao ignorados (nenhuma rede e'
usada de verdade), entao qualquer valor de tomada_ip no config.yaml serve durante o teste.
"""
import asyncio
import random

BASE_POWER_W = 45.0
JITTER_W = 12.0
DISCONNECT_CHANCE = 0.05  # chance de simular uma falha de leitura, pra exercitar o log de erro/reconexao


class _FakeEmulatedPlug:
    def __init__(self):
        self.is_on = True
        self.has_emeter = True
        self._power_w = BASE_POWER_W

    async def update(self):
        # dispositivo real faz uma chamada de rede aqui; simulamos so uma pequena espera
        await asyncio.sleep(0.02)

        if random.random() < DISCONNECT_CHANCE:
            raise ConnectionError("tomada simulada: timeout de rede (falha simulada de proposito)")

    async def get_emeter_realtime(self):
        self._power_w = max(0.0, self._power_w + random.uniform(-JITTER_W, JITTER_W) * 0.3)
        return {"power_mw": round(self._power_w * 1000)}


class Discover:
    @staticmethod
    async def discover_single(host, *, username=None, password=None):
        await asyncio.sleep(0.05)  # simula a latencia de descoberta na rede
        return _FakeEmulatedPlug()