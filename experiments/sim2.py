"""
sim2.py — simulátor se SYNAPTICKOU INTEGRACÍ.

Nález z 13. 9. 2026: upstream `flybrain/simulator.py` počítá I = W @ spikes,
tedy proud trvá jeden krok a v čase se nesčítá. Důsledek: neuron s málo
synapsemi nemůže vystřelit NIKDY, ať je vstup jak chce silný. T5a má 83
excitačních synapsí, takže i při současném výstřelu všech vstupů povyskočí
napětí o 0,513 při prahu 1,0. Proto v celém optickém laloku nestřílely T4
ani T5 a dráha detekce pohybu byla mrtvá.

Referenční model (Shiu et al. 2024) má synaptickou proměnnou g s konstantou
5 ms: `g += w` na každý spike, `dg/dt = -g/tau`. Při trvalém vstupu se
příspěvky sčítají a účinek vzroste zhruba 1/(1-exp(-dt/tau)) ≈ 5,5×.

Tady je to doplněné:
    g[t] = alfa * g[t-1] + W @ spikes[t-1]      (alfa = exp(-dt/tau_syn))
    v[t] = v[t-1] + dt/tau_mbr * (-(v-v_rest) + g[t])

Upstream se NEUPRAVUJE, tohle je samostatná vrstva vedle něj.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import scipy.sparse as sp
import torch

from flybrain.neurons import LIFParams, LIFPopulation
from flybrain.simulator import SimResult, Simulator

logger = logging.getLogger(__name__)

TAU_SYN_MS = 5.0  # Jürgensen et al. 2021, převzato ze Shiu et al. 2024


class Simulator2(Simulator):
    """Simulator z upstreamu, ale s pamětí synaptického proudu."""

    def __init__(self, *a, tau_syn_ms: float = TAU_SYN_MS, **kw) -> None:
        super().__init__(*a, **kw)
        self.alfa = math.exp(-self.pop.params.dt_ms / tau_syn_ms)
        self.g = torch.zeros(self.n, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, external: torch.Tensor | None = None) -> torch.Tensor:
        vstup = torch.sparse.mm(self.W, self.pop.spikes.view(-1, 1)).view(-1)
        self.g = self.alfa * self.g + vstup
        proud = self.g
        if external is not None:
            proud = proud + external.to(device=self.device, dtype=self.dtype)
        return self.pop.step(proud)

    @torch.no_grad()
    def run(self, *a, **kw) -> SimResult:
        self.g.zero_()
        return super().run(*a, **kw)
