"""
flyagent.py — uzavřená smyčka nad skutečným connectomem: vjem → mozek → akce.

Tohle je infrastruktura pro všechno další (včetně hry v prohlížeči), ne
experiment. Agent umí:

    agent.act(stimul)  -> Akce(bilance, vlevo_hz, vpravo_hz, ...)
    agent.odmena(...)  -> změní váhy podle třífaktorového pravidla

Akční kanál = SESTUPNÉ NEURONY (descending, 1290 kusů, 638 vlevo / 644 vpravo).
To je skutečná povelová cesta z mozku do těla; rozdíl levá/pravá je zatáčení.
Vjemový kanál = vizuální neurony po stranách.

POZOR na poctivost: connectome dává ZAPOJENÍ. Učicí pravidlo na vstupech do DN
je NAŠE, ne mouší — moucha se vizuálně neučí takhle. Je to třífaktorové
pravidlo (pre × post × odměna), co je standardní inženýrská volba, ale
biologicky je to konstrukt. Pravidlo v houbovitém tělísku (KC→MBON deprese)
je naproti tomu doložené; to se používá v exp01.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from flybrain.circuits import Circuits
from flybrain.loader import Connectome
from flybrain.simulator import Simulator

logger = logging.getLogger(__name__)

WEIGHT_SCALE = 0.124  # ukotveno, viz UKOTVENI-WEIGHT-SCALE.md


@dataclass
class Akce:
    """Co mozek udělal: frekvence obou DN skupin a z nich odvozené zatáčení."""

    vlevo_hz: float
    vpravo_hz: float
    spiky: np.ndarray  # počty spiků všech neuronů (pro eligibility trace)

    @property
    def bilance(self) -> float:
        """(L − P) / (L + P) v [−1, 1]. Kladné = zatoč doleva."""
        s = self.vlevo_hz + self.vpravo_hz
        return (self.vlevo_hz - self.vpravo_hz) / s if s > 0 else 0.0

    @property
    def smer(self) -> str:
        return "vlevo" if self.bilance > 0 else "vpravo"


class FlyAgent:
    """Octomilka jako agent: dostane vjem, vrátí akci, umí přijmout odměnu."""

    def __init__(self, connectome: Connectome, weight_scale: float = WEIGHT_SCALE,
                 steps: int = 200, amp: float = 3.0) -> None:
        self.c = connectome
        self.weight_scale = weight_scale
        self.steps = steps
        self.amp = amp

        cir = Circuits(connectome)
        self.dn_l = cir.find("DN L", super_class="descending", side="left").indices
        self.dn_r = cir.find("DN R", super_class="descending", side="right").indices
        self.vis_l = cir.find("vis L", class_="visual", side="left").indices
        self.vis_r = cir.find("vis R", class_="visual", side="right").indices

        self.W = connectome.signed_weights().tocsr().astype(np.float32)
        self.W.sort_indices()
        self.W0 = self.W.copy()  # pro reset

        # předpočítej pozice vstupních synapsí do každé DN skupiny
        self._vstupy = {"vlevo": self._radky(self.dn_l), "vpravo": self._radky(self.dn_r)}

    def _radky(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pozice v W.data pro všechny vstupní synapse daných neuronů + jejich zdroje."""
        pos, src = [], []
        for r in rows:
            lo, hi = self.W.indptr[r], self.W.indptr[r + 1]
            pos.append(np.arange(lo, hi))
            src.append(self.W.indices[lo:hi])
        return np.concatenate(pos), np.concatenate(src)

    def reset_vahy(self) -> None:
        self.W = self.W0.copy()

    def act(self, stimul: np.ndarray) -> Akce:
        """Jeden průchod smyčkou: stimuluj vjem, nech mozek běžet, přečti DN."""
        sim = Simulator(self.W, weight_scale=self.weight_scale)
        ext = sim.input_vector(stimul, self.amp) if len(stimul) else None
        res = sim.run(self.steps, external=ext)
        spiky = res.spike_counts.numpy()
        dur = res.duration_s
        return Akce(float(spiky[self.dn_l].mean() / dur),
                    float(spiky[self.dn_r].mean() / dur), spiky)

    def odmena(self, akce: Akce, strana: str, sila: float = 0.05) -> None:
        """Třífaktorové pravidlo (pre × post × odměna) s protitahem.

        Posílí vstupy do DN skupiny `strana` a o totéž zeslabí vstupy do druhé
        strany — úměrně tomu, jak byl presynaptický neuron při TÉHLE akci aktivní
        (eligibility trace). Protitah je tam proto, aby váhy jen nerostly:
        samotné posilování by po pár trialech rozjelo síť do saturace.

        Nebiologické — viz hlavička modulu.
        """
        druha = "vpravo" if strana == "vlevo" else "vlevo"
        for cil, znam in ((strana, +1.0), (druha, -1.0)):
            pos, src = self._vstupy[cil]
            elig = akce.spiky[src]
            if elig.max() > 0:
                elig = elig / elig.max()
            self.W.data[pos] *= (1.0 + znam * sila * elig).astype(np.float32)

    # --- vjemy ---
    def vjem(self, kde: str) -> np.ndarray:
        return {"vlevo": self.vis_l, "vpravo": self.vis_r,
                "nic": np.array([], dtype=int)}[kde]
