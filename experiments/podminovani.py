"""
podminovani.py — společný základ pro učení v houbovitém tělísku.

Vzniklo 20. 9. 2026 po exp29 (první doložené učení). Sdružuje, co se
opakuje: mozek se Shiuovým pravidlem, zraková scéna v pozadí (bez ní je
sosák na podlaze 1–3 Hz), odměna z cukru přes skutečný dopamin, trest
z hořké chuti přes PPL1, a NOVĚ převod obrazu na kód Kenyonových buněk.

Kód obrazu: jas 768 sloupců retiny -> pevná náhodná projekce -> 5 %
nejsilnějších buněk (winner-take-all). To je mechanismus, kterým řídký
kód v houbovitém tělísku vzniká (náhodné vstupy + inhibice APL). Podobné
obrazy dávají překrývající se sady buněk, různé obrazy skoro disjunktní.
JE TO NÁŠ PŘEVOD, ne mouší: zrak k Kenyonovým buňkám v modelu nedojde.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp26_endogenni_odmena import Chut  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from pravidlo_shiu import pouzij_shiu  # noqa: E402
from server_mozku import snimek  # noqa: E402
from sim2 import Simulator2  # noqa: E402

log = logging.getLogger(__name__)
WS = 0.0393
PODIL_KC = 0.05


class Mozek:
    def __init__(self, steps: int = 100, amp_kc: float = 4.0) -> None:
        c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
        pouzij_shiu(c)
        self.c = c
        self.a = Agent2(c, steps=steps); cir = Circuits(c)
        self.ch = Chut(self.a, c)
        poz = {int(j): i for i, j in enumerate(self.a.i_spike)}
        self.sosak = np.array([poz[int(j)] for j in cir.proboscis_motor_neurons().indices if int(j) in poz])
        self.horka = np.array([poz[int(j)] for j in cir.bitter_grns().indices if int(j) in poz])
        self.ppl1 = np.array([poz[int(j)] for j in cir.ppl1_dans().indices if int(j) in poz])
        mb_glob = cir.mbons().indices
        self.ppl1_na_mbon = c.syn_counts[mb_glob][:, cir.ppl1_dans().indices].toarray().astype(np.float32)
        self.amp_kc = amp_kc
        self.scena = self.a.vjem(snimek(-12.0), snimek(-24.0))
        # kód obrazu: retina (levé oko, OFF kanál) -> náhodná projekce -> WTA
        self.ret = self.a.oci["left"]["retina"]
        rng = np.random.default_rng(0)
        uv = self.ret.souradnice("off"); n_sl = len(uv)
        # Prostorové rozmazání PŘED projekcí (σ ≈ 10° zorného pole). Bez něj
        # dává sloup posunutý o 6° skoro disjunktní kód (překryv < 0,15),
        # protože se nepřekrývají už samotné vstupy z retiny — sloup je užší
        # než posun. Receptivní pole širší než posun je to, co dělá podobné
        # obrazy podobnými. (Nalezeno 20. 9. 2026, exp30 K0.)
        d2 = ((uv[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
        sigma = 4.0 / 120.0   # 10° dělalo i vodorovný pruh podobným sloupu (0,24); 4° drží K0 s rezervou
        K = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        self.K = K / K.sum(1, keepdims=True)
        self.R = rng.standard_normal((n_sl, len(self.ch.kc))).astype(np.float32) / np.sqrt(n_sl)

    # --- vstupy ---
    def kc_z_obrazu(self, obraz: np.ndarray):
        """Obraz (2D, 0–1) -> (indexy KC, amplituda). Tma = signál (OFF kanál)."""
        idx, amp = self.ret.stimul(obraz, amp=1.0)
        tma = amp[: len(self.ret.souradnice("off"))]
        z = (self.K @ tma) @ self.R
        k = int(PODIL_KC * len(self.ch.kc))
        vyb = np.argpartition(-z, k)[:k]
        return (self.ch.kc[vyb], self.amp_kc)

    def kc_nahodne(self, rng):
        return (self.ch.kc[rng.choice(len(self.ch.kc), size=int(PODIL_KC * len(self.ch.kc)), replace=False)], self.amp_kc)

    @staticmethod
    def prekryv(kc1, kc2) -> float:
        return len(np.intersect1d(kc1[0], kc2[0])) / len(kc1[0])

    # --- běh ---
    def beh(self, kc=None, cukr=False, horka=False, kroku=None):
        """Vrací (sosák Hz, spiky). Scéna vždy v pozadí."""
        ext = self.scena.copy()
        if kc is not None:
            ext[kc[0]] += kc[1]
        if cukr:
            ext[self.ch.cukr] += 3.0
        if horka:
            ext[self.horka] += 3.0
        res = Simulator2(self.a.W_spike, weight_scale=WS).run(kroku or self.a.steps, external=ext)
        sc = res.spike_counts.numpy().astype(np.float32)
        return float(sc[self.sosak].sum() / res.duration_s / len(self.sosak)), sc

    def odpoved(self, kc) -> float:
        """Sosák na podnět samotný, 200 kroků (odečet)."""
        return self.beh(kc, kroku=200)[0]

    # --- učení ---
    def odmen(self, kc, sila=0.5) -> float:
        _, sc = self.beh(kc, cukr=True)
        return self.ch.uc_z_chuti(sc, sila)

    def potrestej(self, kc, sila=0.5) -> float:
        """Hořká chuť -> PPL1 -> deprese v PPL1 kompartmentech (skutečný dopamin)."""
        _, sc = self.beh(kc, horka=True)
        akt = sc[self.ppl1]
        brana = self.ppl1_na_mbon @ akt
        mx = brana.max()
        if mx <= 0:
            return 0.0
        brana = brana / mx
        for k, (pos, src) in self.ch.useky.items():
            g = float(brana[k])
            if g <= 0:
                continue
            a = sc[src]; m = a.max()
            if m > 0:
                self.a.W_spike.data[pos] *= (1.0 - sila * g * (a / m)).astype(np.float32)
        return float(akt.sum())

    def reset(self):
        self.a.reset_vahy()
