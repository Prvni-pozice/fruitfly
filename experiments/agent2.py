"""
agent2.py — agent podle architektury z NALEZ-ARCHITEKTURA.md.

    snímek -> retina (sloupce) -> detektor pohybu -> T4/T5
           -> graduovaná vrstva (váhy FlyWire)     -> HS/VS/LC
           -> spikující centrální mozek (sim2)     -> DNa02 -> akce

Dělba na graduovanou a spikující část není optimalizace, ale biologie:
celá zraková dráha odpovídá graduovaně, spiky začínají až u sestupných
neuronů. Vedlejší užitek: spikující část je o 45 % synapsí menší a běží
4× rychleji.

Detektor pohybu je NAŠE inženýrská volba (Hassenstein-Reichardt: součin
zpožděného a nezpožděného signálu sousedních sloupců). Nahrazuje flyvis
tam, kde je potřeba rychlost — flyvis staví podnět 23 s, hra potřebuje
desetiny sekundy. Shoda s flyvis se ověřuje zvlášť.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from flybrain.loader import Connectome
from rate_vrstva import RateVrstva
from retina import Retina
from sim2 import Simulator2

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
WS = 0.0393
SMERY = ["a", "b", "c", "d"]  # a/b = vodorovně, c/d = svisle


@dataclass
class Akce2:
    dna02_l: float
    dna02_r: float
    dnp09: float
    mdn: float
    spiky: np.ndarray = field(repr=False, default=None)

    @property
    def zataceni(self) -> float:
        s = self.dna02_l + self.dna02_r
        return (self.dna02_l - self.dna02_r) / s if s > 0 else 0.0


class Agent2:
    def __init__(self, c: Connectome, steps: int = 200, zisk: float = 0.5) -> None:
        self.c, self.steps = c, steps
        n = c.nodes
        pt = n["primary_type"].astype("string")
        W = c.signed_weights().tocsr().astype(np.float32)
        self.N = W.shape[0]

        graded = np.array(n["super_class"].isin(["optic", "visual_centrifugal"]).to_numpy(), copy=True)
        graded |= pt.str.startswith(("HS", "VS", "H2", "LPT")).fillna(False).to_numpy()
        self.i_graded = np.flatnonzero(graded)
        self.i_spike = np.flatnonzero(~graded)
        self.rate = RateVrstva(W, self.i_graded, zisk=zisk)
        self.W_spike = W[self.i_spike][:, self.i_spike].tocsr()
        self.W_g2s = W[self.i_spike][:, self.i_graded].tocsr()   # graduovaná -> spikující
        self.W0 = self.W_spike.copy()
        self._W_g2s0 = self.W_g2s.copy()
        self._wmax = float(np.abs(self.W_spike.data).max())

        # odečty (pozice uvnitř spikující části)
        poz = {int(j): i for i, j in enumerate(self.i_spike)}
        def sel(mask):
            return np.array([poz[int(j)] for j in np.flatnonzero(mask) if int(j) in poz])
        self.dna_l = sel((n["primary_type"] == "DNa02").fillna(False).to_numpy()
                         & (n["side"] == "left").to_numpy())
        self.dna_r = sel((n["primary_type"] == "DNa02").fillna(False).to_numpy()
                         & (n["side"] == "right").to_numpy())
        self.dnp09 = sel((n["primary_type"] == "DNp09").fillna(False).to_numpy())
        self.mdn = sel(pt.str.startswith("MDN").fillna(False).to_numpy())
        self.dn_vse = sel(n["super_class"].eq("descending").to_numpy())

        self._retina_a_sloupce()

    def _retina_a_sloupce(self) -> None:
        """Retina na obě oči + pozice T4/T5 uvnitř graduované vrstvy."""
        ca = pd.read_csv(ROOT.parent / "flybrain" / "data" / "column_assignment.csv.gz")
        rid2idx = {r: i for i, r in enumerate(self.c.root_ids)}
        self.oci = {}
        for hemi in ("left", "right"):
            r = Retina(self.c, hemi)
            uv_l2 = r.souradnice("off")
            d0 = ca[ca["hemisphere"] == hemi]
            t45 = {}
            for pref in ("T4", "T5"):
                for s in SMERY:
                    T = f"{pref}{s}"
                    d = d0[d0["type"] == T]
                    ix = np.array([rid2idx[x] for x in d["root_id"] if x in rid2idx])
                    pq = d[["p", "q"]].values.astype(float)[: len(ix)]
                    a = (pq - pq.min(0)) / (pq.max(0) - pq.min(0))
                    # ke každému T4/T5 najdi nejbližší sloupec retiny
                    t45[T] = (self.rate.poz(ix), cKDTree(uv_l2).query(a)[1])
            # Sousedé podle SKUTEČNÉHO směru v zorném poli, ne podle pořadí.
            # (Chyba nalezená 13. 9. 2026: brát k-tého nejbližšího souseda
            # přiřadí T4a-d náhodné směry a rozlišení vlevo/vpravo se rozpadne.)
            strom = cKDTree(uv_l2)
            _, kand = strom.query(uv_l2, k=7)
            # PRAVÉ OKO JE ZRCADLO. Podtyp T4a preferuje v každém oku svůj
            # vlastní směr vůči tělu, ne vůči obrázku — takže se u pravého
            # oka musí prohodit vodorovná dvojice (a<->b) i svislá (c<->d).
            # Bez toho moucha sleduje cíl jen v levé polovině zorného pole
            # a v pravé zatáčí náhodně (změřeno 14. 9. 2026: správný směr
            # 75–100 % vlevo, ale 12–75 % vpravo). Nález byl zapsaný
            # v NALEZ-ARCHITEKTURA.md z mostu do flyvis, jen se sem
            # nikdy nepromítl.
            smery = {"a": (1, 0), "b": (-1, 0), "c": (0, 1), "d": (0, -1)}
            if hemi == "right":
                smery = {"a": (-1, 0), "b": (1, 0), "c": (0, -1), "d": (0, 1)}
            sous = {}
            for s_, (dx, dy) in smery.items():
                vyb = np.arange(len(uv_l2))
                for i in range(len(uv_l2)):
                    kd = kand[i, 1:]
                    v = uv_l2[kd] - uv_l2[i]
                    nrm = np.linalg.norm(v, axis=1); nrm[nrm == 0] = 1
                    skal = (v / nrm[:, None]) @ np.array([dx, dy], dtype=float)
                    vyb[i] = kd[int(np.argmax(skal))]
                sous[s_] = vyb
            self.oci[hemi] = {"retina": r, "uv": uv_l2, "t45": t45, "sousedi": sous}

    # --- vjem ---
    def pohyb(self, hemi: str, jas_ted: np.ndarray, jas_pred: np.ndarray) -> dict[str, np.ndarray]:
        """Hassenstein-Reichardt po sloupcích: 4 směry, ON i OFF."""
        o = self.oci[hemi]
        sous = o["sousedi"]
        vys = {}
        for s in SMERY:
            j = sous[s]      # soused ve směru, který tenhle podtyp preferuje
            korel = jas_ted * jas_pred[j] - jas_pred * jas_ted[j]  # zpožděný součin
            vys[f"T4{s}"] = np.clip(korel, 0, None)                 # ON
            vys[f"T5{s}"] = np.clip(-korel, 0, None)                # OFF
        return vys

    def vjem(self, snimek: np.ndarray, predchozi: np.ndarray | None) -> np.ndarray:
        """Snímek (2D, 0–1) -> vstupní proud pro spikující část."""
        x = np.zeros(len(self.i_graded), dtype=np.float32)
        for hemi in ("left", "right"):
            o = self.oci[hemi]
            jas = self._jas(o, snimek, hemi)
            jas_p = self._jas(o, predchozi, hemi) if predchozi is not None else jas
            m = self.pohyb(hemi, jas, jas_p)
            for T, (poz, mapa) in o["t45"].items():
                v = m[T][mapa][: len(poz)]
                mx = v.max()
                x[poz] = v / mx if mx > 0 else 0.0
        stav = self.rate.ustaleny_stav(x)
        self._posledni_stav = stav          # pro učení na rozhraní graduovaná -> spikující
        return (self.W_g2s @ stav).astype(np.float32)

    @staticmethod
    def _jas(o, snimek, hemi: str = "left") -> np.ndarray:
        """Každé oko vidí SVOU polovinu zorného pole.

        Do 14. 9. 2026 dostávala obě oči celý obraz, takže cíl vpravo budil
        levé oko úplně stejně jako pravé a lateralizace neměla odkud vzniknout.
        Projevilo se to tak, že moucha sledovala cíl jen v levé půlce pole
        (správný směr 75–100 %) a v pravé zatáčela náhodně (12–75 %).
        Skutečná moucha má oči po stranách hlavy a jejich zorná pole se
        překrývají jen úzkým pruhem vpředu.
        """
        uv = o["uv"]; h, w = snimek.shape
        lo, hi = (0.0, 0.55) if hemi == "left" else (0.45, 1.0)
        x = lo + uv[:, 0] * (hi - lo)
        col = np.clip((x * (w - 1)).round().astype(int), 0, w - 1)
        row = np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
        return snimek[row, col].astype(np.float32)

    # --- akce ---
    def act(self, snimek: np.ndarray, predchozi=None, zesileni: float = 1.0,
            sum_dna: float = 0.0, rng=None) -> Akce2:
        """`sum_dna` vpustí explorační šum PŘÍMO do neuronů DNa02.

        Šum přičtený až k výsledné akci (první verze, 13. 9. 2026) nefunguje:
        váhy se na takovém vybočení nijak nepodílely, takže korelovat je
        s odměnou je odhad gradientu, který žádný gradient neodhaduje.
        Node perturbation vyžaduje šum v NEURONU, ne za ním.
        """
        ext = self.vjem(snimek, predchozi) * zesileni
        if sum_dna and rng is not None:
            self._posledni_sum = {}
            for strana, ix in (("vlevo", self.dna_l), ("vpravo", self.dna_r)):
                e = float(rng.normal(0, sum_dna))
                ext[ix] += e
                self._posledni_sum[strana] = e
        sim = Simulator2(self.W_spike, weight_scale=WS)
        res = sim.run(self.steps, external=ext)
        sc = res.spike_counts.numpy(); dur = res.duration_s
        f = lambda ix: float(sc[ix].sum() / dur) if len(ix) else 0.0
        return Akce2(f(self.dna_l), f(self.dna_r), f(self.dnp09), f(self.mdn), sc)

    # --- učení ---
    def reset_vahy(self) -> None:
        self.W_spike = self.W0.copy()

    def _vstupy(self, rows: np.ndarray):
        pos, src = [], []
        for r in rows:
            lo, hi = self.W_spike.indptr[r], self.W_spike.indptr[r + 1]
            pos.append(np.arange(lo, hi)); src.append(self.W_spike.indices[lo:hi])
        return (np.concatenate(pos), np.concatenate(src)) if len(pos) else (np.array([],int), np.array([],int))

    def uc_rozhrani(self, stopy: dict[str, np.ndarray], delta: float, sila: float = 0.25) -> None:
        """Učí VSTUPY DNa02 z graduované vrstvy — dvanáct synapsí celkem.

        Předchozí pokusy učily 60 tisíc neuronů spikující části a utopily se
        v šumu (exp09–exp11: paired k nerozeznání od placeba na 12 losech).
        Tady se mění jen to, co skutečně vede od zraku k zatáčení: 7 synapsí
        do levého DNa02 a 5 do pravého. Zůstává to omezené connectomem —
        mění se síla existujících spojů, žádný nový nevzniká.
        """
        if not hasattr(self, "_W_g2s_uc"):
            self._W_g2s_uc = self.W_g2s.tolil()
        for strana, ix in (("vlevo", self.dna_l), ("vpravo", self.dna_r)):
            for r in ix:
                lo, hi = self.W_g2s.indptr[r], self.W_g2s.indptr[r + 1]
                if lo == hi:
                    continue
                src = self.W_g2s.indices[lo:hi]
                e = stopy[strana][src]
                mx = np.abs(e).max()
                if mx > 0:
                    e = e / mx
                self.W_g2s.data[lo:hi] *= (1.0 + sila * delta * e).astype(np.float32)

    def reset_rozhrani(self) -> None:
        self.W_g2s = self._W_g2s0.copy()

    def odmena_perturbaci(self, stopy: dict[str, np.ndarray], delta: float,
                          sila: float = 0.08) -> None:
        """Node perturbation: každá strana má vlastní stopu z VLASTNÍHO šumu.

        Stopa strany = předsynaptická aktivita × šum vpuštěný do TÉ strany.
        Kladná odchylka odměny posílí to, co doprovázelo kladné vybočení.
        """
        for strana, ix in (("vlevo", self.dna_l), ("vpravo", self.dna_r)):
            pos, src = self._vstupy(ix)
            if not len(pos):
                continue
            e = stopy[strana][src]
            mx = np.abs(e).max()
            if mx > 0:
                e = e / mx
            self.W_spike.data[pos] *= (1.0 + sila * delta * e).astype(np.float32)
        np.clip(self.W_spike.data, -self._wmax, self._wmax, out=self.W_spike.data)

    def odmena_gradient(self, stopa: np.ndarray, delta: float, sila: float = 0.05) -> None:
        """Učení podle literatury: eligibility trace × chyba predikce odměny.

        Tři věci, které první dvě verze neměly a bez kterých to podle
        literatury nemůže fungovat (ověřeno záporným během 13. 9. 2026):
          1. STOPA S ROZPADEM místo prostého součtu aktivity
             (Izhikevich 2007, distal reward problem).
          2. ODMĚNA JAKO CHYBA PREDIKCE, tedy r − klouzavý průměr, ne r
             (Frémaux & Gerstner 2016: bez odečtení základní hladiny to
             nekonverguje).
          3. EXPLORAČNÍ ŠUM v akci a posilování ODCHYLKY od střední akce
             (node perturbation / REINFORCE). Deterministická síť nemá
             co vybírat — tohle byla hlavní chyba.

        `stopa` je už hotová: součet přes kroky z (šum_akce × předsynaptická
        aktivita), s rozpadem. Aplikuje se protisměrně na obě DNa02.
        """
        for strana, ix, znam in (("vlevo", self.dna_l, +1.0), ("vpravo", self.dna_r, -1.0)):
            pos, src = self._vstupy(ix)
            if not len(pos):
                continue
            e = stopa[src]
            mx = np.abs(e).max()
            if mx > 0:
                e = e / mx
            self.W_spike.data[pos] *= (1.0 + sila * delta * znam * e).astype(np.float32)
        np.clip(self.W_spike.data, -self._wmax, self._wmax, out=self.W_spike.data)

    def odmena_stopou(self, stopa: dict[str, np.ndarray], odmena: float, sila: float = 0.04) -> None:
        """Odložená odměna, třífaktorové pravidlo přes eligibility trace.

        Posílí se to, co bylo aktivní, když se volila DANÁ strana, a to úměrně
        odměně (kladná i záporná). Každá strana má vlastní stopu, takže
        pravidlo umí vyjádřit PODMÍNĚNOU politiku „doleva, když se cíl hnul
        doleva" — na rozdíl od první verze, která posilovala jednu stranu
        globálně za celý blok a podmíněnost vyjádřit neuměla (záporný běh
        13. 9. 2026: paired −3,3, placebo −1,7 p. b., tedy šum).
        """
        for strana, ix in (("vlevo", self.dna_l), ("vpravo", self.dna_r)):
            pos, src = self._vstupy(ix)
            if not len(pos):
                continue
            e = stopa[strana][src]
            mx = e.max()
            if mx > 0:
                e = e / mx
            self.W_spike.data[pos] *= (1.0 + sila * odmena * e).astype(np.float32)
