"""
retina.py — převod obrázku (snímku hry) na stimulaci mouší sítnice.

Stojí na `column_assignment.csv.gz` z FlyWire Codexu: 45 528 neuronů
optického laloku přiřazených ke SLOUPCŮM (785 vlevo, 796 vpravo) s
hexagonálními souřadnicemi p, q. Každý sloupec = jedno ommatidium =
jeden „pixel" mouchy.

Vstupní vrstva: **L2 = dráha OFF** (tma), 768 sloupců, jeden na sloupec.

POZOR, nález z 12. 9. 2026: dráha ON je v tomhle modelu NEPOUŽITELNÁ.
L1 má v predikci neurotransmiterů GLUT (499) / GABA (281), což je podle
mouší konvence INHIBIČNÍ — buzení L1 downstream umlčí (Mi1 střílí 1 neuron
ze 785). L2 je celá ACH a signál nese: L2 -> Tm1, Tm2, Tm4, L5, T1.
Obraz se proto kóduje jako TMA na světlém pozadí. `kanal="on"` zůstává
dostupný pro pokusy, ale ve výchozím stavu se nepoužívá.

Retinotopie je tedy PŘEVZATÁ z publikovaného přiřazení, ne odvozená námi.
Naše je jen mapování hexagonální mřížky na pixely obrázku (rovnoběžná
projekce, bez korekce na zakřivení oka) a volba, že jas budí ON a tma OFF.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "flybrain" / "data"


class Retina:
    def __init__(self, connectome, hemisphere: str = "left") -> None:
        ca = pd.read_csv(DATA / "column_assignment.csv.gz")
        ca = ca[ca["hemisphere"] == hemisphere]
        rid2idx = {r: i for i, r in enumerate(connectome.root_ids)}

        self.hemisphere = hemisphere
        self.vrstvy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.aktivni_kanaly = ("off",)  # ON je inhibiční, viz hlavička
        for typ, kanal in (("L1", "on"), ("L2", "off")):
            d = ca[ca["type"] == typ]
            idx, pq = [], []
            for r, p, q in zip(d["root_id"], d["p"], d["q"]):
                if r in rid2idx:
                    idx.append(rid2idx[r])
                    pq.append((p, q))
            self.vrstvy[kanal] = (np.array(idx), np.array(pq, dtype=float))
            logger.info("retina %s: kanál %s má %d sloupců", hemisphere, kanal, len(idx))

        # společná normalizace hex souřadnic na [0,1]^2 (osa q je „svislá")
        vsechny = np.vstack([pq for _, pq in self.vrstvy.values()])
        self._lo, self._hi = vsechny.min(0), vsechny.max(0)

    def _uv(self, pq: np.ndarray) -> np.ndarray:
        return (pq - self._lo) / (self._hi - self._lo)

    def stimul(self, obraz: np.ndarray, amp: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """Obrázek (2D, hodnoty 0–1) -> (indexy neuronů, amplitudy).

        Jas budí kanál ON, tma kanál OFF. Vzorkuje se nejbližším pixelem —
        mřížka sloupců je řidší než obrázek, takže interpolace nemá co přidat.
        """
        obraz = np.clip(np.asarray(obraz, dtype=float), 0.0, 1.0)
        h, w = obraz.shape
        idx_all, amp_all = [], []
        for kanal in self.aktivni_kanaly:
            idx, pq = self.vrstvy[kanal]
            uv = self._uv(pq)
            col = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
            row = np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
            jas = obraz[row, col]
            hodnota = jas if kanal == "on" else 1.0 - jas
            idx_all.append(idx)
            amp_all.append(amp * hodnota)
        return np.concatenate(idx_all), np.concatenate(amp_all)

    def souradnice(self, kanal: str = "on") -> np.ndarray:
        """Normalizované pozice sloupců v [0,1]^2 — pro kontrolu retinotopie."""
        return self._uv(self.vrstvy[kanal][1])
