"""
rate_vrstva.py — graduovaná (nespikující) vrstva pro zrakovou dráhu.

Nález 13. 9. 2026: CELÁ zraková dráha mouchy je graduovaná — fotoreceptory,
lamina, medulla (T4/T5) i tangenciální buňky lobula plate (HS/VS odpovídají
graduovaným posunem napětí). Spiky začínají až u sestupných neuronů.
Spikující LIF proto zrakovou informaci neunese na ŽÁDNÉ úrovni; není to
otázka parametru.

Architektura, která z toho plyne:
    video -> flyvis (graduovaně, natrénované)        -> T4/T5 po sloupcích
          -> TATO VRSTVA (graduovaně, váhy z FlyWire) -> HS/VS/LC
          -> spikující model (sim2)                   -> DN -> akce

Tady je ta prostřední část: ustálený stav lineárně-prahové sítě nad
skutečnými vahami FlyWire.

    x <- relu(W_vis @ x * g + vstup)      (iterováno do ustálení)

`g` je jediný volný parametr (zisk); ukotvuje se tak, aby se odpovědi HS
pohybovaly v rozsahu pozorovaném u mouchy (jednotky mV graduované odpovědi),
ne podle našeho výsledku.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class RateVrstva:
    def __init__(self, W: sp.csr_matrix, idx_vis: np.ndarray, zisk: float = 0.5) -> None:
        self.idx = np.asarray(idx_vis)
        A = W[self.idx][:, self.idx].astype(np.float32).tocsr()
        # normalizace na součet vstupů: bez ní se lineárně-prahová síť rozkmitá
        # (naměřeno 13. 9. 2026: hodnoty 1e28 při zisku 0,02 na surových vahách)
        soucet = np.abs(A).sum(1).A.ravel()
        soucet[soucet == 0] = 1.0
        self.W = sp.diags(1.0 / soucet) @ A
        self.zisk = zisk  # < 1 je stabilní, spektrální poloměr normované matice ≤ 1
        self._poz = {int(j): i for i, j in enumerate(self.idx)}

    def ustaleny_stav(self, vstup: np.ndarray, kroku: int = 60) -> np.ndarray:
        """Iteruj x <- relu(zisk * W x + vstup) do ustálení."""
        x = np.maximum(vstup, 0).astype(np.float32)
        for _ in range(kroku):
            nove = np.maximum(self.zisk * (self.W @ x) + vstup, 0.0).astype(np.float32)
            if np.allclose(nove, x, atol=1e-5):
                return nove
            x = nove
        return x

    def poz(self, globalni_idx) -> np.ndarray:
        """Přelož globální indexy neuronů na pozice uvnitř téhle vrstvy."""
        return np.array([self._poz[int(j)] for j in globalni_idx if int(j) in self._poz])
