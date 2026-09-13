"""
hra.py — minimální prostředí pro octomilku: udrž cíl ve středu pohledu.

Svět je válec: moucha má směr pohledu `uhel`, ve světě je jeden tmavý sloup
na azimutu `cil`. Scéna se vykresluje do snímku 32×32, který dostane agent.
Akce (zatáčení z DNa02) mění úhel pohledu. Odměna je ODLOŽENÁ — přijde až
na konci bloku a hodnotí, jak dlouho byl cíl blízko středu.

Prostředí je naše, ne biologické. Mozek mezi vjemem a akcí je z connectomu.
"""
from __future__ import annotations

import numpy as np

VELIKOST = 32
ZORNE_POLE = 120.0  # stupňů


class Hra:
    def __init__(self, seed: int = 0, rychlost: float = 18.0, sirka: float = 14.0,
                 drift: float = 9.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.rychlost, self.sirka, self.drift = rychlost, sirka, drift
        self.reset()

    def reset(self) -> np.ndarray:
        self.uhel = 0.0
        self.cil = float(self.rng.uniform(-40, 40))
        # cíl se pohybuje: agent je detektor POHYBU a na statické scéně vydá
        # přesnou nulu (T4/T5 reagují jen na pohyb) -> úloha by se zacyklila
        self.smer = float(self.rng.choice([-1.0, 1.0]))
        self.kroky = 0
        return self.snimek()

    @property
    def odchylka(self) -> float:
        """Kolik stupňů je cíl od středu pohledu."""
        return (self.cil - self.uhel + 180) % 360 - 180

    def snimek(self) -> np.ndarray:
        """Tmavý sloup na světlém poli; mimo zorné pole není vidět."""
        im = np.ones((VELIKOST, VELIKOST), dtype=np.float32)
        d = self.odchylka
        if abs(d) <= ZORNE_POLE / 2:
            stred = (d / ZORNE_POLE + 0.5) * (VELIKOST - 1)
            p = self.sirka / ZORNE_POLE * VELIKOST / 2
            lo, hi = int(np.clip(stred - p, 0, VELIKOST)), int(np.clip(stred + p, 0, VELIKOST))
            im[:, lo:hi] = 0.0
        return im

    def krok(self, zataceni: float) -> tuple[np.ndarray, bool]:
        """Zataceni > 0 otáčí doleva (za cílem vlevo). Vrací snímek a konec."""
        self.uhel -= float(np.clip(zataceni, -1, 1)) * self.rychlost
        self.cil += self.smer * self.drift
        if abs(self.cil) > 70:                    # odraz od okraje světa
            self.smer *= -1.0
            self.cil = float(np.clip(self.cil, -70, 70))
        self.kroky += 1
        return self.snimek(), self.kroky >= 999

    def na_cili(self, prah: float = 20.0) -> bool:
        return abs(self.odchylka) < prah
