"""
pravidlo_shiu.py — znaménka spojů podle Shiu et al. 2024.

Upstream `flybrain` dává modulačním přenašečům (DA, OCT, SER) váhu NULA.
Referenční model Shiu et al. 2024 je počítá jako BUDIVÉ (+1), stejně tak
projekt femaleflybrain (pravidlo `shiu2024`). Naše ukotvení `weight_scale`
= 0,0393 vychází z parametrů Shiu, takže je konzistentní používat i jejich
pravidlo znamének — jinak ukotvujeme na model, který simulujeme jinak.

Důsledek nuly, změřený 17. 9. 2026: 2 332 neuronů a přes milion synapsí
mlčelo, mezi nimi celá odměnová soustava. Cukr proto nedosáhl k PAM.

Použití: `pouzij_shiu(c)` PŘED `Agent2(c)` — agent si znaménka bere
z `c.nodes["nt_sign"]` při stavbě.
"""
from __future__ import annotations
import numpy as np

MONOAMINY = ("DA", "OCT", "SER")


def pouzij_shiu(c) -> dict[str, int]:
    """Nastaví monoaminům znaménko +1. Vrací, kolik neuronů se změnilo."""
    nt = c.nodes["nt_type"].astype("string")
    zmeny = {}
    for t in MONOAMINY:
        m = (nt == t).fillna(False).to_numpy()
        c.nodes.loc[m, "nt_sign"] = 1
        zmeny[t] = int(m.sum())
    return zmeny
