"""
most_flyvis_export.py — 1. půlka mostu: flyvis (optický lalok) -> soubor.

Běží v .venv-flyvis. Pustí podnět skrz předtrénovaný model optického laloku
(Lappalainen et al. 2024, Nature) a uloží aktivitu T4a-d a T5a-d po sloupcích.
Druhá půlka (`most_flyvis_import.py`) to nasadí na naše neurony a pustí
centrálním mozkem.

Proč: optický lalok je z velké části nespikující, takže náš spikující LIF
jím signál neprotlačí (viz NALEZ-OPTICKY-LALOK.md). flyvis je graduovaný,
connectome-constrained a natrénovaný na optický tok — tedy validovaná
náhrada přesně té části, kterou náš model neumí.

Použití:
    .venv-flyvis/bin/python experiments/most_flyvis_export.py
"""

import logging
from pathlib import Path

import numpy as np
import torch

logging.disable(logging.INFO)
from flyvis import NetworkView  # noqa: E402
from flyvis.datasets.moving_bar import MovingBar  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "flyvis_odpovedi.npz"
TYPY = [f"T4{s}" for s in "abcd"] + [f"T5{s}" for s in "abcd"]


def main():
    net = NetworkView("flow/0000/000").init_network()
    cn = net.connectome
    typ = np.array([s.decode() if isinstance(s, bytes) else s for s in cn.nodes.type[:]])
    u, v = np.array(cn.nodes.u[:]), np.array(cn.nodes.v[:])

    ds = MovingBar(widths=[4], offsets=(-10, 11), intensities=[0, 1], speeds=[19],
                   height=9, post_pad_mode="continue", t_pre=0.5, t_post=0.5,
                   dt=1 / 200, angles=[0, 180])
    print(ds.arg_df.to_string())

    data = {}
    for i in range(len(ds)):
        with torch.no_grad():
            r = net.simulate(ds[i][None, :, None, :], dt=ds.dt)[0].numpy()
        # vrchol odpovědi každého neuronu přes čas = síla odpovědi sloupce
        vrchol = r.max(0)
        for T in TYPY:
            m = typ == T
            data[f"{i}|{T}|akt"] = vrchol[m]
            data[f"{i}|{T}|uv"] = np.c_[u[m], v[m]]
        print(f"podnět {i} hotov")

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, popis=np.array(ds.arg_df.to_string()), **data)
    print("uloženo ->", OUT)


if __name__ == "__main__":
    main()
