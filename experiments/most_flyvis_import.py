"""
most_flyvis_import.py — 2. půlka mostu: flyvis -> náš centrální mozek -> DN.

PREREGISTRACE (zapsáno před prvním během, 13. 9. 2026):
Aktivita T4/T5 z flyvis se nasadí na TYTÉŽ typy neuronů v našem connectomu
(párování po sloupcích přes hexagonální souřadnice) a pustí centrálním mozkem
se synaptickou integrací (`sim2.py`) při ukotvené ws = 0,0393.

Otázka: nese odpověď sestupných neuronů SMĚR pohybu (pruh 0° vs 180°)?

  M1: bilance DN se mezi 0° a 180° liší aspoň o 0,05 a se stejným
      znaménkem u obou intenzit podnětu (tmavý i světlý pruh).
  M2: rozdíl je větší než 95. percentil nullu z 20 přehozených párování
      sloupců (tytéž hodnoty, jiné přiřazení).
  M3: `sham` (nulový vstup) dá přesně nulovou aktivitu.
Vyvrácení: když rozdíl nepřekročí null, most nenese směr a je potřeba
buď jiný odečet (populace místo bilance), nebo jiná cílová vrstva.

Použití:
    flybrain/.venv/bin/python experiments/most_flyvis_import.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain"))
sys.path.insert(0, str(ROOT))

from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

WS = 0.0393  # ukotveno pro model se synaptickou integrací (Shiu: 0,275/7)
TYPY = [f"T4{s}" for s in "abcd"] + [f"T5{s}" for s in "abcd"]
OUT = ROOT / "outputs"


def parovani(c, hemisphere: str, data) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pro každý typ: indexy našich neuronů + index odpovídajícího sloupce flyvis."""
    ca = pd.read_csv(ROOT.parent / "flybrain" / "data" / "column_assignment.csv.gz")
    ca = ca[ca["hemisphere"] == hemisphere]
    rid2idx = {r: i for i, r in enumerate(c.root_ids)}
    out = {}
    for T in TYPY:
        d = ca[ca["type"] == T]
        idx = np.array([rid2idx[r] for r in d["root_id"] if r in rid2idx])
        pq = d[[c_ for c_ in ("p", "q")]].values.astype(float)[: len(idx)]
        uv = data[f"0|{T}|uv"]
        # obě mřížky na [0,1]^2 a spárovat nejbližším sousedem
        a = (pq - pq.min(0)) / (pq.max(0) - pq.min(0))
        b = (uv - uv.min(0)) / (uv.max(0) - uv.min(0))
        _, j = cKDTree(b).query(a)
        out[T] = (idx, j)
    return out


def globalni_max(data) -> dict[str, float]:
    """Max aktivity každého typu PŘES VŠECHNY podněty.

    Normalizovat každý podnět zvlášť je chyba: směrová selektivita T4/T5 je
    právě v amplitudě, a per-podnět normalizace ji zahodí. (Nalezeno
    13. 9. 2026 při prvním běhu mostu.)
    """
    out = {}
    for T in TYPY:
        out[T] = max(float(np.clip(data[f"{i}|{T}|akt"], 0, None).max()) for i in range(4)) or 1.0
    return out


def budic(par, data, vzorek: int, amp: float, N: int, perm=None, gmax=None) -> np.ndarray:
    """Sestav vstupní proud z flyvis odpovědí."""
    ext = np.zeros(N, dtype=np.float32)
    for T, (idx, j) in par.items():
        akt = np.clip(data[f"{vzorek}|{T}|akt"], 0, None)
        akt = akt / (gmax[T] if gmax else (akt.max() or 1.0))
        jj = perm[T] if perm is not None else j
        ext[idx] = amp * akt[jj]
    return ext


def main():
    data = np.load(OUT / "flyvis_odpovedi.npz")
    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    cir = Circuits(c)
    dn_l = cir.find("l", super_class="descending", side="left").indices
    dn_r = cir.find("r", super_class="descending", side="right").indices
    W = c.signed_weights().tocsr().astype(np.float32)
    N = W.shape[0]
    par = parovani(c, "left", data)
    logger.info("spárováno: %s", {T: len(v[0]) for T, v in par.items()})

    gmax = globalni_max(data)

    def beh(ext):
        sim = Simulator2(W, weight_scale=WS)
        rt = sim.run(400, external=ext).rates_hz().numpy()
        l, r = rt[dn_l].mean(), rt[dn_r].mean()
        return (l - r) / (l + r) if (l + r) > 0 else 0.0, l, r, float((rt > 0).mean())

    b0, *_ = beh(np.zeros(N, dtype=np.float32))
    logger.info("M3 sham: bilance %.4f", b0)

    amp = 3.0
    vysl, celkem = {}, {}
    for i in range(4):
        bal, l, r, akt = beh(budic(par, data, i, amp, N, gmax=gmax))
        vysl[i] = bal; celkem[i] = l + r
        logger.info("podnět %d: DN L %.2f Hz, P %.2f Hz, bilance %+.4f, aktivních %.2f %%",
                    i, l, r, bal, 100 * akt)

    # 0 a 1 = úhel 0°, 2 a 3 = úhel 180°
    rozdil_tmavy = vysl[0] - vysl[2]
    rozdil_svetly = vysl[1] - vysl[3]
    logger.info("-" * 70)
    logger.info("rozdíl 0° − 180°: tmavý pruh %+.4f, světlý pruh %+.4f", rozdil_tmavy, rozdil_svetly)

    rng = np.random.default_rng(0)
    null, null_c = [], []
    for k in range(20):
        perm = {T: rng.permutation(len(v[1]))[v[1] % len(v[1])] for T, v in par.items()}
        perm = {T: rng.permutation(par[T][1]) for T in par}
        a = beh(budic(par, data, 0, amp, N, perm, gmax))
        b = beh(budic(par, data, 2, amp, N, perm, gmax))
        null.append(a[0] - b[0]); null_c.append((a[1] + a[2]) - (b[1] + b[2]))
    null = np.array(null); null_c = np.array(null_c)
    prah = np.percentile(np.abs(null), 95)
    logger.info("null z 20 přehození: medián |rozdíl| %.4f, 95. percentil %.4f, max %.4f",
                np.median(np.abs(null)), prah, np.abs(null).max())

    m1 = (abs(rozdil_tmavy) >= 0.05 and abs(rozdil_svetly) >= 0.05
          and np.sign(rozdil_tmavy) == np.sign(rozdil_svetly))
    m2 = abs(rozdil_tmavy) > prah
    logger.info("  M1 rozdíl ≥ 0,05 a stejné znaménko : %s", "SPLNĚNO" if m1 else "NESPLNĚNO")
    logger.info("  M2 nad 95. percentilem nullu       : %s", "SPLNĚNO" if m2 else "NESPLNĚNO")
    logger.info("  M3 sham nulový                     : %s", "SPLNĚNO" if abs(b0) < 1e-9 else "NESPLNĚNO")
    # DRUHÝ ODEČET (doplněný po prvním běhu, hlásí se zvlášť): celková aktivita DN
    ct, cs = celkem[0] - celkem[2], celkem[1] - celkem[3]
    prah_c = np.percentile(np.abs(null_c), 95)
    logger.info("  celková aktivita DN, rozdíl 0°−180°: tmavý %+.2f Hz, světlý %+.2f Hz "
                "(null p95 %.2f) -> %s", ct, cs, prah_c,
                "NAD NULLEM" if min(abs(ct), abs(cs)) > prah_c else "v nullu")
    verdikt = ("Most nese směr pohybu" if (m1 and m2)
               else "ZÁPORNÝ: most směr nenese (viz kritéria vyvrácení)")
    logger.info("VERDIKT: %s", verdikt)
    (OUT / "most_flyvis.json").write_text(json.dumps(
        {"verdikt": verdikt, "bilance": {str(k): float(v) for k, v in vysl.items()},
         "rozdil_tmavy": float(rozdil_tmavy), "rozdil_svetly": float(rozdil_svetly),
         "null_p95": float(prah)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
