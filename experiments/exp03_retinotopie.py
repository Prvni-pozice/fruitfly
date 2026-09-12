"""
exp03_retinotopie.py — přežije poloha podnětu cestu do mozku?

PREREGISTRACE (zapsáno před během, 12. 9. 2026):
Retina (`retina.py`) promítá obrázek na 783 sloupců podle publikovaného
přiřazení FlyWire. Otázka: nese aktivita UVNITŘ mozku informaci o tom, KDE
podnět byl — a jak hluboko to vydrží?

Postup: světlá skvrna na mřížce pozic, odečet těžiště aktivity v souřadnicích
sloupců u tří vrstev různě hluboko (Mi1 = 1 synapse od L1, T4 = 2 synapse,
Tm9/LC = dál). Porovná se s pravou pozicí skvrny.

Kontrola (placebo): PŘEHÁZENÉ přiřazení sloupců na vstupu — tytéž neurony,
tytéž amplitudy, jen jiný sloupec. Když vyjde stejně, mapa nic nepřidává.

Predikce s prahy:
  W1: medián chyby dekódované polohy u Tm1 < 0,15 šířky zorného pole.
  W2: korelace pravá vs dekódovaná poloha u Tm1 ≥ 0,8 v OBOU osách.
  (Původně zapsáno na Mi1; po zjištění, že dráha ON je v modelu inhibiční,
   přepsáno na dráhu OFF PŘED novým během. Staré prahy beze změny.)
  W3: přehozená kontrola má chybu ≥ 2× větší a korelaci < 0,3.
  W4: totéž pro hlubší vrstvy se reportuje zvlášť; selhání NENÍ chyba, je to nález
      o tom, jak hluboko retinotopie vydrží.
Vyvrácení: když W3 neplatí (přehozená mapa dekóduje stejně dobře), není to
retinotopie, ale artefakt odečtu.

Použití:
    python experiments/exp03_retinotopie.py --mrizka 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain"))
sys.path.insert(0, str(ROOT))

from flybrain.loader import Connectome  # noqa: E402
from flybrain.simulator import Simulator  # noqa: E402
from retina import Retina  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
logging.getLogger("retina").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OUT = ROOT / "outputs"
WS = 0.124


def skvrna(stred_uv, velikost=32, polomer=0.18):
    """Obrázek s TMAVOU skvrnou na světlém poli.

    Tma, ne světlo: vstupní vrstva je L2 = dráha OFF (dráha ON je v modelu
    inhibiční, viz retina.py). Světlá skvrna na tmavém poli by budila celé
    pozadí — chyba nalezená 12. 9. 2026, 701 sloupců ze 768.
    """
    yy, xx = np.mgrid[0:velikost, 0:velikost] / (velikost - 1)
    yy = 1 - yy
    tma = ((xx - stred_uv[0]) ** 2 + (yy - stred_uv[1]) ** 2) < polomer ** 2
    return 1.0 - tma.astype(float)


def vrstvy_sloupcu(c, hemisphere="left"):
    """Indexy a hex souřadnice pro odečtové vrstvy různě hluboko."""
    ca = pd.read_csv(ROOT.parent / "flybrain" / "data" / "column_assignment.csv.gz")
    ca = ca[ca["hemisphere"] == hemisphere]
    rid2idx = {r: i for i, r in enumerate(c.root_ids)}
    out = {}
    for typ in ("Tm1", "Tm2", "Tm4", "L5", "T1"):
        d = ca[ca["type"] == typ]
        idx, pq = [], []
        for r, p, q in zip(d["root_id"], d["p"], d["q"]):
            if r in rid2idx:
                idx.append(rid2idx[r]); pq.append((p, q))
        if idx:
            pq = np.array(pq, float)
            out[typ] = (np.array(idx), pq)
    return out


def teziste(rates, idx, uv):
    """Těžiště aktivity ve znormalizovaných souřadnicích sloupců."""
    w = rates[idx]
    if w.sum() <= 0:
        return np.array([np.nan, np.nan])
    return (uv * w[:, None]).sum(0) / w.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mrizka", type=int, default=5, help="pozic na osu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--amp", type=float, default=3.0)
    p.add_argument("--nperm", type=int, default=20,
                   help="kolik nezávislých přehození (null rozdělení, ne jedna permutace)")
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    ret = Retina(c, "left")
    vrstvy = vrstvy_sloupcu(c)
    logger.info("odečtové vrstvy: %s", {k: len(v[0]) for k, v in vrstvy.items()})

    # normalizace souřadnic odečtových vrstev stejným rámcem jako retina
    norm = {}
    for k, (idx, pq) in vrstvy.items():
        uv = (pq - pq.min(0)) / (pq.max(0) - pq.min(0))
        norm[k] = (idx, uv)

    W = c.signed_weights().tocsr().astype(np.float32)
    pozice = [(u, v) for u in np.linspace(0.25, 0.75, args.mrizka)
              for v in np.linspace(0.25, 0.75, args.mrizka)]

    rng = np.random.default_rng(0)
    res = []
    rezimy = ["skutecna"] + [f"prehozena{i}" for i in range(args.nperm)]
    for rezim in rezimy:
        # přehozená mapa: tytéž neurony, permutované pozice sloupců
        r2 = Retina(c, "left")
        if rezim.startswith("prehozena"):
            for kanal, (idx, pq) in r2.vrstvy.items():
                r2.vrstvy[kanal] = (idx, pq[rng.permutation(len(pq))])
        for (u, v) in pozice:
            idx_s, amp_s = r2.stimul(skvrna((u, v)), amp=args.amp)
            sim = Simulator(W, weight_scale=WS)
            ext = np.zeros(W.shape[0], dtype=np.float32)
            ext[idx_s] = amp_s
            rates = sim.run(args.steps, external=ext).rates_hz().numpy()
            row = {"rezim": rezim, "u": u, "v": v}
            for k, (idx, uv) in norm.items():
                t = teziste(rates, idx, uv)
                row[f"{k}_u"], row[f"{k}_v"] = float(t[0]), float(t[1])
            res.append(row)
        logger.info("  %s hotovo (%d pozic)", rezim, len(pozice))

    df = pd.DataFrame(res)
    logger.info("=" * 74)
    logger.info("%-8s %-10s %10s %10s %10s", "vrstva", "režim", "chyba", "kor. u", "kor. v")
    souhrn = {}
    for k in norm:
        for rezim in rezimy:
            d = df[df["rezim"] == rezim].dropna(subset=[f"{k}_u", f"{k}_v"])
            if len(d) < 3:
                continue
            err = np.hypot(d[f"{k}_u"] - d["u"], d[f"{k}_v"] - d["v"])
            ku = np.corrcoef(d["u"], d[f"{k}_u"])[0, 1]
            kv = np.corrcoef(d["v"], d[f"{k}_v"])[0, 1]
            souhrn[(k, rezim)] = (float(np.median(err)), float(ku), float(kv))
            if rezim in ("skutecna", "prehozena0"):
                logger.info("%-8s %-11s %10.3f %10.3f %10.3f", k, rezim, np.median(err), ku, kv)

    logger.info("-" * 74)
    logger.info("PREDIKCE Z PREREGISTRACE")
    e_mi, ku_mi, kv_mi = souhrn.get(("Tm1", "skutecna"), (np.nan,) * 3)
    perm = [souhrn[("Tm1", r)] for r in rezimy if r.startswith("prehozena") and ("Tm1", r) in souhrn]
    e_sh = float(np.median([p_[0] for p_ in perm]))
    ku_sh = float(np.median([abs(p_[1]) for p_ in perm]))
    kv_sh = float(np.median([abs(p_[2]) for p_ in perm]))
    kmax = float(np.max([max(abs(p_[1]), abs(p_[2])) for p_ in perm]))
    logger.info("  null z %d přehození: chyba medián %.3f | |kor| medián u %.3f v %.3f, max %.3f",
                len(perm), e_sh, ku_sh, kv_sh, kmax)
    w1 = e_mi < 0.15
    w2 = min(ku_mi, kv_mi) >= 0.8
    w3 = (e_sh >= 2 * e_mi) and max(abs(ku_sh), abs(kv_sh)) < 0.3
    logger.info("  W1 chyba Tm1 < 0,15            : %s (%.3f)", "SPLNĚNO" if w1 else "NESPLNĚNO", e_mi)
    logger.info("  W2 korelace Tm1 ≥ 0,8 obě osy  : %s (u %.3f, v %.3f)",
                "SPLNĚNO" if w2 else "NESPLNĚNO", ku_mi, kv_mi)
    logger.info("  W3 přehozená kontrola selže    : %s (chyba %.3f, kor. u %.3f, v %.3f)",
                "SPLNĚNO" if w3 else "NESPLNĚNO", e_sh, ku_sh, kv_sh)
    for k in ("Tm2", "Tm4", "L5", "T1"):
        if (k, "skutecna") in souhrn:
            e, ku, kv = souhrn[(k, "skutecna")]
            logger.info("  W4 hloubka %-5s             : chyba %.3f, kor. u %.3f, v %.3f", k, e, ku, kv)

    verdikt = ("Retinotopie se do mozku přenáší" if (w1 and w2 and w3)
               else "ZÁPORNÝ/NEROZHODNUTO — viz kritéria")
    logger.info("VERDIKT: %s", verdikt)
    OUT.mkdir(exist_ok=True)
    (OUT / "exp03_retinotopie.json").write_text(
        json.dumps({"verdikt": verdikt, "souhrn": {f"{a}|{b}": v for (a, b), v in souhrn.items()},
                    "data": res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
