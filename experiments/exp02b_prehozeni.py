"""
exp02b_prehozeni.py — je to učení kontingence, nebo jen drift jedním směrem?

PREREGISTRACE (zapsáno před během, 12. 9. 2026):
Exp02 ukázal, že smyčka se naučí zatáčet podle vjemu (20/20 losů, chování
se překlopí v 16/20). To ale ještě neodlišuje učení od driftu do pevného
stavu: pokud pravidlo prostě táhne k „levý vjem → levá akce", vyjde totéž.

Test: natrénovat normálně, pak OTOČIT pravidlo odměny (vlevo→vpravo,
vpravo→vlevo) a trénovat dál. Skutečné učení kontingence se přeučí.

Predikce s prahy:
  R1: po přehození klesne S pod hodnotu po prvním tréninku v ≥ 8 z 10 losů.
  R2: medián S po přehození je ZÁPORNÝ (chování se otočí, ne jen zeslábne).
  R3: kontrolní rameno `pokracuj` (dál stejné pravidlo) S nesníží —
      medián změny ve druhé fázi ≥ 0.
Vyvrácení: když `pokracuj` klesne stejně jako `prehozeni`, je pokles únavou
pravidla (saturace vah), ne přeučením.

Použití:
    python experiments/exp02b_prehozeni.py --seeds 0 1 2 ... --trials 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain"))
sys.path.insert(0, str(ROOT))

from flyagent import FlyAgent  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OUT = ROOT / "outputs"
NORMAL = {"vlevo": "vlevo", "vpravo": "vpravo"}
OTOCENE = {"vlevo": "vpravo", "vpravo": "vlevo"}


def rozevreni(a: FlyAgent) -> float:
    return a.act(a.vjem("vlevo")).bilance - a.act(a.vjem("vpravo")).bilance


def faze(agent: FlyAgent, mapa: dict, trials: int, rng, sila: float) -> None:
    for _ in range(trials):
        for vjem in ("vlevo", "vpravo"):
            plny = agent.vjem(vjem)
            stimul = rng.choice(plny, size=len(plny) // 2, replace=False)
            agent.odmena(agent.act(stimul), mapa[vjem], sila=sila)


def boot_ci(v, stat=np.median, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--sila", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    agent = FlyAgent(c, steps=args.steps)

    res = []
    for seed in args.seeds:
        for arm, mapa2 in (("prehozeni", OTOCENE), ("pokracuj", NORMAL)):
            agent.reset_vahy()
            rng = np.random.default_rng(seed)
            s0 = rozevreni(agent)
            faze(agent, NORMAL, args.trials, rng, args.sila)
            s1 = rozevreni(agent)
            faze(agent, mapa2, args.trials, rng, args.sila)
            s2 = rozevreni(agent)
            logger.info("seed %2d %-10s  S: %+.4f -> %+.4f -> %+.4f  (2. fáze %+.4f)",
                        seed, arm, s0, s1, s2, s2 - s1)
            res.append({"seed": seed, "arm": arm, "S0": s0, "S1": s1, "S2": s2, "d2": s2 - s1})

    pre = np.array([r["d2"] for r in res if r["arm"] == "prehozeni"])
    pok = np.array([r["d2"] for r in res if r["arm"] == "pokracuj"])
    s2_pre = np.array([r["S2"] for r in res if r["arm"] == "prehozeni"])

    logger.info("=" * 74)
    logger.info("ZMĚNA VE 2. FÁZI (ΔS)")
    logger.info("  prehozeni: medián %+.4f, klesá v %d/%d", np.median(pre), int((pre < 0).sum()), len(pre))
    logger.info("  pokracuj : medián %+.4f, klesá v %d/%d", np.median(pok), int((pok < 0).sum()), len(pok))
    lo, hi = boot_ci(pre - pok)
    logger.info("  rozdíl (prehozeni − pokracuj): medián %+.4f, CI [%+.4f, %+.4f]",
                np.median(pre - pok), lo, hi)

    r1 = (pre < 0).sum() >= int(0.8 * len(pre))
    r2 = np.median(s2_pre) < 0
    r3 = np.median(pok) >= 0
    logger.info("-" * 74)
    logger.info("  R1 pokles po přehození (≥80 %% losů) : %s (%d/%d)",
                "SPLNĚNO" if r1 else "NESPLNĚNO", int((pre < 0).sum()), len(pre))
    logger.info("  R2 medián S po přehození záporný    : %s (%+.4f)",
                "SPLNĚNO" if r2 else "NESPLNĚNO", np.median(s2_pre))
    logger.info("  R3 kontrola `pokracuj` neklesá      : %s (%+.4f)",
                "SPLNĚNO" if r3 else "NESPLNĚNO", np.median(pok))

    if r1 and r3 and not (lo <= 0 <= hi):
        verdikt = "Přeučí se — je to učení kontingence, ne drift" + ("" if r2 else " (ale znaménko se neotočí)")
    elif np.median(pok) < 0 and r1:
        verdikt = "ZÁPORNÝ NÁLEZ: klesá i kontrola = saturace vah, ne přeučení"
    else:
        verdikt = "NEROZHODNUTO"
    logger.info("VERDIKT: %s", verdikt)

    OUT.mkdir(exist_ok=True)
    (OUT / "exp02b_prehozeni.json").write_text(
        json.dumps({"verdikt": verdikt, "results": res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
