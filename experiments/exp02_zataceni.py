"""
exp02_zataceni.py — naučí se uzavřená smyčka zatáčet podle vjemu?

Predikce, prahy a kritéria vyvrácení: EXP02-PREREGISTRACE.md (zapsáno PŘED
během). Skript predikce jen vyhodnotí.

Úloha: vjem vlevo → zatoč vlevo, vjem vpravo → zatoč vpravo. Akce se čte
z bilance sestupných neuronů. Trénuje se na náhodných polovinách vizuálních
neuronů, testuje na plné straně (kterou model při učení neviděl).

Použití:
    python experiments/exp02_zataceni.py --seeds 0 1 2 --trials 10
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
SPRAVNA = {"vlevo": "vlevo", "vpravo": "vpravo"}  # vjem -> správná akce


def rozevreni(agent: FlyAgent) -> float:
    """S = bilance(vjem vlevo) − bilance(vjem vpravo). Test na PLNÉ straně."""
    return agent.act(agent.vjem("vlevo")).bilance - agent.act(agent.vjem("vpravo")).bilance


def run_arm(arm: str, agent: FlyAgent, seed: int, args) -> dict:
    agent.reset_vahy()
    rng = np.random.default_rng(seed)
    s_pred = rozevreni(agent)

    for t in range(args.trials):
        for vjem in ("vlevo", "vpravo"):
            plny = agent.vjem(vjem)
            stimul = rng.choice(plny, size=len(plny) // 2, replace=False)  # "snímek"
            akce = agent.act(stimul)
            if arm == "sham":
                continue
            elif arm == "paired":
                strana = SPRAVNA[vjem]
            elif arm == "placebo":
                strana = rng.choice(["vlevo", "vpravo"])
            elif arm == "unpaired":
                strana = "vlevo"
            else:
                raise ValueError(arm)
            agent.odmena(akce, strana, sila=args.sila)

    s_po = rozevreni(agent)
    logger.info("  %-9s S: %+.4f -> %+.4f  (ΔS %+.4f)", arm, s_pred, s_po, s_po - s_pred)
    return {"arm": arm, "seed": seed, "S_pred": s_pred, "S_po": s_po, "dS": s_po - s_pred}


def boot_ci(v, stat=np.median, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def report(results, args):
    seeds = sorted({r["seed"] for r in results})
    by = {(r["seed"], r["arm"]): r for r in results}
    dS = {a: np.array([by[(s, a)]["dS"] for s in seeds]) for a in ("sham", "paired", "placebo", "unpaired")}

    logger.info("=" * 74)
    logger.info("VÝSLEDKY  (ΔS = rozevření po tréninku − před)")
    logger.info("%-10s%10s%10s%10s%14s", "rameno", "medián", "průměr", "sd", "kladných")
    for a in ("sham", "paired", "placebo", "unpaired"):
        v = dS[a]
        logger.info("%-10s%10.4f%10.4f%10.4f%10d/%d", a, np.median(v), v.mean(), v.std(ddof=1),
                    int((v > 0).sum()), len(v))

    par = dS["paired"] - dS["placebo"]
    lo, hi = boot_ci(par)
    logger.info("-" * 74)
    logger.info("PREDIKCE Z PREREGISTRACE")
    sham_ok = np.all(np.abs(dS["sham"]) < 1e-12)
    logger.info("  P3 determinismus (ΔS_sham = 0)      : %s", "SPLNĚNO" if sham_ok else "NESPLNĚNO — běh zahodit")
    n_kl = int((dS["paired"] > 0).sum())
    p1 = n_kl >= int(0.75 * len(seeds))
    logger.info("  P1 ΔS_paired > 0 (≥75 %% losů)       : %s  (%d/%d)", "SPLNĚNO" if p1 else "NESPLNĚNO", n_kl, len(seeds))
    p2 = not (lo <= 0 <= hi) and np.median(par) > 0
    logger.info("  P2 paired−placebo, CI bez nuly      : %s  (medián %+.4f, CI [%+.4f, %+.4f])",
                "SPLNĚNO" if p2 else "NESPLNĚNO", np.median(par), lo, hi)
    p4 = np.median(dS["paired"]) >= 0.05
    logger.info("  P4 medián ΔS_paired ≥ 0,05          : %s  (%+.4f)", "SPLNĚNO" if p4 else "NESPLNĚNO",
                np.median(dS["paired"]))

    if not sham_ok:
        verdikt = "BĚH NEPLATNÝ (nedeterminismus)"
    elif p1 and p2 and p4:
        verdikt = "Smyčka se naučila zatáčet podle vjemu"
    elif p1 and p2:
        verdikt = "Efekt je reálný, ale pod praktickou velikostí (P4)"
    elif np.median(dS["placebo"]) >= np.median(dS["paired"]):
        verdikt = "ZÁPORNÝ NÁLEZ: rozevření dělá změna vah, ne párování s vjemem"
    else:
        verdikt = "NEROZHODNUTO"
    logger.info("-" * 74)
    logger.info("VERDIKT: %s", verdikt)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"exp02_zataceni_sila{args.sila:g}.json"
    path.write_text(json.dumps({"verdikt": verdikt, "args": {k: str(v) for k, v in vars(args).items()},
                                "results": results}, indent=2, ensure_ascii=False))
    logger.info("Uloženo -> %s", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--sila", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    agent = FlyAgent(c, steps=args.steps)
    logger.info("DN vlevo=%d vpravo=%d | vizuál vlevo=%d vpravo=%d",
                len(agent.dn_l), len(agent.dn_r), len(agent.vis_l), len(agent.vis_r))

    results = []
    for seed in args.seeds:
        logger.info("=" * 74)
        logger.info("SEED %d", seed)
        for arm in ("sham", "paired", "placebo", "unpaired"):
            results.append(run_arm(arm, agent, seed, args))
    report(results, args)


if __name__ == "__main__":
    main()
