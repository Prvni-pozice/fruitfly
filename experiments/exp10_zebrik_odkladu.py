"""
exp10_zebrik_odkladu.py — kde přesně pravidlo praskne?

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Exp09 měl platné měřidlo (sham 0,0 p. b. ve všech losech) a přesto se
pravidlo s odloženou odměnou nenaučilo. Otázka teď zní, jestli je vinen
ODKLAD, nebo úloha a pravidlo samo.

Žebřík: odměna přichází po 1, 2, 5 a 10 krocích. Když to funguje při 1 a
praskne u 5, je problém v odkladu a řeší se delší stopou. Když to nefunguje
ani při odkladu 1, odklad s tím nemá co dělat a je potřeba změnit pravidlo
nebo odečet — a celý bod „odložená odměna" v plánu je špatně položený.

  Z1: při odkladu 1 se `paired` zlepší aspoň o 8 p. b. ve ≥ 3 ze 4 losů.
  Z2: `sham` se nehne o víc než 2 p. b. (kontrola měřidla, jako v exp09).
  Z3: zlepšení klesá s délkou odkladu monotónně.
Vyvrácení: když Z1 padne, není to problém odkladu. Reportovat to tak a
nepřidávat další parametry pravidla.

Měří se na téže pevné zkušební sadě jako exp09, aby čísla byla srovnatelná.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp09_pevna_zkouska import ROZPAD, zkouska  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def trenink(a, h, bloku, delka, rezim, rng, sila, sigma):
    """Odměna po každých `delka` krocích. delka=1 znamená bez odkladu."""
    baseline = {"v": 0.0}
    for _ in range(bloku):
        h.reset()
        stopa = np.zeros(len(a.i_spike), np.float32)
        ok = 0
        pred = h.snimek()
        for _ in range(delka):
            s = h.snimek()
            k = a.act(s, pred); pred = s
            sum_noise = float(rng.normal(0, sigma))
            akce = float(np.clip(k.zataceni + sum_noise, -1, 1))
            sp = k.spiky.astype(np.float32); mx = sp.max()
            stopa = ROZPAD * stopa + sum_noise * (sp / mx if mx > 0 else sp)
            h.krok(akce)
            ok += int(h.na_cili())
        r = 2.0 * (ok / delka) - 1.0
        if rezim != "sham":
            delta = float(rng.normal(0, 0.3)) if rezim == "placebo" else r - baseline["v"]
            a.odmena_gradient(stopa, delta, sila=sila)
        baseline["v"] = 0.8 * baseline["v"] + 0.2 * r


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--odklady", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--kroku-celkem", type=int, default=200, help="stejný počet kroků učení pro každý odklad")
    p.add_argument("--zk-kroku", type=int, default=20)
    p.add_argument("--sila", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    vysl = []
    for odklad in args.odklady:
        bloku = max(1, args.kroku_celkem // odklad)   # stejný počet kroků, jiný počet odměn
        for seed in args.seeds:
            for rezim in ("sham", "paired"):
                a.reset_vahy()
                rng = np.random.default_rng(seed)
                pred_ = zkouska(a, args.zk_kroku)
                trenink(a, Hra(seed), bloku, odklad, rezim, rng, args.sila, args.sigma)
                po_ = zkouska(a, args.zk_kroku)
                log.info("odklad %2d, seed %d, %-7s: %.3f -> %.3f (%+.1f p. b.)",
                         odklad, seed, rezim, pred_, po_, 100 * (po_ - pred_))
                vysl.append({"odklad": odklad, "seed": seed, "rezim": rezim,
                             "pred": pred_, "po": po_, "zmena": po_ - pred_})

    log.info("=" * 70)
    def zm(o, r): return np.array([v["zmena"] for v in vysl if v["odklad"] == o and v["rezim"] == r])
    mediany = {}
    for o in args.odklady:
        mediany[o] = float(np.median(zm(o, "paired")))
        log.info("  odklad %2d: paired %s medián %+.1f | sham max %.1f p. b.", o,
                 np.round(100 * zm(o, "paired"), 1), 100 * mediany[o],
                 100 * np.abs(zm(o, "sham")).max())
    z1 = int((zm(args.odklady[0], "paired") >= 0.08).sum()) >= 3
    z2 = max(float(np.abs(zm(o, "sham")).max()) for o in args.odklady) <= 0.02
    poradi = [mediany[o] for o in args.odklady]
    z3 = all(poradi[i] >= poradi[i + 1] for i in range(len(poradi) - 1))
    log.info("  Z1 bez odkladu se učí (≥3 ze 4 losů) : %s (%d/%d)",
             "SPLNĚNO" if z1 else "NESPLNĚNO",
             int((zm(args.odklady[0], "paired") >= 0.08).sum()), len(args.seeds))
    log.info("  Z2 měřidlo drží                       : %s", "SPLNĚNO" if z2 else "NESPLNĚNO")
    log.info("  Z3 zlepšení klesá s odkladem          : %s", "SPLNĚNO" if z3 else "NESPLNĚNO")
    if not z2:
        verdikt = "BĚH NEPLATNÝ: měřidlo se hnulo"
    elif z1:
        verdikt = "Vinen je ODKLAD — pravidlo bez něj funguje" + ("" if z3 else " (ale ne monotónně)")
    else:
        verdikt = "ZÁPORNÝ: odklad za to nemůže, nefunguje ani bez něj -> změnit pravidlo nebo odečet"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp10.json").write_text(
        json.dumps({"verdikt": verdikt, "mediany": mediany, "vysledky": vysl},
                   indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
