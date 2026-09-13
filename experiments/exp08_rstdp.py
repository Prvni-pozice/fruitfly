"""
exp08_rstdp.py — učení s odloženou odměnou podle literatury.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Dva předchozí záporné běhy (exp06) používaly vlastní pravidlo bez tří
prvků, které literatura označuje za nutné. Tahle verze je má:
  1. eligibility trace s rozpadem (Izhikevich 2007),
  2. odměna jako chyba predikce proti klouzavému průměru (Frémaux &
     Gerstner 2016),
  3. explorační šum v akci + posilování odchylky od střední akce
     (node perturbation / REINFORCE).

  R1: `paired` se zlepší mezi prvními a posledními pěti bloky aspoň
      o 8 p. b. v ≥ 3 ze 4 losů.
  R2: medián přírůstku `paired` > medián `placebo` (odměna přeházená mezi
      bloky, tedy stejné rozdělení, žádná vazba na výkon) aspoň o 5 p. b.
  R3: `sham` (bez učení) se nezlepší.
Vyvrácení: když se `placebo` zlepší stejně, dělá to drift vah, ne odměna.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
ROZPAD = 0.8      # rozpad eligibility trace mezi kroky


def blok(a, h, delka, rezim, rng, sila, sigma, baseline):
    stopa = np.zeros(len(a.i_spike), np.float32)
    ok = 0
    pred = h.snimek()
    for _ in range(delka):
        s = h.snimek()
        k = a.act(s, pred); pred = s
        sum_ = k.zataceni
        sum_noise = float(rng.normal(0, sigma))       # explorace
        akce = float(np.clip(sum_ + sum_noise, -1, 1))
        sp = k.spiky.astype(np.float32)
        mx = sp.max()
        stopa = ROZPAD * stopa + (sum_noise * (sp / mx if mx > 0 else sp))
        h.krok(akce)
        ok += int(h.na_cili())
    r = 2.0 * (ok / delka) - 1.0
    if rezim != "sham":
        delta = r - baseline["v"]                      # chyba predikce odměny
        if rezim == "placebo":
            delta = float(rng.normal(0, 0.3))          # stejný rozsah, bez vazby
        a.odmena_gradient(stopa, delta, sila=sila)
    baseline["v"] = 0.8 * baseline["v"] + 0.2 * r
    return ok / delka


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--bloku", type=int, default=24)
    p.add_argument("--delka", type=int, default=10)
    p.add_argument("--sila", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    vysl = []
    for seed in args.seeds:
        for rezim in ("sham", "paired", "placebo"):
            a.reset_vahy()
            rng = np.random.default_rng(seed)
            h = Hra(seed); baseline = {"v": 0.0}; krivka = []
            for _ in range(args.bloku):
                h.reset()
                krivka.append(blok(a, h, args.delka, rezim, rng, args.sila, args.sigma, baseline))
            zac, kon = float(np.mean(krivka[:5])), float(np.mean(krivka[-5:]))
            log.info("seed %d %-8s: %.2f -> %.2f (%+.1f p. b.)", seed, rezim, zac, kon, 100 * (kon - zac))
            vysl.append({"seed": seed, "rezim": rezim, "krivka": krivka,
                         "zacatek": zac, "konec": kon, "prirustek": kon - zac})

    def pr(r): return np.array([v["prirustek"] for v in vysl if v["rezim"] == r])
    log.info("=" * 70)
    for r in ("sham", "paired", "placebo"):
        log.info("  %-8s: %s  medián %+.1f p. b.", r, np.round(100 * pr(r), 1), 100 * np.median(pr(r)))
    r1 = int((pr("paired") >= 0.08).sum()) >= max(3, int(0.75 * len(args.seeds)))
    r2 = (np.median(pr("paired")) - np.median(pr("placebo"))) > 0.05
    r3 = np.median(pr("sham")) < 0.05
    log.info("  R1 paired roste ≥8 p. b. ve ≥3 losech : %s (%d/%d)",
             "SPLNĚNO" if r1 else "NESPLNĚNO", int((pr("paired") >= 0.08).sum()), len(args.seeds))
    log.info("  R2 paired − placebo > 5 p. b.          : %s (%+.1f)",
             "SPLNĚNO" if r2 else "NESPLNĚNO", 100 * (np.median(pr("paired")) - np.median(pr("placebo"))))
    log.info("  R3 sham se nezlepší                    : %s", "SPLNĚNO" if r3 else "NESPLNĚNO")
    verdikt = "Učí se s odloženou odměnou" if (r1 and r2 and r3) else "ZÁPORNÝ/ČÁSTEČNÝ — viz kritéria"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp08_rstdp.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
