"""
exp09_pevna_zkouska.py — učení s odloženou odměnou, měřené na PEVNÉ sadě zkoušek.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Exp06 a exp08 selhaly na měřítku, ne nutně na učení: porovnávaly první a
poslední tréninkové bloky, jenže každý blok má jiný náhodný cíl. Rameno
`sham`, které se vůbec neučí, se tak „zlepšilo" o 9 p. b.

Oprava: zkušební sada je PEVNÁ (8 epizod s pevnými seedy), stejná pro
všechna ramena a pro měření před i po tréninku, a při jejím hraní se
NEUČÍ. Trénink běží na jiných, náhodných epizodách.

  P1: `sham` se na pevné sadě nezmění o víc než 2 p. b. (kontrola měřítka —
      bez toho je běh neplatný, protože měříme šum).
  P2: `paired` se zlepší aspoň o 8 p. b. v ≥ 3 ze 4 losů.
  P3: medián zlepšení `paired` − `placebo` > 5 p. b.
Vyvrácení: když P1 selže, měřítko pořád neměří; když P2 a P3 selžou při
platném P1, pravidlo se s odloženou odměnou nenaučí a je potřeba změnit
pravidlo, ne měření.
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
ROZPAD = 0.8
ZKOUSKA = list(range(1000, 1008))   # pevné seedy zkušebních epizod


def zkouska(a: Agent2, kroku: int) -> float:
    """Pevná sada epizod, bez učení a bez explorace. Stejná pro všechna ramena."""
    ok = n = 0
    for seed in ZKOUSKA:
        h = Hra(seed); pred = h.snimek()
        for _ in range(kroku):
            s = h.snimek()
            z = a.act(s, pred).zataceni
            pred = s
            h.krok(z)
            ok += int(h.na_cili()); n += 1
    return ok / n


def trenink_blok(a: Agent2, h: Hra, delka: int, rezim: str, rng, sila, sigma, baseline) -> None:
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
    p.add_argument("--bloku", type=int, default=20)
    p.add_argument("--delka", type=int, default=10)
    p.add_argument("--zk-kroku", type=int, default=20)
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
            pred_ = zkouska(a, args.zk_kroku)
            h = Hra(seed); baseline = {"v": 0.0}
            for _ in range(args.bloku):
                h.reset()
                trenink_blok(a, h, args.delka, rezim, rng, args.sila, args.sigma, baseline)
            po_ = zkouska(a, args.zk_kroku)
            log.info("seed %d %-8s: zkouška %.3f -> %.3f  (%+.1f p. b.)",
                     seed, rezim, pred_, po_, 100 * (po_ - pred_))
            vysl.append({"seed": seed, "rezim": rezim, "pred": pred_, "po": po_,
                         "zmena": po_ - pred_})

    def zm(r): return np.array([v["zmena"] for v in vysl if v["rezim"] == r])
    log.info("=" * 70)
    for r in ("sham", "paired", "placebo"):
        log.info("  %-8s: %s  medián %+.1f p. b.", r, np.round(100 * zm(r), 1), 100 * np.median(zm(r)))
    p1 = bool(np.abs(zm("sham")).max() <= 0.02)
    p2 = int((zm("paired") >= 0.08).sum()) >= 3
    p3 = bool((np.median(zm("paired")) - np.median(zm("placebo"))) > 0.05)
    log.info("  P1 sham se nehne (≤2 p. b.)          : %s (max %.1f p. b.)",
             "SPLNĚNO" if p1 else "NESPLNĚNO", 100 * np.abs(zm("sham")).max())
    log.info("  P2 paired roste ≥8 p. b. ve ≥3 losech: %s (%d/%d)",
             "SPLNĚNO" if p2 else "NESPLNĚNO", int((zm("paired") >= 0.08).sum()), len(args.seeds))
    log.info("  P3 paired − placebo > 5 p. b.        : %s (%+.1f)",
             "SPLNĚNO" if p3 else "NESPLNĚNO", 100 * (np.median(zm("paired")) - np.median(zm("placebo"))))
    if not p1:
        verdikt = "BĚH NEPLATNÝ: měřítko pořád měří šum"
    elif p2 and p3:
        verdikt = "Učí se i s odloženou odměnou"
    else:
        verdikt = "ZÁPORNÝ: měřítko v pořádku, ale pravidlo se nenaučí"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp09.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
