"""
exp07_optomotor.py — sleduje mozek cíl BEZ učení?

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Po záporném běhu učení s odloženou odměnou (exp06) je otázka jinde: nese
samotné ZAPOJENÍ connectomu chování, které cíl sleduje? U skutečné mouchy je
to optomotorická reakce — vrozená, neučená.

Měří se podíl kroků, kdy je cíl do 20° od středu pohledu, přes 12 epizod.

Ramena:
  skutecna  — správné přiřazení směrů k podtypům T4/T5
  smery     — přeházené přiřazení směrů (zničí pohybovou informaci,
              vše ostatní stejné)   <- hlavní kontrola
  nehybna   — agent vrací vždy 0 (moucha se netočí)  <- dolní mez

Predikce:
  O1: `skutecna` má vyšší úspěšnost než `smery` aspoň v 9 z 12 epizod.
  O2: medián rozdílu skutecna − smery > 5 p. b.
  O3: `skutecna` > `nehybna` (jinak je „sledování" jen tím, že cíl sám
      občas projde středem).
Vyvrácení: když se `skutecna` neliší od `smery`, chování nenese pohybová
informace a jde o artefakt prostředí.
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


def epizoda(a, seed, kroku, rezim, rng) -> tuple[float, list]:
    h = Hra(seed); pred = h.snimek(); ok = 0; stopa = []
    for _ in range(kroku):
        s = h.snimek()
        z = 0.0 if rezim == "nehybna" else a.act(s, pred).zataceni
        pred = s
        stopa.append((round(h.odchylka, 1), round(z, 3)))
        h.krok(z)
        ok += int(h.na_cili())
    return ok / kroku, stopa


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epizod", type=int, default=12)
    p.add_argument("--kroku", type=int, default=30)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    puvodni = {h_: dict(a.oci[h_]["sousedi"]) for h_ in a.oci}
    rng = np.random.default_rng(0)
    vysl = {}
    for rezim in ("skutecna", "smery", "nehybna"):
        if rezim == "smery":
            for h_ in a.oci:
                kl = list(puvodni[h_]); perm = rng.permutation(kl)
                a.oci[h_]["sousedi"] = {k: puvodni[h_][q] for k, q in zip(kl, perm)}
        else:
            for h_ in a.oci:
                a.oci[h_]["sousedi"] = puvodni[h_]
        u = [epizoda(a, s, args.kroku, rezim, rng)[0] for s in range(args.epizod)]
        vysl[rezim] = np.array(u)
        log.info("%-9s úspěšnost: medián %.3f  průměr %.3f  %s", rezim,
                 np.median(u), np.mean(u), np.round(u, 2))

    d = vysl["skutecna"] - vysl["smery"]
    o1 = int((d > 0).sum()) >= int(0.75 * args.epizod)
    o2 = float(np.median(d)) > 0.05
    o3 = float(np.median(vysl["skutecna"])) > float(np.median(vysl["nehybna"]))
    log.info("-" * 70)
    log.info("  O1 skutečná > přeházené směry v ≥75 %% epizod : %s (%d/%d)",
             "SPLNĚNO" if o1 else "NESPLNĚNO", int((d > 0).sum()), args.epizod)
    log.info("  O2 medián rozdílu > 5 p. b.                   : %s (%+.1f p. b.)",
             "SPLNĚNO" if o2 else "NESPLNĚNO", 100 * np.median(d))
    log.info("  O3 skutečná > nehybná                         : %s (%.3f vs %.3f)",
             "SPLNĚNO" if o3 else "NESPLNĚNO", np.median(vysl["skutecna"]), np.median(vysl["nehybna"]))
    verdikt = ("Zapojení nese optomotorické sledování" if (o1 and o2 and o3)
               else "ZÁPORNÝ/ČÁSTEČNÝ — viz kritéria")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp07_optomotor.json").write_text(json.dumps(
        {"verdikt": verdikt, **{k: v.tolist() for k, v in vysl.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
