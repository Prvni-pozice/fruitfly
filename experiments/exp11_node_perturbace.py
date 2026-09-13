"""
exp11_node_perturbace.py — šum do neuronu, ne za něj.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Exp09 a exp10 ukázaly, že pravidlo se nenaučí a že odklad za to nemůže
(měřidlo přitom drží: sham 0,0 p. b. v šestnácti bězích). Diagnóza:
explorační šum se přičítal k AKCI, tedy až za mozkem, takže se na něm
váhy nijak nepodílely a korelace stopy s odměnou nic neodhadovala.

Oprava podle metody: šum jde PŘÍMO do neuronů DNa02 (node perturbation),
stopa = předsynaptická aktivita × velikost šumu na té straně, odměna jako
chyba predikce. Jinak vše stejné jako exp09, aby byla čísla srovnatelná
(táž pevná zkušební sada, tytéž losy).

  N1: sham se nehne o víc než 2 p. b. (kontrola měřidla).
  N2: paired se zlepší aspoň o 8 p. b. ve ≥ 3 ze 4 losů.
  N3: medián paired − placebo > 5 p. b.
Vyvrácení: když N2 a N3 padnou při platném N1, je vyčerpaná i opravená
verze pravidla a učení se přesune do graduované vrstvy (bod 1 vidličky
v PLAN.md). Žádné další ladění parametrů tohohle pravidla.

POZOR: první krok po restartu epizody se přeskakuje. Scéna se ještě
nehýbe, agent je detektor pohybu a vydá přesnou nulu — u exp10 to udělalo
celou příčku žebříku nečitelnou.
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


def blok(a, h, delka, rezim, rng, sila, sigma, baseline):
    stopy = {"vlevo": np.zeros(len(a.i_spike), np.float32),
             "vpravo": np.zeros(len(a.i_spike), np.float32)}
    ok = 0
    h.krok(0.0)                      # rozhýbat scénu, viz hlavička
    pred = h.snimek()
    for _ in range(delka):
        s = h.snimek()
        k = a.act(s, pred, sum_dna=sigma, rng=rng); pred = s
        sp = k.spiky.astype(np.float32); mx = sp.max()
        spn = sp / mx if mx > 0 else sp
        for strana in ("vlevo", "vpravo"):
            stopy[strana] = ROZPAD * stopy[strana] + a._posledni_sum[strana] * spn
        h.krok(k.zataceni)
        ok += int(h.na_cili())
    r = 2.0 * (ok / delka) - 1.0
    if rezim != "sham":
        delta = float(rng.normal(0, 0.3)) if rezim == "placebo" else r - baseline["v"]
        a.odmena_perturbaci(stopy, delta, sila=sila)
    baseline["v"] = 0.8 * baseline["v"] + 0.2 * r


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--bloku", type=int, default=20)
    p.add_argument("--delka", type=int, default=10)
    p.add_argument("--zk-kroku", type=int, default=20)
    p.add_argument("--sila", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.6)
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
                blok(a, h, args.delka, rezim, rng, args.sila, args.sigma, baseline)
            po_ = zkouska(a, args.zk_kroku)
            log.info("seed %d %-8s: %.3f -> %.3f (%+.1f p. b.)",
                     seed, rezim, pred_, po_, 100 * (po_ - pred_))
            vysl.append({"seed": seed, "rezim": rezim, "pred": pred_, "po": po_,
                         "zmena": po_ - pred_})

    def zm(r): return np.array([v["zmena"] for v in vysl if v["rezim"] == r])
    log.info("=" * 70)
    for r in ("sham", "paired", "placebo"):
        log.info("  %-8s: %s medián %+.1f p. b.", r, np.round(100 * zm(r), 1), 100 * np.median(zm(r)))
    n1 = bool(np.abs(zm("sham")).max() <= 0.02)
    n2 = int((zm("paired") >= 0.08).sum()) >= 3
    n3 = bool((np.median(zm("paired")) - np.median(zm("placebo"))) > 0.05)
    log.info("  N1 měřidlo drží                      : %s", "SPLNĚNO" if n1 else "NESPLNĚNO")
    log.info("  N2 paired roste ≥8 p. b. ve ≥3 losech: %s (%d/4)",
             "SPLNĚNO" if n2 else "NESPLNĚNO", int((zm("paired") >= 0.08).sum()))
    log.info("  N3 paired − placebo > 5 p. b.        : %s (%+.1f)",
             "SPLNĚNO" if n3 else "NESPLNĚNO", 100 * (np.median(zm("paired")) - np.median(zm("placebo"))))
    if not n1:
        verdikt = "BĚH NEPLATNÝ: měřidlo se hnulo"
    elif n2 and n3:
        verdikt = "Učí se — šum musel do neuronu, ne za něj"
    else:
        verdikt = "ZÁPORNÝ: ani opravené pravidlo se nenaučí -> učení do graduované vrstvy"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp11.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
