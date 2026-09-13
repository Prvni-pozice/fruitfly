"""
exp12_rozhrani.py — učit dvanáct synapsí místo šedesáti tisíc neuronů.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Exp09 až exp11 učily váhy uvnitř spikující části (60 117 neuronů) a na
dvanácti losech vyšlo paired k nerozeznání od placeba (medián 3,8 proti
4,1; párově 7/12, tedy hod mincí). Měřidlo přitom drželo na přesné nule
ve všech 36 kontrolních bězích, takže problém není v měření.

Nový cíl učení: rozhraní graduované vrstvy na DNa02. Je to celkem
**12 synapsí** (7 do levého, 5 do pravého) a je to přesně ta cesta, kterou
zrak ovládá zatáčení. Zůstává to omezené connectomem — mění se síla
existujících spojů, žádný nový nevzniká.

  R1: měřidlo drží, sham se nehne o víc než 2 p. b.
  R2: medián paired − placebo > 5 p. b. přes VŠECH 12 losů.
  R3: paired > placebo párově aspoň v 9 z 12 losů.
Vyvrácení: když R2 a R3 padnou, neučí se ani dvanáct parametrů a odměnové
učení v téhle podobě končí; hra se postaví na vrozeném chování a učení
se povede jako samostatná větev (bod 3 vidličky v PLAN.md).

Rovnou 12 losů. Čtyři nestačí — v exp11 vypadaly první čtyři na +8,4
proti placebu +2,9 a po dalších osmi z toho bylo +2,5 proti +4,7.
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
    n = len(a.i_graded)
    stopy = {"vlevo": np.zeros(n, np.float32), "vpravo": np.zeros(n, np.float32)}
    ok = 0
    h.krok(0.0)                      # rozhýbat scénu (statická dá nulu)
    pred = h.snimek()
    for _ in range(delka):
        s = h.snimek()
        k = a.act(s, pred, sum_dna=sigma, rng=rng); pred = s
        stav = a._posledni_stav
        mx = stav.max()
        stavn = stav / mx if mx > 0 else stav
        for strana in ("vlevo", "vpravo"):
            stopy[strana] = ROZPAD * stopy[strana] + a._posledni_sum[strana] * stavn
        h.krok(k.zataceni)
        ok += int(h.na_cili())
    r = 2.0 * (ok / delka) - 1.0
    if rezim != "sham":
        delta = float(rng.normal(0, 0.3)) if rezim == "placebo" else r - baseline["v"]
        a.uc_rozhrani(stopy, delta, sila=sila)
    baseline["v"] = 0.8 * baseline["v"] + 0.2 * r


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--bloku", type=int, default=20)
    p.add_argument("--delka", type=int, default=10)
    p.add_argument("--zk-kroku", type=int, default=20)
    p.add_argument("--sila", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    vysl = []
    for seed in args.seeds:
        for rezim in ("sham", "paired", "placebo"):
            a.reset_vahy(); a.reset_rozhrani()
            rng = np.random.default_rng(seed)
            pred_ = zkouska(a, args.zk_kroku)
            h = Hra(seed); baseline = {"v": 0.0}
            for _ in range(args.bloku):
                h.reset()
                blok(a, h, args.delka, rezim, rng, args.sila, args.sigma, baseline)
            po_ = zkouska(a, args.zk_kroku)
            log.info("seed %2d %-8s: %.3f -> %.3f (%+.1f p. b.)",
                     seed, rezim, pred_, po_, 100 * (po_ - pred_))
            vysl.append({"seed": seed, "rezim": rezim, "pred": pred_, "po": po_,
                         "zmena": po_ - pred_})

    def zm(r): return np.array([v["zmena"] for v in vysl if v["rezim"] == r])
    log.info("=" * 70)
    for r in ("sham", "paired", "placebo"):
        log.info("  %-8s: medián %+.1f p. b.  %s", r, 100 * np.median(zm(r)), np.round(100 * zm(r), 1))
    p_, q_ = zm("paired"), zm("placebo")
    r1 = bool(np.abs(zm("sham")).max() <= 0.02)
    r2 = bool((np.median(p_) - np.median(q_)) > 0.05)
    r3 = int((p_ > q_).sum()) >= 9
    log.info("  R1 měřidlo drží                 : %s", "SPLNĚNO" if r1 else "NESPLNĚNO")
    log.info("  R2 medián paired − placebo > 5  : %s (%+.1f p. b.)",
             "SPLNĚNO" if r2 else "NESPLNĚNO", 100 * (np.median(p_) - np.median(q_)))
    log.info("  R3 párově paired > placebo ≥9/12: %s (%d/12)",
             "SPLNĚNO" if r3 else "NESPLNĚNO", int((p_ > q_).sum()))
    if not r1:
        verdikt = "BĚH NEPLATNÝ: měřidlo se hnulo"
    elif r2 and r3:
        verdikt = "Učí se — dvanáct synapsí na rozhraní stačí"
    else:
        verdikt = "ZÁPORNÝ: neučí se ani 12 parametrů -> hra na vrozeném chování"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp12.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
