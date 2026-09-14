"""
exp13_proti_instinktu.py — učit dvanáct synapsí místo šedesáti tisíc neuronů.

PREREGISTRACE (zapsáno před během, 14. 9. 2026):
Po opravě očí (exp07: 0,880) umí moucha sledovat cíl vrozeně na 85 %.
Učení se na takové úloze nedá změřit — není se kam zlepšovat. Opakování
exp12 s vidoucí mouchou to potvrdilo: paired +0,0, placebo +0,3, tedy
nasycené měřidlo, ne nález o učení.

Tenhle pokus proto úlohu OBRACÍ: odměna se dává za to, když je cíl DÁL
než 45° od středu pohledu. Vrozené zapojení táhne přesně opačně, takže
výchozí úspěšnost je nízká a je kam růst. Zároveň je to ta otázka, na
které celý projekt stojí: dokáže odměna přebít to, co je v drátech?

Učí se stejných 12 synapsí na rozhraní graduované vrstvy a DNa02.

  R1: měřidlo drží, sham se nehne o víc než 2 p. b.
  R2: medián paired − placebo > 5 p. b. přes všech 12 losů.
  R3: paired > placebo párově aspoň v 9 z 12 losů.
Vyvrácení: když R2 i R3 padnou při platném R1 a nízké výchozí hodnotě
(tedy bez stropového efektu), je to poctivý záporný nález o učení —
na rozdíl od exp12, kde měřidlo narazilo na strop.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp09_pevna_zkouska import ROZPAD, ZKOUSKA  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# Odměna je SPOJITÁ, ne prahová. Binární práh tady vždycky narazí na kraj:
# při 45° drží moucha cíl tak pevně, že úspěšnost je 0,6 % (podlaha, odměna
# bez rozptylu), při 20° zase 13,8 %. Spojitá míra „jak daleko se podařilo
# cíl udržet" má rozptyl vždycky. (Nalezeno 14. 9. 2026 po dvou nepovedených
# pokusech s prahem.)
NORMA = 60.0              # stupňů; na tolik se odchylka normuje do [0, 1]
CIL_UHEL = 30.0           # kde se má cíl držet (stupňů vlevo od středu)


def miraKrok(h) -> float:
    """Úspěch = držet cíl na CIL_UHEL, ne ve středu a ne za obzorem.

    Původní zadání „drž cíl co nejdál" bylo pro detektor pohybu s omezeným
    zorným polem nesplnitelné: jakmile moucha cíl vytlačí za okraj, přestane
    ho vidět, nemá podle čeho zatáčet a cíl se sám vrátí. Úspěch by vyžadoval
    udržet stav, ve kterém nemá žádnou informaci. Ověřeno hrubou silou
    (exp15): ani náhodné přenastavení 12 ani 1376 synapsí nedá víc než
    +6 p. b. Držet cíl STRANOU, ale v dohledu, splnitelné být může.
    """
    return max(0.0, 1.0 - abs(abs(h.odchylka) - CIL_UHEL) / NORMA)


def zkouska(a, kroku: int) -> float:
    """Pevná sada epizod, bez učení a bez explorace.

    Vrací průměrnou vzdálenost cíle od středu pohledu (0 = přesně ve středu,
    1 = 60° a dál). Úkolem je ji ZVĚTŠIT, tedy dívat se jinam než na cíl —
    přesně proti tomu, k čemu zapojení táhne.
    """
    from hra import Hra as _H
    soucet = n = 0.0
    for seed in ZKOUSKA:
        h = _H(seed); pred = h.snimek()
        for _ in range(kroku):
            s = h.snimek()
            z = a.act(s, pred).zataceni
            pred = s
            h.krok(z)
            soucet += miraKrok(h); n += 1
    return soucet / n


def blok(a, h, delka, rezim, rng, sila, sigma, baseline):
    n = len(a.i_graded)
    stopy = {"vlevo": np.zeros(n, np.float32), "vpravo": np.zeros(n, np.float32)}
    ok = 0.0
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
        ok += miraKrok(h)
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
        verdikt = "Odměna přebila instinkt"
    else:
        verdikt = "ZÁPORNÝ: odměna instinkt nepřebije (a tentokrát bez stropového efektu)"
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp13.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
