"""
exp15_je_to_resitelne.py — jde úloha vůbec vyřešit těmi dvanácti vahami?

Kontrola, kterou si projekt vynutil 14. 9. 2026 po čtyřech záporných
pokusech o učení. Preregistrace a placebo hlídají, jestli je výsledek
pravdivý — nehlídají, jestli je ZADÁNÍ splnitelné. Tenhle skript hledá
řešení hrubou silou: náhodně přenastaví dvanáct učených synapsí a změří,
jestli některé nastavení úlohu řeší.

  S1: aspoň jedno nastavení dá výsledek o 10 p. b. lepší než výchozí.
Když S1 padne, je záporný nález o učení nálezem o ROZHRANÍ: přes dvanáct
synapsí na DNa02 se tahle úloha vyjádřit nedá a učení nemá co najít.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp13_proti_instinktu import zkouska  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pokusu", type=int, default=14)
    p.add_argument("--zk-kroku", type=int, default=15)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--spikujici", action="store_true",
                   help="zkusit 746 synapsí ze SPIKUJÍCÍ části na DNa02 (99 % vstupu)")
    p.add_argument("--siroke", action="store_true",
                   help="místo 12 synapsí na DNa02 zkusit všech 1376 vstupů do sestupných neuronů")
    args = p.parse_args()

    a = Agent2(Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome"), steps=args.steps)
    if args.siroke:
        rady = [("dn", int(r)) for r in a.dn_vse]
        log.info("široké rozhraní: %d sestupných neuronů", len(rady))
    else:
        rady = [(strana, r) for strana, ix in (("l", a.dna_l), ("p", a.dna_r)) for r in ix]

    def prenastav(rng):
        """Náhodně přeškáluje učené synapse (včetně obracení znamének)."""
        if args.spikujici:
            for ix in (a.dna_l, a.dna_r):
                for r in ix:
                    lo, hi = a.W_spike.indptr[r], a.W_spike.indptr[r + 1]
                    if lo == hi:
                        continue
                    skala = rng.uniform(-2.0, 2.0, size=hi - lo).astype(np.float32)
                    a.W_spike.data[lo:hi] = a.W0.data[lo:hi] * skala
            return
        for _, r in rady:
            lo, hi = a.W_g2s.indptr[r], a.W_g2s.indptr[r + 1]
            if lo == hi:
                continue
            skala = rng.uniform(-2.0, 2.0, size=hi - lo).astype(np.float32)
            a.W_g2s.data[lo:hi] = a._W_g2s0.data[lo:hi] * skala

    def reset():
        a.reset_rozhrani(); a.reset_vahy()

    reset()
    vychozi = zkouska(a, args.zk_kroku)
    log.info("výchozí (vrozené zapojení): %.3f", vychozi)

    rng = np.random.default_rng(0)
    nej = vychozi; nej_popis = "výchozí"
    for k in range(args.pokusu):
        reset()
        prenastav(rng)
        v = zkouska(a, args.zk_kroku)
        znacka = ""
        if v > nej:
            nej, nej_popis, znacka = v, f"pokus {k}", "  <- zatím nejlepší"
        log.info("  pokus %2d: %.3f%s", k, v, znacka)

    reset()
    log.info("=" * 60)
    log.info("výchozí %.3f | nejlepší nalezené %.3f (%s) | zlepšení %+.1f p. b.",
             vychozi, nej, nej_popis, 100 * (nej - vychozi))
    s1 = (nej - vychozi) >= 0.10
    log.info("  S1 existuje nastavení o ≥10 p. b. lepší: %s", "ANO" if s1 else "NE")
    log.info("VERDIKT: %s", "Úloha JE přes tohle rozhraní řešitelná — záporný nález platí o učení"
             if s1 else
             "Úloha NENÍ přes tohle rozhraní řešitelná — záporné nálezy o učení jsou nálezy o rozhraní")


if __name__ == "__main__":
    main()
