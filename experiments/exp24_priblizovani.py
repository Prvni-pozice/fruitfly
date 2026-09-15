"""
exp24_priblizovani.py — měříme chování v čase, ne okamžitou zatáčku.

PREREGISTRACE (zapsáno před během, 15. 9. 2026):
Patnáct pokusů o učení (exp09–exp23) měřilo OKAMŽITÝ směr zatáčení na
statický podnět. Plasticita vyšla stimulus-specifická, ale směr se řídit
nedal. Je ale možné, že měříme špatnou veličinu: učení v houbovitém
tělísku je u mouchy popsané jako PŘIBLIŽOVÁNÍ a VYHÝBÁNÍ v čase, ne jako
okamžitá zatáčka.

Tady se tedy po tréninku nechá moucha volně běhat ve hře a měří se
průměrná vzdálenost, na kterou si cíl drží — integrovaně přes celou
epizodu. Přibližování = menší vzdálenost, vyhýbání = větší.

Kenyonovy buňky se budí po celou epizodu daným vzorcem („pach\", ve kterém
se moucha pohybuje). Trénuje se s jedním vzorcem, testuje s trénovaným
i netrénovaným.

  H1: měřidlo drží (sham nezmění nic).
  H2: u trénovaného vzorce se průměrná vzdálenost změní o > 3° v ≥ 2/3 losů.
  H3: změna u trénovaného je aspoň o 2° větší než u netrénovaného.
Vyvrácení: když H2 padne, nejde o špatně zvolený odečet a patnáct
předchozích pokusů mělo pravdu — v tomhle modelu se odměnové učení
do chování nepropisuje.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp18_houbovite import PODNETY, zataceni  # noqa: E402
from exp23_homeostaza import MBHomeo  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393


def epizoda(a: Agent2, seed: int, kc, kroku: int) -> float:
    """Volný běh ve hře s daným vzorcem KC. Vrací průměrnou vzdálenost cíle."""
    h = Hra(seed)
    pred = h.snimek()
    soucet = 0.0
    for _ in range(kroku):
        s = h.snimek()
        ext = a.vjem(s, pred)
        ext[kc[0]] += kc[1]
        res = Simulator2(a.W_spike, weight_scale=WS).run(a.steps, external=ext)
        sc = res.spike_counts.numpy(); dur = res.duration_s
        l = sc[a.dna_l].sum() / dur; p = sc[a.dna_r].sum() / dur
        z = (l - p) / (l + p) if (l + p) > 0 else 0.0
        pred = s
        h.krok(z)
        soucet += abs(h.odchylka)
    return soucet / kroku


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--kol", type=int, default=40)
    p.add_argument("--epizod", type=int, default=3)
    p.add_argument("--kroku", type=int, default=15)
    p.add_argument("--sila", type=float, default=0.5)
    p.add_argument("--amp-kc", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    mb = MBHomeo(a, c); mb.priprav()

    vysl = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        vz = {j: (mb.kc[rng.choice(len(mb.kc), size=int(0.05 * len(mb.kc)), replace=False)],
                  args.amp_kc) for j in ("A", "B")}
        for rezim in ("sham", "paired"):
            a.reset_vahy(); mb.priprav()
            pred = {j: np.mean([epizoda(a, 500 + i, vz[j], args.kroku) for i in range(args.epizod)])
                    for j in vz}
            for _ in range(args.kol):
                _, akt = zataceni(a, PODNETY["vlevo"], vz["A"])
                if rezim == "paired":
                    mb.uc(akt, True, args.sila, homeo=False)
            po = {j: np.mean([epizoda(a, 500 + i, vz[j], args.kroku) for i in range(args.epizod)])
                  for j in vz}
            dA, dB = po["A"] - pred["A"], po["B"] - pred["B"]
            log.info("seed %2d %-7s: trénovaný %.1f° -> %.1f° (%+.1f) | netrénovaný %+.1f | rozdíl %+.1f",
                     seed, rezim, pred["A"], po["A"], dA, dB, dA - dB)
            vysl.append({"seed": seed, "rezim": rezim, "dA": float(dA), "dB": float(dB),
                         "rozdil": float(dA - dB)})

    def v(r, k): return np.array([x[k] for x in vysl if x["rezim"] == r])
    dA, dB, roz = v("paired", "dA"), v("paired", "dB"), v("paired", "rozdil")
    log.info("=" * 74)
    log.info("  sham: trénovaný %s", np.round(v("sham", "dA"), 2))
    log.info("  paired: trénovaný medián %+.2f° | netrénovaný %+.2f° | rozdíl %+.2f°",
             np.median(dA), np.median(dB), np.median(roz))
    h1 = bool(np.abs(v("sham", "dA")).max() < 1e-9)
    h2 = int((np.abs(dA) > 3.0).sum()) >= max(2, int(0.66 * len(args.seeds)))
    h3 = bool(np.median(np.abs(dA)) - np.median(np.abs(dB)) > 2.0)
    log.info("  H1 měřidlo drží                : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 změna > 3° ve ≥2/3 losů     : %s (%d/%d)",
             "SPLNĚNO" if h2 else "NESPLNĚNO", int((np.abs(dA) > 3.0).sum()), len(args.seeds))
    log.info("  H3 specifické vůči netrénovanému: %s", "SPLNĚNO" if h3 else "NESPLNĚNO")
    verdikt = ("Učení se projeví na chování v čase" if (h1 and h2 and h3)
               else "Změna je, ale nespecifická" if (h1 and h2)
               else "ZÁPORNÝ: ani v chování v čase se učení neprojeví")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp24.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
