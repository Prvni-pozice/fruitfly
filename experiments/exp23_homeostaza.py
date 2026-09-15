"""
exp23_homeostaza.py — deprese + homeostáza = rozlišování?

PREREGISTRACE (zapsáno před během, 15. 9. 2026):
Exp18–exp22 používaly pravidlo, které synapse jen ZESLABUJE. Výsledkem byl
plošný posun: zeslabení všech výstupů paměti pohnulo oběma podněty stejným
směrem (−40 a −70 Hz), tedy vychýlení, ne rozlišení.

Literatura (PLOS Comput Biol 2023, „imbalanced associative learning in the
mushroom body compartment\") říká proč: dopaminová plasticita je jednosměrná
a posilování obstarává HOMEOSTÁZA — celková váha vstupů na výstupní neuron
se udržuje konstantní. Zeslabení synapsí jednoho podnětu tím relativně
posílí ostatní, a teprve to dělá rozlišování.

Tady se přidá právě ta homeostáza: po každé depresi se každému MBON
přeškáluje suma vstupů z Kenyonových buněk zpět na původní hodnotu.

  H1: měřidlo drží (sham 0 ve všech losech).
  H2: trénovaný a netrénovaný podnět se posunou OPAČNÝM směrem
      v ≥ 8 z 12 losů (to je rozlišování, ne vychýlení).
  H3: rozdíl mezi jejich posuny > 10 Hz v mediánu.
Vyvrácení: když se oba podněty dál posouvají stejným směrem, homeostáza
rozlišování nepřinese a paměť v tomhle modelu umí jen plošné vychýlení.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp18_houbovite import PODNETY, zataceni  # noqa: E402
from exp19_odmena_trest import MBDve  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class MBHomeo(MBDve):
    """Houbovité tělísko s homeostázou: suma vstupů z KC na MBON je zachovaná."""

    def priprav(self) -> None:
        self.sumy = {k: float(np.abs(self.a.W_spike.data[pos]).sum())
                     for k, (pos, _) in self.useky.items()}

    def uc(self, aktivita: np.ndarray, odmena: bool, sila: float, homeo: bool = True) -> None:
        brana = self.pam if odmena else self.ppl1
        for k, (pos, src) in self.useky.items():
            g = float(brana[k])
            if g <= 0:
                continue
            akt = aktivita[src]
            mx = akt.max()
            if mx <= 0:
                continue
            self.a.W_spike.data[pos] *= (1.0 - sila * g * (akt / mx)).astype(np.float32)
            if homeo:
                # homeostáza: vrať celkovou váhu vstupů zpět na původní sumu.
                # Co se zeslabilo u trénovaného podnětu, to se tím relativně
                # posílí u ostatních — a právě to dělá rozlišování.
                nova = float(np.abs(self.a.W_spike.data[pos]).sum())
                if nova > 0:
                    self.a.W_spike.data[pos] *= np.float32(self.sumy[k] / nova)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--kol", type=int, default=40)
    p.add_argument("--sila", type=float, default=0.5)
    p.add_argument("--amp-kc", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--bez-homeostazy", action="store_true", help="kontrolní běh pro srovnání")
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    mb = MBHomeo(a, c)
    mb.priprav()
    homeo = not args.bez_homeostazy
    log.info("homeostáza: %s", "ZAPNUTA" if homeo else "vypnuta (kontrola)")

    vysl = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        vz = {j: (mb.kc[rng.choice(len(mb.kc), size=int(0.05 * len(mb.kc)), replace=False)],
                  args.amp_kc) for j in PODNETY}
        trenovany = "vlevo" if seed % 2 == 0 else "vpravo"
        netrenovany = "vpravo" if trenovany == "vlevo" else "vlevo"
        for rezim in ("sham", "paired"):
            a.reset_vahy(); mb.priprav()
            pred = {j: zataceni(a, PODNETY[j], vz[j])[0] for j in PODNETY}
            for _ in range(args.kol):
                _, akt = zataceni(a, PODNETY[trenovany], vz[trenovany])
                if rezim == "paired":
                    mb.uc(akt, True, args.sila, homeo)
            po = {j: zataceni(a, PODNETY[j], vz[j])[0] for j in PODNETY}
            dt = po[trenovany] - pred[trenovany]
            dn = po[netrenovany] - pred[netrenovany]
            log.info("seed %2d %-7s: trénovaný %+.0f Hz | netrénovaný %+.0f Hz | rozdíl %+.0f",
                     seed, rezim, dt, dn, dt - dn)
            vysl.append({"seed": seed, "rezim": rezim, "dt": float(dt), "dn": float(dn),
                         "rozdil": float(dt - dn)})

    def v(r, k): return np.array([x[k] for x in vysl if x["rezim"] == r])
    dt, dn, roz = v("paired", "dt"), v("paired", "dn"), v("paired", "rozdil")
    log.info("=" * 74)
    log.info("  sham rozdíly: %s", np.round(v("sham", "rozdil"), 1))
    log.info("  paired: trénovaný medián %+.1f | netrénovaný %+.1f | rozdíl %+.1f Hz",
             np.median(dt), np.median(dn), np.median(roz))
    opacne = int(((dt > 0) & (dn < 0)).sum() + ((dt < 0) & (dn > 0)).sum())
    h1 = bool(np.abs(v("sham", "rozdil")).max() < 1e-9)
    h2 = opacne >= max(8, int(0.66 * len(args.seeds)))
    h3 = bool(np.median(np.abs(roz)) > 10.0)
    log.info("  H1 měřidlo drží                  : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 opačné směry ve ≥2/3 losů     : %s (%d/%d)",
             "SPLNĚNO" if h2 else "NESPLNĚNO", opacne, len(args.seeds))
    log.info("  H3 rozdíl posunů > 10 Hz         : %s (%+.1f)",
             "SPLNĚNO" if h3 else "NESPLNĚNO", np.median(np.abs(roz)))
    verdikt = ("Homeostáza přinesla ROZLIŠOVÁNÍ" if (h1 and h2 and h3)
               else "Rozdíl je, ale směry nejsou opačné" if (h1 and h3)
               else "ZÁPORNÝ: ani s homeostázou to není rozlišování")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / ("exp23.json" if homeo else "exp23_kontrola.json")).write_text(
        json.dumps({"verdikt": verdikt, "homeostaza": homeo, "vysledky": vysl},
                   indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
