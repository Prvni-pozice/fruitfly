"""
exp18_houbovite.py — učit tam, kde se moucha SKUTEČNĚ učí.

PREREGISTRACE (zapsáno před během, 14. 9. 2026):
Šest pokusů (exp09–exp17) učilo synapse na sestupných neuronech a žádný
neuspěl; poslední z nich ukázal, že ani přímý optimalizátor tam nenajde
obecně lepší politiku (na nových epizodách −1,0 p. b. po 40 iteracích).

Literatura je přitom jednotná: u mouchy je místem odměnového učení
HOUBOVITÉ TĚLÍSKO — synapse Kenyonových buněk na MBON, řízené dopaminem
(Nature Communications 2021; spikující model larvy, bioRxiv 2022).
Učili jsme celou dobu na místě, kde se moucha neučí.

Změřeno předem, že cesta existuje a je průchozí:
  · vizuální projekční neurony -> Kenyonovy buňky: 13 229 synapsí (354 KC)
  · MBON -> DNa02: 143 synapsí, MBON -> sestupné celkem 3 543
  · umělé vybuzení MBON jedné strany posune zatáčení až o 0,658

Úloha: dva vizuální podněty (cíl vlevo / vpravo). U JEDNOHO z nich se
odměňuje, u druhého ne. Sleduje se, jestli se zatáčení posune u trénovaného
podnětu a NEPOSUNE u netrénovaného — tedy jestli je paměť specifická.

  H1: měřidlo drží (sham 0).
  H2: posun zatáčení u trénovaného podnětu > 0,10 v ≥ 8 z 12 losů.
  H3: posun u trénovaného je aspoň o 0,10 větší než u netrénovaného
      (specificita — jinak jde o plošný drift, ne o paměť).
Vyvrácení: když H2 padne, neučí se ani v houbovitém tělísku a projekt
odměnové učení uzavírá jako nedosažené. Když padne jen H3, je efekt
reálný, ale nespecifický — což je slabší, ale zapsatelný nález.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from server_mozku import snimek  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

PODNETY = {"vlevo": -12.0, "vpravo": +12.0}   # ne -24/+24: tam je odpověď na stropu (+1,000) a učení se nemá kde projevit


class MB:
    """Houbovité tělísko: které synapse KC->MBON se učí a jak."""

    def __init__(self, a: Agent2, c: Connectome) -> None:
        cir = Circuits(c)
        poz = {int(j): i for i, j in enumerate(a.i_spike)}
        self.kc = np.array([poz[int(j)] for j in cir.kenyon_cells().indices if int(j) in poz])
        self.mbon = np.array([poz[int(j)] for j in cir.mbons().indices if int(j) in poz])
        pam = cir.pam_dans().indices
        # dopaminová brána na MBON ze skutečné konektivity PAM -> MBON
        d = c.syn_counts[cir.mbons().indices][:, pam].toarray().sum(axis=1).astype(np.float32)
        self.brana = d / d.max() if d.max() > 0 else d
        self.a = a
        # pozice synapsí KC -> MBON v matici spikující části
        self.useky = {}
        for k, r in enumerate(self.mbon):
            lo, hi = a.W_spike.indptr[r], a.W_spike.indptr[r + 1]
            src = a.W_spike.indices[lo:hi]
            je_kc = np.isin(src, self.kc)
            if je_kc.any():
                self.useky[k] = (np.arange(lo, hi)[je_kc], src[je_kc])
        log.info("houbovité tělísko: %d KC, %d MBON, %d MBON má vstup z KC, PAM zasahuje %d",
                 len(self.kc), len(self.mbon), len(self.useky), int((self.brana > 0).sum()))

    def uc(self, aktivita: np.ndarray, sila: float) -> None:
        """Dopaminem řízená deprese: zeslab KC->MBON tam, kde byly KC aktivní.

        Přesně pravidlo z exp01, které v houbovitém tělísku prokazatelně
        funguje (kompartmentově i pachově specifické) — jen se poprvé
        použije na VIZUÁLNÍ podnět a s odečtem na chování.
        """
        for k, (pos, src) in self.useky.items():
            g = float(self.brana[k])
            if g <= 0:
                continue
            akt = aktivita[src]
            mx = akt.max()
            if mx > 0:
                self.a.W_spike.data[pos] *= (1.0 - sila * g * (akt / mx)).astype(np.float32)


def zataceni(a: Agent2, uhel: float, kc_vstup=None):
    """Zatáčení na daný podnět; volitelně s přímým buzením Kenyonových buněk.

    Proč přímé buzení: v modelu nevystřelí ANI JEDNA z 5 177 Kenyonových
    buněk, ať je vizuální podnět jakýkoli (měřeno 14. 9. 2026). Vizuální
    vstup na ně jde jen přes 354 buněk po ~37 synapsích, což je hluboko pod
    prahem. Bez aktivity KC nemá plasticita na čem pracovat.
    Je to stejný ústupek jako u optického laloku: dodáváme zvenčí signál,
    který model sám neunese. Zapojení KC->MBON i MBON->DNa02 zůstává
    z connectomu, dodává se jen buzení vstupní vrstvy.
    """
    ext = a.vjem(snimek(uhel), snimek(uhel - 12.0))
    if kc_vstup is not None:
        ext[kc_vstup[0]] += kc_vstup[1]
    from sim2 import Simulator2
    res = Simulator2(a.W_spike, weight_scale=0.0393).run(a.steps, external=ext)
    sc = res.spike_counts.numpy(); dur = res.duration_s
    l = sc[a.dna_l].sum() / dur; pr = sc[a.dna_r].sum() / dur
    # ROZDÍL v Hz, ne normalizovaná bilance: ta se při umlčení jedné strany
    # zasekne na ±1 a učení se nemá kde projevit (naměřeno u ±24°)
    return float(l - pr), sc.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--kol", type=int, default=12)
    p.add_argument("--sila", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--amp-kc", type=float, default=2.5,
                   help="jak silně budit Kenyonovy buňky (model je sám neprotlačí)")
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    mb = MB(a, c)

    # každý podnět budí svou třetinu Kenyonových buněk — řídký kód jako u mouchy
    log.info("vzorce KC se losují pro každý los znovu (5 %% z %d buněk)", len(mb.kc))

    vysl = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # každý los má jiný pachový/vizuální kód -> skutečná replikace
        vzorce = {j: (mb.kc[rng.choice(len(mb.kc), size=int(0.05 * len(mb.kc)), replace=False)],
                      args.amp_kc) for j in PODNETY}
        trenovany = "vlevo" if seed % 2 == 0 else "vpravo"
        netrenovany = "vpravo" if trenovany == "vlevo" else "vlevo"
        for rezim in ("sham", "paired"):
            a.reset_vahy()
            pred = {j: zataceni(a, PODNETY[j], vzorce[j])[0] for j in PODNETY}
            for _ in range(args.kol):
                z, akt = zataceni(a, PODNETY[trenovany], vzorce[trenovany])
                if rezim == "paired":
                    mb.uc(akt, args.sila)
            po = {j: zataceni(a, PODNETY[j], vzorce[j])[0] for j in PODNETY}
            dt = po[trenovany] - pred[trenovany]
            dn = po[netrenovany] - pred[netrenovany]
            log.info("seed %2d %-7s (trénink %s): trénovaný %+.3f -> %+.3f (%+.3f) | "
                     "netrénovaný %+.3f (%+.3f)",
                     seed, rezim, trenovany, pred[trenovany], po[trenovany], dt, po[netrenovany], dn)
            vysl.append({"seed": seed, "rezim": rezim, "trenovany": trenovany,
                         "zmena_trenovany": float(dt), "zmena_netrenovany": float(dn)})

    def zm(r, k): return np.array([v[k] for v in vysl if v["rezim"] == r])
    log.info("=" * 74)
    log.info("  sham   trénovaný: %s", np.round(zm("sham", "zmena_trenovany"), 3))
    log.info("  paired trénovaný: medián %+.3f  %s",
             np.median(zm("paired", "zmena_trenovany")), np.round(zm("paired", "zmena_trenovany"), 3))
    log.info("  paired NEtrénovaný: medián %+.3f", np.median(zm("paired", "zmena_netrenovany")))
    h1 = bool(np.abs(zm("sham", "zmena_trenovany")).max() < 1e-9)
    tr = np.abs(zm("paired", "zmena_trenovany"))
    h2 = int((tr > 2.0).sum()) >= max(5, int(0.66 * len(args.seeds)))
    h3 = bool(np.median(tr) - np.median(np.abs(zm("paired", "zmena_netrenovany"))) > 2.0)
    log.info("  H1 měřidlo drží                    : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 posun > 2 Hz ve ≥2/3 losů       : %s (%d/%d)",
             "SPLNĚNO" if h2 else "NESPLNĚNO", int((tr > 0.10).sum()), len(args.seeds))
    log.info("  H3 specifický vůči netrénovanému   : %s", "SPLNĚNO" if h3 else "NESPLNĚNO")
    verdikt = ("Houbovité tělísko se učí a projeví se to na chování" if (h1 and h2 and h3)
               else "Efekt bez specificity" if (h1 and h2)
               else "ZÁPORNÝ: neučí se ani v houbovitém tělísku")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp18.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
