"""
exp19_odmena_trest.py — odměna a trest vybírají směr změny.

PREREGISTRACE (zapsáno před během, 15. 9. 2026):
Exp18 ukázal, že plasticita v houbovitém tělísku se poprvé projeví až na
zatáčení a je specifická pro trénovaný podnět (6/8 losů nad 2 Hz, netrénovaný
medián 0,000). Znaménka změn ale byla různá, protože pravidlo depresi
aplikovalo pokaždé a nic nevybíralo směr. Bylo to podmiňování, ne učení.

Tady směr vybírá VÝSLEDEK akce, přes dva skutečné dopaminové systémy
z connectomu:
  · akce správná -> PAM (odměna), brána z PAM->MBON konektivity
  · akce špatná  -> PPL1 (trest), brána z PPL1->MBON konektivity
Obě brány jsou změřené z dat (exp01: PAM zasahuje 35 z 96 MBON, PPL1 63,
a jejich cíle jsou fyzicky oddělené). Deprese se tedy uplatní na JINÝCH
kompartmentech podle toho, jak akce dopadla.

Úloha: u podnětu se má moucha otáčet na určenou stranu. Polovina losů má
za cíl stranu, ke které zapojení táhne, druhá polovina opačnou — aby
nešlo uspět jen tím, že se posílí vrozené chování.

  H1: měřidlo drží (sham 0 ve všech losech).
  H2: u trénovaného podnětu se zatáčení posune SPRÁVNÝM směrem
      (k určené straně) v ≥ 8 z 12 losů.
  H3: posun u trénovaného je aspoň o 5 Hz větší než u netrénovaného.
Vyvrácení: když H2 padne při platném H1, dopaminové brány směr nevyberou
a odměnové učení se v tomhle modelu nedaří ani na správném místě.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp18_houbovite import PODNETY, zataceni  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class MBDve:
    """Houbovité tělísko se dvěma dopaminovými branami: odměna a trest."""

    def __init__(self, a: Agent2, c: Connectome) -> None:
        cir = Circuits(c)
        poz = {int(j): i for i, j in enumerate(a.i_spike)}
        mb_glob = cir.mbons().indices
        self.kc = np.array([poz[int(j)] for j in cir.kenyon_cells().indices if int(j) in poz])
        self.mbon = np.array([poz[int(j)] for j in mb_glob if int(j) in poz])
        self.a = a

        def brana(dans):
            d = c.syn_counts[mb_glob][:, dans.indices].toarray().sum(axis=1).astype(np.float32)
            return d / d.max() if d.max() > 0 else d

        self.pam = brana(cir.pam_dans())
        self.ppl1 = brana(cir.ppl1_dans())
        log.info("PAM zasahuje %d z %d MBON, PPL1 %d; překryv %d",
                 int((self.pam > 0).sum()), len(self.pam), int((self.ppl1 > 0).sum()),
                 int(((self.pam > 0) & (self.ppl1 > 0)).sum()))

        self.useky = {}
        for k, r in enumerate(self.mbon):
            lo, hi = a.W_spike.indptr[r], a.W_spike.indptr[r + 1]
            src = a.W_spike.indices[lo:hi]
            je_kc = np.isin(src, self.kc)
            if je_kc.any():
                self.useky[k] = (np.arange(lo, hi)[je_kc], src[je_kc])

    def uc(self, aktivita: np.ndarray, odmena: bool, sila: float) -> None:
        """Deprese KC->MBON v kompartmentech té dopaminové skupiny, která hoří."""
        brana = self.pam if odmena else self.ppl1
        for k, (pos, src) in self.useky.items():
            g = float(brana[k])
            if g <= 0:
                continue
            akt = aktivita[src]
            mx = akt.max()
            if mx > 0:
                self.a.W_spike.data[pos] *= (1.0 - sila * g * (akt / mx)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--kol", type=int, default=40)
    p.add_argument("--sila", type=float, default=0.5)
    p.add_argument("--amp-kc", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    mb = MBDve(a, c)

    vysl = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        vzorce = {j: (mb.kc[rng.choice(len(mb.kc), size=int(0.05 * len(mb.kc)), replace=False)],
                      args.amp_kc) for j in PODNETY}
        trenovany = "vlevo" if seed % 2 == 0 else "vpravo"
        netrenovany = "vpravo" if trenovany == "vlevo" else "vlevo"
        # polovina losů má za cíl stranu PROTI vrozenému tahu
        proti = (seed % 4) >= 2
        for rezim in ("sham", "paired"):
            a.reset_vahy()
            pred = {j: zataceni(a, PODNETY[j], vzorce[j])[0] for j in PODNETY}
            cil_kladny = (pred[trenovany] < 0) if proti else (pred[trenovany] > 0)
            for _ in range(args.kol):
                z, akt = zataceni(a, PODNETY[trenovany], vzorce[trenovany])
                if rezim == "paired":
                    spravne = (z > 0) == cil_kladny
                    mb.uc(akt, spravne, args.sila)
            po = {j: zataceni(a, PODNETY[j], vzorce[j])[0] for j in PODNETY}
            dt = po[trenovany] - pred[trenovany]
            smer = 1.0 if cil_kladny else -1.0
            log.info("seed %2d %-7s cíl %s (%s): trénovaný %+.0f -> %+.0f Hz (posun %+.0f, "
                     "správným směrem %+.0f) | netrénovaný %+.0f",
                     seed, rezim, "kladný" if cil_kladny else "záporný",
                     "proti instinktu" if proti else "po instinktu",
                     pred[trenovany], po[trenovany], dt, smer * dt,
                     po[netrenovany] - pred[netrenovany])
            vysl.append({"seed": seed, "rezim": rezim, "proti": bool(proti),
                         "posun_spravnym_smerem": float(smer * dt),
                         "posun_netrenovany": float(abs(po[netrenovany] - pred[netrenovany]))})

    def v(r, k): return np.array([x[k] for x in vysl if x["rezim"] == r])
    sp = v("paired", "posun_spravnym_smerem")
    log.info("=" * 74)
    log.info("  sham posun: %s", np.round(v("sham", "posun_spravnym_smerem"), 1))
    log.info("  paired posun správným směrem: medián %+.1f Hz  %s", np.median(sp), np.round(sp, 1))
    log.info("  z toho losy PROTI instinktu: %s",
             np.round(np.array([x["posun_spravnym_smerem"] for x in vysl
                                if x["rezim"] == "paired" and x["proti"]]), 1))
    h1 = bool(np.abs(v("sham", "posun_spravnym_smerem")).max() < 1e-9)
    h2 = int((sp > 0).sum()) >= max(8, int(0.66 * len(args.seeds)))
    h3 = bool(np.median(np.abs(sp)) - np.median(v("paired", "posun_netrenovany")) > 5.0)
    log.info("  H1 měřidlo drží                  : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 posun správným směrem ≥2/3    : %s (%d/%d)",
             "SPLNĚNO" if h2 else "NESPLNĚNO", int((sp > 0).sum()), len(args.seeds))
    log.info("  H3 specifický vůči netrénovanému : %s", "SPLNĚNO" if h3 else "NESPLNĚNO")
    verdikt = ("Odměna a trest vybírají směr — moucha se učí" if (h1 and h2 and h3)
               else "Posun správným směrem, ale nespecifický" if (h1 and h2)
               else "ZÁPORNÝ: dopaminové brány směr nevyberou")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp19.json").write_text(
        json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
