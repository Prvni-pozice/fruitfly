"""
exp26_endogenni_odmena.py — moucha se učí, protože OCHUTNALA cukr.

PREREGISTRACE (zapsáno před během, 19. 9. 2026):
Ve všech dosavadních pokusech o učení (exp09–exp24) jsem dopaminovou bránu
nastavoval ručně: „akce správná -> deprese v PAM kompartmentech". Tady bránu
vyrábí sám connectome: za správnou akci dostane moucha CUKR na chuťové
neurony, cukr přes zapojení (se Shiuovým pravidlem, exp25) rozsvítí PAM,
a deprese KC->MBON se řídí tím, kolik dopaminu který MBON skutečně dostal
(Σ frekvence PAM × synapse PAM->MBON). Žádné číslo nevkládám zvenčí.

Co zůstává umělé a je to zapsané: Kenyonovy buňky se budí přímo, protože
je vizuální vstup v modelu neprotlačí (exp18). Zapojení KC->MBON,
PAM->MBON i MBON->DNa02 je z connectomu.

Kvůli změřené páce paměti na zatáčení (exp21–22: 4–17 %) se NEPŘEDPOVÍDÁ
velký posun. Předpovídá se SMĚR — to je to, co dosud nikdy nevyšlo:
  H1: měřidlo drží (sham 0 ve všech losech).
  H2: posun zatáčení u trénovaného podnětu míří ke správné straně
      v ≥ 9 z 12 losů (dosud 6/12 = mince).
  H3: placebo (cukr náhodně, bez vazby na akci) správný směr v < 9/12.
Vyvrácení: H2 padne -> ani skutečná chuť cukru směr nevybere; učení
v tomhle modelu končí s tím, že paměť se mění, ale volant nedrží.

Výsledky se ukládají po losech (--pridat), běh lze dělit na kusy.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp18_houbovite import PODNETY  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from pravidlo_shiu import pouzij_shiu  # noqa: E402
from server_mozku import snimek  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393
VYSTUP = ROOT / "outputs" / "exp26.json"


class Chut:
    """Cukr -> PAM -> deprese KC->MBON, celé přes zapojení."""

    def __init__(self, a: Agent2, c: Connectome) -> None:
        cir = Circuits(c)
        poz = {int(j): i for i, j in enumerate(a.i_spike)}
        self.a = a
        self.kc = np.array([poz[int(j)] for j in cir.kenyon_cells().indices if int(j) in poz])
        mb_glob = cir.mbons().indices
        self.mbon = np.array([poz[int(j)] for j in mb_glob if int(j) in poz])
        pam_glob = cir.pam_dans().indices
        self.pam = np.array([poz[int(j)] for j in pam_glob if int(j) in poz])
        self.cukr = np.array([poz[int(j)] for j in cir.sugar_grns().indices if int(j) in poz])
        # anatomie PAM -> MBON (řádky MBON, sloupce PAM), z počtů synapsí
        self.pam_na_mbon = c.syn_counts[mb_glob][:, pam_glob].toarray().astype(np.float32)
        self.useky = {}
        for k, r in enumerate(self.mbon):
            lo, hi = a.W_spike.indptr[r], a.W_spike.indptr[r + 1]
            src = a.W_spike.indices[lo:hi]
            je_kc = np.isin(src, self.kc)
            if je_kc.any():
                self.useky[k] = (np.arange(lo, hi)[je_kc], src[je_kc])
        log.info("chuť: %d KC, %d MBON, %d PAM, %d chuťových neuronů na cukr",
                 len(self.kc), len(self.mbon), len(self.pam), len(self.cukr))

    def krok(self, uhel: float, kc_vzor, s_cukrem: bool, sum_dna: float = 0.0, rng=None):
        """Jeden průchod: vjem (+KC vzor) (+cukr) (+explorační šum do DNa02).

        Šum je nutný: bez něj je moucha deterministická, u poloviny losů
        nikdy neudělá „správnou" akci, nikdy neochutná cukr a dopamin je
        celkem 0 (změřeno 19. 9. 2026, losy 1 a 2). Bez explorace není odměna.
        """
        ext = self.a.vjem(snimek(uhel), snimek(uhel - 12.0))
        ext[kc_vzor[0]] += kc_vzor[1]
        if sum_dna and rng is not None:
            ext[self.a.dna_l] += float(rng.normal(0, sum_dna))
            ext[self.a.dna_r] += float(rng.normal(0, sum_dna))
        if s_cukrem:
            ext[self.cukr] += 3.0
        res = Simulator2(self.a.W_spike, weight_scale=WS).run(self.a.steps, external=ext)
        sc = res.spike_counts.numpy().astype(np.float32); dur = res.duration_s
        l = sc[self.a.dna_l].sum() / dur; p = sc[self.a.dna_r].sum() / dur
        return float(l - p), sc

    def uc_z_chuti(self, sc: np.ndarray, sila: float) -> float:
        """Deprese řízená SKUTEČNÝM dopaminem: kolik ho který MBON dostal.

        brána_k = Σ_j frekvence(PAM_j) × synapse(PAM_j -> MBON_k), normováno.
        Vrací celkovou dopaminovou aktivitu (pro log).
        """
        pam_akt = sc[self.pam]
        brana = self.pam_na_mbon @ pam_akt
        mx = brana.max()
        if mx <= 0:
            return 0.0
        brana = brana / mx
        for k, (pos, src) in self.useky.items():
            g = float(brana[k])
            if g <= 0:
                continue
            akt = sc[src]
            m = akt.max()
            if m > 0:
                self.a.W_spike.data[pos] *= (1.0 - sila * g * (akt / m)).astype(np.float32)
        return float(pam_akt.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--kol", type=int, default=30)
    p.add_argument("--sila", type=float, default=0.5)
    p.add_argument("--amp-kc", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sum", type=float, default=1.5, help="explorační šum do DNa02 při tréninku")
    p.add_argument("--pridat", action="store_true", help="připojit k dřívějším výsledkům")
    p.add_argument("--jen-report", action="store_true")
    args = p.parse_args()

    vysl = json.loads(VYSTUP.read_text())["vysledky"] if (args.pridat or args.jen_report) and VYSTUP.exists() else []
    if not args.jen_report:
        c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
        pouzij_shiu(c)
        a = Agent2(c, steps=args.steps)
        ch = Chut(a, c)
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            vz = {j: (ch.kc[rng.choice(len(ch.kc), size=int(0.05 * len(ch.kc)), replace=False)], args.amp_kc)
                  for j in PODNETY}
            tren = "vlevo" if seed % 2 == 0 else "vpravo"
            netren = "vpravo" if tren == "vlevo" else "vlevo"
            for rezim in ("sham", "paired", "placebo"):
                a.reset_vahy()
                pred = {j: ch.krok(PODNETY[j], vz[j], False)[0] for j in PODNETY}
                # cíl: u poloviny losů posunout doleva (kladně), u druhé doprava
                cil_kladny = (seed % 4) < 2
                dopamin = 0.0; odmen = 0
                for _ in range(args.kol):
                    z, _ = ch.krok(PODNETY[tren], vz[tren], False, args.sum, rng)
                    spravne = (z > 0) == cil_kladny
                    if rezim == "sham":
                        continue
                    dej = spravne if rezim == "paired" else bool(rng.random() < 0.5)
                    if dej:
                        odmen += 1
                        _, sc = ch.krok(PODNETY[tren], vz[tren], True)   # ochutná cukr
                        dopamin += ch.uc_z_chuti(sc, args.sila)
                po = {j: ch.krok(PODNETY[j], vz[j], False)[0] for j in PODNETY}
                dt = po[tren] - pred[tren]; dn = po[netren] - pred[netren]
                smer = dt if cil_kladny else -dt
                log.info("seed %2d %-7s cíl %-7s: trénovaný %+.0f -> %+.0f Hz (správným směrem %+.0f) | "
                         "netrénovaný %+.0f | dopamin celkem %.0f | odměn %d/%d",
                         seed, rezim, "doleva" if cil_kladny else "doprava", pred[tren], po[tren], smer, dn, dopamin,
                         odmen, args.kol)
                vysl.append({"seed": seed, "rezim": rezim, "spravnym_smerem": float(smer),
                             "netrenovany": float(dn), "dopamin": float(dopamin)})
            VYSTUP.write_text(json.dumps({"vysledky": vysl}, indent=2, ensure_ascii=False))

    def v(r): return np.array([x["spravnym_smerem"] for x in vysl if x["rezim"] == r])
    n = len({x["seed"] for x in vysl})
    log.info("=" * 74)
    for r in ("sham", "paired", "placebo"):
        s = v(r)
        log.info("  %-8s správným směrem: %d/%d kladných, medián %+.1f Hz  %s",
                 r, int((s > 0).sum()), len(s), np.median(s) if len(s) else 0, np.round(s, 0))
    h1 = bool(len(v("sham")) and np.abs(v("sham")).max() < 1e-9)
    h2 = int((v("paired") > 0).sum()) >= int(np.ceil(0.75 * n))
    h3 = int((v("placebo") > 0).sum()) < int(np.ceil(0.75 * n))
    log.info("  H1 měřidlo drží                    : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 paired správný směr ≥ 3/4 losů  : %s (%d/%d)", "SPLNĚNO" if h2 else "NESPLNĚNO",
             int((v("paired") > 0).sum()), n)
    log.info("  H3 placebo pod prahem              : %s (%d/%d)", "SPLNĚNO" if h3 else "NESPLNĚNO",
             int((v("placebo") > 0).sum()), n)
    verdikt = ("Chuť cukru vybírá směr učení" if (h1 and h2 and h3)
               else "Směr vybírá i placebo — není to odměna, ale cukr sám" if (h1 and h2)
               else "ZÁPORNÝ: ani skutečná chuť cukru směr nevybere")
    log.info("VERDIKT (%d losů): %s", n, verdikt)
    VYSTUP.write_text(json.dumps({"verdikt": verdikt, "losu": n, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
