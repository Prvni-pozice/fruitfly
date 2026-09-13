"""
exp06_hra.py — uzavřená smyčka nad hrou s ODLOŽENOU odměnou.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Agent (`agent2.py`) vidí scénu, zatáčí podle DNa02 a má udržet cíl ve středu.
Odměna nepřijde po každém kroku, ale až na konci bloku N kroků — to je ten
případ, který skutečná hra vyžaduje. Přemostí se eligibility trace: každý
krok si pamatuje součin pre×post aktivity, odměna na konci ho použije.

  H0 (ověření vstupu): pohyb cíle doleva vyvolá jiné zatáčení než doprava,
     rozdíl > 0,15 a nad nullem z 10 přehozených map sloupců.
  H1: úspěšnost (podíl kroků s cílem do 20° od středu) vzroste mezi prvními
     a posledními pěti bloky aspoň o 10 p. b. ve 3 ze 3 losů.
  H2: přírůstek u `paired` je větší než u `placebo` (odměna náhodná).
  H3: `sham` (bez učení) se nezlepší (přírůstek < 5 p. b.).
Vyvrácení: když placebo doroste stejně, zlepšení nedělá odměna, ale drift
vah; reportovat jako záporný nález.
"""
from __future__ import annotations
import json, logging, sys, time
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
OUT = ROOT / "outputs"


def h0_overeni(a: Agent2, n_pozic: int = 9, n_null: int = 10) -> dict:
    """Sleduje zatáčení polohu cíle? S nullem z přehozených map."""
    h = Hra(0)
    def mer(smer, mapa_perm=None):
        z = []
        for d in np.linspace(-30, 30, n_pozic):
            h.uhel, h.cil = 0.0, float(d)
            s = h.snimek()
            h.cil = float(d) + smer * 12.0
            z.append(a.act(h.snimek(), s).zataceni)
        return np.array(z)
    zl, zp = mer(-1), mer(+1)
    poz = list(np.linspace(-30, 30, n_pozic))
    zat = list(zl - zp)
    r = float(np.mean(zl - zp))          # kladné = pohyb doleva -> zatáčí doleva
    # NULL: přeházet přiřazení SMĚRŮ k podtypům T4/T5, ne sloupce.
    # (Chyba nalezená 13. 9. 2026: přeházení sloupců směr nezničí, protože
    # podnět je širokoúhlý — T4a dostane pořád směr „a", jen z jiných míst.)
    null = []
    puvodni = {hemi: dict(a.oci[hemi]["sousedi"]) for hemi in a.oci}
    rng = np.random.default_rng(0)
    for _ in range(n_null):
        for hemi in a.oci:
            klice = list(puvodni[hemi])
            perm = rng.permutation(klice)
            a.oci[hemi]["sousedi"] = {k: puvodni[hemi][p_] for k, p_ in zip(klice, perm)}
        null.append(abs(float(np.mean(mer(-1) - mer(+1)))))
    for hemi in a.oci:
        a.oci[hemi]["sousedi"] = puvodni[hemi]
    p95 = float(np.percentile(null, 95))
    log.info("H0: rozdíl zatáčení (pohyb doleva − doprava) = %+.3f | null p95 = %.3f -> %s",
             r, p95, "SPLNĚNO" if r > 0.15 and r > p95 else "NESPLNĚNO")
    return {"r": r, "null_p95": p95, "pozice": list(map(float, poz)), "zataceni": list(map(float, zat))}


def blok(a: Agent2, h: Hra, delka: int, rezim: str, rng, sila: float) -> float:
    """Jeden blok kroků + odložená odměna na konci. Vrací úspěšnost bloku."""
    stopa = {"vlevo": np.zeros(len(a.i_spike), np.float32),
             "vpravo": np.zeros(len(a.i_spike), np.float32)}
    uspech = 0
    pred = h.snimek()
    for _ in range(delka):
        s = h.snimek()
        k = a.act(s, pred); pred = s
        # eligibility trace: co bylo aktivní, když se zatáčelo na tu stranu
        strana = "vlevo" if k.zataceni > 0 else "vpravo"
        sp = k.spiky.astype(np.float32)
        stopa[strana] += sp / (sp.max() or 1.0)
        h.krok(k.zataceni)
        uspech += int(h.na_cili())
    if rezim != "sham":
        # odměna přijde až teď a hodnotí celý blok
        r = 2.0 * (uspech / delka) - 1.0            # v [-1, 1]
        if rezim == "placebo":
            r = float(rng.uniform(-1, 1))           # stejná velikost, bez vazby na výkon
        a.odmena_stopou(stopa, r, sila=sila)
    return uspech / delka


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bloku", type=int, default=20)
    p.add_argument("--delka", type=int, default=12)
    p.add_argument("--sila", type=float, default=0.04)
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    log.info("agent: graduovaná %d, spikující %d", len(a.i_graded), len(a.i_spike))

    h0 = h0_overeni(a)
    vysl = []
    for seed in args.seeds:
        for rezim in ("sham", "paired", "placebo"):
            a.reset_vahy()
            rng = np.random.default_rng(seed)
            h = Hra(seed)
            t0 = time.time(); krivka = []
            for b in range(args.bloku):
                h.reset()
                krivka.append(blok(a, h, args.delka, rezim, rng, args.sila))
            zac, kon = np.mean(krivka[:5]), np.mean(krivka[-5:])
            log.info("seed %d %-8s: %.2f -> %.2f (%+.1f p. b.)  [%.0f s]",
                     seed, rezim, zac, kon, 100 * (kon - zac), time.time() - t0)
            vysl.append({"seed": seed, "rezim": rezim, "krivka": krivka,
                         "zacatek": float(zac), "konec": float(kon), "prirustek": float(kon - zac)})

    def pr(r): return np.array([v["prirustek"] for v in vysl if v["rezim"] == r])
    log.info("=" * 70)
    for r in ("sham", "paired", "placebo"):
        log.info("  %-8s přírůstek: %s (medián %+.1f p. b.)", r,
                 np.round(100 * pr(r), 1), 100 * np.median(pr(r)))
    h1 = bool((pr("paired") >= 0.10).all())
    h2 = bool(np.median(pr("paired")) > np.median(pr("placebo")))
    h3 = bool(np.abs(pr("sham")).max() < 0.05)
    log.info("  H1 paired roste ≥10 p. b. ve všech losech : %s", "SPLNĚNO" if h1 else "NESPLNĚNO")
    log.info("  H2 paired > placebo                       : %s", "SPLNĚNO" if h2 else "NESPLNĚNO")
    log.info("  H3 sham se nezlepší                       : %s", "SPLNĚNO" if h3 else "NESPLNĚNO")
    verdikt = "Učí se i s odloženou odměnou" if (h1 and h2 and h3) else "ZÁPORNÝ/ČÁSTEČNÝ — viz kritéria"
    log.info("VERDIKT: %s", verdikt)
    OUT.mkdir(exist_ok=True)
    (OUT / "exp06_hra.json").write_text(json.dumps(
        {"verdikt": verdikt, "H0": h0, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
