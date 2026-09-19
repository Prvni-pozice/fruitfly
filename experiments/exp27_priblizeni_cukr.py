"""
exp27_priblizeni_cukr.py — cukr za přiblížení: úloha, kterou paměť umí.

PREREGISTRACE (zapsáno před během, 19. 9. 2026):
Exp26 ukázal, že houbovité tělísko nevybírá vlevo/vpravo: dopaminová
deprese umí vliv MBON jen ODEBRAT, ne přidat, takže odměna za „doleva"
posunula zatáčení doprava (−20, −10 Hz). U skutečné mouchy dělá paměť
PŘIBLÍŽENÍ a VYHÝBÁNÍ: MBON ve výchozím stavu ženou k vyhýbání a odměna
(PAM) je zeslabí -> přiblížení. Tahle úloha je s tím sladěná.

Moucha běhá volně ve hře s „pachem" (vzorec KC) A. Kdykoli má cíl blízko
(|odchylka| < 20°), dosáhla na jídlo: dostane CUKR na chuťové neurony,
cukr přes zapojení rozsvítí PAM a deprese KC->MBON se řídí skutečným
dopaminem (exp26). Odečet = průměrná vzdálenost cíle při volném běhu
(exp24), s pachem A (trénovaný) i B (netrénovaný).

Explorace přichází sama: cíl se hýbe a moucha zatáčí, takže se stav mění
a odměna občas nastane i bez umělého šumu — počet odměn se loguje a když
je u nějakého losu 0, ten los je neplatný.

  H1: měřidlo drží (sham 0 ve všech losech).
  H2: u pachu A se průměrná vzdálenost ZMENŠÍ (přiblížení) v ≥ 3/4 losů
      s aspoň jednou odměnou.
  H3: u pachu B se vzdálenost změní o méně než u A (specificita).
  H4: placebo (cukr v náhodných krocích) přiblížení v < 3/4 losů.
Vyvrácení: H2 padne -> ani úloha sladěná s paměťí a odměna z chuti
nezmění chování; učení v tomhle modelu končí.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp26_endogenni_odmena import Chut  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402
from pravidlo_shiu import pouzij_shiu  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393
VYSTUP = ROOT / "outputs" / "exp27.json"
BLIZKO = 20.0


def prubeh(ch: Chut, h: Hra, s: np.ndarray, pred: np.ndarray, kc, s_cukrem: bool):
    a = ch.a
    ext = a.vjem(s, pred)
    ext[kc[0]] += kc[1]
    if s_cukrem:
        ext[ch.cukr] += 3.0
    res = Simulator2(a.W_spike, weight_scale=WS).run(a.steps, external=ext)
    sc = res.spike_counts.numpy().astype(np.float32); dur = res.duration_s
    l = sc[a.dna_l].sum() / dur; p = sc[a.dna_r].sum() / dur
    return ((l - p) / (l + p) if (l + p) > 0 else 0.0), sc


def epizoda(ch, seed, kc, kroku, rezim=None, rng=None, sila=0.5):
    """Volný běh. Když rezim je dán, trénuje se: cukr za blízkost (paired),
    náhodně (placebo), nikdy (sham). Vrací průměrnou vzdálenost a počet odměn."""
    h = Hra(seed); pred = h.snimek(); soucet = 0.0; odmen = 0
    for _ in range(kroku):
        s = h.snimek()
        z, _ = prubeh(ch, h, s, pred, kc, False)
        h.krok(z)
        soucet += abs(h.odchylka)
        if rezim in ("paired", "placebo"):
            dej = (abs(h.odchylka) < BLIZKO) if rezim == "paired" else bool(rng.random() < 0.35)
            if dej:
                odmen += 1
                _, sc = prubeh(ch, h, h.snimek(), s, kc, True)   # ochutná cukr tam, kde je
                ch.uc_z_chuti(sc, sila)
        pred = s
    return soucet / kroku, odmen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tren-epizod", type=int, default=2)
    p.add_argument("--tren-kroku", type=int, default=25)
    p.add_argument("--test-epizod", type=int, default=2)
    p.add_argument("--test-kroku", type=int, default=10)
    p.add_argument("--sila", type=float, default=0.5)
    p.add_argument("--amp-kc", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--pridat", action="store_true")
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
                  for j in ("A", "B")}
            for rezim in ("sham", "paired", "placebo"):
                a.reset_vahy()
                test = lambda: {j: float(np.mean([epizoda(ch, 500 + i, vz[j], args.test_kroku)[0]
                                                  for i in range(args.test_epizod)])) for j in vz}
                pred = test()
                odmen = sum(epizoda(ch, 100 + seed * 10 + i, vz["A"], args.tren_kroku, rezim, rng, args.sila)[1]
                            for i in range(args.tren_epizod))
                po = test()
                dA, dB = po["A"] - pred["A"], po["B"] - pred["B"]
                log.info("seed %2d %-7s: A %.1f° -> %.1f° (%+.1f) | B %+.1f | odměn %d",
                         seed, rezim, pred["A"], po["A"], dA, dB, odmen)
                vysl.append({"seed": seed, "rezim": rezim, "dA": float(dA), "dB": float(dB), "odmen": int(odmen)})
            VYSTUP.write_text(json.dumps({"vysledky": vysl}, indent=2, ensure_ascii=False))

    def v(r, k): return np.array([x[k] for x in vysl if x["rezim"] == r])
    platne = [x for x in vysl if x["rezim"] == "paired" and x["odmen"] > 0]
    n = len({x["seed"] for x in vysl}); npl = len(platne)
    dA = np.array([x["dA"] for x in platne]); dB = np.array([x["dB"] for x in platne])
    dPl = v("placebo", "dA")
    log.info("=" * 74)
    log.info("  sham A: %s", np.round(v("sham", "dA"), 2))
    log.info("  paired (losy s odměnou %d/%d): A medián %+.2f°, přiblížení v %d/%d | B medián %+.2f°",
             npl, n, np.median(dA) if npl else 0, int((dA < 0).sum()), npl, np.median(dB) if npl else 0)
    log.info("  placebo: A medián %+.2f°, přiblížení v %d/%d", np.median(dPl) if len(dPl) else 0, int((dPl < 0).sum()), len(dPl))
    h1 = bool(len(v("sham", "dA")) and np.abs(v("sham", "dA")).max() < 1e-9)
    h2 = npl > 0 and int((dA < 0).sum()) >= int(np.ceil(0.75 * npl))
    h3 = npl > 0 and bool(np.median(np.abs(dA)) > np.median(np.abs(dB)))
    h4 = int((dPl < 0).sum()) < int(np.ceil(0.75 * max(len(dPl), 1)))
    for k, val in (("H1 měřidlo drží", h1), ("H2 přiblížení k A v ≥3/4 platných losů", h2),
                   ("H3 specifické (A > B)", h3), ("H4 placebo pod prahem", h4)):
        log.info("  %-42s: %s", k, "SPLNĚNO" if val else "NESPLNĚNO")
    verdikt = ("Cukr za přiblížení učí přiblížení" if all((h1, h2, h3, h4))
               else "Přiblížení dělá i placebo — cukr sám, ne odměna" if (h1 and h2 and h3)
               else "ZÁPORNÝ: ani sladěná úloha s chutí cukru chování nezmění")
    log.info("VERDIKT (%d losů): %s", n, verdikt)
    VYSTUP.write_text(json.dumps({"verdikt": verdikt, "losu": n, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
