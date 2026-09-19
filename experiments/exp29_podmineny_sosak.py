"""
exp29_podmineny_sosak.py — klasické podmiňování: pach + cukr -> pach sám
vysune sosák.

PREREGISTRACE (zapsáno před během, 19. 9. 2026):
Mapa vlivu (exp28) ukázala, kam paměť dosáhne: na motoneurony sosáku
(24 %) a polykání (26 %), ne na zatáčení (2 %). A se správným znaménkem —
zeslabení MBON krmení UVOLNÍ, protože MBON ho ve výchozím stavu brzdí.
To je přesně mechanismus apetitivního podmiňování octomilky (Tempel 1983,
Kirkhart & Scott 2015): pach spárovaný s cukrem začne sám vyvolávat
vysunutí sosáku (PER).

Postup: pach A (vzorec KC) se párově podává s cukrem na chuťové neurony;
cukr přes zapojení rozsvítí PAM a deprese KC->MBON se řídí skutečným
dopaminem (exp26). Test: pach A SÁM, bez cukru — frekvence motoneuronů
sosáku před a po. Pach B nikdy s cukrem nebyl (specificita).

  H1: měřidlo drží (sham 0 ve všech losech).
  H2: pach A po tréninku zvýší odpověď sosáku (Δ > 0) v ≥ 9 z 12 losů.
  H3: Δ u pachu A je v mediánu aspoň o 2 Hz větší než u pachu B.
  H4: nepárové rameno (cukr bez pachu, „unpaired") zvýší odpověď na A
      v < 9 z 12 losů — jinak to dělá cukr sám, ne asociace.
Vyvrácení: H2 padne -> ani na výstupu s největší pákou se učení neprojeví.
Když projde H2 i H3, ale H4 ne, je to senzitizace cukrem, ne paměť pachu.

Kenyonovy buňky se budí přímo (model je neprotlačí, exp18) — zapsáno.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp26_endogenni_odmena import Chut  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from pravidlo_shiu import pouzij_shiu  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393
VYSTUP = ROOT / "outputs" / "exp29.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--kol", type=int, default=20)
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
        a = Agent2(c, steps=args.steps); cir = Circuits(c); ch = Chut(a, c)
        poz = {int(j): i for i, j in enumerate(a.i_spike)}
        sosak = np.array([poz[int(j)] for j in cir.proboscis_motor_neurons().indices if int(j) in poz])
        N = len(a.i_spike)

        from server_mozku import snimek
        scena = a.vjem(snimek(-12.0), snimek(-24.0))   # táž scéna jako v mapě vlivu

        def beh(kc=None, cukr=False, kroku=None):
            """Se zrakovou scénou v pozadí. Bez ní klesne sosák na 0,8–2,9 Hz
            (2–7 spiků za měření) a odečet je na podlaze — první běh 19. 9.
            2026 tím pádem neměřil nic. V mapě vlivu (exp28) byl se scénou
            výchozí sosák 19,8 Hz a páka paměti 24 %. Testy navíc běží
            200 kroků, aby se odečet nekvantoval na jednotlivé spiky."""
            ext = scena.copy()
            if kc is not None:
                ext[kc[0]] += kc[1]
            if cukr:
                ext[ch.cukr] += 3.0
            res = Simulator2(a.W_spike, weight_scale=WS).run(kroku or a.steps, external=ext)
            sc = res.spike_counts.numpy().astype(np.float32)
            return float(sc[sosak].sum() / res.duration_s / len(sosak)), sc

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            vz = {j: (ch.kc[rng.choice(len(ch.kc), size=int(0.05 * len(ch.kc)), replace=False)], args.amp_kc)
                  for j in ("A", "B")}
            for rezim in ("sham", "paired", "unpaired"):
                a.reset_vahy()
                pred = {j: beh(vz[j], kroku=200)[0] for j in vz}
                dopamin = 0.0
                for _ in range(args.kol):
                    if rezim == "paired":
                        _, sc = beh(vz["A"], cukr=True)          # pach A + cukr zároveň
                        dopamin += ch.uc_z_chuti(sc, args.sila)
                    elif rezim == "unpaired":
                        beh(vz["A"])                              # pach sám
                        _, sc = beh(None, cukr=True)             # cukr sám, jindy
                        dopamin += ch.uc_z_chuti(sc, args.sila)
                po = {j: beh(vz[j], kroku=200)[0] for j in vz}
                dA, dB = po["A"] - pred["A"], po["B"] - pred["B"]
                log.info("seed %2d %-8s: sosák na A %.1f -> %.1f Hz (%+.1f) | na B %+.1f | dopamin %.0f",
                         seed, rezim, pred["A"], po["A"], dA, dB, dopamin)
                vysl.append({"seed": seed, "rezim": rezim, "predA": pred["A"], "dA": float(dA),
                             "dB": float(dB), "dopamin": float(dopamin)})
            VYSTUP.write_text(json.dumps({"vysledky": vysl}, indent=2, ensure_ascii=False))

    def v(r, k): return np.array([x[k] for x in vysl if x["rezim"] == r])
    n = len({x["seed"] for x in vysl}); prah = int(np.ceil(0.75 * n))
    dA, dB, dU = v("paired", "dA"), v("paired", "dB"), v("unpaired", "dA")
    log.info("=" * 74)
    log.info("  sham A: %s", np.round(v("sham", "dA"), 2))
    log.info("  paired:   A medián %+.2f Hz, roste v %d/%d | B medián %+.2f Hz", np.median(dA), int((dA > 0).sum()), n, np.median(dB))
    log.info("  unpaired: A medián %+.2f Hz, roste v %d/%d", np.median(dU), int((dU > 0).sum()), n)
    h1 = bool(len(v("sham", "dA")) and np.abs(v("sham", "dA")).max() < 1e-9)
    h2 = int((dA > 0).sum()) >= prah
    h3 = bool((np.median(dA) - np.median(dB)) >= 2.0)
    h4 = int((dU > 0).sum()) < prah
    for k, val in (("H1 měřidlo drží", h1), (f"H2 sosák na A roste v ≥{prah}/{n}", h2),
                   ("H3 A − B ≥ 2 Hz", h3), ("H4 unpaired pod prahem", h4)):
        log.info("  %-36s: %s", k, "SPLNĚNO" if val else "NESPLNĚNO")
    verdikt = ("PODMÍNĚNÝ SOSÁK: pach spárovaný s cukrem sám vyvolá krmení" if all((h1, h2, h3, h4))
               else "Senzitizace cukrem, ne asociace s pachem" if (h1 and h2 and h3)
               else "Roste, ale nespecificky" if (h1 and h2)
               else "ZÁPORNÝ: ani sosák se nepodmíní")
    log.info("VERDIKT (%d losů): %s", n, verdikt)
    VYSTUP.write_text(json.dumps({"verdikt": verdikt, "losu": n, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
