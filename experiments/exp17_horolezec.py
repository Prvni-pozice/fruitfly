"""
exp17_horolezec.py — dokáže chování zlepšit VŮBEC nějaký optimalizátor?

Rozhoduje jedinou otevřenou otázku projektu. Šest pokusů o učení z odměny
selhalo, ale u širokého rozhraní (746 synapsí na DNa02) nešlo odlišit
„pravidlo selhává" od „úloha je nesplnitelná" — osm náhodných vzorků
v sedmi stech rozměrech nenajde nic.

Tenhle skript hledá cíleně: (1+1) evoluční strategie. Vezme nejlepší známé
váhy, mírně je rozhýbe, a když je výsledek lepší, ponechá je. Krok se
adaptuje podle úspěšnosti (pravidlo jedné pětiny).

Když horolezec vyšplhá, je úloha řešitelná a chyba je v našem učicím
pravidle. Když ani on nikam nedojde, je problém v zadání nebo v modelu.

Stav se průběžně ukládá, takže běh jde kdykoli přerušit a navázat —
prostředí tenhle stroj opakovaně zabíjí.

Použití:
    python experiments/exp17_horolezec.py --minut 8
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp13_proti_instinktu import miraKrok  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import Hra  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
STAV = ROOT / "outputs" / "exp17_stav.npz"
ZKOUSKA = list(range(1000, 1006))


def hodnot(a: Agent2, kroku: int, seedy=None) -> float:
    """Když se `seedy` nezadá, měří se na pevné sadě.

    POZOR: optimalizovat na pevné sadě znamená naučit se ji nazpaměť.
    Změřeno 14. 9. 2026: horolezec vyšplhal na trénovaných epizodách
    z 0,692 na 0,859, ale na nových jen z 0,700 na 0,702. Proto se
    v režimu --nahodne losují epizody pro každé hodnocení znovu.
    """
    soucet = n = 0.0
    for seed in (seedy if seedy is not None else ZKOUSKA):
        h = Hra(seed); pred = h.snimek()
        for _ in range(kroku):
            s = h.snimek()
            z = a.act(s, pred).zataceni
            pred = s
            h.krok(z)
            soucet += miraKrok(h); n += 1
    return soucet / n


def useky(a: Agent2):
    """Pozice v W_spike.data, které patří vstupům do DNa02."""
    poz = []
    for ix in (a.dna_l, a.dna_r):
        for r in ix:
            poz.append(np.arange(a.W_spike.indptr[r], a.W_spike.indptr[r + 1]))
    return np.concatenate(poz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minut", type=float, default=8.0)
    p.add_argument("--kroku", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--od-znovu", action="store_true")
    p.add_argument("--nahodne", action="store_true",
                   help="losovat epizody pro každé hodnocení znovu (nutí to hledat obecné řešení)")
    args = p.parse_args()

    a = Agent2(Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome"), steps=args.steps)
    poz = useky(a)
    zaklad = a.W0.data[poz].copy()
    log.info("optimalizuje se %d synapsí na DNa02", len(poz))

    if STAV.exists() and not args.od_znovu:
        d = np.load(STAV)
        nej_v, nej_skore, krok, iterace = d["vahy"], float(d["skore"]), float(d["krok"]), int(d["iterace"])
        log.info("navazuji: iterace %d, nejlepší %.4f, krok %.3f", iterace, nej_skore, krok)
    else:
        a.W_spike.data[poz] = zaklad
        nej_v, nej_skore, krok, iterace = zaklad.copy(), hodnot(a, args.kroku), 0.25, 0
        log.info("výchozí (vrozené zapojení): %.4f", nej_skore)

    historie = []
    rng = np.random.default_rng(iterace + 1)
    konec = time.time() + args.minut * 60
    uspechy = 0
    zacatek_skore = nej_skore
    while time.time() < konec:
        iterace += 1
        # v náhodném režimu se obě strany porovnávají na TÝCHŽ losech,
        # jinak by rozhodoval šum epizod, ne kvalita vah
        davka = (list(rng.integers(3000, 9000, size=6)) if args.nahodne else None)
        if args.nahodne:
            a.W_spike.data[poz] = nej_v
            nej_skore = hodnot(a, args.kroku, davka)
        kandidat = nej_v + krok * np.abs(zaklad) * rng.standard_normal(len(poz)).astype(np.float32)
        a.W_spike.data[poz] = kandidat
        v = hodnot(a, args.kroku, davka)
        if v > nej_skore:
            nej_v, nej_skore = kandidat, v
            uspechy += 1
            log.info("  iterace %3d: %.4f  <- lepší (krok %.3f)", iterace, v, krok)
        # pravidlo jedné pětiny: úspěšnost pod 20 % -> menší krok, nad -> větší
        if iterace % 20 == 0:
            # kontrola na NOVÝCH epizodách — jediné číslo, které něco znamená
            a.W_spike.data[poz] = nej_v
            val = hodnot(a, args.kroku, list(range(2000, 2008)))
            a.W_spike.data[poz] = a.W0.data[poz]
            zaklad_val = hodnot(a, args.kroku, list(range(2000, 2008)))
            log.info("  === iterace %3d: na nových epizodách %.4f proti výchozím %.4f (%+.1f p. b.)",
                     iterace, val, zaklad_val, 100 * (val - zaklad_val))
            historie.append([iterace, float(val), float(zaklad_val)])
        if iterace % 10 == 0:
            krok *= 1.3 if uspechy > 2 else 0.8
            krok = float(np.clip(krok, 0.02, 2.0))
            log.info("  iterace %3d: nejlepší %.4f, úspěchů %d/10, nový krok %.3f",
                     iterace, nej_skore, uspechy, krok)
            uspechy = 0

    np.savez(STAV, vahy=nej_v, skore=nej_skore, krok=krok, iterace=iterace)
    if historie:
        h = ROOT / "outputs" / "exp17_historie.json"
        stara = json.loads(h.read_text()) if h.exists() else []
        h.write_text(json.dumps(stara + historie, ensure_ascii=False))
    log.info("=" * 60)
    log.info("iterací celkem %d | nejlepší %.4f (na začátku tohoto běhu %.4f)",
             iterace, nej_skore, zacatek_skore)
    (ROOT / "outputs" / "exp17.json").write_text(json.dumps(
        {"iterace": iterace, "nejlepsi": nej_skore, "krok": krok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
