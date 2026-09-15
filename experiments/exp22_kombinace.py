"""
exp22_kombinace.py — jak daleko dojde zeslabení VÍC výstupů paměti najednou?

Exp21 testoval MBON po jednom: největší posun zatáčení byl 10 Hz při
výchozích 230 Hz, tedy páka 4 %. Kombinace ale testované nebyly a mohou
dát víc — buď proto, že se účinky sčítají, nebo proto, že se jich musí
vypnout víc, než se síť přepne do jiného režimu.

Měří se čtyři věci:
  1. horní mez: zeslabit VŠECH 84 najednou,
  2. náhodné podmnožiny různých velikostí,
  3. nejlepších k podle jednotlivého účinku (hladová aproximace),
  4. nejlepších k s OPAČNÝM znaménkem účinku (kdyby se rušily).

Smysl: když ani vypnutí všeho nepřesune zatáčení výrazně, je páka
paměťového centra na tohle chování malá bez ohledu na učicí pravidlo
a odměnové učení touhle cestou nejde. To je tvrdý strukturální závěr,
ne selhání konkrétního pokusu.
"""
from __future__ import annotations
import json, logging, sys
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nahodnych", type=int, default=12)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--zeslabeni", type=float, default=0.1)
    args = p.parse_args()

    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=args.steps)
    mb = MBDve(a, c)
    rng = np.random.default_rng(0)
    vz = {j: (mb.kc[rng.choice(len(mb.kc), size=int(0.05 * len(mb.kc)), replace=False)], 4.0)
          for j in PODNETY}
    klice = list(mb.useky)

    def zmer(sada) -> dict:
        a.reset_vahy()
        for k in sada:
            pos, _ = mb.useky[k]
            a.W_spike.data[pos] *= args.zeslabeni
        out = {j: zataceni(a, PODNETY[j], vz[j])[0] for j in PODNETY}
        a.reset_vahy()
        return out

    zaklad = zmer([])
    log.info("výchozí: vlevo %+.0f Hz, vpravo %+.0f Hz", zaklad["vlevo"], zaklad["vpravo"])

    vysl = []

    def uloz(jm, sada):
        v = zmer(sada)
        dl, dp = v["vlevo"] - zaklad["vlevo"], v["vpravo"] - zaklad["vpravo"]
        log.info("  %-28s (n=%2d): vlevo %+5.0f Hz, vpravo %+5.0f Hz", jm, len(sada), dl, dp)
        vysl.append({"jmeno": jm, "n": len(sada), "posun_vlevo": dl, "posun_vpravo": dp})
        return max(abs(dl), abs(dp))

    log.info("1) horní mez — všechny výstupy paměti najednou:")
    uloz("všech 84", klice)

    log.info("2) náhodné podmnožiny:")
    for n in (10, 20, 40, 60):
        for i in range(max(1, args.nahodnych // 4)):
            uloz(f"náhodných {n} (#{i})", list(rng.choice(klice, size=n, replace=False)))

    log.info("3) nejlepších k podle jednotlivého účinku:")
    jedn = []
    for k in klice:
        v = zmer([k])
        jedn.append((k, v["vlevo"] - zaklad["vlevo"]))
    serazene = [k for k, _ in sorted(jedn, key=lambda x: x[1])]
    for n in (5, 10, 20, 40):
        uloz(f"top {n} (posun doprava)", serazene[:n])
    for n in (5, 10, 20, 40):
        uloz(f"top {n} (posun doleva)", serazene[-n:])

    nej = max(vysl, key=lambda x: max(abs(x["posun_vlevo"]), abs(x["posun_vpravo"])))
    log.info("=" * 74)
    log.info("největší dosažený posun: %s -> vlevo %+.0f Hz, vpravo %+.0f Hz",
             nej["jmeno"], nej["posun_vlevo"], nej["posun_vpravo"])
    paka = max(abs(nej["posun_vlevo"]) / max(abs(zaklad["vlevo"]), 1),
               abs(nej["posun_vpravo"]) / max(abs(zaklad["vpravo"]), 1))
    log.info("páka paměťového centra na zatáčení: %.0f %% (po jednom to byla 4 %%)", 100 * paka)
    (ROOT / "outputs" / "exp22.json").write_text(
        json.dumps({"zaklad": zaklad, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
