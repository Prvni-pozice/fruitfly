"""
exp31_tri_pameti.py — tři vstupy, tři různé osudy: odměna, trest, nic.

PREREGISTRACE (zapsáno před během, 20. 9. 2026):
Exp29 doložil jednu paměť (pach A + cukr -> sosák). Skutečná paměť musí
unést víc stop najednou a s různým znaménkem. Tři vzorce KC:
  A — párováno s CUKREM (PAM, odměna)
  B — párováno s HOŘKOU chutí (PPL1, trest)
  C — podáváno samo (neutrální kontrola)
prokládaně 20 kol. Odečet: sosák na každý vzorec sám, před a po.
Nejdřív se ověří, že hořká chuť vůbec rozsvítí PPL1 (log); když ne, trest
se v modelu nedá vyrobit z chuti a rameno B je neplatné.

  H1: sham 0 ve všech losech.
  H2: A roste v ≥ 6/8 (replikace exp29).
  H3: B klesne nebo se nezmění: medián ΔB ≤ 0 a ΔB < ΔA v ≥ 6/8.
  H4: C v mediánu |ΔC| < |ΔA| / 2 (neutrální se nemění).
Vyvrácení: H3 padne -> trest z hořké chuti sosák neubere, paměť umí jen
jedno znaménko. H4 padne -> paměť není specifická při třech stopách.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from podminovani import Mozek  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader", "podminovani"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def main():
    m = Mozek()
    # předběžně: rozsvítí hořká chuť PPL1?
    _, sc = m.beh(None, horka=True)
    ppl1_h = float(sc[m.ppl1].sum()); _, sc0 = m.beh(None); ppl1_0 = float(sc0[m.ppl1].sum())
    log.info("hořká chuť -> PPL1 spiky: %.0f (bez podnětu %.0f) -> %s", ppl1_h, ppl1_0,
             "trest z chuti JE k dispozici" if ppl1_h > ppl1_0 else "trest z chuti NENÍ (rameno B neplatné)")

    vysl = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        vz = {j: m.kc_nahodne(rng) for j in "ABC"}
        for rezim in ("sham", "paired"):
            m.reset()
            pred = {j: m.odpoved(vz[j]) for j in vz}
            if rezim == "paired":
                for _ in range(20):
                    m.odmen(vz["A"]); m.potrestej(vz["B"]); m.beh(vz["C"])
            po = {j: m.odpoved(vz[j]) for j in vz}
            d = {j: po[j] - pred[j] for j in vz}
            log.info("seed %d %-6s: A (cukr) %+5.1f | B (hořká) %+5.1f | C (nic) %+5.1f Hz", seed, rezim, d["A"], d["B"], d["C"])
            vysl.append({"seed": seed, "rezim": rezim, **{f"d{j}": float(v) for j, v in d.items()}})

    P = [x for x in vysl if x["rezim"] == "paired"]
    dA = np.array([x["dA"] for x in P]); dB = np.array([x["dB"] for x in P]); dC = np.array([x["dC"] for x in P])
    log.info("=" * 70)
    log.info("  A medián %+.2f (roste %d/8) | B medián %+.2f (B<A v %d/8) | C medián %+.2f",
             np.median(dA), int((dA > 0).sum()), np.median(dB), int((dB < dA).sum()), np.median(dC))
    h1 = bool(max(abs(x["dA"]) for x in vysl if x["rezim"] == "sham") < 1e-9)
    h2 = int((dA > 0).sum()) >= 6
    h3 = np.median(dB) <= 0 and int((dB < dA).sum()) >= 6
    h4 = bool(np.median(np.abs(dC)) < np.median(np.abs(dA)) / 2)
    for k, v in (("H1 měřidlo", h1), ("H2 A roste ≥6/8", h2), ("H3 B klesá / < A", h3), ("H4 C se nemění", h4)):
        log.info("  %-22s: %s", k, "SPLNĚNO" if v else "NESPLNĚNO")
    verdikt = ("TŘI PAMĚTI: odměna zvedá, trest sráží, neutrální stojí" if all((h1, h2, h3, h4))
               else "Odměna funguje, trest ne (jen jedno znaménko)" if (h1 and h2 and h4)
               else "Odměna funguje, ale stopy se míchají" if (h1 and h2)
               else "ZÁPORNÝ")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp31.json").write_text(json.dumps({"verdikt": verdikt, "ppl1_horka": ppl1_h, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
