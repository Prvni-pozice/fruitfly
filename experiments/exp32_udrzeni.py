"""
exp32_udrzeni.py — vydrží paměť další učení? A zapomíná?

PREREGISTRACE (zapsáno před během, 20. 9. 2026):
Fáze 1: pach A + cukr 20× -> test A.
Fáze 2: pach B + cukr 20× -> test A i B. Přepíše nová stopa starou?
Fáze 3: pach A sám 20× (vyhasínání) -> test A. V modelu není žádný
mechanismus zapomínání bez dopaminu, takže se předpovídá NULA změny —
je to vlastnost modelu, ne mouchy, a zapisuje se tak.

  H1: sham 0.
  H2: A po fázi 1 roste v ≥ 6/8 (replikace).
  H3: A po fázi 2 si drží aspoň polovinu (A2 ≥ 0,5 × A1) v ≥ 6/8 — stopy
      se nepřepisují.
  H4: B po fázi 2 roste v ≥ 6/8 — druhá stopa se přidá.
  H5: fáze 3 změní A o méně než 1 Hz v mediánu (žádné vyhasínání bez
      dopaminu — očekávaná vlastnost modelu).
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
    m = Mozek(); vysl = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        A, B = m.kc_nahodne(rng), m.kc_nahodne(rng)
        for rezim in ("sham", "paired"):
            m.reset()
            a0, b0 = m.odpoved(A), m.odpoved(B)
            if rezim == "paired":
                for _ in range(20): m.odmen(A)
            a1 = m.odpoved(A)
            if rezim == "paired":
                for _ in range(20): m.odmen(B)
            a2, b2 = m.odpoved(A), m.odpoved(B)
            for _ in range(20): m.beh(A)            # vyhasínání: A bez cukru
            a3 = m.odpoved(A)
            log.info("seed %d %-6s: A %5.1f -> po A %5.1f -> po B %5.1f -> po vyhasínání %5.1f | B %5.1f -> %5.1f",
                     seed, rezim, a0, a1, a2, a3, b0, b2)
            vysl.append({"seed": seed, "rezim": rezim, "dA1": a1 - a0, "dA2": a2 - a0, "dA3": a3 - a2, "dB": b2 - b0})

    P = [x for x in vysl if x["rezim"] == "paired"]
    dA1 = np.array([x["dA1"] for x in P]); dA2 = np.array([x["dA2"] for x in P])
    dA3 = np.array([x["dA3"] for x in P]); dB = np.array([x["dB"] for x in P])
    drzi = (dA2 >= 0.5 * dA1) & (dA1 > 0)
    log.info("=" * 70)
    log.info("  A po fázi 1: medián %+.2f, roste %d/8 | A po fázi 2: medián %+.2f, drží ≥½ v %d/8 | B: %+.2f, roste %d/8 | vyhasínání ΔA: %+.2f",
             np.median(dA1), int((dA1 > 0).sum()), np.median(dA2), int(drzi.sum()), np.median(dB), int((dB > 0).sum()), np.median(dA3))
    h1 = bool(max(abs(x["dA1"]) for x in vysl if x["rezim"] == "sham") < 1e-9)
    h2 = int((dA1 > 0).sum()) >= 6; h3 = int(drzi.sum()) >= 6; h4 = int((dB > 0).sum()) >= 6
    h5 = bool(abs(np.median(dA3)) < 1.0)
    for k, v in (("H1 měřidlo", h1), ("H2 A se naučí", h2), ("H3 A vydrží učení B", h3), ("H4 B se přidá", h4), ("H5 bez dopaminu nezapomíná", h5)):
        log.info("  %-28s: %s", k, "SPLNĚNO" if v else "NESPLNĚNO")
    verdikt = ("PAMĚŤ DRŽÍ: dvě stopy vedle sebe, bez dopaminu se nemění" if all((h1, h2, h3, h4, h5))
               else "Nová stopa přepisuje starou" if (h1 and h2 and h4 and not h3)
               else "ČÁSTEČNÝ — viz kritéria")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp32.json").write_text(json.dumps({"verdikt": verdikt, "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
