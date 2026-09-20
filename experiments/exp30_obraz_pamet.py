"""
exp30_obraz_pamet.py — naučí se moucha rozpoznat OBRÁZEK?

PREREGISTRACE (zapsáno před během, 20. 9. 2026):
Exp29 podmínil sosák na náhodný vzorec Kenyonových buněk. Tady je vstupem
obraz (tmavý sloup v určité poloze), převedený na kód KC náhodnou projekcí
s výběrem 5 % (podminovani.py). Otázky: naučí se odpovídat na TENTO obraz,
zobecní na PODOBNÝ (sloup posunutý o 6°), a nezobecní na JINÝ (sloup na
druhé straně, vodorovný pruh)?

Nejdřív se změří překryv kódů — bez něj nemá smysl nic předpovídat:
  K0: překryv kódu podobného obrazu s trénovaným > 0,4; jiného < 0,15.
Pak podmiňování (20× obraz + cukr), 8 losů (los = jiná trénovaná poloha):
  H1: sham 0 ve všech losech.
  H2: odpověď na trénovaný obraz vzroste v ≥ 6/8.
  H3: podobný obraz vzroste aspoň o polovinu toho co trénovaný (medián
      poměru ≥ 0,5) — generalizace.
  H4: jiný obraz vzroste o méně než čtvrtinu trénovaného (medián ≤ 0,25)
      — diskriminace.
Vyvrácení: K0 padne -> kód nenese podobnost, obrazová paměť se nedá
testovat tímhle převodem. H2 padne -> obraz se nepodmíní.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from podminovani import Mozek  # noqa: E402
from server_mozku import snimek  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader", "podminovani"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def vodorovny(v=32):
    im = np.ones((v, v), dtype=np.float32); im[12:19, :] = 0.0; return im


def main():
    m = Mozek()
    polohy = [-36, -24, -12, 0, 12, 24, 36, 48]   # 8 losů = 8 trénovaných poloh
    log.info("K0 — překryv kódů (trénovaný vs podobný ±6°, jiný na druhé straně):")
    prek_pod, prek_jin = [], []
    for o in polohy:
        kt = m.kc_z_obrazu(snimek(o)); kp = m.kc_z_obrazu(snimek(o + 6)); kj = m.kc_z_obrazu(snimek(-o if o else 40))
        prek_pod.append(m.prekryv(kt, kp)); prek_jin.append(m.prekryv(kt, kj))
    log.info("  podobný medián %.2f | jiný medián %.2f | vodorovný pruh vs sloup %.2f",
             np.median(prek_pod), np.median(prek_jin), m.prekryv(m.kc_z_obrazu(snimek(0)), m.kc_z_obrazu(vodorovny())))
    k0 = np.median(prek_pod) > 0.4 and np.median(prek_jin) < 0.15
    log.info("  K0: %s", "SPLNĚNO" if k0 else "NESPLNĚNO")

    vysl = []
    for o in polohy:
        sady = {"trénovaný": m.kc_z_obrazu(snimek(o)), "podobný": m.kc_z_obrazu(snimek(o + 6)),
                "jiný": m.kc_z_obrazu(snimek(-o if o else 40)), "pruh": m.kc_z_obrazu(vodorovny())}
        for rezim in ("sham", "paired"):
            m.reset()
            pred = {k: m.odpoved(v) for k, v in sady.items()}
            if rezim == "paired":
                for _ in range(20):
                    m.odmen(sady["trénovaný"])
            po = {k: m.odpoved(v) for k, v in sady.items()}
            d = {k: po[k] - pred[k] for k in sady}
            log.info("poloha %+3d %-6s: trénovaný %+5.1f | podobný %+5.1f | jiný %+5.1f | pruh %+5.1f Hz",
                     o, rezim, d["trénovaný"], d["podobný"], d["jiný"], d["pruh"])
            vysl.append({"poloha": o, "rezim": rezim, **{f"d_{k}": float(v) for k, v in d.items()}})

    P = [x for x in vysl if x["rezim"] == "paired"]
    dt = np.array([x["d_trénovaný"] for x in P]); dp = np.array([x["d_podobný"] for x in P])
    dj = np.array([x["d_jiný"] for x in P]); dr = np.array([x["d_pruh"] for x in P])
    kladne = dt > 0
    pomer = lambda x: np.median(x[kladne] / dt[kladne]) if kladne.any() else float("nan")
    log.info("=" * 74)
    log.info("  trénovaný: medián %+.2f Hz, roste %d/8 | podobný/trénovaný %.2f | jiný/trénovaný %.2f | pruh/trénovaný %.2f",
             np.median(dt), int(kladne.sum()), pomer(dp), pomer(dj), pomer(dr))
    h1 = bool(np.abs([x["d_trénovaný"] for x in vysl if x["rezim"] == "sham"]).max() < 1e-9)
    h2 = int(kladne.sum()) >= 6
    h3 = pomer(dp) >= 0.5
    h4 = max(pomer(dj), pomer(dr)) <= 0.25
    for k, v in (("H1 měřidlo", h1), ("H2 trénovaný roste ≥6/8", h2), ("H3 generalizace ≥0,5", h3), ("H4 diskriminace ≤0,25", h4)):
        log.info("  %-28s: %s", k, "SPLNĚNO" if v else "NESPLNĚNO")
    verdikt = ("OBRAZOVÁ PAMĚŤ: rozpozná, zobecní i rozliší" if all((k0, h1, h2, h3, h4))
               else "Rozpozná, ale nezobecní nebo nerozliší" if (k0 and h1 and h2)
               else "ZÁPORNÝ" if k0 else "KÓD NENESE PODOBNOST — test neplatný")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp30.json").write_text(json.dumps({"verdikt": verdikt, "K0": [float(np.median(prek_pod)), float(np.median(prek_jin))], "vysledky": vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
