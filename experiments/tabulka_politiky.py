"""
tabulka_politiky.py — předpočítá rozhodnutí mozku do tabulky.

Mozek je deterministický: na stejný vjem odpoví vždy stejně. Jeho politiku
jde proto vyčerpávajícím způsobem otabelovat a hra pak běží přímo
v prohlížeči bez serveru — a přesto ji řídí skutečná rozhodnutí
connectomu, ne napsaná pravidla.

Vjem má dva rozměry: kde cíl JE (odchylka od středu pohledu) a kam se
POSUNUL od minulého snímku. Druhý rozměr je nutný, protože moucha je
detektor pohybu a na nehybné scéně vydá přesnou nulu.
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from server_mozku import snimek  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Mřížka je záměrně hrubá a prohlížeč mezi body dopočítá lineárně.
# Krok 3° se ukázal jako neúnosný: čím blíž je cíl středu, tím víc neuronů
# střílí a tím pomalejší je simulace — 27 s na řádek u okraje, ale 497 s
# ve čtvrtině cesty ke středu. Celkem přes hodinu, a ta přesnost by ve hře
# stejně nebyla vidět.
ODCHYLKY = np.arange(-60, 61, 6.0)      # kde cíl je (21 hodnot)
POSUNY = np.arange(-24, 25, 6.0)        # o kolik se posunul (9 hodnot)


def main():
    a = Agent2(Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome"), steps=100)
    tab = np.zeros((len(ODCHYLKY), len(POSUNY)), dtype=np.float32)
    dna = np.zeros((len(ODCHYLKY), len(POSUNY), 2), dtype=np.float32)
    celkem = tab.size
    t0 = time.time()
    for i, o in enumerate(ODCHYLKY):
        for j, d in enumerate(POSUNY):
            k = a.act(snimek(float(o)), snimek(float(o - d)))
            tab[i, j] = k.zataceni
            dna[i, j] = (k.dna02_l, k.dna02_r)
        hotovo = (i + 1) * len(POSUNY)
        log.info("odchylka %+.0f° hotova (%d/%d, %.0f s, zbývá ~%.0f min)",
                 o, hotovo, celkem, time.time() - t0,
                 (celkem - hotovo) * (time.time() - t0) / hotovo / 60)
    out = {
        "popis": "Rozhodnutí mozku octomilky (FlyWire connectome) otabelovaná "
                 "pro každou kombinaci polohy a pohybu cíle.",
        "odchylky": ODCHYLKY.tolist(),
        "posuny": POSUNY.tolist(),
        "zataceni": np.round(tab, 4).tolist(),
        "dna": np.round(dna, 1).tolist(),
        "kroku_simulace": 100,
        "neuronu": {"spikujicich": len(a.i_spike), "graduovanych": len(a.i_graded)},
    }
    cesta = ROOT.parent / "web" / "src" / "data" / "politika.json"
    cesta.write_text(json.dumps(out, ensure_ascii=False))
    log.info("uloženo -> %s (%.0f kB, %.1f min)", cesta,
             cesta.stat().st_size / 1024, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
