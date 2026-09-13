"""zaznam_hry.py — odehraje epizody a uloží průběh pro vizualizaci."""
from __future__ import annotations
import json, logging, sys
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


def epizoda(a: Agent2, seed: int, kroku: int = 40) -> list[dict]:
    h = Hra(seed); pred = h.snimek(); zaznam = []
    for i in range(kroku):
        s = h.snimek()
        k = a.act(s, pred); pred = s
        zaznam.append({"krok": i, "uhel": round(h.uhel, 2), "cil": round(h.cil, 2),
                       "odchylka": round(h.odchylka, 2), "zataceni": round(k.zataceni, 4),
                       "dna_l": k.dna02_l, "dna_r": k.dna02_r,
                       "na_cili": bool(h.na_cili())})
        h.krok(k.zataceni)
    return zaznam


def main():
    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    a = Agent2(c, steps=120)
    out = {"popis": "octomilka sleduje pohyblivý cíl; mozek z connectomu FlyWire",
           "epizody": {}}
    for seed in (0, 1, 2):
        z = epizoda(a, seed)
        out["epizody"][str(seed)] = z
        log.info("epizoda %d: na cíli %d/%d kroků", seed, sum(x["na_cili"] for x in z), len(z))
    (ROOT / "outputs" / "zaznam_hry.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info("uloženo")


if __name__ == "__main__":
    main()
