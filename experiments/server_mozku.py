"""
server_mozku.py — mozek jako služba: stav hry dovnitř, akce ven.

Hra běží v prohlížeči, mozek na tomhle stroji. Prohlížeč pošle, kde je
moucha natočená a kde je cíl; server z toho vykreslí, co moucha vidí,
protáhne to celou architekturou (retina -> detektor pohybu -> graduovaná
vrstva -> spikující centrální mozek) a vrátí zatáčení plus frekvence
neuronů DNa02.

Bez závislostí navíc, jen standardní knihovna — na VPS nejde instalovat
systémové balíčky a tohle je dost.

Spuštění:
    flybrain/.venv/bin/python experiments/server_mozku.py --port 5188
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from hra import VELIKOST, ZORNE_POLE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

AGENT: Agent2 | None = None
ZAMEK = threading.Lock()          # mozek je jeden, požadavky se řadí
POVOLENE = "*"                    # web běží na jiné doméně (Vercel)


def snimek(odchylka: float, sirka: float = 14.0) -> np.ndarray:
    """Tmavý sloup na světlém poli podle odchylky cíle od středu pohledu."""
    im = np.ones((VELIKOST, VELIKOST), dtype=np.float32)
    if abs(odchylka) <= ZORNE_POLE / 2:
        stred = (odchylka / ZORNE_POLE + 0.5) * (VELIKOST - 1)
        p = sirka / ZORNE_POLE * VELIKOST / 2
        lo, hi = int(np.clip(stred - p, 0, VELIKOST)), int(np.clip(stred + p, 0, VELIKOST))
        im[:, lo:hi] = 0.0
    return im


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _odpoved(self, kod: int, telo: dict) -> None:
        data = json.dumps(telo, ensure_ascii=False).encode()
        self.send_response(kod)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", POVOLENE)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", POVOLENE)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/zdravi"):
            self._odpoved(200, {"stav": "ok", "neuronu_spikujicich": len(AGENT.i_spike),
                                "neuronu_graduovanych": len(AGENT.i_graded)})
        else:
            self._odpoved(404, {"chyba": "neznámá cesta"})

    def do_POST(self):
        if not self.path.startswith("/akce"):
            return self._odpoved(404, {"chyba": "neznámá cesta"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            vstup = json.loads(self.rfile.read(n) or b"{}")
            odch = float(vstup.get("odchylka", 0.0))
            odch_pred = float(vstup.get("odchylka_pred", odch))
        except Exception as e:
            return self._odpoved(400, {"chyba": f"špatný vstup: {e}"})

        t0 = time.time()
        with ZAMEK:
            k = AGENT.act(snimek(odch), snimek(odch_pred))
        self._odpoved(200, {
            "zataceni": round(k.zataceni, 4),
            "dna_vlevo": round(k.dna02_l, 1),
            "dna_vpravo": round(k.dna02_r, 1),
            "ms": round(1000 * (time.time() - t0)),
        })

    def log_message(self, *a):    # ticho, vlastní logování výš
        pass


def main():
    global AGENT
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5188)
    p.add_argument("--steps", type=int, default=100, help="kroků simulace na jednu akci")
    args = p.parse_args()

    log.info("načítám connectome ...")
    AGENT = Agent2(Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome"),
                   steps=args.steps)
    log.info("mozek připraven: %d spikujících, %d graduovaných neuronů",
             len(AGENT.i_spike), len(AGENT.i_graded))
    t0 = time.time(); AGENT.act(snimek(10), snimek(0)); log.info("první akce za %.2f s", time.time() - t0)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
