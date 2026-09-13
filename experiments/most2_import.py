"""
most2_import.py — širokoúhlý pohyb -> flyvis -> naše T4/T5 -> HS -> DNa02.

PREREGISTRACE (zapsáno před během, 13. 9. 2026):
Oprava návrhu po záporném běhu `most_flyvis_import.py`. Tam byl podnět úzký
pruh a odečet hrubá bilance 1 290 sestupných neuronů. Mouší zatáčení ale
pohání širokoúhlý pohyb, který čtou HS buňky lobula plate, a ty jdou přímo
na DNa02 — v datech: T4→HS 17 342 synapsí, HS→DNa02 rozdělené po stranách
(18/38/14 na jeden, 54/29/24 na druhý).

  N1: DNa02 vlevo vs vpravo se mezi 0° a 180° přehodí (znaménko rozdílu
      se otočí) u OBOU intenzit.
  N2: |rozdíl| nad 95. percentilem nullu z 20 přehozených párování sloupců.
  N3: HS buňky samy vykazují směrovou preferenci (kontrola, že signál
      dorazil až k nim).
Vyvrácení: když ani DNa02 ani HS směr nerozliší, nenese ho už vstup z flyvis
a problém je v mostu, ne v odečtu.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from flybrain.loader import Connectome  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393
TYPY = [f"T4{s}" for s in "abcd"] + [f"T5{s}" for s in "abcd"]
OUT = ROOT / "outputs"


def main():
    data = np.load(OUT / "flyvis_edge.npz")
    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    n = c.nodes; pt = n["primary_type"].astype("string")
    hs = np.flatnonzero(pt.str.startswith("HS").fillna(False).to_numpy())
    dna = np.flatnonzero((n["primary_type"] == "DNa02").fillna(False).to_numpy())
    A = c.syn_counts.tocsr()
    # která DNa02 patří ke které HS (podle synapsí) -> strany
    vaz = A[dna][:, hs].toarray()
    log.info("HS: %d, DNa02: %d, vazba HS->DNa02:\n%s", len(hs), len(dna), vaz.astype(int))

    ca = pd.read_csv(ROOT.parent / "flybrain" / "data" / "column_assignment.csv.gz")
    rid2idx = {r: i for i, r in enumerate(c.root_ids)}
    W = c.signed_weights().tocsr().astype(np.float32); N = W.shape[0]

    par = {}
    for hemi in ("left", "right"):
        d0 = ca[ca["hemisphere"] == hemi]
        for T in TYPY:
            d = d0[d0["type"] == T]
            idx = np.array([rid2idx[r] for r in d["root_id"] if r in rid2idx])
            pq = d[["p", "q"]].values.astype(float)[: len(idx)]
            uv = data[f"0|{T}|uv"]
            a = (pq - pq.min(0)) / (pq.max(0) - pq.min(0))
            b = (uv - uv.min(0)) / (uv.max(0) - uv.min(0))
            if hemi == "right":       # druhé oko vidí zrcadlově
                a[:, 0] = 1 - a[:, 0]
            par[(hemi, T)] = (idx, cKDTree(b).query(a)[1])

    gmax = {T: max(float(np.clip(data[f"{i}|{T}|akt"], 0, None).max()) for i in range(4)) or 1.0
            for T in TYPY}

    def budic(i, perm=None):
        ext = np.zeros(N, dtype=np.float32)
        for (hemi, T), (idx, j) in par.items():
            akt = np.clip(data[f"{i}|{T}|akt"], 0, None) / gmax[T]
            ext[idx] = 3.0 * akt[perm[(hemi, T)] if perm else j]
        return ext

    def beh(ext):
        rt = Simulator2(W, weight_scale=WS).run(400, external=ext).rates_hz().numpy()
        return rt[dna], rt[hs], float((rt > 0).mean())

    vysl = {}
    for i in range(4):
        d, h, akt = beh(budic(i))
        vysl[i] = (d, h)
        log.info("podnět %d: DNa02 %s | HS %s | aktivních %.2f %%",
                 i, np.round(d, 2), np.round(h, 2), 100 * akt)

    def rozdil_dna(a, b): return float((vysl[a][0][0] - vysl[a][0][1]) - (vysl[b][0][0] - vysl[b][0][1]))
    rt_, rs_ = rozdil_dna(0, 2), rozdil_dna(1, 3)
    log.info("-" * 70)
    log.info("DNa02 (levá−pravá), rozdíl 0°−180°: tmavá hrana %+.3f, světlá %+.3f", rt_, rs_)

    rng = np.random.default_rng(0); null = []
    for _ in range(20):
        perm = {k: rng.permutation(v[1]) for k, v in par.items()}
        a = beh(budic(0, perm))[0]; b = beh(budic(2, perm))[0]
        null.append(float((a[0] - a[1]) - (b[0] - b[1])))
    null = np.abs(np.array(null)); prah = float(np.percentile(null, 95))
    log.info("null z 20 přehození: medián %.3f, p95 %.3f, max %.3f", np.median(null), prah, null.max())

    hs_rozdil = float(np.abs(vysl[0][1] - vysl[2][1]).max())
    n1 = np.sign(rt_) == np.sign(rs_) and min(abs(rt_), abs(rs_)) > 0
    n2 = abs(rt_) > prah
    n3 = hs_rozdil > 0.5
    log.info("  N1 DNa02 se přehodí u obou intenzit : %s", "SPLNĚNO" if n1 else "NESPLNĚNO")
    log.info("  N2 nad 95. percentilem nullu        : %s (%.3f vs %.3f)",
             "SPLNĚNO" if n2 else "NESPLNĚNO", abs(rt_), prah)
    log.info("  N3 HS rozlišují směr                : %s (max rozdíl %.2f Hz)",
             "SPLNĚNO" if n3 else "NESPLNĚNO", hs_rozdil)
    verdikt = "Most nese směr až na DNa02" if (n1 and n2 and n3) else "ZÁPORNÝ/ČÁSTEČNÝ — viz kritéria"
    log.info("VERDIKT: %s", verdikt)
    (OUT / "most2.json").write_text(json.dumps(
        {"verdikt": verdikt, "dna_rozdil_tmava": rt_, "dna_rozdil_svetla": rs_,
         "null_p95": prah, "hs_rozdil": hs_rozdil}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
