"""
exp28_mapa_vlivu.py — kam všude paměť dosáhne? Mapa vlivu MBON.

Otázka (19. 9. 2026): učení jsme dosud odečítali jen na zatáčení, které
řídí reflex (poloha podnětu vysvětluje 77 % zatáčení). Paměť ho možná
nepřehlasuje, i když sama funguje. Tady se změří, KTERÉ populace mozku se
pohnou, když se výstupy paměti (KC->MBON) zeslabí — a jak moc vůči své
výchozí aktivitě. Kde je páka velká, tam má smysl učení měřit.

Postup: pach (vzorec KC) -> aktivita všech neuronů; pak totéž po depresi
všech KC->MBON synapsí na desetinu. Pro každou populaci: výchozí Hz,
Hz po depresi, relativní změna. Šest různých pachů, medián.

Populace: motoneurony sosáku a polykání (jídlo), sestupné neurony po
rodinách (chůze, zatáčení, únik…), laterální roh, centrální komplex,
vzestupné neurony, MBON samy (kontrola, že deprese působí).
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp26_endogenni_odmena import Chut  # noqa: E402
from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from pravidlo_shiu import pouzij_shiu  # noqa: E402
from server_mozku import snimek  # noqa: E402
from sim2 import Simulator2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for jm in ("flybrain.simulator", "retina", "flybrain.loader"):
    logging.getLogger(jm).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
WS = 0.0393


def populace(a: Agent2, c: Connectome, cir: Circuits) -> dict[str, np.ndarray]:
    n = c.nodes; poz = {int(j): i for i, j in enumerate(a.i_spike)}
    pt = n["primary_type"].astype("string")

    def sel(mask):
        return np.array([poz[int(j)] for j in np.flatnonzero(mask) if int(j) in poz])
    pop = {
        "MBON (kontrola)": sel((n["class"] == "MBON").fillna(False).to_numpy()),
        "sosák MN (jídlo)": sel(((n["super_class"] == "motor") & (n["sub_class"] == "proboscis_motor_neuron")).fillna(False).to_numpy()),
        "polykání MN": sel(((n["super_class"] == "motor") & (n["sub_class"] == "ingestion_motor_neuron")).fillna(False).to_numpy()),
        "laterální roh (LHLN)": sel((n["class"] == "LHLN").fillna(False).to_numpy()),
        "centrální komplex (CX)": sel((n["class"] == "CX").fillna(False).to_numpy()),
        "vzestupné (AN)": sel((n["super_class"] == "ascending").fillna(False).to_numpy()),
        "DNa02 (zatáčení)": sel((n["primary_type"] == "DNa02").fillna(False).to_numpy()),
    }
    dn = (n["super_class"] == "descending").fillna(False).to_numpy()
    for pref, popis in (("DNa", "DNa (zatáčení/chůze)"), ("DNb", "DNb (chůze)"), ("DNg", "DNg (modulace)"),
                        ("DNp", "DNp (únik aj.)"), ("DNge", "DNge (ezofag.)"), ("MDN", "MDN (couvání)")):
        m = dn & pt.str.startswith(pref).fillna(False).to_numpy()
        if pref == "DNg":
            m &= ~pt.str.startswith("DNge").fillna(False).to_numpy()
        if pref == "DNp":
            m &= ~pt.str.startswith("DNpe").fillna(False).to_numpy()
        pop[popis] = sel(m)
    return {k: v for k, v in pop.items() if len(v)}


def main():
    c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
    pouzij_shiu(c)
    a = Agent2(c, steps=100); cir = Circuits(c); ch = Chut(a, c)
    pop = populace(a, c, cir)
    log.info("populací: %s", {k: len(v) for k, v in pop.items()})
    vsechny_pos = np.concatenate([pos for pos, _ in ch.useky.values()])

    def beh(kc):
        ext = a.vjem(snimek(-12.0), snimek(-24.0)); ext[kc[0]] += kc[1]
        res = Simulator2(a.W_spike, weight_scale=WS).run(a.steps, external=ext)
        return res.spike_counts.numpy() / res.duration_s

    zmeny = {k: [] for k in pop}; zaklady = {k: [] for k in pop}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        kc = (ch.kc[rng.choice(len(ch.kc), size=int(0.05 * len(ch.kc)), replace=False)], 4.0)
        a.reset_vahy(); r0 = beh(kc)
        a.W_spike.data[vsechny_pos] *= 0.1; r1 = beh(kc); a.reset_vahy()
        for k, ix in pop.items():
            b = float(r0[ix].mean()); zaklady[k].append(b)
            zmeny[k].append(float(r1[ix].mean() - b))

    log.info("=" * 78)
    log.info("%-26s %6s %10s %12s %10s", "populace", "n", "výchozí Hz", "Δ po depresi", "rel. změna")
    radky = []
    for k in pop:
        b = float(np.median(zaklady[k])); d = float(np.median(zmeny[k]))
        rel = abs(d) / b if b > 0 else float("nan")
        radky.append((k, len(pop[k]), b, d, rel))
    for k, n_, b, d, rel in sorted(radky, key=lambda r: -(r[4] if r[4] == r[4] else -1)):
        log.info("%-26s %6d %10.2f %+12.2f %9.0f %%", k, n_, b, d, 100 * rel if rel == rel else 0)
    (ROOT / "outputs" / "exp28.json").write_text(json.dumps(
        [{"populace": k, "n": n_, "vychozi": b, "delta": d, "rel": rel} for k, n_, b, d, rel in radky],
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
