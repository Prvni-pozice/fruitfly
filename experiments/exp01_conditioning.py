"""
exp01_conditioning.py — propíše se naučená změna v houbovitém tělísku až do
motorického výstupu přes zbytek skutečného connectomu?

Design, predikce a kritéria vyvrácení: EXP01-PREREGISTRACE.md (zapsáno PŘED
během). Tenhle skript predikce jen vyhodnotí, nemění je.

Rozdíl proti upstream scripts/add_plasticity.py: tam se odečítá přímo MBON,
tedy vrstva, do které se deprese ručně vkládá. Tady se odečítá až MOTORICKÝ
výstup po propagaci celým connectomem — cesta MBON → … → motoneurony není
součástí učicího pravidla.

Použití:
    python experiments/exp01_conditioning.py
    python experiments/exp01_conditioning.py --seeds 0 1 2 --trials 18
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

FLYBRAIN_ROOT = Path(__file__).resolve().parent.parent / "flybrain"
sys.path.insert(0, str(FLYBRAIN_ROOT))

from flybrain.circuits import Circuits, NeuronSet  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from flybrain.simulator import Simulator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)  # jinak loguje každý build
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def da_gate(c: Connectome, mbon_idx: np.ndarray, dans: NeuronSet) -> np.ndarray:
    """Dopaminová brána na MBON v [0,1] ze skutečných DAN→MBON počtů synapsí."""
    d = c.syn_counts[mbon_idx][:, dans.indices].toarray().sum(axis=1).astype(np.float32)
    return d / d.max() if d.max() > 0 else d


def depress(W: sp.csr_matrix, rows: np.ndarray, col_mask: np.ndarray, factor: np.ndarray) -> None:
    """Vynásob W[rows, cols kde col_mask] po řádcích faktorem — in-place, sparse.

    Matice se NIKDY nedensifikuje; sahá se jen na data 96 řádků MBON.
    """
    for row, f in zip(rows, factor):
        lo, hi = W.indptr[row], W.indptr[row + 1]
        seg = W.data[lo:hi]           # view do W.data
        seg[col_mask[W.indices[lo:hi]]] *= f


def make_odors(n_kc: int, sparsity: float, overlap: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Dva řídké pachové vzorce nad KC s řízeným překryvem (indexy do KC setu)."""
    n_active = max(1, int(sparsity * n_kc))
    plus = rng.choice(n_kc, size=n_active, replace=False)
    n_shared = int(overlap * n_active)
    shared = rng.choice(plus, size=n_shared, replace=False) if n_shared else np.array([], dtype=int)
    rest = rng.choice(np.setdiff1d(np.arange(n_kc), plus), size=n_active - n_shared, replace=False)
    return plus, np.concatenate([shared, rest]).astype(int)


def rates(W: sp.csr_matrix, stim_idx: np.ndarray, readouts: dict[str, np.ndarray],
          weight_scale: float, amp: float, steps: int) -> dict[str, float]:
    """Jeden běh celého mozku: stimuluj KC vzorec, vrať průměrné frekvence odečtů."""
    sim = Simulator(W, weight_scale=weight_scale)
    res = sim.run(steps, external=sim.input_vector(stim_idx, amp))
    r = res.rates_hz().numpy()
    return {name: float(r[idx].mean()) for name, idx in readouts.items()}


def run_arm(arm: str, W0: sp.csr_matrix, kc_idx, mbon_rows, gate, plus_kc, minus_kc,
            readouts, args, rng) -> dict:
    """Odehraj jedno rameno: natrénuj a změř odezvu na CS+ i CS− v několika bodech."""
    W = W0.copy()
    n = W.shape[0]

    # které KC sloupce se depresují a jakým gate — rozdíl mezi rameny je jen tady
    if arm == "sham":
        target_kc, g = None, None
    elif arm == "paired":
        target_kc, g = plus_kc, gate
    elif arm == "placebo":
        target_kc, g = plus_kc, rng.permutation(gate)   # stejná váha, špatné kompartmenty
    elif arm == "unpaired":
        target_kc, g = minus_kc, gate                   # odměna s jiným pachem
    else:
        raise ValueError(arm)

    col_mask = np.zeros(n, dtype=bool)
    if target_kc is not None:
        col_mask[kc_idx[target_kc]] = True
        factor = (1.0 - args.lr * g).astype(np.float32)

    probes = {"CS+": kc_idx[plus_kc], "CS-": kc_idx[minus_kc]}
    checkpoints = [0, args.trials] if args.fast else sorted(
        {0, args.trials // 3, 2 * args.trials // 3, args.trials})
    hist: dict[str, list[dict[str, float]]] = {p: [] for p in probes}

    done = 0
    for cp in checkpoints:
        while done < cp:
            if target_kc is not None:
                depress(W, mbon_rows, col_mask, factor)
            done += 1
        for name, stim in probes.items():
            hist[name].append(rates(W, stim, readouts, args.weight_scale, args.amp, args.steps))
        logger.info("  %-9s trial %2d | CS+ motor %.2f Hz | CS- motor %.2f Hz | CS+ MBON %.2f Hz",
                    arm, cp, hist["CS+"][-1]["motor"], hist["CS-"][-1]["motor"], hist["CS+"][-1]["MBON"])

    return {"arm": arm, "checkpoints": checkpoints, "hist": hist}


def delta(hist, probe, readout) -> float:
    """Změna frekvence odečtu mezi koncem a začátkem tréninku (Hz)."""
    return hist[probe][-1][readout] - hist[probe][0][readout]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--trials", type=int, default=18)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--kc-sparsity", type=float, default=0.05)
    p.add_argument("--overlap", type=float, default=0.1)
    p.add_argument("--weight-scale", type=float, default=0.2, help="volný parametr, viz preregistrace")
    p.add_argument("--amp", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fast", action="store_true", help="jen začátek a konec tréninku, bez křivky")
    p.add_argument("--cache-dir", type=Path, default=FLYBRAIN_ROOT / "outputs" / "connectome")
    args = p.parse_args()

    logger.info("Načítám connectome ...")
    c = Connectome.load(args.cache_dir)
    cir = Circuits(c)
    kcs, mbons, pam = cir.kenyon_cells(), cir.mbons(), cir.pam_dans()
    readouts = {
        "motor": cir.motor().indices,
        "proboscis": cir.proboscis_motor_neurons().indices,
        "MBON": mbons.indices,
    }
    gate = da_gate(c, mbons.indices, pam)
    logger.info("KC=%d  MBON=%d (PAM zasahuje %d)  motoneuronů=%d",
                len(kcs), len(mbons), int((gate > 0).sum()), len(readouts["motor"]))

    W0 = c.signed_weights().tocsr().astype(np.float32)
    W0.sort_indices()

    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        plus_kc, minus_kc = make_odors(len(kcs), args.kc_sparsity, args.overlap, rng)
        logger.info("=" * 78)
        logger.info("SEED %d  (CS+ %d KC, CS- %d KC, překryv %d)",
                    seed, len(plus_kc), len(minus_kc), len(np.intersect1d(plus_kc, minus_kc)))
        for arm in ("sham", "paired", "placebo", "unpaired"):
            r = run_arm(arm, W0, kcs.indices, mbons.indices, gate, plus_kc, minus_kc,
                        readouts, args, np.random.default_rng(seed + 1000))
            r["seed"] = seed
            r["D_motor"] = delta(r["hist"], "CS+", "motor") - delta(r["hist"], "CS-", "motor")
            r["D_mbon"] = delta(r["hist"], "CS+", "MBON") - delta(r["hist"], "CS-", "MBON")
            results.append(r)

    report(results, args)


def report(results: list[dict], args) -> None:
    """Vyhodnoť predikce P1–P4 z preregistrace. Prahy jsou dané, ne dopočítané."""
    by = {(r["seed"], r["arm"]): r for r in results}
    seeds = sorted({r["seed"] for r in results})

    logger.info("=" * 78)
    logger.info("VÝSLEDKY  (D = Δrate(CS+) − Δrate(CS−), Hz)")
    logger.info("%-6s %-9s %10s %10s %12s", "seed", "rameno", "D_motor", "D_MBON", "CS+ MBON %")
    for s in seeds:
        for arm in ("sham", "paired", "placebo", "unpaired"):
            r = by[(s, arm)]
            h = r["hist"]["CS+"]
            pct = 100 * (h[-1]["MBON"] - h[0]["MBON"]) / h[0]["MBON"] if h[0]["MBON"] else float("nan")
            logger.info("%-6d %-9s %10.3f %10.3f %11.1f%%", s, arm, r["D_motor"], r["D_mbon"], pct)

    logger.info("=" * 78)
    logger.info("PREDIKCE Z PREREGISTRACE")

    # P3 nejdřív — když simulace není deterministická, zbytek nemá smysl číst
    sham_ok = all(abs(by[(s, "sham")]["D_motor"]) < 1e-9 for s in seeds)
    logger.info("  P3 determinismus (D_sham = 0)            : %s", "SPLNĚNO" if sham_ok else "NESPLNĚNO — běh zahodit")

    # P1: odor-specifita na MBON, v procentech poklesu
    p1 = []
    for s in seeds:
        h = by[(s, "paired")]["hist"]
        d_plus = 100 * (h["CS+"][-1]["MBON"] - h["CS+"][0]["MBON"]) / h["CS+"][0]["MBON"]
        d_minus = 100 * (h["CS-"][-1]["MBON"] - h["CS-"][0]["MBON"]) / h["CS-"][0]["MBON"]
        p1.append(d_minus - d_plus)
    logger.info("  P1 odor-specifita na MBON (≥ 10 p. b.)   : %s  (%s)",
                "SPLNĚNO" if min(p1) >= 10 else "NESPLNĚNO",
                ", ".join(f"{v:+.1f}" for v in p1))

    # P2: hlavní predikce, po seedech
    p2 = []
    for s in seeds:
        dp = abs(by[(s, "paired")]["D_motor"])
        ctrl = max(abs(by[(s, "placebo")]["D_motor"]), abs(by[(s, "unpaired")]["D_motor"]))
        p2.append(dp >= 1.0 and dp > 2 * ctrl)
        logger.info("     seed %d: |D_paired|=%.3f  vs 2×max(kontroly)=%.3f -> %s",
                    s, dp, 2 * ctrl, "ano" if p2[-1] else "ne")
    logger.info("  P2 efekt přežije kontroly (≥1 Hz a 2×)   : %s  (%d/%d seedů)",
                "SPLNĚNO" if all(p2) else "NESPLNĚNO", sum(p2), len(p2))

    signs = {np.sign(round(by[(s, "paired")]["D_motor"], 6)) for s in seeds}
    logger.info("  P4 stabilní znaménko D_paired            : %s  (%s)",
                "SPLNĚNO" if len(signs) == 1 else "NESPLNĚNO", signs)

    logger.info("-" * 78)
    if not sham_ok:
        verdikt = "BĚH NEPLATNÝ (nedeterminismus)"
    elif all(p2) and len(signs) == 1:
        verdikt = "MB učení se propisuje do motorického výstupu"
    elif sum(p2) <= 1:
        verdikt = "ZÁPORNÝ NÁLEZ: efekt neprojde kontrolami (viz kritéria vyvrácení)"
    else:
        verdikt = "NEROZHODNUTO: efekt jen v části seedů = šum"
    logger.info("VERDIKT: %s", verdikt)
    logger.info("Pozor: weight_scale=%g je volný parametr, connectome váhy nenese.", args.weight_scale)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "exp01_conditioning.json"
    path.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                "verdikt": verdikt, "results": results}, indent=2, ensure_ascii=False))
    logger.info("Uloženo -> %s", path)


if __name__ == "__main__":
    main()
