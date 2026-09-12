"""
exp01d_rozklad.py — rozklad rozptylu D_motor na zdroje + poctivý test chaosu.

Vzniklo na námitku, že (a) zeslabení JEDNÉ synapse nic neříká o kolektivní
dynamice, (b) „rozptyl dělá los pachu" je zatím hypotéza, ne měření.

Tři zdroje rozptylu, každý měřený zvlášť:
  1. LOS PACHU     — mění se, které KC pach aktivuje (ostatní fixní)
  2. DYNAMIKA      — pach fixní, kolektivně rozhýbané VŠECHNY váhy o ±eps
                     (tohle je ten chybějící test chaosu: mikroskopická
                     perturbace rozprostřená přes 3,7 M synapsí)
  3. PAM GATE      — pach fixní, přeházená dopaminová brána mezi MBON

Když 2 vyjde srovnatelně s 1, je odečet chaotický a exp01 se nedá číst.

Použití:
    python experiments/exp01d_rozklad.py --n 10 --eps 0.001
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

FLYBRAIN_ROOT = Path(__file__).resolve().parent.parent / "flybrain"
sys.path.insert(0, str(FLYBRAIN_ROOT))

from flybrain.circuits import Circuits  # noqa: E402
from flybrain.loader import Connectome  # noqa: E402
from flybrain.simulator import Simulator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("flybrain.simulator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def motor_rate(W, stim, motor_idx, ws, amp, steps) -> float:
    sim = Simulator(W, weight_scale=ws)
    r = sim.run(steps, external=sim.input_vector(stim, amp)).rates_hz().numpy()
    return float(r[motor_idx].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="opakování na zdroj")
    p.add_argument("--eps", type=float, default=0.001, help="relativní rozhýbání všech vah")
    p.add_argument("--weight-scale", type=float, default=0.2)
    p.add_argument("--amp", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()

    c = Connectome.load(FLYBRAIN_ROOT / "outputs" / "connectome")
    cir = Circuits(c)
    kcs, mbons = cir.kenyon_cells(), cir.mbons()
    motor = cir.motor().indices
    W0 = c.signed_weights().tocsr().astype(np.float32)
    W0.sort_indices()

    n_act = int(0.05 * len(kcs))

    # --- 1. los pachu ---
    vals = []
    for s in range(args.n):
        rng = np.random.default_rng(s)
        stim = kcs.indices[rng.choice(len(kcs), size=n_act, replace=False)]
        vals.append(motor_rate(W0, stim, motor, args.weight_scale, args.amp, args.steps))
    v_odor = np.array(vals)

    # --- 2. dynamika: pach fixní, všechny váhy rozhýbané o ±eps ---
    rng0 = np.random.default_rng(0)
    stim_fix = kcs.indices[rng0.choice(len(kcs), size=n_act, replace=False)]
    vals = []
    for s in range(args.n):
        rng = np.random.default_rng(10_000 + s)
        W = W0.copy()
        W.data *= (1.0 + args.eps * rng.standard_normal(W.data.size)).astype(np.float32)
        vals.append(motor_rate(W, stim_fix, motor, args.weight_scale, args.amp, args.steps))
    v_dyn = np.array(vals)

    # --- 3. PAM gate: pach fixní, přeházená brána (aplikovaná jako v exp01) ---
    d = c.syn_counts[mbons.indices][:, cir.pam_dans().indices].toarray().sum(axis=1).astype(np.float32)
    gate = d / d.max()
    kc_mask = np.zeros(W0.shape[0], dtype=bool)
    kc_mask[stim_fix] = True
    vals = []
    for s in range(args.n):
        rng = np.random.default_rng(20_000 + s)
        W = W0.copy()
        factor = (1.0 - 0.2 * rng.permutation(gate)).astype(np.float32)
        for row, f in zip(mbons.indices, factor):
            lo, hi = W.indptr[row], W.indptr[row + 1]
            seg = W.data[lo:hi]
            seg[kc_mask[W.indices[lo:hi]]] *= f ** 18   # 18 trialů najednou
        vals.append(motor_rate(W, stim_fix, motor, args.weight_scale, args.amp, args.steps))
    v_gate = np.array(vals)

    print(f"\nROZKLAD ROZPTYLU frekvence motoneuronů (n={args.n} na zdroj, ws={args.weight_scale})")
    print("-" * 74)
    print(f"{'zdroj':<34}{'průměr':>9}{'sd':>9}{'rozpětí':>12}")
    for name, v in (("1. los pachu (které KC)", v_odor),
                    (f"2. dynamika (všechny váhy ±{args.eps:g})", v_dyn),
                    ("3. PAM gate (přeházená brána)", v_gate)):
        print(f"{name:<34}{v.mean():>9.3f}{v.std(ddof=1):>9.3f}{v.max()-v.min():>12.3f}")

    print("-" * 74)
    pomer = v_dyn.std(ddof=1) / v_odor.std(ddof=1) if v_odor.std(ddof=1) else float("inf")
    print(f"  poměr sd(dynamika) / sd(los pachu) = {pomer:.3f}")
    if pomer > 0.5:
        print("  -> odečet je citlivý na mikroskopickou perturbaci = CHAOTICKÝ, exp01 nečitelný")
    elif pomer > 0.1:
        print("  -> dynamika přispívá nezanedbatelně, uvádět jako spodní mez šumu")
    else:
        print("  -> rozptyl NEdělá dynamika; dominuje los pachu (hypotéza z exp01 potvrzena)")


if __name__ == "__main__":
    main()
