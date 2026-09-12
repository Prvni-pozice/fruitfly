"""
exp01b_citlivost.py — je motorický odečet celého mozku vůbec čitelný, nebo je
chaotický?

Vzniklo jako diagnostika po záporném nálezu exp01: D_motor skákal mezi seedy
o ±12 Hz. Test: zeslab JEDINOU KC→MBON synapsi o 0,1 % (změna, která nemůže mít
biologický význam) a změř, o kolik se pohne frekvence motoneuronů. Když se
pohne srovnatelně s efektem celého učení, odečet měří dynamiku sítě, ne učení.

Použití:
    python experiments/exp01b_citlivost.py
"""

from __future__ import annotations

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

WEIGHT_SCALE, AMP, STEPS = 0.2, 2.0, 300


def motor_rate(W, stim, motor_idx) -> float:
    sim = Simulator(W, weight_scale=WEIGHT_SCALE)
    r = sim.run(STEPS, external=sim.input_vector(stim, AMP)).rates_hz().numpy()
    return float(r[motor_idx].mean())


def main() -> None:
    c = Connectome.load(FLYBRAIN_ROOT / "outputs" / "connectome")
    cir = Circuits(c)
    kcs, mbons = cir.kenyon_cells(), cir.mbons()
    motor = cir.motor().indices

    rng = np.random.default_rng(0)
    stim = kcs.indices[rng.choice(len(kcs), size=int(0.05 * len(kcs)), replace=False)]

    W0 = c.signed_weights().tocsr().astype(np.float32)
    W0.sort_indices()
    base = motor_rate(W0, stim, motor)
    logger.info("výchozí frekvence motoneuronů: %.3f Hz", base)

    # najdi nejsilnější KC->MBON synapsi a zeslab ji o zadané promile
    kc_mask = np.zeros(W0.shape[0], dtype=bool)
    kc_mask[kcs.indices] = True
    best = (0.0, None)
    for row in mbons.indices:
        lo, hi = W0.indptr[row], W0.indptr[row + 1]
        sel = kc_mask[W0.indices[lo:hi]]
        if sel.any():
            k = int(np.argmax(np.abs(W0.data[lo:hi][sel])))
            pos = np.flatnonzero(sel)[k] + lo
            if abs(W0.data[pos]) > best[0]:
                best = (abs(float(W0.data[pos])), pos)
    _, pos = best
    logger.info("nejsilnější KC→MBON synapse: váha %.1f", W0.data[pos])

    logger.info("%-14s %12s %12s", "zeslabení", "motor (Hz)", "Δ (Hz)")
    for frac in (0.001, 0.01, 0.1, 0.5):
        W = W0.copy()
        W.data[pos] *= (1.0 - frac)
        r = motor_rate(W, stim, motor)
        logger.info("%-14s %12.3f %12.3f", f"{frac*100:g} % (1 synapse)", r, r - base)

    logger.info("-" * 44)
    logger.info("Srovnávej s efektem celého učení v exp01 (D_motor byl 1–12 Hz).")
    logger.info("Když jedna synapse hýbe podobně, odečet měří dynamiku sítě, ne učení.")


if __name__ == "__main__":
    main()
