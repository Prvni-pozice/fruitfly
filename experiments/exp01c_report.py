"""
exp01c_report.py — statistický report exp01 nad hotovými 20 losy.

Nepouští simulaci, jen čte outputs/exp01_conditioning.json. Vzniklo na
námitku, že medián a rozdělení říkají něco jiného než průměr, a že efekt
může táhnout pár extrémních pachů.

Počítá: rozdělení D_motor po ramenech, medián, bootstrap CI, PÁROVÉ rozdíly
paired−placebo a paired−unpaired (týž los pachu = pravý párový design),
effect size, podíl správných znamének a robustnost po odstranění extrémů.

Použití:
    python experiments/exp01c_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"


def boot_ci(v: np.ndarray, stat=np.median, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Percentilový bootstrap CI 95 % — na n=20 je to jediné poctivé CI."""
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def line(name: str, v: np.ndarray) -> None:
    lo, hi = boot_ci(v)
    print(f"{name:<22}{np.median(v):>9.2f}{v.mean():>9.2f}{v.std(ddof=1):>8.2f}"
          f"{f'[{lo:.2f}, {hi:.2f}]':>18}{int((v > 0).sum()):>6}/{v.size}")


def main() -> None:
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else "exp01_conditioning.json"
    print(f"### {fn}")
    R = json.load(open(OUT / fn))["results"]
    seeds = sorted({r["seed"] for r in R})
    by = {(r["seed"], r["arm"]): r for r in R}
    D = {a: np.array([by[(s, a)]["D_motor"] for s in seeds]) for a in ("sham", "paired", "placebo", "unpaired")}

    print(f"n = {len(seeds)} losů pachu\n")
    print(f"{'veličina':<22}{'medián':>9}{'průměr':>9}{'sd':>8}{'95% CI (med)':>18}{'kladných':>12}")
    print("-" * 78)
    for a in ("sham", "paired", "placebo", "unpaired"):
        line(f"D_motor {a}", D[a])

    print("\nPÁROVÉ rozdíly (týž los pachu, tedy pravý párový design):")
    print("-" * 78)
    dpp = D["paired"] - D["placebo"]
    dpu = D["paired"] - D["unpaired"]
    line("paired − placebo", dpp)
    line("paired − unpaired", dpu)

    # effect size na párových rozdílech (Cohenovo d_z)
    for name, v in (("paired−placebo", dpp), ("paired−unpaired", dpu)):
        dz = v.mean() / v.std(ddof=1)
        print(f"  Cohenovo d_z {name:<18}: {dz:+.2f}   (|d_z| < 0,2 = zanedbatelné)")

    print("\nTÁHNOU VÝSLEDEK EXTRÉMNÍ PACHY?  (medián D_paired−D_placebo po ořezu)")
    print("-" * 78)
    order = np.argsort(np.abs(dpp))
    for k in (0, 1, 2, 3):
        keep = order[: len(order) - k] if k else order
        print(f"  po odstranění {k} nejextrémnějších: medián {np.median(dpp[keep]):+6.2f} Hz "
              f"(n={len(keep)})")

    print("\nPO SEEDECH  (D_motor, Hz)")
    print("-" * 78)
    print(f"{'seed':>5}{'paired':>9}{'placebo':>9}{'unpaired':>10}{'p−pl':>8}{'p−un':>8}")
    for i, s in enumerate(seeds):
        print(f"{s:>5}{D['paired'][i]:>9.2f}{D['placebo'][i]:>9.2f}{D['unpaired'][i]:>10.2f}"
              f"{dpp[i]:>8.2f}{dpu[i]:>8.2f}")

    print("\nZÁVĚR")
    print("-" * 78)
    lo, hi = boot_ci(dpp)
    nula_v_ci = lo <= 0 <= hi
    print(f"  medián paired−placebo = {np.median(dpp):+.2f} Hz, CI [{lo:.2f}, {hi:.2f}]")
    print(f"  nula {'JE' if nula_v_ci else 'NENÍ'} v intervalu -> "
          f"{'efekt neprokázán' if nula_v_ci else 'efekt nad kontrolou'}")
    print(f"  správné znaménko (paired > placebo): {int((dpp > 0).sum())}/{len(dpp)}")


if __name__ == "__main__":
    main()
