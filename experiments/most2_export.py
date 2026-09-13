"""most2_export.py — širokoúhlý pohyb (MovingEdge) přes flyvis -> soubor."""
import logging
from pathlib import Path
import numpy as np, torch
logging.disable(logging.INFO)
from flyvis import NetworkView
from flyvis.datasets.moving_bar import MovingEdge

OUT = Path(__file__).resolve().parent / "outputs" / "flyvis_edge.npz"
TYPY = [f"T4{s}" for s in "abcd"] + [f"T5{s}" for s in "abcd"]

net = NetworkView("flow/0000/000").init_network()
cn = net.connectome
typ = np.array([s.decode() if isinstance(s, bytes) else s for s in cn.nodes.type[:]])
u, v = np.array(cn.nodes.u[:]), np.array(cn.nodes.v[:])

ds = MovingEdge(offsets=(-10, 11), intensities=[0, 1], speeds=[19], height=9,
                dt=1/200, angles=[0, 180], post_pad_mode="continue", t_pre=0.5, t_post=0.5)
print(ds.arg_df.to_string())
data = {}
for i in range(len(ds)):
    with torch.no_grad():
        r = net.simulate(ds[i][None, :, None, :], dt=ds.dt)[0].numpy()
    for T in TYPY:
        m = typ == T
        data[f"{i}|{T}|akt"] = r.max(0)[m]
        data[f"{i}|{T}|uv"] = np.c_[u[m], v[m]]
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, popis=np.array(ds.arg_df.to_string()), **data)
print("uloženo ->", OUT)
