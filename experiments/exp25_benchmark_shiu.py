"""
exp25_benchmark_shiu.py — co se změní, když monoaminy počítáme jako Shiu?

PREREGISTRACE (zapsáno před během, 19. 9. 2026):
Přechod z pravidla „monoaminy = 0" na Shiuovo „monoaminy = +1". Čtyři
benchmarky, u každého se předem řekne, co znamená, že přechod NEŠKODÍ:

  B1 (Shiu, cukr -> sosák): MN9/sosákové motoneurony na cukr > 30 Hz
      a hořká je tlumí. Podle Shiu (kalibrace na ~80 % maxima MN9) a
      fly-brain-minecraft (30–90 Hz).
  B2 (odměna): cukr budí PAM víc než PPL1 v součtu aktivity
      (PAM průměr × 261 > PPL1 průměr × 16). Bez toho cukr není odměna.
  B3 (řízení): směr zatáčení správný v ≥ 7 z 8 poloh (dosud 5/5 na užší sadě).
  B4 (optomotorika): skutečné směry > přeházené o ≥ 20 p. b. (dosud +52).
Vyvrácení: když B3 nebo B4 padne, Shiuovo pravidlo rozbíjí řízení a
nesmí se nasadit plošně; když padne B2, monoaminy +1 odměnu nevyrobí
a je potřeba modulaci modelovat jinak než jako spoj.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flybrain")); sys.path.insert(0, str(ROOT))
from agent2 import Agent2  # noqa: E402
from exp07_optomotor import epizoda  # noqa: E402
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


def bench(c, jmeno: str) -> dict:
    cir = Circuits(c)
    W = c.signed_weights().tocsr().astype(np.float32); N = W.shape[0]
    cukr, horka = cir.sugar_grns(), cir.bitter_grns()
    mn = cir.proboscis_motor_neurons(); pam, ppl1 = cir.pam_dans(), cir.ppl1_dans()

    def beh(stim):
        e = np.zeros(N, dtype=np.float32)
        for ns in stim: e[ns.indices] = 3.0
        return Simulator2(W, weight_scale=WS).run(300, external=e).rates_hz().numpy()

    r_c, r_ch, r_0 = beh([cukr]), beh([cukr, horka]), beh([])
    out = {"MN_cukr": float(r_c[mn.indices].mean()), "MN_cukr_horka": float(r_ch[mn.indices].mean()),
           "MN_nic": float(r_0[mn.indices].mean()),
           "PAM": float(r_c[pam.indices].mean()), "PPL1": float(r_c[ppl1.indices].mean()),
           "aktivnich_pct": float(100 * (r_c > 0).mean())}
    log.info("[%s] B1 sosák: nic %.1f | cukr %.1f | cukr+hořká %.1f Hz | aktivních %.1f %%",
             jmeno, out["MN_nic"], out["MN_cukr"], out["MN_cukr_horka"], out["aktivnich_pct"])
    log.info("[%s] B2 odměna: PAM %.2f Hz ×261 = %.0f | PPL1 %.2f Hz ×16 = %.0f",
             jmeno, out["PAM"], out["PAM"] * 261, out["PPL1"], out["PPL1"] * 16)

    a = Agent2(c, steps=100)
    spr = []
    for o in (-48, -36, -24, -12, 12, 24, 36, 48):
        z = [a.act(snimek(o), snimek(o - d)).zataceni for d in (-12, -6, 6, 12)]
        nz = [x for x in z if x != 0]
        spr.append(bool(nz) and np.mean([np.sign(x) == np.sign(-o) for x in nz]) >= 0.5)
    out["rizeni_spravne"] = int(sum(spr))
    log.info("[%s] B3 řízení: správný směr v %d/8 poloh", jmeno, out["rizeni_spravne"])

    puv = {h_: dict(a.oci[h_]["sousedi"]) for h_ in a.oci}
    rng = np.random.default_rng(0)
    u = {}
    for rezim in ("skutecna", "smery"):
        for h_ in a.oci:
            if rezim == "smery":
                kl = list(puv[h_]); perm = rng.permutation(kl)
                a.oci[h_]["sousedi"] = {k: puv[h_][q] for k, q in zip(kl, perm)}
            else:
                a.oci[h_]["sousedi"] = puv[h_]
        u[rezim] = float(np.median([epizoda(a, s, 20, rezim, rng)[0] for s in range(8)]))
    out["opto_skutecna"], out["opto_smery"] = u["skutecna"], u["smery"]
    log.info("[%s] B4 optomotorika: skutečné %.3f | přeházené %.3f | rozdíl %+.1f p. b.",
             jmeno, u["skutecna"], u["smery"], 100 * (u["skutecna"] - u["smery"]))
    return out


def main():
    vysl = {}
    for jmeno in ("nula", "shiu"):
        c = Connectome.load(ROOT.parent / "flybrain" / "outputs" / "connectome")
        if jmeno == "shiu":
            log.info("Shiuovo pravidlo: %s", pouzij_shiu(c))
        vysl[jmeno] = bench(c, jmeno)

    s = vysl["shiu"]
    b1 = s["MN_cukr"] > 30 and s["MN_cukr_horka"] < s["MN_cukr"]
    b2 = s["PAM"] * 261 > s["PPL1"] * 16
    b3 = s["rizeni_spravne"] >= 7
    b4 = (s["opto_skutecna"] - s["opto_smery"]) >= 0.20
    log.info("=" * 74)
    for k, v in (("B1 cukr -> sosák > 30 Hz, hořká tlumí", b1), ("B2 cukr je odměna (PAM > PPL1 v součtu)", b2),
                 ("B3 řízení správně ≥ 7/8", b3), ("B4 optomotorika ≥ +20 p. b.", b4)):
        log.info("  %-42s: %s", k, "SPLNĚNO" if v else "NESPLNĚNO")
    verdikt = ("Shiuovo pravidlo neškodí a vyrobí odměnu" if all((b1, b2, b3, b4))
               else "Shiuovo pravidlo neškodí, ale odměnu nevyrobí" if (b1 and b3 and b4)
               else "Shiuovo pravidlo ROZBÍJÍ řízení/zrak — nenasazovat plošně")
    log.info("VERDIKT: %s", verdikt)
    (ROOT / "outputs" / "exp25.json").write_text(json.dumps({"verdikt": verdikt, **vysl}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
