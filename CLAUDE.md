# CLAUDE.md — fruitfly

Simulace celého connectomu octomilky (FlyWire FAFB v783) jako sparse LIF sítě.
Cíl: ověřit reálný connectome → plasticita (učení) → tělo → časem hry
„minecraftového" typu.

## Struktura
- `flybrain/` — klon https://github.com/Tharusha101/flybrain (vlastní git, v parent
  repu ignorován). Upstream skripty se NEUPRAVUJÍ.
- `flybrain/data/` — FlyWire CSV v783 (licencované, gitignored).
- `flybrain/outputs/` — cache sparse matice + výsledky (gitignored).
- `experiments/` — NAŠE vlastní experimenty, tady se píše nový kód.

## Prostředí
- Bez sudo, bez `python3 -m venv` (chybí ensurepip). Venv se dělá přes **uv**:
  `uv venv .venv -p 3.12` + `uv pip install -p .venv/bin/python …`.
- Interpret: `/data/bot/fruitfly/flybrain/.venv/bin/python`.
- torch 2.14.0+**cpu** (VPS nemá GPU). Celý mozek běží na CPU v sekundách.

## Data — kde se berou bez Codex loginu
Codex vyžaduje přihlášení, ale existuje veřejný mirror:
`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/{connections_princeton,neurons,classification,consolidated_cell_types}.csv.gz`
(~69 MB). Schéma sedělo s loaderem beze změny.

## Pravidla
- **Nikdy nedensifikovat** matici 138 584 × 138 584 (~78 GB). Vše sparse.
- Váhy nejsou v datech: `weight_scale` je VOLNÝ parametr, který si skript
  sweepuje (vyšlo 0,2). Výsledky se s ním hýbou — při každém novém
  experimentu ho reportovat.
- Neurotransmiter je PREDIKCE s confidence, ne fakt. 961 neuronů zůstalo
  bez znaménka, 31 591 bez třídy.

## Stav (12. 9. 2026) — vše ověřeno na VPS
| krok | výsledek |
|---|---|
| `build_graph.py` | 138 584 neuronů, 3 732 460 pre→post párů, 10 s |
| `simulate_taste.py` | cukr → sosák 90 Hz vs 0 Hz baseline, latence **15 ms** (sedí s Shiu et al. 2024); hořká tlumí na 31,4 Hz |
| `explore.py` | degree distribuce, neuropily (ME_L největší) |
| `add_plasticity.py` | PAM odměna: CS+ v inervovaných MBON −67 %, CS− jen −5 % (odor-specific), corr(DA gate, deprese) = 0,91 |

Nezkoušeno: `animate_cascade.py`, `phase3_rigor.py`, embodiment (MuJoCo neinstalováno).

## Experiment 1 (12. 9. 2026) — vlastní, `experiments/`
Otázka: propíše se MB učení do motorického výstupu? **Ne.** 20 losů pachu,
paired vs placebo je hod mincí (10/20 párově, median |D| 4,22 vs 3,90 Hz).
Odor-specifita na MBON ale PLATÍ (+23,7 p. b. proti placebu +5,6).
Podrobně `experiments/EXP01-VYSLEDEK.md`. Co si odnést do každé další studie:
- **Tři seedy nestačí** — seed 0 sám dal čistě vypadající +12,4 Hz, na 20 losech
  to zmizelo. Los pachu je silnější zdroj rozptylu než testovaný zásah.
- **Placebo rameno** (stejná odebraná váha, přeházené kompartmenty) rozhodlo.
- **Odečet není chaotický**: jedna synapse zeslabená o 50 % nepohne motorickým
  výstupem o 0,001 Hz (`exp01b_citlivost.py`).
- Prahy formulovat na medián/podíl seedů, ne na minimum přes všechny.
