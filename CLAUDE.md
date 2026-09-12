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
Otázka: propíše se MB učení do motorického výstupu? **Odpověď závisí na
`weight_scale`, což je volný parametr** — medián paired−placebo je +2,31 Hz
(19/20 losů, CI nad nulou) při ws 0,12, ale −7,58 Hz při ws 0,30. Podrobně
`experiments/EXP01-VYSLEDEK.md`. Co si odnést do každé další studie:
- **Volný parametr ukotvit PŘED experimentem**, nezávisle na měřené veličině.
  Pro `weight_scale` to je hotové: **0,124**, odvozeno z publikovaných parametrů
  Shiu et al. 2024 (`experiments/UKOTVENI-WEIGHT-SCALE.md`). Výchozí hodnota
  0,2 z upstream `simulate_taste.py` je NEUKOTVENÁ a dává jiné závěry.
- **Tři seedy nestačí** — seed 0 sám dal čistě vypadající +12,4 Hz, na 20 losech
  zmizel. Los pachu je největší zdroj rozptylu (sd 7,85 Hz).
- **Párové rozdíly** (týž los v obou ramenech), ne porovnání mediánů ramen.
- **Podlaha šumu je 2,36 Hz** — rozhýbání všech vah o 0,1 % (`exp01d_rozklad.py`).
  Efekt menší než tohle je nečitelný. Jedna synapse přitom nedělá nic, takže
  z ní na (ne)chaotičnost usuzovat NELZE.
- Prahy formulovat na medián/podíl seedů, ne na minimum přes všechny.
