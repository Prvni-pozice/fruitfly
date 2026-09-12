# Experiment 2 — naučí se smyčka zatáčet podle vjemu?

Zapsáno 12. 9. 2026 PŘED prvním během. Nemění se podle výsledku.

## Proč tenhle krok
Cíl projektu je nechat octomilku hrát hru v prohlížeči. Do toho chybí
UZAVŘENÁ SMYČKA: vjem → mozek → akce → odměna → změna vah. Exp01 ukázal,
že učení v houbovitém tělísku se k motorickému výstupu spolehlivě nedostane,
takže akční kanál musí být jiný.

Akční kanál = **sestupné neurony** (1 290, z toho 638 vlevo a 644 vpravo).
To je skutečná povelová cesta z mozku do těla a rozdíl stran je zatáčení.
Změřeno předem: vizuál vlevo dá bilanci −0,209, vpravo −0,003, tedy kanál
JE dekódovatelný (rozdíl 0,21). Obojí ale zatáčí doprava — úkolem učení je
ten rozdíl rozevřít a otočit znaménko u levého vjemu.

## Úloha
Vjem vlevo → zatoč VLEVO (bilance > 0). Vjem vpravo → zatoč VPRAVO (< 0).
Stimul každého trialu je náhodná polovina vizuálních neuronů dané strany
(jako různé snímky téže scény); testuje se na PLNÉ straně, kterou model
při učení neviděl.

Metrika: **rozevření S = bilance(vjem vlevo) − bilance(vjem vpravo)**,
měřené před a po tréninku. Sleduje se **ΔS = S_po − S_před**.

## Ramena
| rameno | odměna |
|---|---|
| `sham` | žádná (kontrola determinismu) |
| `paired` | správná strana podle vjemu |
| `placebo` | náhodná strana (stejná velikost změny, bez vazby na vjem) |
| `unpaired` | vždy vlevo, bez ohledu na vjem |

## Predikce s prahy
- **P1**: ΔS_paired > 0 aspoň v 15 z 20 losů.
- **P2**: medián (ΔS_paired − ΔS_placebo) > 0 a 95% CI **neobsahuje nulu**
  (párově, týž los).
- **P3**: ΔS_sham = 0,000 přesně.
- **P4**: medián ΔS_paired ≥ 0,05 (tedy aspoň čtvrtina už naměřeného
  rozdílu 0,21 mezi stranami — jinak je to efekt bez praktické velikosti).

## Kritéria vyvrácení
- ΔS_placebo ≥ ΔS_paired: rozevření dělá samotná změna vah, ne párování
  vjemu s odměnou. Záporný nález, nehledat jiný odečet.
- P1 splněno, ale P4 ne: efekt existuje a je k ničemu — reportovat takhle.
- Jakmile padne P3, běh je neplatný.

## Co experiment NEMĚŘÍ
Že se takhle učí moucha. Učicí pravidlo na vstupech do DN je NAŠE
inženýrská volba (třífaktorové pre × post × odměna s protitahem), ne
doložená biologie. Doložené je jen zapojení. Platí i ukotvení
`weight_scale = 0,124` a podlaha šumu z exp01.
