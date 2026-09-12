# Ukotvení `weight_scale` (12. 9. 2026)

Po exp01, kde se na tomhle volném parametru obracelo ZNAMÉNKO závěru
(`EXP01-VYSLEDEK.md`), bylo potřeba ho ukotvit nezávisle na našem odečtu.

## Zdroj: referenční implementace papíru, který projekt reprodukuje

Shiu et al. 2024, Nature 634:210–219 — kód `github.com/philshiu/Drosophila_brain_model`,
soubor `model.py`. Parametry jsou tam s citacemi na elektrofyziologii:

| parametr | hodnota | zdroj v jejich kódu |
|---|---|---|
| klidový potenciál | −52 mV | Kakaria & de Bivort 2017 |
| práh | −45 mV | tamtéž |
| membránová konstanta | 20 ms | tamtéž |
| synaptická konstanta | 5 ms | Jürgensen et al. 2021 |
| refrakterní perioda | 2,2 ms | Lazar et al. 2021 |
| synaptické zpoždění | 1,8 ms | Paul et al. 2015 |
| **váha na synapsi** | **0,275 mV** | označeno „Free parameter" |

Vzdálenost k prahu je tedy **7 mV** a jedna synapse dodá 0,275 mV do
proměnné `g`, která teprve nabíjí membránu (alfa-synapse, tau 5 ms).

## Převod na náš model

Náš LIF má klid 0 a práh 1, jeden krok dělá `dv = (dt/tau)·I = 0,05·n·ws`.
Srovnávanou veličinou je **vrchol EPSP jako podíl vzdálenosti k prahu** —
to je to, co rozhoduje o vystřelení, a jde porovnat i mezi různými
formalismy. Numerickou integrací jejich rovnic:

| spoj o n synapsích | vrchol EPSP (Shiu) | % k prahu |
|---|---|---|
| 1 | 0,043 mV | 0,62 % |
| 10 | 0,433 mV | 6,19 % |
| 125 | 5,416 mV | 77,4 % |

Náš model dá totéž při **ws = 0,0062 / 0,05 = 0,1238**.

| ws | 1 synapse | 10 synapsí | 125 synapsí |
|---|---|---|---|
| **0,1238** | **0,62 %** | **6,19 %** | **77,4 %** |
| 0,2 | 1,00 % | 10,0 % | 125 % ← jeden spike sám vystřelí cíl |
| 0,3 | 1,50 % | 15,0 % | 188 % |

## Druhé, datové kritérium

Jediný presynaptický spike nesmí sám vystřelit cílový neuron (až na
výjimky). Rozdělení spojů: medián 8 synapsí, 99. percentil 87, maximum 2 633.

| ws | nadprahových spojů | práh překročí spoj od |
|---|---|---|
| 0,1238 | 9 011 (0,24 %) | 162 synapsí |
| 0,2 | 28 618 (0,77 %) | 100 synapsí |
| 0,3 | 62 605 (1,68 %) | 67 synapsí |

Při 0,3 by jediný spike sám vystřelil cíl u 1,7 % všech spojů, což je nad
99. percentilem rozdělení. Obě kritéria ukazují na dolní konec.

## Závěr

**`weight_scale = 0,124` je od teď výchozí hodnota**, odvozená ze
zveřejněných parametrů, ne z našich výsledků. Náhodná shoda s hodnotou
0,12 ze sweepu v exp01 — tedy s režimem, kde efekt vycházel čistě
(19/20 losů, CI nad nulou) — je příznivá, ale POZOR: exp01 byl odehraný
dřív a kotva se hledala až potom. Kdyby se kotva trefila do 0,3,
platil by opačný závěr.

## Meze téhle kotvy

1. **Shiu sami `w_syn` označují jako volný parametr.** Ukotvujeme se na
   obecně používaný model, ne na měření. Jejich hodnota je vybraná tak,
   aby model reprodukoval chování — je to tedy kotva o řád lepší než naše
   ruční volba, ale pořád ne biologické měření.
2. **Dynamika se liší.** Oni mají alfa-synapsi s tau 5 ms a zpoždění
   1,8 ms, my okamžitý proud v jednom kroku. Srovnal jsem vrchol EPSP,
   což je pro vystřelení rozhodující veličina, ale integrál pod křivkou
   a chování při vysokých frekvencích se liší — u opakované stimulace
   se rozdíl projeví.
3. **Neověřeno proti frekvencím.** Třetí nezávislé kritérium by bylo
   srovnat rozdělení frekvencí s publikovanými záznamy z mouchy. To jsme
   neudělali, takže kotva stojí na dvou nohách, ne na třech.

## Co z toho plyne pro exp01
Závěr exp01 se má číst při ws = 0,124, tedy: **medián paired−placebo
+2,31 Hz, 19/20 losů, CI [0,78; 2,95] nad nulou, d_z = 1,12.** Efekt nad
placebem existuje. Předtím zapsané kritérium P2 (≥ 1 Hz a 2× nad
kontrolami po seedech) ale i tady prošlo jen v 1/20 — na jednotlivém losu
je efekt pod podlahou šumu, čitelný je až v agregátu přes losy.
