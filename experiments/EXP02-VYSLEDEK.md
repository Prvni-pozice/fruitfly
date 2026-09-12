# Experiment 2 — uzavřená smyčka se naučí zatáčet podle vjemu (12. 9. 2026)

Preregistrace `EXP02-PREREGISTRACE.md`, kód `exp02_zataceni.py`,
`exp02b_prehozeni.py`, smyčka `flyagent.py`. ws = 0,124 (ukotvené).
20 losů, trénink na náhodných polovinách vizuálních neuronů, TEST na plné
straně (při učení neviděné).

## Výsledek: úloha se naučí, a je to učení kontingence

| rameno | medián ΔS | kladných | chování překlopeno (S > 0) |
|---|---|---|---|
| sham | 0,0000 | 0/20 | — |
| **paired** | **+0,2678** | **20/20** | **16/20** |
| placebo | +0,0415 | 17/20 | 1/20 |
| unpaired | +0,0549 | 18/20 | 0/20 |

Párově paired − placebo = **+0,1964**, CI [0,1734; 0,2589]. Všechny čtyři
predikce P1–P4 splněny.

Před tréninkem smyčka zatáčela DOPRAVA na oba vjemy (S = −0,2057, shodně
ve všech losech). Po tréninku zatáčí doleva na levý vjem a doprava na pravý.

## Test přehození (exp02b) — odlišuje učení od driftu

Placebo nechytí případ, kdy pravidlo prostě táhne do pevného stavu
„levý vjem → levá akce". Proto: natrénovat, pak OTOČIT pravidlo odměny.

| 2. fáze | medián ΔS | klesá |
|---|---|---|
| přehozené pravidlo | **−0,1939** | **10/10** |
| kontrola: dál stejné pravidlo | **+0,3761** | **0/10** |

Rozdíl −0,5340, CI [−0,6146; −0,5051]. R1–R3 splněny. Chování se po
přehození otočí i znaménkem (medián S = −0,142).

**Není to tedy drift ani saturace vah** — kdyby šlo o saturaci, klesala by
i kontrola, a ta místo toho dál roste.

## Co to znamená a co ne

ZNAMENÁ: smyčka vjem → mozek → akce → odměna → změna vah nad skutečným
connectomem funguje a mění chování podle toho, co odměna říká. To je
stavební kámen pro hru.

NEZNAMENÁ, že je to chytré. Úloha je konstrukčně snadná: oba vjemy jsou
disjunktní množiny neuronů, tedy lineárně oddělitelné zadání. A učicí
pravidlo (třífaktorové pre × post × odměna s protitahem na vstupech do
sestupných neuronů) je NAŠE inženýrská volba, ne mouší biologie. Doložené
je jen zapojení.

## Otevřené
- Trénink i test používají stejnou jednoduchou geometrii (celá levá vs celá
  pravá strana zorného pole). Skutečný snímek ze hry je něco jiného — to je
  krok 4 v `PLAN.md`.
- Akce se čte z hrubé bilance 638 vs 644 neuronů. Jmenovité povelové
  neurony (DNa02, DNp09, MDN) budou čistší — krok 3.
