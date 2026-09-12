# Experiment 1 — výsledek (12. 9. 2026)

Predikce viz `EXP01-PREREGISTRACE.md` (zapsané před během, nezměněné).
20 losů pachu, 4 ramena, `weight_scale = 0,2`, 18 trialů, lr = 0,2.
Data: `outputs/exp01_conditioning.json`, log `outputs/exp01_20seedu.log`.

## Verdikt: ZÁPORNÝ na hlavní otázce

**Naučená změna v houbovitém tělísku se do motorického výstupu nepropisuje
způsobem, který by šel odlišit od placeba.**

| rameno | mean D_motor | median &#124;D&#124; | sd | kladných |
|---|---|---|---|---|
| sham | 0,00 | 0,00 | 0,00 | 0/20 |
| paired | +1,73 | 4,22 | 6,83 | 10/20 |
| placebo | −1,44 | 3,90 | 7,01 | 9/20 |
| unpaired | −2,01 | 4,66 | 6,13 | 6/20 |

Párově je |D_paired| > |D_placebo| v **10 z 20** losů — hod mincí.
P2 (předem: ≥ 1 Hz a 2× nad kontrolami) prošla ve 4/20. P4 padla,
znaménko se obrací. P3 (determinismus) splněna přesně.

## Co naopak PLATÍ: odor-specifita na úrovni MBON

Táž data, odečet o vrstvu výš:

| | median (p. b.) | nad prahem 10 |
|---|---|---|
| paired | **+23,7** | 14/20 |
| placebo | +5,6 | 8/20 |

CS+ klesá o ~24 p. b. víc než nepárovaný CS−, a placebo (stejná odebraná
váha, přeházené kompartmenty) to nedokáže. Učení tedy reálné a pachově
specifické je — jen se zastaví před motorickým výstupem.

## Proč to není šum měření

`exp01b_citlivost.py`: zeslabení JEDINÉ nejsilnější KC→MBON synapse
o 0,1 / 1 / 10 / 50 % nepohnulo frekvencí motoneuronů ani o 0,001 Hz.
Odečet není chaotický ani hypersenzitivní. Rozptyl ±12 Hz mezi losy dělá
**který pach** se losuje, ne nestabilita sítě.

## Poučení pro další experimenty

1. **Tři seedy nestačí.** Seed 0 dal +12,42 Hz proti kontrolám 3,43 a −4,92 —
   čistě vypadající pozitivní nález, který na 20 losech zmizel. Los pachu
   je v tomhle modelu silnější zdroj rozptylu než testovaný zásah.
2. **Placebo rameno rozhodlo.** Bez něj by median |D_paired| = 4,22 Hz
   vypadal jako efekt; placebo má 3,90.
3. **Odečet o vrstvu dál od zásahu je jiná veličina, ne slabší verze téže.**
   Na MBON efekt je, na motoru není. Reportovat obojí zvlášť.
4. **Kritérium „ve VŠECH seedech" je moc tvrdé** na měření s tímhle
   rozptylem (P1 takhle padla, i když v agregátu platí). Příště prahy
   formulovat na medián nebo podíl seedů, ne na minimum.

## Otevřené
Nevíme, jestli nález platí i pro `weight_scale` mimo 0,2 a jestli by se
efekt neobjevil při silnějším učení (lr, trials). To je exp02.
