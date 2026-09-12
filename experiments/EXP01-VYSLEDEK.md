# Experiment 1 — výsledek (12. 9. 2026)

Predikce viz `EXP01-PREREGISTRACE.md` (zapsané před během, nezměněné).
20 losů pachu × 4 ramena × 3 hodnoty `weight_scale`, 18 trialů, lr = 0,2.
Data `outputs/exp01_conditioning_ws*.json`, analýza `exp01c_report.py`.

## Hlavní závěr: odpověď závisí na volném parametru, ne na mozku

`weight_scale` není v datech — connectome nese počty synapsí, ne váhy.
Je to volný parametr. Na něm ale závisí i ZNAMÉNKO výsledku:

| ws | medián paired−placebo | 95 % CI | správné znam. | d_z | sd D_paired |
|---|---|---|---|---|---|
| 0,12 | **+2,31 Hz** | [0,78; 2,95] | **19/20** | +1,12 | 1,49 |
| 0,20 | +2,80 Hz | [−0,80; 6,79] | 13/20 | +0,62 | 6,83 |
| 0,30 | **−7,58 Hz** | [−14,16; 0,81] | 7/20 | −0,40 | 21,68 |

Při 0,12 čistý efekt nad placebem, při 0,20 neprokázáno, při 0,30 se obrací.
**Dokud není `weight_scale` ukotvený nezávisle, není z motorického odečtu
co reportovat.** Totéž platí pro P1 (odor-specifita na MBON): při ws 0,12
platí ve 20/20 losů, při 0,30 je v 11/20 záporná.

Režimy (stimulace 258 KC, 300 kroků):

| ws | aktivních neuronů | průměr celého mozku | medián aktivních |
|---|---|---|---|
| 0,12 | 0,60 % | 0,19 Hz | 23,3 Hz |
| 0,20 | 2,91 % | 0,55 Hz | 10,0 Hz |
| 0,30 | 9,72 % | 4,22 Hz | 23,3 Hz |

Který z nich je biologicky správný, z dat nevíme. Reálná moucha má řídkou
aktivitu s nízkými frekvencemi, což mluví spíš pro dolní konec — ale to je
argument z literatury, ne měření, a nesmí se použít k výběru parametru,
který dává hezčí výsledek.

## Co platí napříč všemi režimy

1. **Placebo je nutné.** Při ws 0,12 má paired 19/20 kladných a placebo
   9/20 — bez placeba by to vypadalo jako důkaz dopaminového učení, přitom
   část efektu dělá samo odebrání synaptické váhy.
2. **`unpaired` je nejostřejší kontrola.** Při ws 0,12 d_z = 2,53, znaménko
   20/20. Odměna se špatným pachem dává systematicky opačný posun.
3. **Determinismus** (P3): D_sham = 0,000 ve všech 60 bězích.

## Podlaha šumu (`exp01d_rozklad.py`)

Rozklad rozptylu frekvence motoneuronů, n = 10 na zdroj, ws = 0,2:

| zdroj | sd (Hz) |
|---|---|
| los pachu (které KC) | 7,85 |
| dynamika (VŠECHNY váhy ±0,1 %) | **2,36** |
| PAM gate (přeházená brána) | 3,89 |

Rozptyl tedy dominantně dělá los pachu (poměr 0,30), ale dynamický šum
**2,36 Hz je srovnatelný s celým měřeným efektem 2,80 Hz** při ws 0,2.
Při ws 0,12 je sd celého ramene 1,49 Hz, tedy pod touhle podlahou — proto
tam efekt vychází čistě.

**Oprava dřívějšího tvrzení:** `exp01b_citlivost.py` (jedna synapse
zeslabená až o 50 % nepohne odečtem o 0,001 Hz) NEDOKAZUJE, že odečet není
chaotický. Ukazuje jen, že na jedné synapsi nevisí. Kolektivní perturbace
rozprostřená přes 3,7 M synapsí měřitelný efekt MÁ.

## Poučení pro další experimenty

1. **Volný parametr ukotvit PŘED experimentem**, nezávisle na měřené
   veličině — jinak se závěr vybírá spolu s ním.
2. **Tři seedy nestačí.** Seed 0 při ws 0,2 dal +12,42 Hz proti kontrolám
   3,43 a −4,92; na 20 losech medián spadl na +2,80 s nulou v CI.
3. **Párové rozdíly, ne mediány ramen.** Týž los pachu v obou ramenech je
   pravý párový design a je citlivější: při ws 0,2 dává 13/20 a d_z 0,62,
   zatímco porovnání mediánů ramen vypadalo jako čirý šum.
4. **Prahy formulovat na medián nebo podíl seedů, ne na minimum přes
   všechny** (P1 takhle padla i tam, kde v agregátu platí).
5. **Efekt vždy srovnat s podlahou šumu** z kolektivní perturbace, ne jen
   s kontrolním ramenem.

## Otevřené
Ukotvení `weight_scale` nezávislým kritériem (např. cílová řídkost aktivity
a rozdělení frekvencí proti publikovaným záznamům z mouchy). Teprve pak má
smysl exp02 s lr/trials.
