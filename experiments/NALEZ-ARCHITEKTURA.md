# Architektura: kde končí graduovaná část a začínají spiky (13. 9. 2026)

## Hlavní závěr

**Celá zraková dráha mouchy je graduovaná — spiky začínají až u sestupných
neuronů.** Doloženo pro obě úrovně, na které jsme narazili:
- T4/T5 (medulla/lobula): záznamy ukazují jen graduované podprahové
  odpovědi, transienty 1–2 mV nešlo potvrdit jako spiky.
- HS/VS (tangenciální buňky lobula plate): odpovídají graduovaným posunem
  membránového napětí axonu.

Spikující LIF proto zrakovou informaci neunese na ŽÁDNÉ úrovni. Není to
otázka parametru — v našich měřeních T4/T5 ani HS nevystřelily ani jednou,
ať byl podnět jakýkoli.

## Architektura, která z toho plyne

    video
      -> flyvis                      graduovaně, natrénované, 45 669 neuronů
      -> T4/T5 po sloupcích
      -> rate_vrstva.py              graduovaně, váhy z FlyWire, 86 014 neuronů
      -> HS / VS / LC
      -> sim2.py                     spikující, synaptická integrace
      -> sestupné neurony -> akce

## Co je hotové a ověřené

**flyvis** (Lappalainen et al., Nature 2024) běží na CPU, 50 předtrénovaných
modelů. Směrová selektivita ověřena: T4b reaguje 1,8× víc na 0° než 180°,
T4a naopak. Nemá LC ani HS — končí u T4/T5, proto je potřeba prostřední
vrstva.

**Most po sloupcích** funguje: `column_assignment.csv.gz` na naší straně,
hexagonální u,v na straně flyvis, párování nejbližším sousedem.
Spárováno 712–775 sloupců na typ.

**`rate_vrstva.py`**: ustálený stav lineárně-prahové sítě nad skutečnými
vahami FlyWire. Nutná normalizace na součet vstupů — bez ní se síť rozkmitá
(naměřeno: hodnoty 1e28). Zisk 0,5 je stabilní.

**Zrcadlení druhého oka musí prohodit směrové typy** (T4a↔T4b, T5a↔T5b),
nejen souřadnice. Bez toho lateralizace nevznikne. Fyzikálně to odpovídá
rozdílu mezi posuvem (obě oči vidí totéž) a otáčením (každé oko opačně).

## Výsledek: HS nesou směr a jsou lateralizované

| podnět | HS vlevo | HS vpravo | rozdíl |
|---|---|---|---|
| 0° tmavá hrana | 0,0122 | 0,0099 | +0,0023 |
| 0° světlá | 0,0160 | 0,0109 | +0,0051 |
| 180° tmavá | 0,0209 | 0,0062 | **+0,0147** |
| 180° světlá | 0,0223 | 0,0078 | **+0,0146** |

Rozdíl levá−pravá je u 180° **3–6× větší** než u 0°, shodně u obou intenzit.
To je použitelný signál pro zatáčení.

## Zatáčecí dráha je v datech celá

HS: 6 buněk (HSN, HSE, HSS po stranách). T4 → HS: 17 342 synapsí.
HS → DNa02: rozdělené přesně po stranách (18/38/14 na jeden DNa02,
54/29/24 na druhý). Odečet tedy patří na DNa02, ne na hrubou bilanci
všech 1 290 sestupných neuronů.

## Dva záporné běhy, které k tomu vedly
1. `most_flyvis_import.py`: úzký pruh + bilance všech DN → směr nenese
   (tmavý +0,05, světlý −0,60, opačná znaménka, null p95 0,21).
2. `most2_import.py`: širokoúhlá hrana + DNa02 → HS nevystřelily vůbec,
   protože byly ve spikujícím režimu. To vedlo k nálezu výše.

Chyba v mém kódu, kterou odhalil první běh: normalizace aktivity zvlášť pro
každý podnět zahazuje amplitudu, a právě v ní je směrová selektivita.
Správně je normalizovat globálním maximem přes všechny podněty.
