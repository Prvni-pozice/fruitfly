# Plán: od connectomu k hraní hry v prohlížeči

Cíl: octomilka (simulovaný connectome FlyWire, 138 584 neuronů) hraje naše
hry minecraftového typu v prohlížeči. Postupuje se po malých krocích, každý
má vlastní preregistraci a kontrolní ramena. Krok se nezapočítá, dokud
neprojde kontrolami — záporný nález je taky výsledek a zapisuje se.

## Hotovo
- **0. Instalace a ověření** (12. 9. 2026). FlyWire v783 na VPS, sparse LIF,
  cukr → sosák s latencí 15 ms. `CLAUDE.md`.
- **1. Preregistrovaný experiment o učení** (exp01). Zjištěno: učení
  v houbovitém tělísku se k motoneuronům spolehlivě nepropisuje; los pachu
  je největší zdroj rozptylu (sd 7,85 Hz); podlaha šumu 2,36 Hz;
  na jednotlivém losu je efekt nečitelný. `experiments/EXP01-VYSLEDEK.md`.
- **1b. Ukotvení `weight_scale` = 0,124** ze zveřejněných parametrů
  Shiu et al. 2024, nezávisle na našem odečtu. Bez toho se obracelo
  znaménko závěru. `experiments/UKOTVENI-WEIGHT-SCALE.md`.

## Kde jsme
- **2. Uzavřená smyčka + řiditelný akční kanál** (`experiments/flyagent.py`).
  Akce = bilance sestupných neuronů (638 vlevo / 644 vpravo), vjem = vizuální
  neurony po stranách. Změřeno: kanál je dekódovatelný (rozdíl bilance 0,21).
  BĚŽÍ exp02 — naučí se smyčka zatáčet podle vjemu? Preregistrace hotová.

## Dál
- **3. Prostor akcí větší než dvě.** ČÁSTEČNĚ VYŘEŠENO 12. 9. 2026:
  všech 1 290 sestupných neuronů má v datech pojmenovaný typ a jsou mezi
  nimi doložené povelové neurony, každý po jednom na stranu:
  | typ | n | funkce podle literatury |
  |---|---|---|
  | DNa02 | 2 (1+1) | zatáčení, klasický steering neuron |
  | DNa01 | 2 (1+1) | zatáčení |
  | DNp09 | 2 (1+1) | zastavení / freezing |
  | MDN | 4 (2+2) | couvání |
  Akční prostor {vpřed, vlevo, vpravo, stát, vzad} jde tedy číst z
  JMENOVITÝCH neuronů, ne z hrubého průměru přes 638 DN. Zbývá ověřit, že
  jdou budit a číst nezávisle na sobě (exp03).
- **4. Vjem z obrazovky.** Převod snímku hry na stimulaci vizuálních
  neuronů s respektem k retinotopii (optický lalok má 77 812 neuronů
  a strukturu; nesmí se z toho udělat náhodné plácnutí do poolu).
  Kontrola: dva různé snímky musí dát různý vzorec aktivity, dva podobné
  podobný.
- **5. Úloha s odloženou odměnou.** Ve hře nepřijde odměna hned. Potřebuje
  to eligibility trace přes víc kroků. Tady čekám největší problém.
- **6. Rychlost.** Teď jeden průchod smyčkou trvá ~1,5 s na CPU. Hra
  potřebuje aspoň 10 kroků za sekundu → 15× zrychlení, nebo běh v dávkách
  mimo reálný čas (hra se přehraje offline, ne živě).
- **7. Napojení na hru.** Server drží mozek, hra v prohlížeči posílá snímky
  a dostává akce. Až sem, ne dřív.

## Pravidla, která platí pro každý krok
1. Preregistrace s prahy PŘED během. Kritéria vyvrácení taky.
2. Vždy placebo rameno (stejná velikost zásahu, bez obsahu) a kontrola
   determinismu (`sham` musí dát přesně 0).
3. Minimálně 20 losů. Tři nestačí — exp01 ukázal falešný pozitivní nález
   na prvním seedu.
4. Efekt porovnat s podlahou šumu z kolektivní perturbace, ne jen s kontrolou.
5. Volné parametry ukotvit nezávisle na měřené veličině.
6. Co je z connectomu a co je naše inženýrská volba, se píše zvlášť.
   Zapojení je doložené. Učicí pravidlo mimo houbovité tělísko není.
