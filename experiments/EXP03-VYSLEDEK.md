# Experiment 3 — obraz do mozku a retinotopie (12.–13. 9. 2026)

Preregistrace v hlavičce `exp03_retinotopie.py`, převodník `retina.py`.
49 pozic skvrny (mřížka 7×7), 20 nezávislých přehození jako null.

## Retina stojí na publikovaných datech

`column_assignment.csv.gz` z Codexu: 45 528 neuronů optického laloku
přiřazených ke sloupcům (785 vlevo, 796 vpravo) s hexagonálními
souřadnicemi p, q. L1 i L2 mají přesně 1 neuron na sloupec = vrstva pixelů.
Mřížka je pravidelná, CV vzdálenosti k sousedovi **0,004**.

Dvě slepé uličky před tím (obě zapsané, ať se neopakují):
- **Ze souřadnic to nejde.** `coordinates.csv.gz` dává jeden reprezentativní
  bod neuronu, ne pozici sloupce; proložení koulí dalo zorné pole 229°×316°.
- **Z konektivity taky ne.** Sloupce jsou paralelní kanály, L1 spolu sdílejí
  cíle jen s podobností 0,017 — graf sousednosti z toho nevznikne.

## Nález: dráha ON je v tomhle modelu nepoužitelná

L1 má v predikci neurotransmiterů GLUT (499) a GABA (281) = podle mouší
konvence INHIBIČNÍ. Buzení L1 downstream umlčí: Mi1 střílí 1 neuron ze 785,
T4 a Tm9 nula. L2 je celá ACH a signál nese (L2 → Tm1, Tm2, Tm4, L5, T1).
**Obraz se proto kóduje jako TMA na světlém poli.**

## Výsledek: poloha se v mozku zachová

| vrstva | chyba polohy | kor. u | kor. v |
|---|---|---|---|
| Tm1 | **0,013** | 0,991 | 0,993 |
| Tm2 | 0,025 | 0,991 | 0,991 |
| Tm4 | 0,020 | 0,989 | 0,986 |
| L5 | 0,041 | 0,992 | 0,991 |
| T1 | 0,040 | 0,988 | 0,986 |

Null z 20 přehození: chyba medián 0,232, |kor| medián 0,39/0,28, **max 0,693**.

W1 (chyba < 0,15) a W2 (korelace ≥ 0,8) SPLNĚNY s velkou rezervou.

## W3 NESPLNĚNA — a je to chyba kritéria, ne výsledku

W3 zněla „přehozená kontrola má korelaci pod 0,3". Null ale má medián
0,28–0,39 a maximum 0,69, takže práh byl nesplnitelný bez ohledu na to,
jak experiment dopadne. Napsal jsem ho, aniž bych null předtím změřil.

Věcně je oddělení jednoznačné: skutečná mapa 0,991 leží nad maximem nullu
0,693 a chyba je **18× menší** (0,013 proti 0,232).

**Poučení: práh pro kontrolní rameno se nesmí střílet od boku — null se má
napřed změřit.** A dál: zhuštění mřížky z 25 na 49 pozic null NEZÚŽILO
(max 0,67 → 0,69), protože přehození je pevná bijekce a vztah pozice →
těžiště má i tak systematickou strukturu (okrajové efekty, různý počet
buzených sloupců). **Pro tenhle typ úlohy je chyba dekódování mnohem lepší
metrika než korelace** — odděluje 18×, zatímco korelace jen 1,4×.

## Chyba v podnětu, kterou odhalil až kontrolní výpočet
První verze používala SVĚTLOU skvrnu na tmavém poli. Kanál OFF pak budil
celé pozadí (701 sloupců ze 768) a korelace vyšla −0,92. Málem jsem to
přečetl jako souřadnicový posun. Správně je tmavá skvrna na světlém poli.
