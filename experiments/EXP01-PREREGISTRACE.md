# Experiment 1 — diferenciální conditioning, PŘEDEM zapsané predikce

Zapsáno 12. 9. 2026 PŘED prvním během. Nemění se podle výsledku.

## Otázka
Upstream `add_plasticity.py` ukazuje, že dopaminová deprese KC→MBON je
kompartmentově specifická. Jenže míra deprese je do pravidla vložená ručně
(faktor = 1 − lr·gate), takže korelace 0,91 je poloviční tautologie — měří
se tím pravidlo, ne mozek.

**Naše otázka je jiná:** propíše se naučená změna v houbovitém tělísku
až do MOTORICKÉHO výstupu přes zbytek skutečného connectomu?
Cesta MBON → … → motoneurony není součástí učicího pravidla, takže na ní
se dá měřit něco, co jsme tam nevložili.

## Design
Odečet = frekvence motoneuronů po přímé stimulaci pachového vzorce KC
(5 % Kenyonových buněk), propagovaná celým connectomem (138 584 neuronů).
`weight_scale = 0,2` (stejná jako `simulate_taste.py`), 18 trialů, lr = 0,2.

Ramena:
| rameno | co se depresuje | čím je kontrola |
|---|---|---|
| `sham` | nic | determinismus; musí dát přesně 0 |
| `paired` | CS+ sloupce, gate z reálné PAM→MBON konektivity | ostré rameno |
| `placebo` | CS+ sloupce, gate PŘEHÁZENÝ mezi MBON | stejná odebraná váha, špatné kompartmenty |
| `unpaired` | CS− sloupce, reálný gate | odměna dorazí s jiným pachem |

Metrika: **D = Δrate(CS+) − Δrate(CS−)** na motoneuronech (Hz),
proti stavu před učením. Tři seedy (jiné losované pachy) = test-retest.

## Predikce s prahy
- **P1** (MBON, sanity): v `paired` klesne CS+ odpověď v PAM-inervovaných
  MBON aspoň o 10 p. b. víc než CS−. Bez toho experiment nemá co propagovat.
- **P2** (hlavní): |D_paired| > 2× max(|D_placebo|, |D_unpaired|) **a**
  zároveň |D_paired| ≥ 1,0 Hz.
- **P3** (determinismus): D_sham = 0,000 přesně. Cokoli jiného znamená
  nedeterministickou simulaci a celý běh se zahazuje.
- **P4** (stabilita): znaménko D_paired stejné ve všech třech seedech.

## Kritéria vyvrácení
- Když |D_placebo| ≥ |D_paired|: kompartmentová specificita se do chování
  NEPROPISUJE. Reportovat jako záporný nález, ne hledat jiný odečet.
- Když |D_paired| < 1 Hz ve všech seedech: MB učení je pro motorický výstup
  za těchhle podmínek neměřitelné. Taky záporný nález.
- P2 splněná jen v 1 ze 3 seedů = šum, ne efekt.

## Co experiment NEMĚŘÍ
Chování mouchy. Měří změnu frekvence motoneuronů v modelu s jednotnou
biofyzikou a volným parametrem `weight_scale`. Váhy connectome nenese.
