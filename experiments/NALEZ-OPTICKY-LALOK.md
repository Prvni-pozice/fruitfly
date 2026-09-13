# Nález: optickým lalokem signál ve spikujícím modelu neprojde (13. 9. 2026)

Vzniklo při kroku 5 plánu (spojit vjem s akcí). Je to nejdůležitější
omezení, na které projekt zatím narazil, a mění cestu k cíli.

## Co se stalo

1. **Statický podnět přes retinu k sestupným neuronům nedojde použitelně.**
   Populace 1 290 DN nerozliší ani to, které oko bylo osvětleno:
   **8/16 správně, tedy náhoda.** DNa02 (klasický steering neuron) dává
   1–7 spiků bez vzoru. Bilance DN na azimutu nezávisí (−0,07, −0,06,
   −0,06, −0,32, −0,22, −0,30, −0,03 napříč zorným polem).
2. **Pohyblivý pruh dal nula spiků v DN** (proti 193 u statického).
3. **T4 a T5 nevystřelí ani jednou**, ať je podnět jakýkoli.

## Příčina 1: upstream simulátor nemá synaptickou integraci

`flybrain/simulator.py` počítá `I = W @ spikes`, proud tedy trvá jeden krok
a v čase se NESČÍTÁ. Neuron potřebuje ~1/(0,05·ws) synapsí střílejících
současně; při ws = 0,124 je to **161 synapsí**. T5a jich má excitačních
dohromady **83** — nemůže vystřelit nikdy, bez ohledu na sílu vstupu.

Referenční model (Shiu et al. 2024) má synaptickou proměnnou s konstantou
5 ms, takže se příspěvky sčítají (zesílení ≈ 5,5×). Doplněno v
`experiments/sim2.py` (upstream se neupravuje). S integrací T5 vystřelí
1 727 spiků a DN 28 189.

**Důsledek pro ukotvení:** s integrací je struktura rovnic totožná se
Shiuem, takže kotva se odvodí napřímo: **ws = 0,275/7 = 0,0393** (ne 0,124,
což platilo pro model bez integrace).

## Příčina 2 (hlavní): optický lalok je z velké části NESPIKUJÍCÍ

Při ukotvené hodnotě 0,0393 je mozek příliš tichý (T5 nestřílí, DN mlčí);
při 0,124, kde T5 střílí, je aktivních **21,6 % neuronů**, což je na mouchu
nereálné. Není to tedy otázka parametru.

Vysvětlení je biologické: neurony optického laloku pracují s GRADUOVANÝM
napětím, ne s akčními potenciály. Záznamy z T5 vykazují jen graduované
podprahové odpovědi; ojedinělé transienty 1–2 mV nešlo potvrdit jako spiky
(eLife 2019, výpočet směrové selektivity v OFF dráze). Totéž platí pro
fotoreceptory, L1–L3 a většinu neuronů medully.

**Spikující LIF je na optický lalok špatný nástroj.** Není to chyba dat ani
naše chyba v kódu — je to mez modelu.

## Cesta dál: optický lalok obejít

Vizuální PROJEKČNÍ neurony (LC, LPLC; 7 682 kusů) spikují a jdou do
centrálního mozku. Buzeny přímo při ukotvené ws = 0,0393 dávají v DN
1,7–5,8 Hz při jen ~1 % aktivních neuronů (biologicky věrohodná řídkost)
a odpověď je lateralizovaná:

| buzeno | DN L | DN P | bilance |
|---|---|---|---|
| LPLC2 vlevo | 3,20 | 2,96 | +0,040 |
| LPLC2 vpravo | 2,45 | 1,70 | **+0,181** |
| LC4 vlevo | 4,64 | 5,21 | −0,058 |
| LC4 vpravo | 2,66 | 2,08 | **+0,122** |

Návrh: rysy ze snímku (poloha, pohyb, rozpínání) spočítat MIMO model a
nasadit je na příslušné LC/LPLC populace podle jejich známé funkce
(LPLC2 = rozpínání/looming, LC4 = rychlé přibližování, LC11 = malý objekt).

**To je ale velký ústupek a musí se psát nahlas:** optický lalok — 77 812
neuronů, víc než polovina mozku — se tím z modelu vyřazuje. Moucha pak
nevidí connectomem, vidí naším kódem. Zůstává skutečný centrální mozek,
skutečná cesta LC → DN a skutečné zapojení. Není to „moucha hraje hru",
je to „centrální mozek mouchy dostává vizuální rysy a řídí akci".
