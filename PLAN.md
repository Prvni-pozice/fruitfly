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

- **2. Uzavřená smyčka + učení nad ní** (exp02, HOTOVO 12. 9. 2026).
  `flyagent.py`: vjem → mozek → akce → odměna → změna vah. Smyčka se naučí
  zatáčet podle vjemu (paired +0,268 proti placebu +0,042, 20/20 losů,
  chování překlopeno v 16/20) a po obrácení pravidla se PŘEUČÍ (10/10,
  kontrola 0/10) — je to učení kontingence, ne drift.
  `experiments/EXP02-VYSLEDEK.md`.

- **4. Vjem z obrazovky** (exp03, HOTOVO 13. 9. 2026). `retina.py` promítá
  obrázek na 768 sloupců podle publikovaného přiřazení FlyWire. Poloha se
  v mozku zachová (chyba 0,013, korelace 0,991) i několik synapsí hluboko.
  Dráha ON je v modelu nepoužitelná (L1 je inhibiční) — kóduje se tma.
  `experiments/EXP03-VYSLEDEK.md`.

- **5. Spojit vjem s akcí** (13. 9. 2026): ZÁPORNÝ NÁLEZ, mění plán.
  Retinotopický podnět k sestupným neuronům použitelně NEDOJDE (populace DN
  nerozliší ani osvětlené oko: 8/16 = náhoda). Příčina: optický lalok je
  z velké části nespikující, spikující LIF je na něj špatný nástroj.
  Vedlejší nález: upstream simulátor nemá synaptickou integraci, takže T4/T5
  nemohou vystřelit vůbec — doplněno v `sim2.py`, kotva se tím mění na
  **ws = 0,0393**. Detailně `experiments/NALEZ-OPTICKY-LALOK.md`.

- **5b. Graduovaná vrstva místo obcházení** (13. 9. 2026, HOTOVO).
  Optický lalok se neobchází, nahrazuje se validovaným modelem: flyvis
  (Lappalainen et al., Nature 2024) + vlastní graduovaná vrstva nad vahami
  FlyWire. HS buňky nesou směr a jsou lateralizované (rozdíl L−P u 180°
  třikrát až šestkrát větší než u 0°). `experiments/NALEZ-ARCHITEKTURA.md`.

## Kde jsme
- **6. Uzavřít smyčku přes novou architekturu.** HS/LC → spikující model →
  DNa02 → akce, a nad tím zopakovat učení z exp02. Teprve tohle je
  „vidí a řídí".
- ~~5b. Obejít optický lalok~~ (nahrazeno bodem 5b výše). Budit rovnou vizuální projekční neurony
  (LC/LPLC, 7 682 kusů) — ty spikují a v DN dávají 1,7–5,8 Hz při ~1 %
  aktivity, lateralizovaně (LPLC2 vpravo +0,181, LC4 vpravo +0,122).
  Rysy ze snímku se spočítají mimo model a nasadí na LC podle jejich známé
  funkce. Cena: 77 812 neuronů optického laloku se z modelu vyřazuje a
  moucha „vidí" naším kódem. Musí se to psát nahlas u každého výsledku.
  Nejdřív ověřit preregistrovaně, že LC → DN nese SMĚR (levý vs pravý
  podnět) nad rámec nullu z přehozených map.

## Dál — cesta k cíli

Cíl zůstává: octomilka hraje hru v prohlížeči. Zbývá k němu pět kroků a
jeden rozcestník. Pořadí je dané závislostmi, ne chutí.

### 6. Učení s odloženou odměnou — UZAVŘENO 13. 9. 2026, ZÁPORNĚ

Čtyři pokusy (exp09–exp12), pokaždé s platným měřidlem (`sham` 0,000
ve všech 48 kontrolních bězích). Postupně vyloučeno: odklad odměny
(exp10), hrubost odečtu, rozptyl mezi epizodami (exp09), špatně umístěný
explorační šum (exp11). Poslední pokus učil jen 12 synapsí na rozhraní
a dal **náznak pod prahem**: paired +5,6 proti placebu +2,2, tedy rozdíl
+3,4 p. b. proti zapsanému prahu 5, párově 7/12.

**Práh se dodatečně neohýbá.** Odměnové učení v téhle podobě se uzavírá
jako nepotvrzené. Kdyby se k němu projekt vracel, cesta je jasná: efekt
+3,4 p. b. potřebuje k rozhodnutí zhruba 40 losů místo 12, což je ~5 hodin
strojového času — udělat to až tehdy, když bude čím podepřít, že to stojí za to.

### 6c. Proč se nic nenaučilo — VYSVĚTLENO 14. 9. 2026

Čtyři záporné pokusy o učení měly společnou příčinu, kterou žádná
preregistrace ani placebo nechytily, protože obojí hlídá pravdivost
výsledku, ne splnitelnost zadání.

**Učené rozhraní neslo 1 % vstupu.** Dvanáct synapsí z graduované vrstvy
na DNa02 má váhu 151 a 87; zbylých 746 synapsí ze spikující části má
13 784 a 14 380. Tedy **1,1 % a 0,6 %**. Učil jsem jedno procento drive
rozhodujícího neuronu a divil se, že se chování nehýbe.

**Test splnitelnosti** (`exp15_je_to_resitelne.py`) to odhalil hrubou
silou: náhodné přenastavení učených vah, včetně obracení znamének,
nezlepší výsledek o víc než 6 p. b. — a to u tří různých úloh shodně.
U dvanácti parametrů je takový test průkazný; u 746 už ne (osm vzorků
v sedmi stech rozměrech nenajde nic), takže tam absence nálezu nic
nedokazuje.

**Dvě úlohy byly navíc špatně položené:**
- „drž cíl co nejdál" je pro detektor pohybu s omezeným zorným polem
  nesplnitelná: za okrajem pohledu moucha nevidí, nemá podle čeho
  zatáčet a cíl se vrátí. Úspěch by vyžadoval udržet stav bez informace.
- „drž cíl ve středu" má strop: po opravě očí to moucha umí na 85 %.

**Nové pravidlo pro každý další pokus o učení:** před během ověřit hrubou
silou, že existuje nastavení učených parametrů, které úlohu řeší. Bez toho
je záporný nález nálezem o zadání, ne o učení.

### 6b. Hra na vrozeném chování — HOTOVO
Moucha sleduje cíl, protože to má v zapojení (exp07), ne protože se to
naučila. Je to míň, než byl původní cíl, ale je to hratelné a poctivé.
Učení zůstává otevřenou samostatnou větví, ne podmínkou hry.

Pevná zkušební sada 8 epizod, stejná pro všechna ramena, při jejím hraní se
neučí. První predikce je kontrola měřidla: `sham` se nesmí pohnout o víc než
2 p. b., jinak je běh neplatný.

**Rozcestník podle výsledku:**

| výsledek | co dál |
|---|---|
| měřidlo neplatné (P1 padne) | zvětšit zkušební sadu a zkrátit epizody; bez funkčního měřidla nemá smysl zkoušet jiná pravidla |
| měřidlo dobré, učení funguje | rovnou krok 7 — složitější úloha (dva cíle, vyhýbání) |
| měřidlo dobré, učení nefunguje | **vidlička níž** — tři možnosti, v tomhle pořadí |

Vidlička, když se pravidlo nenaučí (nejlevnější první):
1. **Učit graduovanou vrstvu, ne spikující.** Mapa vjem → akce sedí na
   přechodu LC/HS → DN. Gradient se tam počítá snáz a je to jen 6 buněk HS
   proti 60 tisícům neuronů. Levné, rychlé, dobře měřitelné.
2. **Zmenšit úlohu na jeden krok s odměnou.** Ověřit, že pravidlo vůbec
   funguje bez odkladu (exp02 to ukázal na jiné úloze — zopakovat na téhle),
   a odklad přidávat po jednom kroku, dokud to nepraskne. Tím se zjistí,
   kde přesně je hranice.
3. **Přijmout vrozené chování a učení oddělit.** Hra může stát na
   optomotorickém reflexu (exp07), který v zapojení už je, a učení řešit
   jako samostatnou větev. Poctivé, ale je to ústup od cíle.

### 7. Úloha, která se podobá hře
Dva cíle místo jednoho (jeden odměňuje, druhý trestá), aby akce musela být
podmíněná vjemem, ne jen reflexem. Kontrola: rameno s prohozeným významem
cílů — když se moucha přeučí, je to politika, ne reflex.

### 8. Rychlost na živý provoz
Teď 0,25 s na krok (4 kroky/s). Na hru stačí, ale s rezervou:
- dávkování více kroků do jednoho násobení matic,
- zkrátit simulaci ze 120 na 60 kroků a ověřit, že odečet drží,
- profilovat, kolik času žere graduovaná vrstva proti spikující.
Cíl: 10 kroků/s. Když to nepůjde, hra poběží zpomaleně — to je přijatelné,
moucha není akční hráč.

### 9. Napojení na hru
Server drží mozek a vystavuje jediný koncový bod: snímek dovnitř, akce ven.
Hra běží v prohlížeči a volá ho. Nasazení vedle webu (`web/` na Vercelu,
mozek na našem VPS). **Až sem, ne dřív** — dokud není co řídit, je to jen
hezčí obal.

### 10. Průběžně: web a zápis
Po každém pokusu přibude článek do `web/src/data/pokusy.json` a výsledky se
pushnou. Záporné nálezy se zapisují stejně jako kladné — zatím jsou
zajímavější.

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
7. **Práh pro kontrolu se nesmí střílet od boku — null se napřed změří.**
   V exp03 byl práh 0,3 nesplnitelný, protože null měl medián 0,35.
8. **Pro polohové úlohy měřit chybu dekódování, ne korelaci.** Chyba
   oddělila signál od nullu 18×, korelace jen 1,4×.
9. **Rešerši stavu poznání dělat PŘED stavěním.** Graduovanost optického
   laloku i hotový model (flyvis) byly publikované; objevovali jsme je
   znovu experimentem. Známé znalosti použít, pak ověřit experimentem.
