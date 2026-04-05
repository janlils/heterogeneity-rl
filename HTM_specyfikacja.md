# HTM Benchmark — Specyfikacja projektu
## Stan na: koniec sesji projektowej (do odtworzenia w kolejnym chacie)

---

## 1. Kontekst badawczy

### Pytanie badawcze
Która architektura Multi-Agent Reinforcement Learning (MARL) najlepiej radzi sobie z heterogenicznością agentów w symulacjach ekonomicznych?

Konkretnie: czy lepiej żeby każdy agent miał **własny, niezależny model** (per-agent, np. SARSA, IPPO), czy jeden **globalny model** uczył się ze wszystkich akcji (np. PPO)?

### Cel końcowy
Artykuł konferencyjny (ICAIF lub AAMAS), docelowo rozszerzony do journal paper.

### Kluczowy wkład
Systematyczna analiza heterogeniczności jako **kontrolowanej zmiennej niezależnej** (Diversity Score D) — tego nikt wcześniej nie zrobił w kontekście porównania architektur MARL. Poprzednie prace (TaxAI, EPyMARL) traktują heterogeniczność jako stałe tło, nie zmienną eksperymentalną.

---

## 2. Model rynkowy — WAŻNA DECYZJA PODJĘTA NA KOŃCU SESJI

### Model który był (G&S — rynek dóbr) — ODRZUCONY
- Agenci mają stałe role: kupiec lub sprzedawca
- Kupcy zawsze mają wyceny powyżej eq, sprzedawcy poniżej
- Problem: efficiency = 1.0 zawsze, SARSA nie ma czego się uczyć

### Model który będzie (spekulacyjny — rynek finansowy) — DO IMPLEMENTACJI
Każdy agent jest **jednocześnie potencjalnym kupcem i sprzedawcą**.

**Kluczowa idea:** Agent ma prywatną wycenę fundamentalnej wartości aktywa (`valuation`). Porównuje ją z aktualną ceną rynkową i decyduje:

```
jeśli valuation - cena_rynkowa > threshold  → KUP (aktywo niedowartościowane)
jeśli cena_rynkowa - valuation > threshold  → SPRZEDAJ (aktywo przewartościowane)
jeśli |valuation - cena| < threshold        → PASS (nie handluj)
```

**Uzasadnienie ekonomiczne:** Heterogeneous beliefs literature (De Long et al. 1990, Scheinkman & Xiong 2003), Santa Fe Artificial Stock Market. Handel powstaje z **różnicy przekonań** — przy D=0 wszyscy mają tę samą wycenę co rynek i nikt nie handluje (Milgrom-Stokey no-trade theorem).

**Co to zmienia dla pytania badawczego:**
- Agent musi nauczyć się TRZECH rzeczy: czy handlować, w którym kierunku, po jakiej cenie
- SARSA per-agent z indywidualnym threshold nauczy się własnej strategii
- PPO globalny z jednym threshold dla wszystkich będzie systematycznie błędny przy wysokim D
- To jest silniejszy argument dla heterogeniczności niż różne gamma

### Przestrzeń akcji — DO IMPLEMENTACJI
Akcje **relatywne względem private_value**, nie absolutne w skali [0,1]:

```
Kupiec (valuation > price):
  action_idx ∈ [0, N-1]
  bid = valuation * (action_idx / (N-1))
  action 0   → bid = 0           (nie wejdzie w transakcję)
  action N-1 → bid = valuation   (gwarantuje transakcję)

Sprzedawca (valuation < price):
  action_idx ∈ [0, N-1]
  ask = valuation + (1 - valuation) * (action_idx / (N-1))
  action 0   → ask = valuation  (gwarantuje transakcję)
  action N-1 → ask = 1.0        (nie wejdzie w transakcję)

Akcja specjalna "PASS":
  action_idx = N  → agent nie składa oferty w tym kroku
  reward = 0
```

---

## 3. Architektura systemu

### Pliki projektu

```
htm_project/
├── config.py                    ← GOTOWE (zaktualizowane)
├── envs/
│   ├── __init__.py
│   └── double_auction.py        ← GOTOWE (zaktualizowane), DO PRZEPISANIA (model spekulacyjny)
├── agents/
│   ├── __init__.py
│   ├── sarsa_agent.py           ← GOTOWE (tablica Q, do zamiany na deep SARSA)
│   └── rllib_configs.py         ← DO NAPISANIA (PPO, IPPO, MAPPO)
├── experiments/
│   ├── __init__.py
│   ├── train_sarsa.py           ← GOTOWE (naprawione)
│   └── grid_search.py           ← DO NAPISANIA
├── analysis/
│   └── metrics.py               ← DO NAPISANIA
├── logs/                        ← tworzone automatycznie
├── plots/                       ← tworzone automatycznie
└── results/                     ← tworzone automatycznie
```

### Stan gotowości plików

| Plik | Stan | Uwagi |
|------|------|-------|
| `config.py` | ✅ Gotowy | MarketDynamics, N jako parametr, max_steps=f(N) |
| `double_auction.py` | ⚠️ Wymaga przepisania | Działa ale model G&S — zamienić na spekulacyjny |
| `sarsa_agent.py` | ⚠️ Działa ale słaby | Q-tablica — zamienić na deep SARSA (sieć neuronowa) |
| `train_sarsa.py` | ✅ Gotowy | Używa parallel_step(), spójny z API |
| `rllib_configs.py` | ❌ Brak | PPO, IPPO, MAPPO — do napisania |
| `grid_search.py` | ❌ Brak | Pełny grid eksperymentów |

---

## 4. Kluczowe decyzje projektowe

### 4.1 Środowisko

**Typ:** ParallelEnv (nie AEC) — wszyscy agenci działają jednocześnie.
Ekonomicznie bliższe CDA (continuous double auction).

**max_steps:** Automatycznie obliczane: `round_multiplier × n_agents`
- `round_multiplier = 2.0` → ~90% efficiency dla ZI
- Zmiana N agentów automatycznie skaluje czas epizodu

**Cena równowagi:** Dynamiczna (nie stała 0.5).
Trzy warunki środowiskowe:
- `MarketDynamics.stable()` — eq=0.5 zawsze (G&S replication)
- `MarketDynamics.random_eq()` — eq losowane per epizod z [0.3, 0.7]
- `MarketDynamics.drifting()` — eq zmienia się w trakcie epizodu (szoki)

**Ograniczenie budżetowe:** Kupiec nie może licytować ponad `min(valuation, wealth)`.
`wealth` jest teraz aktywnym ograniczeniem, nie dekoracją.

**Reward:** Czysty surplus z transakcji. Bez `gamma^step` — to niszczyło uczenie (przy step=30, gamma=0.9: reward × 0.04 ≈ 0).

### 4.2 Agenci

**Heterogeniczność (Diversity Score D ∈ [0,1]):**
- D=0: wszyscy identyczni (representative agent — punkt wyjścia)
- D=1: maksymalne zróżnicowanie wszystkich parametrów

**Cztery wymiary heterogeniczności (kontrolowane osobno przez DiversityConfig):**
1. `valuation_heterogeneity` — prywatne wyceny fundamentalne (nowy model: bez podziału na kupców/sprzedawców)
2. `gamma_heterogeneity` — horyzonty czasowe [0.5, 0.99]
3. `wealth_heterogeneity` — majątek z rozkładu Pareto
4. `belief_heterogeneity` — parametry poznawcze:
   - `update_speed` — szybkość aktualizacji oczekiwań (EMA)
   - `anchoring_bias` — zakotwiczenie do pierwszej ceny (Kahneman)
   - `loss_aversion` — straty bolą loss_aversion × mocniej [1.0, 3.0]
   - `panic_factor` — irracjonalna panika przy spadku ceny
   - `patience` — czekanie na lepszą cenę przy trendzie spadkowym

**Eksperymenty ablacyjne:** `DiversityConfig.economic_only()`, `DiversityConfig.beliefs_only()`

### 4.3 Algorytmy RL

**Do porównania:**

| Model | Typ | Heterogeniczność | Charakter |
|-------|-----|-----------------|-----------|
| ZI baseline | brak uczenia | — | punkt zero G&S |
| Deep SARSA | per-agent, on-policy, sieć NN | pełna | indywidualistyczny |
| PPO globalny | jeden model | brak | centralistyczny |
| IPPO | per-agent PPO | pełna | hybrydowy |
| MAPPO | centralny krytyk | częściowa | hybrydowy |

**Deep SARSA (priorytet implementacji):**
- Mała sieć neuronowa (2 warstwy, 64 neurony) per agent zamiast tablicy Q
- Zachowanie on-policy i per-agent (indywidualne gamma)
- Uogólnianie między stanami — kluczowe przy krótkich epizodach
- Uzasadnienie: tablica Q przy 46656 stanach i 2-4 turach na epizod prawie nic się nie uczy

**Dlaczego deep SARSA a nie Q-table:**
Przy N=20 agentach i max_steps=40, każdy agent dostaje ~2 tury per epizod.
Przestrzeń stanów: 6^6 = 46,656. Większość stanów nigdy nie odwiedzona.
Sieć NN uogólnia między nieodwiedzonymi stanami — Q-tablica nie.

### 4.4 Grid eksperymentów

```
Model RL:           [ZI, DeepSARSA, PPO, IPPO, MAPPO]
Diversity Score D:  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N agentów:          [20, 50, 100]   ← N jako trzecia oś
Warunek rynku:      [stable, random_eq, drifting]
Seeds:              30 (konferencja), 5 (szybki test)
```

**Dla konferencji:** N=[20,50], warunki=[stable, random_eq], seeds=30
**Dla journal:** pełny grid z N=100 i drifting

### 4.5 Metryki

**Ekonomiczne (główne):**
- `allocative_efficiency` — suma surplusów / max możliwy surplus (główna)
- `gini_coefficient` — nierówność wyników między agentami
- `price_discovery_steps` — ile kroków do osiągnięcia ceny równowagi

**ML:**
- `sample_efficiency` — ile epizodów do osiągnięcia X% efficiency
- `mean_td_error` — zbieżność uczenia
- `mean_epsilon` — poziom eksploracji

---

## 5. Narracja artykułu

```
Stabilny rynek (eq=0.5), D=0:
  → Wszystkie modele podobne, PPO globalny wystarczy

Stabilny rynek, D=1:
  → Per-agent modele lepsze (główna hipoteza)
  → SARSA uczy się indywidualnego threshold, PPO uśrednia

Niestabilny rynek (random eq), D=1:
  → Deep SARSA i IPPO wygrywają (uogólnianie na nowe eq)
  → Q-table SARSA zawodzi (nie widziała tego stanu)

Niestabilny + drift, D=1:
  → MAPPO może wygrać (centralny krytyk wykrywa zmianę reżimu)
```

**Kluczowy wykres:**
```
Efficiency
  ^
  |  IPPO ────────────────────────
  |  DeepSARSA ──────────────────╲
  |  MAPPO ─────────────╲         ╲
  |  PPO ────────╲        ╲        ╲
  |               ╲
  +──────────────────────────────────> D
     0.0   0.2   0.4   0.6   0.8   1.0
```

---

## 6. Co zrobić jako pierwszy krok w kolejnym chacie

### Priorytet 1 — Przepisać AgentPopulation na model spekulacyjny

Zmienić `_generate()` w `double_auction.py`:

```python
# Zamiast stałych ról kupiec/sprzedawca:
# Każdy agent ma valuation ~ Uniform losowane z rozkładu
# zależnego od D (rozrzut wycen rośnie z D)

# D=0: wszyscy mają valuation = eq (nikt nie handluje — no-trade theorem)
# D=0.5: wyceny w [eq-0.2, eq+0.2] — umiarkowane różnice
# D=1: wyceny w [0.1, 0.9] — duże różnice

# Brak podziału na buyers/sellers w populacji
# Rola (buy/sell/pass) wynika z decyzji agenta w każdym kroku
```

### Priorytet 2 — Nowa przestrzeń akcji

Zmienić w `parallel_step()` mapowanie action_idx → price:

```python
# Akcja N = pass (nie handluj)
# Akcje 0..N-1 = agresywność oferty (0=najagressywniejsza, N-1=najskromniejsza)

# Jeśli agent decyduje "buy":
#   bid = market_price + (valuation - market_price) * (action / (N-1))
#   action=0: bid = market_price (minimalnie powyżej, może nie trafić)
#   action=N-1: bid = valuation (gwarantuje transakcję)

# Jeśli agent decyduje "sell":
#   ask = market_price - (market_price - valuation) * (action / (N-1))
#   action=0: ask = market_price (minimalnie poniżej)
#   action=N-1: ask = valuation (gwarantuje transakcję)
```

### Priorytet 3 — Deep SARSA

Zamienić Q-tablicę na sieć neuronową w `sarsa_agent.py`:

```python
class QNetwork(nn.Module):
    def __init__(self, n_obs=12, n_actions=21):  # 21 = 20 poziomów + pass
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
            nn.Linear(64, n_actions),
        )
```

---

## 7. Odniesienia do literatury

| Praca | Znaczenie |
|-------|-----------|
| Gode & Sunder (1993) | Baseline ZI: ~82% efficiency z losowymi agentami |
| TaxAI (Mi et al. 2024) | MARL w modelu Bewley-Aiyagari — nie porównuje architektur |
| EPyMARL (Papoudakis 2021) | Porównanie MARL ale na środowiskach kooperacyjnych (SMAC) |
| HANK models | Makroekonomia HA — bez RL |
| De Long et al. (1990) | Heterogeneous beliefs → handel na rynkach |
| Kahneman & Tversky (1979) | Loss aversion, anchoring — podstawa BeliefState |
| Scheinkman & Xiong (2003) | Spekulacja z heterogenicznymi przekonaniami |

---

## 8. Uwagi techniczne

**Zależności Python:**
```bash
pip install numpy pandas matplotlib torch ray[rllib] pettingzoo gymnasium wandb
```

**Uruchomienie walidacji środowiska:**
```bash
cd htm_project
python envs/double_auction.py
```

**Uruchomienie treningu SARSA:**
```bash
python experiments/train_sarsa.py
```

**Parametry szybkiego testu (2 min):**
```python
DIVERSITY_SCORES = [0.0, 0.5, 1.0]
N_EPISODES = 100
N_SEEDS = 3
MARKET = MarketDynamics.stable()
```

**Ważna uwaga o reward:**
Reward = czysty surplus BEZ `gamma^step`. Gamma jest używana tylko w aktualizacji TD:
`Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]`
Nie dyskontuje się nagrody przed podaniem do agenta.

**Ważna uwaga o max_steps:**
`max_steps` to właściwość obliczana (nie pole) w `EnvConfig`:
```python
@property
def max_steps(self) -> int:
    return int(self.round_multiplier * self.n_agents)
```
Zmiana `n_buyers`/`n_sellers` automatycznie skaluje czas epizodu.
