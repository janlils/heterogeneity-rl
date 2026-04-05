# HTM Benchmark — Stan projektu
## Do przekazania do nowego chatu

---

## 1. CEL BADAWCZY

**Pytanie:** Która architektura Multi-Agent Reinforcement Learning (MARL) najlepiej radzi sobie z heterogenicznością agentów w symulacjach ekonomicznych?

**Konkretnie:** Czy lepiej żeby każdy agent miał własny, niezależny model (per-agent, np. Deep SARSA, IPPO), czy jeden globalny model uczył się ze wszystkich akcji naraz (np. PPO globalny)?

**Wkład naukowy:** Systematyczna analiza heterogeniczności jako kontrolowanej zmiennej niezależnej (Diversity Score D ∈ [0,1]) — tego nikt wcześniej nie zrobił w kontekście porównania architektur MARL. Poprzednie prace (TaxAI 2024, EPyMARL 2021) traktują heterogeniczność jako stałe tło, nie zmienną eksperymentalną.

**Cel publikacyjny:** ICAIF lub AAMAS (konferencja), docelowo rozszerzenie do journal paper.

---

## 2. MODEL EKONOMICZNY — Rynek spekulacyjny

### Kluczowa decyzja: odrzucenie modelu G&S

Poprzednia wersja używała modelu Gode & Sunder (1993) — rynek dóbr ze stałymi rolami kupiec/sprzedawca. Problem: efficiency = 1.0 zawsze, RL nie miał czego się uczyć.

### Aktualny model: heterogeneous beliefs (rynek finansowy)

**Brak stałych ról.** Każdy agent ma prywatną wycenę fundamentalną aktywa (`valuation`). Rola wynika dynamicznie:

```
jeśli valuation - cena_rynkowa > threshold  → KUP
jeśli cena_rynkowa - valuation > threshold  → SPRZEDAJ
jeśli |valuation - cena| < threshold        → PASS
```

**Uzasadnienie ekonomiczne:**
- De Long et al. (1990): handel wynika z różnicy przekonań
- Milgrom-Stokey no-trade theorem: przy D=0 wszyscy mają tę samą wycenę → brak transakcji (POPRAWNIE)
- Santa Fe Artificial Stock Market — klasyczny benchmark ABM

**Diversity Score D ∈ [0,1] kontroluje rozrzut wycen:**
```
D=0: wszyscy mają valuation = eq (no-trade theorem → 0 transakcji)
D=0.5: wyceny w [eq-0.2, eq+0.2] → umiarkowany handel
D=1: wyceny w [0.1, 0.9] → intensywny handel, duże różnice przekonań
```

---

## 3. ŚRODOWISKO — double_auction.py

### Parametry (EnvConfig)
```python
n_agents         = 20          # agentów (bez podziału na role)
round_multiplier = 2.0         # max_steps = 2.0 × 20 = 40 kroków/rundę
n_aggression_levels = 10       # poziomów agresywności oferty
n_actions        = 11          # 10 poziomów + 1 PASS (indeks 10)
trade_threshold_base = 0.02    # min |val - price| żeby handlować
n_obs            = 12          # wymiar wektora obserwacji
```

### Przestrzeń akcji (relatywna do własnej wyceny)
```
Kupujący (val > price):
  action=0:   bid = 0              (konserwatywny, prawie nigdy nie trafia)
  action=5:   bid = val × 5/9     (umiarkowany)
  action=9:   bid = val × 9/9     (agresywny, gwarantuje transakcję)
  action=10:  PASS

Sprzedający (val < price):
  action=0:   ask = 1.0            (konserwatywny)
  action=5:   ask = 1-(1-val)×5/9 (umiarkowany)
  action=9:   ask = val            (agresywny, gwarantuje transakcję)
  action=10:  PASS
```

### Obserwacja (12D) per agent
```
[0]  valuation         własna wycena aktywa
[1]  ref_price         aktualna cena rynkowa
[2]  value_signal      clip(val - price + 0.5, 0,1) — siła i kierunek sygnału
[3]  best_bid          najlepsza oferta kupna
[4]  best_ask          najlepsza oferta sprzedaży
[5]  spread            best_ask - best_bid
[6]  frac_traded       ułamek agentów którzy już handlowali
[7]  gamma             własny discount factor
[8]  wealth_norm       majątek znormalizowany
[9]  expected_price    oczekiwana cena wg przekonań (EMA)
[10] price_trend       szacowany trend (znormalizowany)
[11] price_momentum    momentum z ostatnich transakcji
```

### Reward
Czysty surplus z transakcji. **BEZ gamma^step** (niszczyło uczenie). PASS = 0.

### Dynamika rynku (MarketDynamics)
```python
stable()    # eq=0.5 zawsze (baseline G&S)
random_eq() # eq losowane per rundę z [0.32, 0.68]
drifting()  # eq zmienia się w trakcie rundy (szoki)
```

### Kluczowe metody
```python
da.reset(diversity_score=d, seed=s)   # tworzy nową populację
da.reset_market_only()                # reset tylko rynku, agenci niezmienieni (Opcja B)
da.parallel_step(actions)             # wszyscy agenci jednocześnie → (obs, rew, dones, infos)
da.episode_metrics()                  # efficiency, gini, n_trades, action fractions
```

---

## 4. AGENCI — parametry heterogeniczności

Każdy agent (AgentParams):
```python
valuation   # prywatna wycena [0,1] — główny wymiar heterogeniczności
threshold   # min |val-price| do handlu — różna "wrażliwość"
gamma       # discount factor [0.5, 0.99] — horyzont czasowy
wealth      # majątek (Pareto przy D=1) — ograniczenie budżetowe kupca
belief:
  update_speed    # EMA alpha — jak szybko zmienia oczekiwania
  anchoring_bias  # zakotwiczenie do pierwszej ceny (Kahneman)
  loss_aversion   # straty bolą X razy mocniej [1.0, 3.0]
  panic_factor    # panika przy gwałtownym spadku ceny
  patience        # czekanie na niższą cenę przy spadku
```

DiversityConfig — każdy wymiar można włączyć/wyłączyć osobno (eksperymenty ablacyjne):
```python
valuation_spread  = True
threshold_spread  = True
gamma_spread      = True
wealth_spread     = True
belief_spread     = True
```

---

## 5. ALGORYTM — Deep SARSA

### Architektura sieci (per agent, numpy bez PyTorch)
```
Input(12) → Dense(64, ReLU) → Dense(64, ReLU) → Output(11)
~5700 parametrów per agent
Dla 20 agentów: ~114 000 parametrów łącznie
```

### Aktualizacja SARSA (on-policy TD(0))
```
Q(s,a) ← Q(s,a) + α × [r + γ × Q(s',a') - Q(s,a)]
gdzie a' = π(s') — akcja z aktualnej polityki (on-policy, nie argmax)
```

### Hiperparametry
```python
hidden_size   = 64
lr            = 3e-3
epsilon_start = 0.35
epsilon_end   = 0.05
epsilon_decay = 0.993   # raz per epizod
grad_clip     = 1.0
```

---

## 6. PĘTLA TRENINGOWA — kluczowe decyzje

### Opcja B: stała populacja per (D, seed)
```
Jedna populacja przez 500 epizodów.
sieć_0 zawsze = agent z valuation=0.73, gamma=0.87
sieć_1 zawsze = agent z valuation=0.45, gamma=0.62
...

Każda sieć uczy się strategii JEDNEGO konkretnego agenta.
(poprzednio: co epizod nowa populacja → sieć uśredniała po wszystkich agentach
 → de facto model globalny, sprzeczne z hipotezą badawczą)
```

### Multi-round: N_ROUNDS = 5 rund per epizod
```
Epizod = 5 rund × 40 kroków = 200 kroków
Aktualizacji per agent per epizod: ~10 (vs. 1-2 poprzednio)
Po 500 epizodach: ~5000 aktualizacji (vs. ~500-1000 poprzednio)
```

### Parametry treningu
```python
DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]
N_AGENTS         = 20
N_EPISODES       = 500
N_ROUNDS         = 5      # rund per epizod
N_SEEDS          = 3
```

---

## 7. PLIKI PROJEKTU

```
htm_project/
├── config.py                        ✅ gotowy
├── envs/
│   └── double_auction.py            ✅ gotowy (model spekulacyjny + reset_market_only)
├── agents/
│   └── deep_sarsa.py                ✅ gotowy (sieć numpy, on-policy)
├── experiments/
│   └── train_deep_sarsa.py          ✅ gotowy (Opcja B + multi-round)
├── analysis/
│   └── visualize.py                 ✅ gotowy (8 wykresów diagnostycznych)
├── logs/   plots/   results/        ✅ tworzone automatycznie
```

### Uruchomienie
```bash
cd htm_project
pip install numpy matplotlib pandas        # jedyne zależności

python envs/double_auction.py              # walidacja (~0.2s)
python analysis/visualize.py              # wykresy diagnostyczne (~2-3 min)
python experiments/train_deep_sarsa.py    # pełny trening (~30-60 min)
```

---

## 8. CO ZOSTAŁO DO ZROBIENIA

### Priorytet 1 — brakujące algorytmy (bez nich nie ma artykułu)
```
PPO globalny   — jeden model dla wszystkich agentów (główny "przeciwnik")
IPPO           — per-agent PPO (podobny do Deep SARSA ale off-policy z clippingiem)
MAPPO          — centralny krytyk (najtrudniejszy, najciekawszy teoretycznie)
```

### Priorytet 2 — grid eksperymentów
```
D × N_agentów × warunek_rynku × algorytm × seed
Dla konferencji: D=[0..1], N=[20,50], warunek=[stable, random_eq], seeds=30
```

### Priorytet 3 — wykresy artykułu
Główny wykres: efficiency jako funkcja D dla każdego algorytmu.
Hipoteza do weryfikacji:
```
D=0: wszystkie modele podobne
D=1: per-agent (Deep SARSA, IPPO) >> PPO globalny
```

### Opcja 2 (future work) — multi-unit trading
Agenci mogą handlować wielokrotnie w epizodzie (mają budżet i inventory).
Bliższe prawdziwemu rynkowi finansowemu. Wymaga refaktoryzacji środowiska.

---

## 9. KRÓTKI ABSTRAKT (wersja robocza)

*Does Agent Heterogeneity Break Centralized Reinforcement Learning?
Evidence from a Speculative Multi-Agent Market*

Economic systems are populated by agents with heterogeneous beliefs about asset values — a feature central to modern finance theory (De Long et al., 1990) yet largely ignored in comparative evaluations of multi-agent reinforcement learning (MARL). We introduce the Heterogeneous Trader Market (HTM): a speculative double-auction environment in which agents hold private valuations of a risky asset and decide whether to buy, sell, or abstain based on the gap between their valuation and the current market price. Agent diversity is parameterized by a scalar Diversity Score D ∈ [0, 1], enabling systematic study of how growing heterogeneity affects learning outcomes. We compare four MARL architectures spanning the centralization spectrum — Deep SARSA (fully independent, per-agent networks), PPO (single global policy), Independent PPO, and MAPPO — across six levels of D and multiple agent population sizes. Evaluation relies on allocative efficiency (validated against a zero-intelligence baseline consistent with the Milgrom-Stokey no-trade theorem at D=0), outcome inequality (Gini coefficient), and sample efficiency. We hypothesize that centralized architectures degrade systematically as D increases, while per-agent models maintain higher efficiency under strong heterogeneity.

---

## 10. LITERATURA KLUCZOWA

| Praca | Rola |
|-------|------|
| Gode & Sunder (1993) | ZI baseline: ~82% efficiency z losowymi agentami |
| De Long et al. (1990) | Heterogeneous beliefs → handel spekulacyjny |
| Milgrom & Stokey (1982) | No-trade theorem: D=0 → brak transakcji (weryfikacja modelu) |
| Kahneman & Tversky (1979) | Loss aversion, anchoring w BeliefState |
| TaxAI (Mi et al. 2024) | MARL w modelu Bewley-Aiyagari — nie porównuje architektur |
| EPyMARL (Papoudakis 2021) | Porównanie MARL ale na środowiskach kooperacyjnych |
| Scheinkman & Xiong (2003) | Spekulacja z heterogenicznymi przekonaniami |
