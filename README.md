# HTM — Heterogeneous Trader Market

Benchmark MARL do porównania architektur per-agent vs globalnych w środowiskach z heterogenicznymi agentami.

## Instalacja

```bash
pip install numpy matplotlib pandas
```

Brak zależności GPU — kod działa w czystym numpy.

## Uruchomienie

```bash
# Walidacja środowiska (~0.2s)
python -m codes.double_auction

# Wykresy z wyników treningu
python -m codes.visualize

# Wykresy z szybkiego treningu
python -m codes.visualize --quick

# Trening Deep SARSA (równoległy, ~4-6 min)
python -m codes.train_deep_sarsa

# Szybki test Deep SARSA (~1 min)
python -m codes.train_deep_sarsa --quick
```

## Struktura

```
htm_project/
├── codes/
│   ├── config.py               # centralna konfiguracja
│   ├── double_auction.py       # środowisko spekulacyjne
│   ├── deep_sarsa.py           # per-agent Deep SARSA (numpy)
│   ├── train_deep_sarsa.py     # pętla treningowa z multiprocessing
│   └── visualize.py            # wykresy diagnostyczne
├── experiments/
│   └── train_deep_sarsa.py     # wrapper zgodności do codes.train_deep_sarsa
└── tests/
    └── test_env.py             # testy środowiska
```

Katalogi `plots/`, `results/`, `logs/` tworzone automatycznie.

## Model

Rynek spekulacyjny — brak stałych ról. Każdy agent ma prywatną wycenę fundamentalną aktywa (`valuation`). Handel wynika z różnic przekonań (De Long et al. 1990).

**Diversity Score D ∈ [0,1]** kontroluje heterogeniczność populacji:
- D=0: wszyscy identyczni → brak transakcji (Milgrom-Stokey no-trade theorem)
- D=1: duże różnice wycen, gamm, thresholdów, przekonań behawioralnych

**Przestrzeń akcji (5 akcji):** PASS | MARKET | LIMIT_TIGHT | LIMIT_MED | LIMIT_FAR
