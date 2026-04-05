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
python envs/double_auction.py

# Wykresy diagnostyczne (~3 min)
python analysis/visualize.py

# Trening Deep SARSA (równoległy, ~4-6 min)
python experiments/train_deep_sarsa.py
```

## Struktura

```
htm_project/
├── config.py                   # centralna konfiguracja
├── envs/
│   └── double_auction.py       # środowisko spekulacyjne
├── agents/
│   └── deep_sarsa.py           # per-agent Deep SARSA (numpy)
├── experiments/
│   └── train_deep_sarsa.py     # pętla treningowa z multiprocessing
└── analysis/
    └── visualize.py            # 8 wykresów diagnostycznych
```

Katalogi `plots/`, `results/`, `logs/` tworzone automatycznie.

## Model

Rynek spekulacyjny — brak stałych ról. Każdy agent ma prywatną wycenę fundamentalną aktywa (`valuation`). Handel wynika z różnic przekonań (De Long et al. 1990).

**Diversity Score D ∈ [0,1]** kontroluje heterogeniczność populacji:
- D=0: wszyscy identyczni → brak transakcji (Milgrom-Stokey no-trade theorem)
- D=1: duże różnice wycen, gamm, thresholdów, przekonań behawioralnych

**Przestrzeń akcji (5 akcji):** PASS | MARKET | LIMIT_TIGHT | LIMIT_MED | LIMIT_FAR
