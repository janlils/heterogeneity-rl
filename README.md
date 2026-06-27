# HTM — Heterogeneous Trader Market

Projekt do porownania algorytmow RL w heterogenicznym srodowisku tradingowym.

Aktualnie repo zawiera:
- `Deep SARSA` jako independent per-agent value learner w czystym `numpy`
- `PPO` jako shared-policy baseline
- `IPPO` jako light independent policy-gradient baseline
- `SignalRule` jako prosty benchmark regulowy oparty o prywatny sygnal

Kod historyczny i stare entrypointy zostaly przeniesione do `codes/old/` i nie sa czescia glownego pipeline'u.

## Zaleznosci

```bash
pip install numpy pandas matplotlib torch
```

`Deep SARSA` i rdzen srodowiska dzialaja w `numpy`.
`PPO` i `IPPO` wymagaja `torch`.

## Najwazniejsze komendy

Pelny benchmark wszystkich aktualnych algorytmow:

```bash
./venv/bin/python -m codes.train_all --run-tag full_run
```

Szybki smoke test:

```bash
./venv/bin/python -m codes.train_all --quick --run-tag quick_all
```

Tryb pośredni do testów rewardu i kierunku uczenia:

```bash
./venv/bin/python -m codes.train_all --medium --run-tag medium_all
```

To samo z minimalnym kosztem transakcyjnym na fill:

```bash
./venv/bin/python -m codes.train_all --medium --transaction-cost 0.0001 --run-tag medium_cost
```

Opcjonalnie mozna wlaczyc dodatkowy spread `gamma`:

```bash
./venv/bin/python -m codes.train_all --medium --gamma-spread --run-tag medium_gamma
```

Opcjonalnie mozna wymusic wspolna gamme dla wszystkich agentow:

```bash
./venv/bin/python -m codes.train_all --medium --fixed-gamma 0.90 --run-tag medium_fixed_gamma
```

Opcjonalnie mozna tez zwiekszyc limit pozycji:

```bash
./venv/bin/python -m codes.train_all --medium --max-position 2 --run-tag medium_maxpos2
```

Pojedynczy algorytm:

```bash
./venv/bin/python -m codes.main train --algo sarsa --run-tag sarsa_run
./venv/bin/python -m codes.main train --algo ppo --run-tag ppo_run
./venv/bin/python -m codes.main train --algo ippo --run-tag ippo_run
./venv/bin/python -m codes.main train --algo signal_rule --run-tag rule_run
```

Przydatne override'y:

```bash
./venv/bin/python -m codes.train_all --run-tag custom --seeds 5 --episodes 200 --steps 300 --workers 8
./venv/bin/python -m codes.train_all --run-tag custom_pos2 --max-position 2
./venv/bin/python -m codes.train_all --run-tag custom_gamma --fixed-gamma 0.90
./venv/bin/python -m codes.main train --algo ppo --agent-id-features --run-tag ppo_agent_id
./venv/bin/python -m codes.main train --algo sarsa --eval-new-population --run-tag sarsa_eval_new_pop
```

## Aktualne domyslne ustawienia

Tryb `full`:
- `D = [0.0, 0.5, 1.0]`
- `N = 50`
- `episodes = 500`
- `steps = 500`
- `seeds = 5`
- `ZI baseline = 30`
- `eval episodes = 30`

Tryb `quick`:
- `D = [0.0, 0.5, 1.0]`
- `N = 50`
- `episodes = 20`
- `steps = 150`
- `seeds = 1`
- `ZI baseline = 5`
- `eval episodes = 10`

Tryb `medium`:
- `D = [0.0, 0.5, 1.0]`
- `N = 50`
- `episodes = 100`
- `steps = 250`
- `seeds = 2`
- `ZI baseline = 10`
- `eval episodes = 10`

Uwaga:
- `PPO` ma opcjonalny przełącznik `--agent-id-features`
- `SignalRule` nie ma treningu, zapisuje tylko wyniki eval

## Model srodowiska

Aktualna wersja rynku to model `v2`:
- fundamentalna wartosc `V_t` (`eq_price`) jest egzogeniczna
- cena rynkowa `P_t` (`ref_price`) reaguje na flow agentow przed egzekucja
- po egzekucji cena dryfuje w kierunku `V_t` z dodatkowym szumem i rzadkimi szokami

Prywatny sygnal agenta:

```text
signal_i = clip((V_t - P_t + noise_i) / signal_scale, -1, 1)
```

Heterogenicznosc jest budowana przez:
- `sigma_i` — jak bardzo zaszumiony jest prywatny sygnal
- `gamma` — horyzont czasowy agenta

Domyslny benchmark:
- uzywa `sigma-only`, czyli `sigma_i` jest zroznicowane, ale `gamma` jest stale
- to daje uczciwsze porownanie z `SignalRule`, ktory i tak nie uzywa `gamma`

Tryb opcjonalny:
- `--gamma-spread` wlacza wariant `sigma+gamma`
- `--fixed-gamma 0.90` ustawia wspolna gamme dla wszystkich agentow
- jesli podasz jednoczesnie `--gamma-spread` i `--fixed-gamma`, to `fixed-gamma` nadpisuje spread gamma

## Obserwacja i akcje

Obserwacja ma 6 wymiarow:

```text
[signal_i, pos_norm, unrealized, time_rem, price_vs_start, trend_short]
```

Wazna zgodnosc implementacyjna:
- `obs[1]` to `position_norm`
- maskowanie akcji korzysta wlasnie z tego indeksu

Akcje:
- `HOLD`
- `BUY`
- `SELL`

Przy `max_position = 1`:
- `BUY` zwieksza pozycje o `+1`
- `SELL` zmniejsza pozycje o `-1`

Przy `max_position > 1`:
- `BUY` i `SELL` nadal zmieniaja pozycje tylko o `1` na krok
- obserwacja dalej uzywa `pos_norm = position / max_position`, wiec maski akcji skaluja sie poprawnie
- domyslny benchmark zostaje na `max_position = 1`; wyzsze wartosci sa trybem eksperymentalnym

## Reward i PnL

Reward kroku w srodowisku:

```text
reward_t = realized_pnl_this_step + mtm_weight * position * (P_{t+1} - P_exec)
```

Czyli reward laczy:
- zrealizowany PnL z domkniec pozycji
- mark-to-market dla otwartej pozycji po ruchu ceny

W logach epizodowych:
- `trade_accuracy` mierzy udzial zyskownych zamknietych transakcji
- `mean_total_pnl` oznacza sredni koncowy PnL per agent w epizodzie
- `mean_total_pnl_gross` oznacza ten sam PnL przed odjeciem kosztow transakcyjnych
- `mean_transaction_cost` oznacza sredni laczny koszt transakcyjny per agent

## Wyniki

Kazdy run tworzy osobny folder:

```text
results/run_YYYYMMDD_HHMMSS_tag/
```

Typowe pliki:
- `episodes.csv` — metryki train / eval / zi_baseline
- `agents_sample.csv` — probka danych agentowych z eval
- `agent_eval_summary.csv` — summary per agent po eval
- `decision_feature_summary.csv` — korelacje cech obserwacji z kierunkiem decyzji
- `env_steps.csv` — agregaty srodowiska per krok
- `run_config.json` — konfiguracja pojedynczego benchmarku
- `train_all_config.json` — konfiguracja wspolnego runu `train_all`

## Struktura kodu

```text
htm_project/
├── codes/
│   ├── config.py         # centralna konfiguracja
│   ├── market_env.py     # srodowisko HTM i ZI baseline
│   ├── algorithms.py     # Deep SARSA + wspolne helpery RL
│   ├── ppo_core.py       # implementacje PPO i IPPO
│   ├── experiment.py     # wspolna orkiestracja eksperymentow
│   ├── main.py           # pojedynczy punkt wejscia dla treningu algorytmow
│   ├── train_all.py      # runner wszystkich benchmarkow
│   ├── reporting.py      # wykresy i raporty
│   ├── results.py        # zapis wynikow do run folderow
│   └── old/              # starsze, nieuzywane wersje kodu
├── results/
├── plots/
├── logs/
└── README.md
```

## Uwagi praktyczne

- Glowny aktualny interfejs to `codes.main` i `codes.train_all`
- `codes/old/` jest zachowane tylko referencyjnie
- katalogi `results/`, `plots/` i `logs/` sa generowane automatycznie
- lokalne cache `.matplotlib_cache/` i `.cache/` sa ignorowane przez Git
