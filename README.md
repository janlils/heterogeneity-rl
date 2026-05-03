# HTM — Heterogeneous Trader Market

Projekt do porównania architektur MARL w heterogenicznym środowisku tradingowym:
- `Deep SARSA` jako independent per-agent learner
- `PPO` jako shared-policy baseline
- `IPPO` jako light independent policy-gradient baseline
- `MAPPO` jako centralized-critic PPO
- `SignalRule` jako prosty benchmark regułowy oparty o prywatny sygnał

Aktualna wersja środowiska jest oparta na:
- fundamentalnej wartości `V_t` (`eq_price`)
- cenie rynkowej `P_t` (`ref_price`)
- prywatnym sygnale agenta:
  `signal_i = clip((V_t - P_t + noise_i) / signal_scale, -1, 1)`
- heterogeniczności przez `sigma_i`:
  niski `sigma_i` = lepszy sygnał, wysoki `sigma_i` = bardziej zaszumiony sygnał

## Zależności

```bash
pip install numpy matplotlib pandas torch
```

Kod środowiska i SARSA działa w czystym `numpy`.
PPO / IPPO / MAPPO wymagają `torch`.

## Najważniejsze entrypointy

```bash
# Walidacja środowiska
python -m codes.double_auction

# Szybki run SARSA
python -m codes.train_deep_sarsa --quick --run-tag quick_sarsa

# Szybki run PPO
python -m codes.train_ppo --quick --run-tag quick_ppo

# Szybki run IPPO
python -m codes.train_ippo --quick --run-tag quick_ippo

# Szybki run MAPPO
python -m codes.train_mappo --quick --run-tag quick_mappo

# Szybki run SignalRule
python -m codes.train_signal_rule --quick --run-tag quick_signal_rule

# Uruchom wszystkie benchmarki
python -m codes.train_all --quick --run-tag quick_all

# Wykresy z najnowszego runu
python -m codes.visualize

# Debug jednej trajektorii z najnowszego runu
python -m codes.analyze_debug_run
```

## Struktura kodu

```text
htm_project/
├── codes/
│   ├── config.py                 # dataclasses i konfiguracja bazowa
│   ├── double_auction.py         # środowisko HTM + populacja agentów
│   ├── deep_sarsa.py             # implementacja per-agent Deep SARSA
│   ├── ppo.py                    # implementacja PPO
│   ├── train_ippo.py             # entrypoint light-IPPO
│   ├── train_mappo.py            # entrypoint MAPPO
│   ├── train_signal_rule.py      # entrypoint benchmarku SignalRule
│   ├── rule_policies.py          # proste polityki regułowe
│   ├── experiment_settings.py    # wspólne quick/full settings
│   ├── evaluation.py             # wspólna warstwa eval
│   ├── experiment_runner.py      # wspólna orkiestracja runów
│   ├── results_store.py          # zapis wyników do run folderów
│   ├── rl_common.py              # wspólne helpery RL / logowania
│   ├── train_deep_sarsa.py       # entrypoint SARSA
│   ├── train_ppo.py              # entrypoint PPO
│   ├── train_all.py              # entrypoint uruchamiający wszystkie benchmarki
│   ├── visualize.py              # wykresy high-level
│   └── analyze_debug_run.py      # analiza debugowego epizodu krok-po-kroku
├── results/
├── plots/
└── tests/
```

## Wyniki i logowanie

Każdy run tworzy osobny folder:

```text
results/run_YYYYMMDD_HHMMSS_tag/
```

W środku są:
- `episodes.csv` — metryki epizodowe dla train / eval / zi_baseline
- `agent_eval_summary.csv` — lekki summary per agent z końcowego eval (PnL, accuracy, buy/sell/hold, signal alignment)
- `decision_feature_summary.csv` — lekki summary per `(algorithm, D, seed)` z siłą predykcyjną cech obserwacji względem kierunku decyzji
- `article_summary.csv` — zbiorczy summary pod sekcję Results (`algorithm × D`)
- `agents_sample.csv` — pełny debug ostatniego epizodu eval dla `D=1.0`, `seed=0`
- `env_steps.csv` — agregaty środowiska per krok dla tego samego debug epizodu
- `run_config.json` — konfiguracja runu

## Aktualny model środowiska

### Obserwacja

Obserwacja ma 8 wymiarów:

```text
[signal_i, pos_norm, unrealized, time_rem,
 gamma, price_vs_start, trend_short, sigma_norm]
```

Ważna zgodność:
- `obs[1]` to nadal `position_norm`
- maskowanie akcji w SARSA i helperach używa właśnie tego indeksu

### Akcje

Akcje są trzy:
- `HOLD`
- `BUY`
- `SELL`

Przy `max_position = 1` są to de facto akcje zmiany inventory:
- `BUY` zwiększa pozycję o `+1`, jeśli agent nie jest już max long
- `SELL` zmniejsza pozycję o `-1`, jeśli agent nie jest już max short

### Cena i wartość

- `V_t` dryfuje w czasie jako proces AR(1)
- `P_t` przechodzi do kolejnego kroku przez mean reversion do `V_t` oraz szum ceny
- obecnie decyzje agentów nie mają permanentnego impactu na cenę, jeśli `perm_impact = 0`

### Reward

Reward kroku:

```text
reward = realized_pnl_this_step + mtm_weight * position * (P_{t+1} - P_t)
```

czyli:
- `realized` — zysk/strata z zamknięć pozycji
- `MTM` — mark-to-market dla otwartej pozycji po ruchu ceny

## Debug mechanizmu sygnału

Najbardziej użyteczne narzędzie do sprawdzenia:
- czy sygnał przewiduje ruch ceny,
- czy agenci reagują zgodnie z sygnałem,
- czy te decyzje dają reward / PnL,

to:

```bash
python -m codes.analyze_debug_run
```

Skrypt generuje w `results/run_.../debug_analysis/`:
- `summary.json`
- `sigma_bucket_summary.csv`
- `signal_quality_by_sigma.csv`
- `agent_decision_summary.csv`
- wykresy diagnostyczne PNG

Najważniejsze pola w `summary.json`:
- `env_signal.corr_mean_signal_to_price_delta`
- `agent_decisions.signal_action_alignment`
- `agent_decisions.reward_when_aligned`
- `agent_reward.corr_sigma_to_reward`
- `diagnostic_checks.*`

## Wykresy

High-level wykresy z run folderów:

```bash
python -m codes.visualize
```

Wolniejsze wykresy z nowych symulacji:

```bash
python -m codes.visualize --simulate-diagnostics
```

## Uwagi praktyczne

- `quick` używa obecnie tylko `D = [0.5]`
- `quick` i `full` są teraz definiowane centralnie w `codes/experiment_settings.py`
- runnery SARSA/PPO/IPPO/MAPPO są cienkimi entrypointami, a wspólna orkiestracja jest w `codes/experiment_runner.py`
- eval same-population i debug logging przechodzą przez wspólne `codes/evaluation.py`
- `SignalRule` nie ma treningu; zapisuje tylko rekordy `eval_same_population` i opcjonalnie `eval_new_population`
