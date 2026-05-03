from __future__ import annotations

import numpy as np


class SignalRulePolicy:
    """
    Prosta polityka regułowa oparta na prywatnym sygnale.

    Dla max_position=1 działa jak target inventory:
      - BUY jeśli sygnał jest dodatni i agent nie jest już long
      - SELL jeśli sygnał jest ujemny i agent nie jest już short
      - HOLD w przeciwnym razie
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def act(self, obs: np.ndarray) -> int:
        signal_i = float(obs[0])
        pos_norm = float(obs[1])

        if signal_i > self.threshold and pos_norm < 0.99:
            return 1
        if signal_i < -self.threshold and pos_norm > -0.99:
            return 2
        return 0
