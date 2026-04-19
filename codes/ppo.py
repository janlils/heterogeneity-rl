"""
Shared-policy PPO dla HTM.

Jedna sieć actor-critic obsługuje wszystkich agentów. Rollout używa tego
samego równoległego protokołu co Deep SARSA:
get_observation -> execute_parallel_actions -> compute_step_rewards.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from codes.config import PPOConfig
from codes.rl_common import action_mask_from_obs, append_agent_id_feature


MASK_VALUE = -1e9


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    masked_logits = logits.masked_fill(~mask, MASK_VALUE)
    return Categorical(logits=masked_logits)


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        logits, value = self.forward(obs)
        dist = masked_categorical(logits, action_mask)
        action = torch.argmax(logits.masked_fill(~action_mask.bool(), MASK_VALUE), dim=-1)
        if not deterministic:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.data: List[dict] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        reward: float,
        done: bool,
        next_value: float,
        action_mask: np.ndarray,
        agent_id: str,
        episode_idx: int,
        step_idx: int,
    ) -> None:
        self.data.append({
            "obs": obs.astype(np.float32, copy=False),
            "action": int(action),
            "logprob": float(logprob),
            "value": float(value),
            "reward": float(reward),
            "done": bool(done),
            "next_value": float(next_value),
            "action_mask": action_mask.astype(bool, copy=False),
            "agent_id": agent_id,
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
        })

    def __len__(self) -> int:
        return len(self.data)

    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float) -> None:
        grouped: Dict[tuple[int, str], List[int]] = defaultdict(list)
        for i, row in enumerate(self.data):
            grouped[(row["episode_idx"], row["agent_id"])].append(i)

        advantages = np.zeros(len(self.data), dtype=np.float32)
        returns = np.zeros(len(self.data), dtype=np.float32)

        for idxs in grouped.values():
            idxs.sort(key=lambda i: self.data[i]["step_idx"])
            gae = 0.0
            for i in reversed(idxs):
                row = self.data[i]
                nonterminal = 0.0 if row["done"] else 1.0
                delta = (
                    row["reward"]
                    + gamma * row["next_value"] * nonterminal
                    - row["value"]
                )
                gae = delta + gamma * gae_lambda * nonterminal * gae
                advantages[i] = gae
                returns[i] = gae + row["value"]

        for i, row in enumerate(self.data):
            row["advantage"] = float(advantages[i])
            row["return"] = float(returns[i])

    def tensors(self, device: torch.device) -> dict:
        return {
            "obs": torch.as_tensor(np.stack([r["obs"] for r in self.data]), dtype=torch.float32, device=device),
            "actions": torch.as_tensor([r["action"] for r in self.data], dtype=torch.long, device=device),
            "old_logprobs": torch.as_tensor([r["logprob"] for r in self.data], dtype=torch.float32, device=device),
            "values": torch.as_tensor([r["value"] for r in self.data], dtype=torch.float32, device=device),
            "returns": torch.as_tensor([r["return"] for r in self.data], dtype=torch.float32, device=device),
            "advantages": torch.as_tensor([r["advantage"] for r in self.data], dtype=torch.float32, device=device),
            "action_masks": torch.as_tensor(np.stack([r["action_mask"] for r in self.data]), dtype=torch.bool, device=device),
        }

    def get_minibatches(
        self,
        minibatch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ) -> Iterable[dict]:
        tensors = self.tensors(device)
        n = len(self.data)
        idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for start in range(0, n, minibatch_size):
            mb_idx = idx[start:start + minibatch_size]
            yield {k: v[mb_idx] for k, v in tensors.items()}


class SharedPPOTrainer:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: Optional[PPOConfig] = None,
        seed: int = 42,
        agent_ids: Optional[List[str]] = None,
    ):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        torch.manual_seed(seed)
        self.model = PPOActorCritic(obs_dim, n_actions, self.cfg.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam([
            {
                "params": list(self.model.trunk.parameters())
                       + list(self.model.policy_head.parameters()),
                "lr": self.cfg.actor_lr,
            },
            {
                "params": self.model.value_head.parameters(),
                "lr": self.cfg.critic_lr,
            },
        ])
        self.buffer = RolloutBuffer()
        self.training_step = 0
        self.agent_to_idx = (
            {aid: i for i, aid in enumerate(agent_ids)}
            if self.cfg.use_agent_id_features and agent_ids
            else None
        )
        self.last_update_stats: Dict[str, float] = {}

    def _policy_obs(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        return append_agent_id_feature(obs, agent_id, self.agent_to_idx)

    @torch.no_grad()
    def act_np(
        self,
        obs: np.ndarray,
        agent_id: str,
        deterministic: bool = False,
    ) -> tuple[int, float, float, np.ndarray]:
        policy_obs = self._policy_obs(obs, agent_id)
        mask = action_mask_from_obs(obs)
        obs_t = torch.as_tensor(policy_obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        action, logprob, value = self.model.act(obs_t, mask_t, deterministic=deterministic)
        return int(action.item()), float(logprob.item()), float(value.item()), mask

    @torch.no_grad()
    def value_np(self, obs: np.ndarray, agent_id: str) -> float:
        policy_obs = self._policy_obs(obs, agent_id)
        obs_t = torch.as_tensor(policy_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.model(obs_t)
        return float(value.item())

    def collect_rollout(
        self,
        da,
        agent_ids: List[str],
        rng: np.random.Generator,
        deterministic: bool = False,
        rollout_episodes: Optional[int] = None,
    ) -> List[dict]:
        self.buffer.clear()
        episode_metrics: List[dict] = []
        n_rollout = rollout_episodes or self.cfg.rollout_episodes

        for rollout_ep in range(n_rollout):
            da.reset_episode()
            step_idx = 0

            while not da.done:
                obs_at_action = {}
                actions_taken = {}
                logprobs = {}
                values = {}
                masks = {}

                # Parallel: wszyscy obserwują ten sam P_t przed egzekucją.
                for aid in agent_ids:
                    obs = da.get_observation(aid)
                    action, logprob, value, mask = self.act_np(
                        obs, aid, deterministic=deterministic
                    )
                    if not mask[action]:
                        raise RuntimeError(f"PPO selected illegal action {action} for {aid}")
                    obs_at_action[aid] = self._policy_obs(obs, aid)
                    actions_taken[aid] = action
                    logprobs[aid] = logprob
                    values[aid] = value
                    masks[aid] = mask

                da.execute_parallel_actions(actions_taken)
                rewards, dones = da.compute_step_rewards()

                for aid in agent_ids:
                    done = bool(dones.get(aid, False))
                    next_value = 0.0 if done else self.value_np(da.get_observation(aid), aid)
                    self.buffer.add(
                        obs=obs_at_action[aid],
                        action=actions_taken[aid],
                        logprob=logprobs[aid],
                        value=values[aid],
                        reward=float(rewards.get(aid, 0.0)),
                        done=done,
                        next_value=next_value,
                        action_mask=masks[aid],
                        agent_id=aid,
                        episode_idx=rollout_ep,
                        step_idx=step_idx,
                    )

                step_idx += 1

            episode_metrics.append(da.episode_metrics())

        self.buffer.compute_advantages_and_returns(self.cfg.gamma, self.cfg.gae_lambda)
        return episode_metrics

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}

        stats: Dict[str, List[float]] = defaultdict(list)
        for _ in range(self.cfg.update_epochs):
            for batch in self.buffer.get_minibatches(self.cfg.minibatch_size, self.device):
                advantages = batch["advantages"]
                if self.cfg.normalize_advantages and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                logits, values = self.model(batch["obs"])
                dist = masked_categorical(logits, batch["action_masks"])
                new_logprobs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - batch["old_logprobs"])
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, batch["returns"])
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch["old_logprobs"] - new_logprobs).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float().mean()
                    )
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_fraction"].append(float(clip_fraction.item()))
                stats["mean_advantage"].append(float(batch["advantages"].mean().item()))
                stats["mean_return"].append(float(batch["returns"].mean().item()))

        self.training_step += 1
        self.last_update_stats = {k: float(np.mean(v)) for k, v in stats.items()}
        return self.last_update_stats

    def save(self, path: str | Path, episode: int = 0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode": episode,
            "config": asdict(self.cfg),
            "agent_to_idx": self.agent_to_idx,
        }, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = int(checkpoint.get("training_step", 0))
        self.agent_to_idx = checkpoint.get("agent_to_idx", self.agent_to_idx)
