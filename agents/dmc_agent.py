"""
marriage/agents/dmc_agent.py
Deep Monte Carlo (DMC / DouZero-style) comparator agent.
Learns state-action value Q(s, a) directly from episodic terminal Monte Carlo rollouts,
ranking legal actions by predicted terminal zero-sum return.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from marriage.agents.base_agent import BaseAgent
from marriage.models.flat_mlp import FlatMLP

class DeepMonteCarloAgent(BaseAgent):
    """
    Deep Monte Carlo agent for 21-card Nepali Marriage.
    """
    def __init__(
        self,
        name: str = "Deep-Monte-Carlo",
        device: Optional[torch.device] = None,
    ):
        super().__init__(name=name)
        self.device = device if device is not None else torch.device("cpu")
        # Uses capacity-matched FlatMLP Q-network
        self.q_net = FlatMLP().to(self.device)

    def _get_q_values(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            x = self.q_net.extract_features_from_obs(obs, self.device)
            logits, _ = self.q_net(x)
        return logits

    def _select_best_legal_action(self, q_values: torch.Tensor, mask: List[int]) -> int:
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        neg_inf = torch.tensor(-1e8, dtype=torch.float32, device=self.device)
        masked_q = torch.where(mask_tensor, q_values, neg_inf)
        return torch.argmax(masked_q, dim=-1).item()

    def act_draw(self, obs: Dict[str, Any], mask: List[int]) -> int:
        q = self._get_q_values(obs)
        return self._select_best_legal_action(q["draw"], mask)

    def act_see(self, obs: Dict[str, Any], mask: List[int]) -> int:
        q = self._get_q_values(obs)
        return self._select_best_legal_action(q["see"], mask)

    def act_discard(self, obs: Dict[str, Any], mask: List[int]) -> int:
        q = self._get_q_values(obs)
        return self._select_best_legal_action(q["discard"], mask)

    def act_win(self, obs: Dict[str, Any], mask: List[int]) -> int:
        q = self._get_q_values(obs)
        return self._select_best_legal_action(q["win"], mask)
