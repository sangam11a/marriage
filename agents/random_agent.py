"""
marriage/agents/random_agent.py
Uniform legal action sampler baseline.
Samples purely uniformly from all legal actions enabled by the environment's action masks.
"""

import random
from typing import Dict, Any, List
from marriage.agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Uniform random baseline agent. Always chooses uniformly at random among legal actions.
    """
    def __init__(self, name: str = "Random", seed: int = 42):
        super().__init__(name=name)
        self.rng = random.Random(seed)

    def _sample_from_mask(self, mask: List[int]) -> int:
        legal_actions = [i for i, val in enumerate(mask) if val == 1]
        if not legal_actions:
            return 0
        return self.rng.choice(legal_actions)

    def act_draw(self, obs: Dict[str, Any], mask: List[int]) -> int:
        return self._sample_from_mask(mask)

    def act_see(self, obs: Dict[str, Any], mask: List[int]) -> int:
        # If legal to see, always see or sample
        return self._sample_from_mask(mask)

    def act_discard(self, obs: Dict[str, Any], mask: List[int]) -> int:
        return self._sample_from_mask(mask)

    def act_win(self, obs: Dict[str, Any], mask: List[int]) -> int:
        return self._sample_from_mask(mask)
