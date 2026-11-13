"""
marriage/agents/base_agent.py
Abstract base class defining the observation, action masking,
and decision interfaces for all Marriage AI agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAgent(ABC):
    """
    Abstract interface for Marriage AI agents.
    Enforces that all actions respect the environment's legal action masks.
    """
    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act_draw(self, obs: Dict[str, Any], mask: List[int]) -> int:
        """
        Chooses draw action: 0 for Stock, 1 for Discard Top.
        Must satisfy mask[action] == 1.
        """
        pass

    @abstractmethod
    def act_see(self, obs: Dict[str, Any], mask: List[int]) -> int:
        """
        Chooses whether to declare Seen: 0 for Pass, 1 for Declare Seen.
        Must satisfy mask[action] == 1.
        """
        pass

    @abstractmethod
    def act_discard(self, obs: Dict[str, Any], mask: List[int]) -> int:
        """
        Chooses which card type to discard: integer 0..52.
        Must satisfy mask[action] == 1.
        """
        pass

    @abstractmethod
    def act_win(self, obs: Dict[str, Any], mask: List[int]) -> int:
        """
        Chooses whether to declare Show (Win): 0 for Pass, 1 for Declare Show.
        Must satisfy mask[action] == 1.
        """
        pass

    def reset_round(self):
        """Called at the start of each round to clear internal state/memory."""
        pass
