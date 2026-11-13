"""
marriage/agents/heuristic_tuned.py
Tuned heuristic baseline for 21-card Nepali Marriage.
Balances Pure Sequence rush, Tunella hoarding, and high-rank deadwood reduction
using parameterized scoring weights.
"""

from typing import Dict, Any, List
from collections import Counter
from marriage.agents.base_agent import BaseAgent
from marriage.core.card import Card, Suit, Rank

class HeuristicTunedAgent(BaseAgent):
    """
    Tuned heuristic baseline. Evaluates card synergy using parameterized weights:
    - w_seq: bonus for consecutive rank partners in same suit
    - w_dup: bonus for duplicate cards (pair/triplet potential)
    - w_wild: bonus for retaining wildcards
    - w_deadwood: penalty for isolated cards
    """
    def __init__(
        self,
        name: str = "Heuristic-Tuned",
        w_seq: float = 3.5,
        w_dup: float = 4.0,
        w_wild: float = 8.0,
        w_deadwood: float = 1.0,
    ):
        super().__init__(name=name)
        self.w_seq = w_seq
        self.w_dup = w_dup
        self.w_wild = w_wild
        self.w_deadwood = w_deadwood

    def act_draw(self, obs: Dict[str, Any], mask: List[int]) -> int:
        # If discard draw is not legal, must draw stock
        if mask[1] == 0:
            return 0

        # Check if top discard card provides strong synergy to own hand
        top_type = obs["context"]["top_discard_type"]
        if top_type < 0:
            return 0

        hand_counts = obs["hand_vec"]
        # If we already have a duplicate of this card, drawing creates a pair or tunella!
        if hand_counts[top_type] >= 1:
            return 1

        # Check if it completes a consecutive sequence in the same suit
        if top_type < 52:
            suit = top_type // 13
            rank = top_type % 13
            # Check adjacent ranks
            left = suit * 13 + (rank - 1) if rank > 0 else None
            right = suit * 13 + (rank + 1) if rank < 12 else None
            if (left is not None and hand_counts[left] > 0) or (right is not None and hand_counts[right] > 0):
                return 1

        return 0

    def act_see(self, obs: Dict[str, Any], mask: List[int]) -> int:
        # Always declare Seen when legally qualified
        return 1 if mask[1] == 1 else 0

    def act_win(self, obs: Dict[str, Any], mask: List[int]) -> int:
        # Always declare Win when legally qualified
        return 1 if mask[1] == 1 else 0

    def act_discard(self, obs: Dict[str, Any], mask: List[int]) -> int:
        hand_counts = obs["hand_vec"]
        wild_flags = obs["wildcard_vec"]
        is_seen = obs["context"]["is_seen"]

        best_card_type = -1
        max_discard_score = -1e9

        for type_id in range(53):
            if mask[type_id] == 0:
                continue

            # Calculate keep value of this card
            keep_value = 0.0

            # 1. Wildcard value
            if is_seen and wild_flags[type_id] == 1:
                keep_value += self.w_wild * 2.0
            elif type_id == 52:  # Printed joker
                keep_value += self.w_wild

            # 2. Duplicate bonus
            cnt = hand_counts[type_id]
            if cnt >= 3:
                keep_value += self.w_dup * 3.0  # Tunella!
            elif cnt == 2:
                keep_value += self.w_dup * 1.5  # Pair / Dublee

            # 3. Sequence bonus
            if type_id < 52:
                suit = type_id // 13
                rank = type_id % 13
                left = suit * 13 + (rank - 1) if rank > 0 else None
                right = suit * 13 + (rank + 1) if rank < 12 else None
                has_seq_partner = False
                if left is not None and hand_counts[left] > 0:
                    keep_value += self.w_seq
                    has_seq_partner = True
                if right is not None and hand_counts[right] > 0:
                    keep_value += self.w_seq
                    has_seq_partner = True

                # 4. Deadwood penalty (face value / 13)
                if not has_seq_partner and cnt == 1:
                    deadwood_val = (rank + 1) / 13.0
                    keep_value -= self.w_deadwood * deadwood_val

            # Discard score is inverse of keep value (we want to discard the lowest keep value)
            discard_score = -keep_value
            if discard_score > max_discard_score:
                max_discard_score = discard_score
                best_card_type = type_id

        return best_card_type if best_card_type >= 0 else next(i for i, v in enumerate(mask) if v == 1)
