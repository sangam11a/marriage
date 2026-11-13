"""
marriage/agents/heuristic_expert.py
Strong Domain-Expert Heuristic Agent for 21-card Nepali Marriage.
Implements:
- Exact residual card counting (tracking cards latent in stock vs already discarded)
- Combinatorial completion probability estimation
- Dynamic risk hedging upon observing opponent Seen declarations (pivoting from Tunella hoarding to Seen rush)
"""

from typing import Dict, Any, List
from marriage.agents.base_agent import BaseAgent
from marriage.core.card import Card, Suit, Rank

class HeuristicExpertAgent(BaseAgent):
    """
    Strong domain heuristic with card counting and opponent-adaptive risk hedging.
    """
    def __init__(self, name: str = "Expert-CardCounter"):
        super().__init__(name=name)

    def act_draw(self, obs: Dict[str, Any], mask: List[int]) -> int:
        if mask[1] == 0:
            return 0  # Discard pickup illegal

        top_type = obs["context"]["top_discard_type"]
        if top_type < 0:
            return 0

        hand_counts = obs["hand_vec"]
        resid = obs["residual_vec"]

        # If top card directly completes a pair into a Tunella or completes a 2-card sequence
        if hand_counts[top_type] == 2:
            return 1  # Instant Tunella

        if top_type < 52:
            suit = top_type // 13
            rank = top_type % 13
            # Check 2-card sequence completion
            left1 = suit * 13 + (rank - 1) if rank > 0 else None
            left2 = suit * 13 + (rank - 2) if rank > 1 else None
            right1 = suit * 13 + (rank + 1) if rank < 12 else None
            right2 = suit * 13 + (rank + 2) if rank < 11 else None

            # forms (left1, top, right1) or (left2, left1, top) or (top, right1, right2)
            if left1 is not None and right1 is not None and hand_counts[left1] > 0 and hand_counts[right1] > 0:
                return 1
            if left1 is not None and left2 is not None and hand_counts[left1] > 0 and hand_counts[left2] > 0:
                return 1
            if right1 is not None and right2 is not None and hand_counts[right1] > 0 and hand_counts[right2] > 0:
                return 1

        return 0

    def act_see(self, obs: Dict[str, Any], mask: List[int]) -> int:
        return 1 if mask[1] == 1 else 0

    def act_win(self, obs: Dict[str, Any], mask: List[int]) -> int:
        return 1 if mask[1] == 1 else 0

    def act_discard(self, obs: Dict[str, Any], mask: List[int]) -> int:
        hand_counts = obs["hand_vec"]
        wild_flags = obs["wildcard_vec"]
        resid = obs["residual_vec"]
        ctx = obs["context"]

        is_seen = ctx["is_seen"]
        opponents_seen = any(ctx["opponents_seen"])
        stock_size = max(1, ctx["stock_size"])

        best_card_type = -1
        max_discard_score = -1e9

        for type_id in range(53):
            if mask[type_id] == 0:
                continue

            keep_value = 0.0

            # 1. Wildcards
            if is_seen and wild_flags[type_id] == 1:
                keep_value += 35.0
            elif type_id == 52:
                keep_value += 20.0

            cnt = hand_counts[type_id]

            # 2. Tunellas and Pairs
            if cnt >= 3:
                keep_value += 40.0  # Tunella completed
            elif cnt == 2:
                # Pair: estimate probability of drawing 3rd copy from stock
                p_tunella = resid[type_id] / float(stock_size)
                if not is_seen and opponents_seen:
                    # Risk hedging: If opponents are seen, do not hold isolated pairs with 0 residuals
                    keep_value += 10.0 + (p_tunella * 25.0)
                else:
                    keep_value += 15.0 + (p_tunella * 30.0)

            # 3. Sequences
            if type_id < 52:
                suit = type_id // 13
                rank = type_id % 13

                left1 = suit * 13 + (rank - 1) if rank > 0 else None
                left2 = suit * 13 + (rank - 2) if rank > 1 else None
                right1 = suit * 13 + (rank + 1) if rank < 12 else None
                right2 = suit * 13 + (rank + 2) if rank < 11 else None

                # Check if already in 3-card pure sequence
                is_in_complete_seq = False
                if left1 is not None and right1 is not None and hand_counts[left1] > 0 and hand_counts[right1] > 0:
                    is_in_complete_seq = True
                elif left1 is not None and left2 is not None and hand_counts[left1] > 0 and hand_counts[left2] > 0:
                    is_in_complete_seq = True
                elif right1 is not None and right2 is not None and hand_counts[right1] > 0 and hand_counts[right2] > 0:
                    is_in_complete_seq = True

                if is_in_complete_seq:
                    keep_value += 30.0
                else:
                    # 2-card partial sequence: check availability of completing card
                    seq_draw_potential = 0.0
                    if left1 is not None and hand_counts[left1] > 0:
                        # Needs right1 or left2
                        outs = (resid[right1] if right1 is not None else 0) + (resid[left2] if left2 is not None else 0)
                        seq_draw_potential += outs / float(stock_size)
                    if right1 is not None and hand_counts[right1] > 0:
                        outs = (resid[left1] if left1 is not None else 0) + (resid[right2] if right2 is not None else 0)
                        seq_draw_potential += outs / float(stock_size)

                    keep_value += 12.0 * seq_draw_potential

                # 4. Deadwood penalty (penalize isolated high rank cards, especially if opponents are seen!)
                if not is_in_complete_seq and cnt == 1:
                    deadwood_penalty = (rank + 1) * (2.0 if opponents_seen and not is_seen else 1.0)
                    keep_value -= deadwood_penalty

            discard_score = -keep_value
            if discard_score > max_discard_score:
                max_discard_score = discard_score
                best_card_type = type_id

        return best_card_type if best_card_type >= 0 else next(i for i, v in enumerate(mask) if v == 1)
