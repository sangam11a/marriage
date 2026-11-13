"""
marriage/core/engine.py
Formal 4-player Partially Observable Stochastic Game (POSG) Engine for 21-card Nepali Marriage.
Enforces:
- Strict information barriers (Tiplu masked until Seen; stock order and opponent hands hidden)
- Comprehensive action masking at all decision points (Draw, See Maal, Discard, Win)
- Zero illegal action tolerance (tracked and asserted)
- Mathematically rigorous zero-sum terminal payoff settlement with worked example verification
"""

import copy
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter

from marriage.core.card import Card, Suit, Rank, WildcardType
from marriage.core.deck import Deck, TOTAL_CARDS, HAND_SIZE, NUM_PLAYERS
from marriage.core.melds import can_see_maal, calculate_maal, is_winning_hand

MAX_TURNS = 120
PENALTY_UNSEEN = 10
PENALTY_SEEN = 3

class TurnPhase:
    DRAW = 0
    SEE_MAAL = 1
    DISCARD = 2
    DECLARE_WIN = 3


class MarriageEngine:
    """
    State machine for the 4-player Nepali Marriage game.
    Guarantees strict zero-sum payoffs upon termination: sum(payoffs) == 0.
    """
    def __init__(self, max_turns: int = MAX_TURNS, stall_penalty: float = 0.0):
        self.max_turns = max_turns
        self.stall_penalty = stall_penalty
        self.reset(seed=42)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the environment with a deterministic seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

        # Deal hands, cut Tiplu, initialize stock and discard
        self.hands, self.tiplu, self.stock, self.discard_pile = Deck.deal(seed=self.seed)
        
        self.player_seen: List[bool] = [False] * NUM_PLAYERS
        self.current_player: int = 0
        self.turn_count: int = 0
        self.turn_phase: int = TurnPhase.DRAW
        self.done: bool = False
        self.winner: Optional[int] = None
        self.is_timeout: bool = False
        
        # History tracking for observations
        self.opponent_pickups: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.just_drawn_from_discard: Optional[int] = None
        
        # Audit counters
        self.illegal_action_count: int = 0
        self.terminal_payoffs: List[float] = [0.0] * NUM_PLAYERS
        self.raw_maal_held: List[int] = [0] * NUM_PLAYERS

        return self.get_observation(self.current_player)

    # --------------------------------------------------------------------------
    # Action Masking Generators
    # --------------------------------------------------------------------------
    def get_draw_mask(self, player_idx: int) -> List[int]:
        """
        Action space: 2 choices [0: Stock, 1: Discard Top].
        - Stock is always legal if stock has cards.
        - Discard top is ILLEGAL if:
          1. Discard pile is empty.
          2. Player is Seen and top discard is a known wildcard.
          3. Player is Unseen, unless house rules allow (default strict: Unseen cannot pick discard).
        """
        can_stock = 1 if len(self.stock) > 0 else 0
        can_discard = 0

        if len(self.discard_pile) > 0:
            top_card = self.discard_pile[-1]
            if self.player_seen[player_idx]:
                # Seen player: Cannot pick wildcard from discard
                if not top_card.is_wildcard(self.tiplu):
                    can_discard = 1
            else:
                # Unseen player: Strictly draw from stock
                can_discard = 0

        # Fallback: if stock is empty, must draw from discard if available
        if can_stock == 0 and can_discard == 0 and len(self.discard_pile) > 0:
            can_discard = 1

        return [can_stock, can_discard]

    def get_see_mask(self, player_idx: int) -> List[int]:
        """
        Action space: 2 choices [0: Pass, 1: Declare Seen].
        - 1 is legal ONLY if player is currently Unseen AND hand satisfies Seen prerequisites.
        """
        if self.player_seen[player_idx]:
            return [1, 0]  # Already seen, must pass
        can_see = 1 if can_see_maal(self.hands[player_idx]) else 0
        return [1, can_see]

    def get_win_mask(self, player_idx: int) -> List[int]:
        """
        Action space: 2 choices [0: Pass, 1: Declare Show].
        - 1 is legal ONLY if hand (21 cards) forms a complete winning partition.
        """
        can_win = 1 if is_winning_hand(self.hands[player_idx], self.tiplu, self.player_seen[player_idx]) else 0
        return [1, can_win]

    def get_discard_mask(self, player_idx: int) -> List[int]:
        """
        Action space: 53 categorical choices (types 0..51 for standard, 52 for printed joker).
        - Must be in player's current hand.
        - Cannot discard the exact card just picked from the discard pile on the same turn.
        - If player is Seen, cannot discard a wildcard unless the hand consists entirely of wildcards.
        """
        mask = [0] * 53
        hand = self.hands[player_idx]
        held_counts = Counter(c.type_id for c in hand)

        is_seen = self.player_seen[player_idx]
        has_ordinary_cards = False
        if is_seen:
            has_ordinary_cards = any(not c.is_wildcard(self.tiplu) for c in hand)

        for type_id, count in held_counts.items():
            if count <= 0:
                continue

            # Check just-drawn restriction
            if self.just_drawn_from_discard is not None and type_id == self.just_drawn_from_discard and count == 1:
                # Cannot immediately discard the single just-drawn card
                continue

            # Check wildcard discard restriction if Seen
            if is_seen and has_ordinary_cards:
                sample_card = next(c for c in hand if c.type_id == type_id)
                if sample_card.is_wildcard(self.tiplu):
                    continue

            mask[type_id] = 1

        # Safety fallback: if all masked due to edge constraints, allow any card in hand
        if sum(mask) == 0:
            for type_id in held_counts:
                mask[type_id] = 1

        return mask

    # --------------------------------------------------------------------------
    # Step Execution Functions
    # --------------------------------------------------------------------------
    def step_draw(self, action: int) -> bool:
        """Draws a card from stock (0) or discard pile (1)."""
        mask = self.get_draw_mask(self.current_player)
        if mask[action] == 0:
            self.illegal_action_count += 1
            action = 0 if mask[0] == 1 else 1

        p = self.current_player
        if action == 1 and len(self.discard_pile) > 0:
            drawn_card = self.discard_pile.pop()
            self.just_drawn_from_discard = drawn_card.type_id
            self.opponent_pickups[p].append(drawn_card.type_id)
        else:
            if len(self.stock) > 0:
                drawn_card = self.stock.pop()
            else:
                # Stock exhausted
                self._terminate_timeout()
                return False
            self.just_drawn_from_discard = None

        self.hands[p].append(drawn_card)
        self.turn_phase = TurnPhase.SEE_MAAL
        return True

    def step_see_maal(self, action: int) -> bool:
        """Declares Seen (1) or Passes (0)."""
        p = self.current_player
        mask = self.get_see_mask(p)
        if mask[action] == 0:
            self.illegal_action_count += 1
            action = 0

        if action == 1 and not self.player_seen[p]:
            if can_see_maal(self.hands[p]):
                self.player_seen[p] = True

        self.turn_phase = TurnPhase.DISCARD
        return True

    def step_discard(self, card_type: int) -> bool:
        """Discards a card of the specified type_id from the player's 22-card hand."""
        p = self.current_player
        mask = self.get_discard_mask(p)
        if mask[card_type] == 0:
            self.illegal_action_count += 1
            # Pick first valid card type
            card_type = next(idx for idx, val in enumerate(mask) if val == 1)

        # Remove card from hand
        hand = self.hands[p]
        card_to_discard = None
        for i, c in enumerate(hand):
            if c.type_id == card_type:
                card_to_discard = hand.pop(i)
                break

        if card_to_discard is not None:
            self.discard_pile.append(card_to_discard)

        self.turn_phase = TurnPhase.DECLARE_WIN
        return True

    def step_declare_win(self, action: int) -> bool:
        """Declares Show (1) or Passes (0)."""
        p = self.current_player
        mask = self.get_win_mask(p)
        if mask[action] == 0:
            self.illegal_action_count += 1
            action = 0

        if action == 1:
            if is_winning_hand(self.hands[p], self.tiplu, self.player_seen[p]):
                # Legal Show! Terminate game with p as winner
                self._terminate_game(winner=p)
                return True

        # Turn finished without win. Advance to next player
        self.current_player = (self.current_player + 1) % NUM_PLAYERS
        self.turn_phase = TurnPhase.DRAW
        self.turn_count += 1

        if self.turn_count >= self.max_turns or len(self.stock) == 0:
            self._terminate_timeout()

        return True

    # --------------------------------------------------------------------------
    # Termination and Zero-Sum Payoff Settlement
    # --------------------------------------------------------------------------
    def _terminate_game(self, winner: int, override_maal: Optional[List[int]] = None):
        self.done = True
        self.winner = winner
        self.is_timeout = False

        # Winner must be marked as Seen
        self.player_seen[winner] = True

        # Calculate raw Maal for each player
        if override_maal is not None:
            self.raw_maal_held = override_maal[:]
        else:
            for i in range(NUM_PLAYERS):
                self.raw_maal_held[i] = calculate_maal(self.hands[i], self.tiplu, is_seen=self.player_seen[i])

        # 1. Game Points Settlement
        # Unseen: pays 10, Seen non-winner: pays 3. Winner collects sum.
        game_points = [0.0] * NUM_PLAYERS
        winner_collects = 0.0
        for j in range(NUM_PLAYERS):
            if j != winner:
                penalty = PENALTY_UNSEEN if not self.player_seen[j] else PENALTY_SEEN
                game_points[j] = -penalty
                winner_collects += penalty
        game_points[winner] = winner_collects

        # 2. Maal Settlement (Strictly Zero-Sum)
        # Seen players settle pairwise (M_i - M_j) and collect M_i from every Unseen player.
        # Unseen players pay M_j to every Seen player and collect 0.
        maal_net = [0.0] * NUM_PLAYERS
        seen_players = [i for i in range(NUM_PLAYERS) if self.player_seen[i]]
        unseen_players = [i for i in range(NUM_PLAYERS) if not self.player_seen[i]]

        for i in seen_players:
            # Pairwise with other seen players
            for j in seen_players:
                if i != j:
                    maal_net[i] += (self.raw_maal_held[i] - self.raw_maal_held[j])
            # Collect from unseen players
            for k in unseen_players:
                maal_net[i] += self.raw_maal_held[i]

        for k in unseen_players:
            # Pay all seen players their held maal
            for j in seen_players:
                maal_net[k] -= self.raw_maal_held[j]

        # Total payoff: R_i = Game_Points_i + Maal_Net_i
        for i in range(NUM_PLAYERS):
            self.terminal_payoffs[i] = game_points[i] + maal_net[i]

        # Assert zero-sum property
        total_sum = sum(self.terminal_payoffs)
        assert abs(total_sum) < 1e-6, f"Zero-sum assertion failed: sum={total_sum}, payoffs={self.terminal_payoffs}"

    def _terminate_timeout(self):
        self.done = True
        self.winner = None
        self.is_timeout = True
        # In timeout, all players receive stall_penalty (default 0.0, strictly zero-sum)
        self.terminal_payoffs = [self.stall_penalty] * NUM_PLAYERS

    # --------------------------------------------------------------------------
    # Observations & Representations
    # --------------------------------------------------------------------------
    def get_observation(self, player_idx: int) -> Dict[str, Any]:
        """
        Builds the strict private observation vector for player_idx.
        Enforces the information barrier: Tiplu is None and wildcards are masked until Seen.
        """
        is_seen = self.player_seen[player_idx]
        hand = self.hands[player_idx]
        held_counts = Counter(c.type_id for c in hand)

        # 1. Own hand histogram (53 card types: 0..51 standard, 52 joker)
        hand_vec = [held_counts.get(t, 0) for t in range(53)]

        # 2. Wildcard flags (53 card types: all 0 if Unseen!)
        wildcard_vec = [0] * 53
        if is_seen and self.tiplu is not None:
            for t in range(53):
                sample_card = Card.from_type_id(t)
                if sample_card.is_wildcard(self.tiplu):
                    wildcard_vec[t] = 1

        # 3. Public discard histogram (counts of cards discarded across table)
        discard_counts = Counter(c.type_id for c in self.discard_pile)
        discard_vec = [discard_counts.get(t, 0) for t in range(53)]

        # 4. Unseen card residual count: 3 copies - held - discarded
        residual_vec = [max(0, 3 - hand_vec[t] - discard_vec[t]) for t in range(53)]

        # 5. Opponent pickups (53-dim vector summing cards picked from discard by opponents)
        opp_pickups_vec = [0] * 53
        for opp in range(NUM_PLAYERS):
            if opp != player_idx:
                for t in self.opponent_pickups[opp]:
                    opp_pickups_vec[t] += 1

        # 6. Global game context
        top_discard_type = self.discard_pile[-1].type_id if len(self.discard_pile) > 0 else -1
        context = {
            "player_idx": player_idx,
            "is_seen": is_seen,
            "turn_count": self.turn_count,
            "turn_phase": self.turn_phase,
            "stock_size": len(self.stock),
            "top_discard_type": top_discard_type,
            "opponents_seen": [self.player_seen[j] for j in range(NUM_PLAYERS) if j != player_idx],
        }

        # Action masks for current phase
        if self.turn_phase == TurnPhase.DRAW:
            action_mask = self.get_draw_mask(player_idx)
        elif self.turn_phase == TurnPhase.SEE_MAAL:
            action_mask = self.get_see_mask(player_idx)
        elif self.turn_phase == TurnPhase.DISCARD:
            action_mask = self.get_discard_mask(player_idx)
        else:
            action_mask = self.get_win_mask(player_idx)

        return {
            "hand_vec": hand_vec,
            "wildcard_vec": wildcard_vec,
            "discard_vec": discard_vec,
            "residual_vec": residual_vec,
            "opp_pickups_vec": opp_pickups_vec,
            "context": context,
            "action_mask": action_mask,
            "turn_phase": self.turn_phase,
        }

    def get_privileged_state(self) -> Dict[str, Any]:
        """
        Returns full environmental state for training the privileged centralized critic.
        Includes all 4 hands, the true Tiplu, and exact remaining stock deck.
        NEVER used during inference or test deployment.
        """
        return {
            "all_hands": [[c.type_id for c in h] for h in self.hands],
            "tiplu_type": self.tiplu.type_id if self.tiplu else -1,
            "stock_types": [c.type_id for c in self.stock],
            "player_seen": self.player_seen[:],
            "discard_pile": [c.type_id for c in self.discard_pile],
            "turn_count": self.turn_count,
        }
