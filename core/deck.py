"""
marriage/core/deck.py
Implements the 3-deck (159 cards) multi-deck system with deterministic shuffling,
duplicate deal generation, and strict deal-seed isolation.
"""

import random
from typing import List, Tuple
from marriage.core.card import Card, Suit, Rank

TOTAL_CARDS = 159  # 3 * 52 + 3 printed jokers
HAND_SIZE = 21
NUM_PLAYERS = 4

class Deck:
    """
    Represents the 159-card multi-deck structure used in Nepalese Marriage.
    Contains 3 full standard 52-card decks and 3 printed jokers.
    """
    def __init__(self):
        self.cards: List[Card] = self._create_deck()

    @staticmethod
    def _create_deck() -> List[Card]:
        cards = []
        card_id = 0
        for deck_id in range(3):
            # 52 standard cards per deck
            for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
                for rank_val in range(1, 14):
                    cards.append(Card(suit=suit, rank=Rank(rank_val), deck_id=deck_id, card_id=card_id))
                    card_id += 1
            # 1 printed joker per deck
            cards.append(Card(suit=Suit.JOKER, rank=Rank.JOKER, deck_id=deck_id, card_id=card_id))
            card_id += 1
        return cards

    @classmethod
    def deal(cls, seed: int, num_players: int = NUM_PLAYERS, hand_size: int = HAND_SIZE) -> Tuple[List[List[Card]], Card, List[Card], List[Card]]:
        """
        Deterministically deals hands, cuts the Tiplu, and initializes stock and discard piles.
        
        Args:
            seed: Integer random seed for reproducibility.
            num_players: Number of players (default 4).
            hand_size: Initial hand size (default 21).

        Returns:
            hands: List of 4 player hands (each containing 21 Card objects).
            tiplu: The cut wildcard indicator Card.
            stock: Ordered list of remaining cards in the draw pile.
            discard_pile: List of cards in the discard pile (starts with 1 card).
        """
        deck = cls._create_deck()
        rng = random.Random(seed)
        rng.shuffle(deck)

        # Deal 21 cards to each player
        hands = [[] for _ in range(num_players)]
        for _ in range(hand_size):
            for p in range(num_players):
                hands[p].append(deck.pop())

        # The cut card (Tiplu) is taken from the top or middle of remaining stock
        # In standard rules, if cut card is printed joker, it is placed and another cut or handled per rules.
        tiplu = deck.pop()

        # The first open card on the discard pile
        first_discard = deck.pop()
        discard_pile = [first_discard]

        # Remaining cards form the stock pile
        stock = deck

        return hands, tiplu, stock, discard_pile
