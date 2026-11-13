"""
marriage/core/card.py
Defines playing cards, suits, ranks, exchangeable type IDs (0..52),
and the formal Nepali Marriage wildcard classification hierarchy:
- Tiplu (3 Maal points)
- Poplu (2 Maal points)
- Alter (5 Maal points)
- Ordinary Jokers / Printed Jokers (1 Maal point)
"""

from enum import IntEnum, Enum
from typing import Optional, Tuple, Dict, Any

class Suit(IntEnum):
    SPADES = 0    # ♠ Black
    HEARTS = 1    # ♥ Red
    DIAMONDS = 2  # ♦ Red
    CLUBS = 3     # ♣ Black
    JOKER = 4     # Printed Joker

    @classmethod
    def from_str(cls, s: str) -> "Suit":
        s_clean = s.strip().upper()
        mapping = {
            "S": cls.SPADES, "♠": cls.SPADES, "SPADE": cls.SPADES, "SPADES": cls.SPADES,
            "H": cls.HEARTS, "♥": cls.HEARTS, "HEART": cls.HEARTS, "HEARTS": cls.HEARTS,
            "D": cls.DIAMONDS, "♦": cls.DIAMONDS, "DIAMOND": cls.DIAMONDS, "DIAMONDS": cls.DIAMONDS,
            "C": cls.CLUBS, "♣": cls.CLUBS, "CLUB": cls.CLUBS, "CLUBS": cls.CLUBS,
            "J": cls.JOKER, "JK": cls.JOKER, "JOKER": cls.JOKER
        }
        return mapping[s_clean]

    def to_symbol(self) -> str:
        symbols = {0: "♠", 1: "♥", 2: "♦", 3: "♣", 4: "★"}
        return symbols[self.value]

    def is_black(self) -> bool:
        return self.value in (Suit.SPADES, Suit.CLUBS)

    def is_red(self) -> bool:
        return self.value in (Suit.HEARTS, Suit.DIAMONDS)

    def complementary_same_color_suit(self) -> "Suit":
        """Returns the suit of the same color (for Alter determination: Spades <-> Clubs, Hearts <-> Diamonds)."""
        if self.value == Suit.SPADES:
            return Suit.CLUBS
        elif self.value == Suit.CLUBS:
            return Suit.SPADES
        elif self.value == Suit.HEARTS:
            return Suit.DIAMONDS
        elif self.value == Suit.DIAMONDS:
            return Suit.HEARTS
        return Suit.JOKER


class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    JOKER = 14

    @classmethod
    def from_str(cls, r: str) -> "Rank":
        r_clean = r.strip().upper()
        mapping = {
            "A": cls.ACE, "1": cls.ACE, "2": cls.TWO, "3": cls.THREE, "4": cls.FOUR,
            "5": cls.FIVE, "6": cls.SIX, "7": cls.SEVEN, "8": cls.EIGHT, "9": cls.NINE,
            "10": cls.TEN, "T": cls.TEN, "J": cls.JACK, "Q": cls.QUEEN, "K": cls.KING,
            "JK": cls.JOKER, "JOKER": cls.JOKER
        }
        return mapping[r_clean]

    def to_str(self) -> str:
        mapping = {
            1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
            8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "JK"
        }
        return mapping[self.value]

    def next_rank(self) -> "Rank":
        """Next rank for Poplu determination (wraps King -> Ace)."""
        if self.value == Rank.KING:
            return Rank.ACE
        elif self.value in range(1, 13):
            return Rank(self.value + 1)
        return Rank.JOKER


class WildcardType(Enum):
    NONE = "none"
    TIPLU = "tiplu"                    # Cut card itself (3 Maal)
    POPLU = "poplu"                    # Same suit, rank + 1 (2 Maal)
    ALTER = "alter"                    # Same rank, same color opposite suit (5 Maal)
    ORDINARY_JOKER = "ordinary_joker"  # Same rank, different color suits (1 Maal)
    PRINTED_JOKER = "printed_joker"    # Printed joker cards (1 Maal)

    def base_maal(self) -> int:
        maal_map = {
            WildcardType.NONE: 0,
            WildcardType.TIPLU: 3,
            WildcardType.POPLU: 2,
            WildcardType.ALTER: 5,
            WildcardType.ORDINARY_JOKER: 1,
            WildcardType.PRINTED_JOKER: 1,
        }
        return maal_map[self]


class Card:
    """
    Represents a playing card in the 21-card Marriage game.
    There are 3 identical 52-card decks + 3 printed jokers = 159 cards.
    Exchangeable type ID is in range 0..51 for standard cards, and 52 for printed jokers.
    """
    __slots__ = ('suit', 'rank', 'deck_id', 'card_id', 'type_id', 'point_value')

    def __init__(self, suit: Suit, rank: Rank, deck_id: int = 0, card_id: Optional[int] = None):
        self.suit = suit
        self.rank = rank
        self.deck_id = deck_id
        
        # Calculate exchangeable type_id (0..51 for standard, 52 for printed jokers)
        if self.suit == Suit.JOKER or self.rank == Rank.JOKER:
            self.type_id = 52
            self.point_value = 0
        else:
            self.type_id = self.suit.value * 13 + (self.rank.value - 1)
            # Point value for deadwood calculation (A=1, 2-10 face value, J=11, Q=12, K=13)
            self.point_value = self.rank.value

        # Physical card id (0..158)
        if card_id is not None:
            self.card_id = card_id
        else:
            if self.type_id == 52:
                self.card_id = 156 + (deck_id % 3)
            else:
                self.card_id = self.deck_id * 52 + self.type_id

    @property
    def is_printed_joker(self) -> bool:
        return self.type_id == 52

    def classify_wildcard(self, tiplu: Optional["Card"]) -> WildcardType:
        """
        Classifies this card relative to the cut Tiplu card according to strict Nepali Marriage rules.
        If Tiplu is None (i.e. Unseen phase), returns NONE (or PRINTED_JOKER for printed jokers).
        """
        if self.is_printed_joker:
            return WildcardType.PRINTED_JOKER
        if tiplu is None or tiplu.is_printed_joker:
            return WildcardType.NONE

        # 1. Tiplu: Exact match in suit and rank
        if self.suit == tiplu.suit and self.rank == tiplu.rank:
            return WildcardType.TIPLU

        # 2. Poplu: Same suit, one rank higher (wraps K -> A)
        if self.suit == tiplu.suit and self.rank == tiplu.rank.next_rank():
            return WildcardType.POPLU

        # 3. Alter: Same rank, complementary suit of the same color
        # e.g., if Tiplu is 7♠ (Black Spade), Alter is 7♣ (Black Club).
        # e.g., if Tiplu is 7♥ (Red Heart), Alter is 7♦ (Red Diamond).
        alter_suit = tiplu.suit.complementary_same_color_suit()
        if self.rank == tiplu.rank and self.suit == alter_suit:
            return WildcardType.ALTER

        # 4. Ordinary Jokers: Same rank as Tiplu in the remaining suits
        if self.rank == tiplu.rank:
            return WildcardType.ORDINARY_JOKER

        return WildcardType.NONE

    def is_wildcard(self, tiplu: Optional["Card"]) -> bool:
        w_type = self.classify_wildcard(tiplu)
        return w_type != WildcardType.NONE

    def maal_value(self, tiplu: Optional["Card"]) -> int:
        return self.classify_wildcard(tiplu).base_maal()

    def same_type(self, other: "Card") -> bool:
        """Returns True if both cards share the exact same rank and suit (regardless of deck_id)."""
        return self.type_id == other.type_id

    @classmethod
    def from_type_id(cls, type_id: int, deck_id: int = 0) -> "Card":
        if type_id == 52:
            return cls(Suit.JOKER, Rank.JOKER, deck_id=deck_id)
        suit_val = type_id // 13
        rank_val = (type_id % 13) + 1
        return cls(Suit(suit_val), Rank(rank_val), deck_id=deck_id)

    @classmethod
    def from_str(cls, card_str: str, deck_id: int = 0) -> "Card":
        c = card_str.strip()
        if c.upper() in ("JK", "JOKER", "★"):
            return cls(Suit.JOKER, Rank.JOKER, deck_id=deck_id)
        suit_str = c[-1]
        rank_str = c[:-1]
        return cls(Suit.from_str(suit_str), Rank.from_str(rank_str), deck_id=deck_id)

    def __str__(self) -> str:
        if self.type_id == 52:
            return "★JK"
        return f"{self.rank.to_str()}{self.suit.to_symbol()}"

    def __repr__(self) -> str:
        return f"Card({str(self)}, deck={self.deck_id})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Card):
            return False
        return self.card_id == other.card_id

    def __hash__(self) -> int:
        return self.card_id
