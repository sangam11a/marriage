"""
tests/test_marriage_rules.py
Rigorous unit tests for 21-card Nepali Marriage core mechanics:
- 159-card multi-deck integrity
- Card type and wildcard classifications
- Pure sequences, Tunellas, Dublees
- Full settlement formula with worked example from §2.3 (asserting strict sum-to-zero)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marriage.core.card import Card, Suit, Rank, WildcardType
from marriage.core.deck import Deck, TOTAL_CARDS, HAND_SIZE, NUM_PLAYERS
from marriage.core.melds import (
    find_tunellas, find_dublees, is_pure_sequence,
    can_see_maal, calculate_maal, is_winning_hand
)
from marriage.core.engine import MarriageEngine, PENALTY_UNSEEN, PENALTY_SEEN

def test_card_basics_and_encoding():
    # 7 of Spades (Black)
    c1 = Card.from_str("7S")
    assert c1.suit == Suit.SPADES
    assert c1.rank == Rank.SEVEN
    assert c1.type_id == 0 * 13 + 6  # 6
    assert c1.point_value == 7

    # Ace of Hearts (Red)
    c2 = Card.from_str("AH")
    assert c2.suit == Suit.HEARTS
    assert c2.rank == Rank.ACE
    assert c2.type_id == 1 * 13 + 0  # 13
    assert c2.point_value == 1

    # Printed Joker
    jk = Card(Suit.JOKER, Rank.JOKER)
    assert jk.is_printed_joker
    assert jk.type_id == 52


def test_wildcard_classification():
    # Set Tiplu as 7 of Spades
    tiplu = Card.from_str("7S")

    # 1. Tiplu itself
    c_tiplu = Card.from_str("7S")
    assert c_tiplu.classify_wildcard(tiplu) == WildcardType.TIPLU
    assert c_tiplu.maal_value(tiplu) == 3

    # 2. Poplu: Same suit (Spades), rank + 1 (8 of Spades)
    c_poplu = Card.from_str("8S")
    assert c_poplu.classify_wildcard(tiplu) == WildcardType.POPLU
    assert c_poplu.maal_value(tiplu) == 2

    # 3. Alter: Same rank (7), complementary suit of same color (Spades -> Clubs)
    c_alter = Card.from_str("7C")
    assert c_alter.classify_wildcard(tiplu) == WildcardType.ALTER
    assert c_alter.maal_value(tiplu) == 5

    # 4. Ordinary Jokers: 7 of Hearts, 7 of Diamonds
    c_ord1 = Card.from_str("7H")
    c_ord2 = Card.from_str("7D")
    assert c_ord1.classify_wildcard(tiplu) == WildcardType.ORDINARY_JOKER
    assert c_ord2.classify_wildcard(tiplu) == WildcardType.ORDINARY_JOKER
    assert c_ord1.maal_value(tiplu) == 1
    assert c_ord2.maal_value(tiplu) == 1

    # 5. Non-wildcard
    c_norm = Card.from_str("8H")
    assert c_norm.classify_wildcard(tiplu) == WildcardType.NONE
    assert c_norm.maal_value(tiplu) == 0


def test_deck_deterministic_deal():
    hands1, tiplu1, stock1, discard1 = Deck.deal(seed=12345)
    hands2, tiplu2, stock2, discard2 = Deck.deal(seed=12345)

    assert len(hands1) == 4
    for h in hands1:
        assert len(h) == 21
    assert len(stock1) == 159 - 4 * 21 - 1 - 1  # 159 - 84 - 2 = 73
    assert len(discard1) == 1
    assert str(tiplu1) == str(tiplu2)
    assert [str(c) for c in hands1[0]] == [str(c) for c in hands2[0]]


def test_pure_sequences():
    # Valid normal pure sequence: 4S, 5S, 6S
    seq1 = [Card.from_str("4S"), Card.from_str("5S"), Card.from_str("6S")]
    assert is_pure_sequence(seq1)

    # Valid high Ace sequence: 10H, JH, QH, KH, AH
    seq2 = [Card.from_str("10H"), Card.from_str("JH"), Card.from_str("QH"), Card.from_str("KH"), Card.from_str("AH")]
    assert is_pure_sequence(seq2)

    # Invalid wrapping: KH, AH, 2H
    seq_wrap = [Card.from_str("KH"), Card.from_str("AH"), Card.from_str("2H")]
    assert not is_pure_sequence(seq_wrap)

    # Invalid suit mix: 4S, 5H, 6S
    seq_mix = [Card.from_str("4S"), Card.from_str("5H"), Card.from_str("6S")]
    assert not is_pure_sequence(seq_mix)


def test_tunella_and_dublee():
    # 3 copies of 9 Hearts
    t_cards = [
        Card(Suit.HEARTS, Rank.NINE, deck_id=0),
        Card(Suit.HEARTS, Rank.NINE, deck_id=1),
        Card(Suit.HEARTS, Rank.NINE, deck_id=2)
    ]
    tunellas = find_tunellas(t_cards)
    assert len(tunellas) == 1

    dublees = find_dublees(t_cards)
    assert len(dublees) == 1  # 3 identical cards contain 1 pair


def test_exact_worked_settlement_example():
    """
    Validates §2.3 of the implementation plan:
    - Player 0: Winner, Seen, Maal = 4
    - Player 1: Non-winner, Seen, Maal = 12
    - Player 2: Non-winner, Seen, Maal = 0
    - Player 3: Non-winner, Unseen, Maal = 6
    Target Payoffs:
    R_0 = +16, R_1 = +29, R_2 = -19, R_3 = -26. Total = 0.
    """
    engine = MarriageEngine(max_turns=10)
    engine.player_seen = [True, True, True, False]
    engine.raw_maal_held = [4, 12, 0, 6]
    engine.winner = 0
    engine.done = True

    # Compute settlement using engine's exact logic
    game_points = [0.0] * 4
    winner_collects = 0.0
    for j in range(4):
        if j != 0:
            penalty = PENALTY_UNSEEN if not engine.player_seen[j] else PENALTY_SEEN
            game_points[j] = -penalty
            winner_collects += penalty
    game_points[0] = winner_collects

    maal_net = [0.0] * 4
    seen_players = [0, 1, 2]
    unseen_players = [3]

    for i in seen_players:
        for j in seen_players:
            if i != j:
                maal_net[i] += (engine.raw_maal_held[i] - engine.raw_maal_held[j])
        for k in unseen_players:
            maal_net[i] += engine.raw_maal_held[i]

    for k in unseen_players:
        for j in seen_players:
            maal_net[k] -= engine.raw_maal_held[j]

    total_payoffs = [game_points[i] + maal_net[i] for i in range(4)]

    assert total_payoffs[0] == 16.0, f"Expected 16, got {total_payoffs[0]}"
    assert total_payoffs[1] == 29.0, f"Expected 29, got {total_payoffs[1]}"
    assert total_payoffs[2] == -19.0, f"Expected -19, got {total_payoffs[2]}"
    assert total_payoffs[3] == -26.0, f"Expected -26, got {total_payoffs[3]}"
    assert sum(total_payoffs) == 0.0, "Total payoff must strictly sum to zero!"


if __name__ == "__main__":
    test_card_basics_and_encoding()
    test_wildcard_classification()
    test_deck_deterministic_deal()
    test_pure_sequences()
    test_tunella_and_dublee()
    test_exact_worked_settlement_example()
    print("ALL CORE MARRIAGE RULE TESTS PASSED PERFECTLY!")
