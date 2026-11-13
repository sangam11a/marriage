"""
marriage/core/melds.py
Fast combinatorial partition solver, meld validators, and Maal calculators for 21-card Nepali Marriage:
- Pure sequences (consecutive same-suit cards, no wildcards)
- Tunellas (3 identical cards: exact same rank AND suit)
- Dublees (pairs of identical cards: exact same rank AND suit)
- Impure sequences and sets (using wildcards)
- Primary qualification check (>= 3 pure sequences/tunellas OR >= 7 dublees)
- Full 21-card winning partition solver (Show condition)
- Comprehensive Maal point evaluation (Tiplu, Poplu, Alter, Jokers, Tunellas, Marriage)
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any
from marriage.core.card import Card, Suit, Rank, WildcardType

class MeldType:
    PURE_SEQUENCE = "pure_sequence"
    TUNELLA = "tunella"
    DUBLEE = "dublee"
    IMPURE_SEQUENCE = "impure_sequence"
    SET = "set"
    MARRIAGE = "marriage"


def find_tunellas(hand: List[Card]) -> List[Tuple[Card, Card, Card]]:
    """
    Finds all valid Tunellas in a hand.
    A Tunella consists of exactly 3 identical cards (same suit and rank).
    """
    counts = Counter(card.type_id for card in hand)
    tunellas = []
    by_type = defaultdict(list)
    for card in hand:
        by_type[card.type_id].append(card)

    for type_id, count in counts.items():
        if count >= 3:
            cards = by_type[type_id]
            tunellas.append((cards[0], cards[1], cards[2]))
    return tunellas


def find_dublees(hand: List[Card]) -> List[Tuple[Card, Card]]:
    """
    Finds all valid Dublees (identical pairs) in a hand.
    A Dublee is a pair of cards with the exact same rank and suit.
    """
    counts = Counter(card.type_id for card in hand)
    dublees = []
    by_type = defaultdict(list)
    for card in hand:
        by_type[card.type_id].append(card)

    for type_id, count in counts.items():
        if count >= 2:
            cards = by_type[type_id]
            # Each pair of 2 forms one dublee
            num_pairs = count // 2
            for p in range(num_pairs):
                dublees.append((cards[2 * p], cards[2 * p + 1]))
    return dublees


def is_pure_sequence(cards: List[Card]) -> bool:
    """
    Validates if a list of 3+ cards forms a strict Pure Sequence:
    - Same suit
    - Strictly consecutive ranks (Ace can be 1 or 14, no wrapping across King-Ace-2)
    - No wildcards / printed jokers used as substitutes
    - Distinct physical cards
    """
    if len(cards) < 3:
        return False
    if any(c.is_printed_joker for c in cards):
        return False
    
    first_suit = cards[0].suit
    if any(c.suit != first_suit for c in cards):
        return False

    # Extract distinct ranks
    ranks = [c.rank.value for c in cards]
    if len(set(ranks)) != len(cards):
        # Ranks must be distinct in a pure sequence
        return False

    sorted_ranks = sorted(ranks)
    # Check regular consecutive: e.g. [4, 5, 6]
    is_regular_seq = all(sorted_ranks[i] == sorted_ranks[i-1] + 1 for i in range(1, len(sorted_ranks)))
    if is_regular_seq:
        return True

    # Check high-Ace consecutive: [10, 11, 12, 13, 1] -> [10, 11, 12, 13, 14]
    if 1 in sorted_ranks:
        high_ranks = sorted([14 if r == 1 else r for r in sorted_ranks])
        if all(high_ranks[i] == high_ranks[i-1] + 1 for i in range(1, len(high_ranks))):
            return True

    return False


def find_all_pure_sequences(hand: List[Card]) -> List[List[Card]]:
    """
    Finds all unique 3-card pure sequences that can be extracted from the hand.
    """
    by_suit = defaultdict(list)
    for c in hand:
        if not c.is_printed_joker:
            by_suit[c.suit.value].append(c)

    pure_sequences = []
    for suit_val, suit_cards in by_suit.items():
        if len(suit_cards) < 3:
            continue
        
        # Group available cards by rank (can have up to 3 copies per rank in 3 decks)
        rank_to_cards = defaultdict(list)
        for c in suit_cards:
            rank_to_cards[c.rank.value].append(c)

        ranks_present = sorted(rank_to_cards.keys())

        # Check for sequences of length 3 (including A-2-3 and Q-K-A)
        # Normal consecutive sequences
        for i in range(len(ranks_present)):
            r = ranks_present[i]
            if (r + 1) in rank_to_cards and (r + 2) in rank_to_cards:
                c1 = rank_to_cards[r][0]
                c2 = rank_to_cards[r + 1][0]
                c3 = rank_to_cards[r + 2][0]
                pure_sequences.append([c1, c2, c3])

        # High Ace sequence (Q-K-A: 12, 13, 1)
        if 12 in rank_to_cards and 13 in rank_to_cards and 1 in rank_to_cards:
            c1 = rank_to_cards[12][0]
            c2 = rank_to_cards[13][0]
            c3 = rank_to_cards[1][0]
            pure_sequences.append([c1, c2, c3])

    return pure_sequences


def can_see_maal(hand: List[Card]) -> bool:
    """
    Evaluates whether the hand satisfies the prerequisite to 'See Maal':
    - At least 3 primary sets (Pure Sequences or Tunellas), OR
    - At least 7 Dublees.
    """
    # 1. Check Dublee condition (>= 7 dublees)
    dublees = find_dublees(hand)
    if len(dublees) >= 7:
        return True

    # 2. Check 3 Primary sets (Tunellas + Pure Sequences)
    # Note: Tunellas and Pure sequences must be mutually disjoint
    tunellas = find_tunellas(hand)
    if len(tunellas) >= 3:
        return True

    # Check combinations of disjoint pure sequences and tunellas
    # Represent hand as card_id set or multiset
    card_counts = Counter(c.type_id for c in hand)
    
    # Greedy / branch search to find 3 disjoint primary sets
    return _has_k_disjoint_primary_sets(hand, k=3)


def _has_k_disjoint_primary_sets(hand: List[Card], k: int = 3) -> bool:
    """
    Recursively tests if at least k disjoint primary sets (pure sequence or tunella) exist.
    """
    # Fast path: check tunellas
    by_type = Counter(c.type_id for c in hand)
    tunella_types = [t for t, cnt in by_type.items() if cnt >= 3]
    if len(tunella_types) >= k:
        return True

    # Group cards by suit and rank for fast pure sequence extraction
    cards_by_suit = defaultdict(lambda: defaultdict(int))
    for c in hand:
        if not c.is_printed_joker:
            cards_by_suit[c.suit.value][c.rank.value] += 1

    # Backtracking search for k disjoint sets
    def backtrack(found_count: int, suit_idx: int) -> bool:
        if found_count >= k:
            return True
        if suit_idx > 3:
            return False

        # Try to find a sequence in current suit
        ranks_in_suit = cards_by_suit[suit_idx]
        for r in range(1, 14):
            # Normal sequence: r, r+1, r+2
            if r <= 11 and ranks_in_suit[r] > 0 and ranks_in_suit[r+1] > 0 and ranks_in_suit[r+2] > 0:
                ranks_in_suit[r] -= 1
                ranks_in_suit[r+1] -= 1
                ranks_in_suit[r+2] -= 1
                if backtrack(found_count + 1, suit_idx):
                    ranks_in_suit[r] += 1
                    ranks_in_suit[r+1] += 1
                    ranks_in_suit[r+2] += 1
                    return True
                ranks_in_suit[r] += 1
                ranks_in_suit[r+1] += 1
                ranks_in_suit[r+2] += 1

            # High Ace: Q, K, A (12, 13, 1)
            if r == 12 and ranks_in_suit[12] > 0 and ranks_in_suit[13] > 0 and ranks_in_suit[1] > 0:
                ranks_in_suit[12] -= 1
                ranks_in_suit[13] -= 1
                ranks_in_suit[1] -= 1
                if backtrack(found_count + 1, suit_idx):
                    ranks_in_suit[12] += 1
                    ranks_in_suit[13] += 1
                    ranks_in_suit[1] += 1
                    return True
                ranks_in_suit[12] += 1
                ranks_in_suit[13] += 1
                ranks_in_suit[1] += 1

            # Check Tunella in current rank
            if ranks_in_suit[r] >= 3:
                ranks_in_suit[r] -= 3
                if backtrack(found_count + 1, suit_idx):
                    ranks_in_suit[r] += 3
                    return True
                ranks_in_suit[r] += 3

        # Move to next suit
        return backtrack(found_count, suit_idx + 1)

    return backtrack(0, 0)


def calculate_maal(hand: List[Card], tiplu: Optional[Card], is_seen: bool) -> int:
    """
    Calculates the total raw Maal points held in a player's hand according to formal Nepali Marriage rules:
    - Tiplu: 3 Maal each
    - Poplu: 2 Maal each
    - Alter: 5 Maal each
    - Ordinary Jokers / Printed Jokers: 1 Maal each
    - Pure Tunella (non-wildcard): 5 Maal
    - Wildcard Tunella: 15 Maal (Tiplu, Poplu, or Alter tunella)
    - Marriage / Phun (Tiplu + Poplu + Alter in same hand): +10 Maal bonus
    
    Note: If player is NOT Seen, their effective Maal for settlement is 0,
    but this function returns raw potential Maal held.
    """
    if tiplu is None:
        return 0

    total_maal = 0
    type_counts = Counter(c.type_id for c in hand)
    
    # 1. Base card Maal
    has_tiplu = False
    has_poplu = False
    has_alter = False

    for card in hand:
        w_type = card.classify_wildcard(tiplu)
        base = w_type.base_maal()
        total_maal += base
        if w_type == WildcardType.TIPLU:
            has_tiplu = True
        elif w_type == WildcardType.POPLU:
            has_poplu = True
        elif w_type == WildcardType.ALTER:
            has_alter = True

    # 2. Marriage Bonus (Holding Tiplu + Poplu + Alter)
    if has_tiplu and has_poplu and has_alter:
        total_maal += 10  # Marriage bonus

    # 3. Tunellas (3 identical cards)
    for type_id, count in type_counts.items():
        if count >= 3:
            # Check if this card type is a wildcard
            sample_card = Card.from_type_id(type_id)
            w_type = sample_card.classify_wildcard(tiplu)
            if w_type in (WildcardType.TIPLU, WildcardType.POPLU, WildcardType.ALTER):
                total_maal += 15  # Wildcard tunella bonus
            elif w_type in (WildcardType.ORDINARY_JOKER, WildcardType.PRINTED_JOKER):
                total_maal += 10  # Joker tunella
            else:
                total_maal += 5   # Standard pure tunella

    return total_maal


def is_winning_hand(hand: List[Card], tiplu: Optional[Card], is_seen: bool) -> bool:
    """
    Evaluates if a 21-card hand is a complete winning hand ('Show').
    Requirements:
    1. The player must already be Seen, or have just qualified to be Seen.
    2. Hand size must be 21 cards.
    3. Either:
       a) Standard Game: 7 valid melds of size 3 (with at least 3 primary sets: pure sequences or tunellas).
       b) Dublee Game: At least 8 Dublees + 1 triplet/pure sequence.
    """
    if len(hand) != 21:
        return False

    # Dublee Win Check
    dublees = find_dublees(hand)
    if len(dublees) >= 8:
        # 8 dublees = 16 cards. Remaining 5 cards must form a valid meld (e.g. sequence of 3 or tunella)
        # In full 21 card rules, 8 dublees + 1 triplet/tunella = 19 cards + 2 cards = 21 cards (or 7 dublees + 1 tunella + 1 sequence of 4)
        if len(dublees) >= 10:  # 10 dublees = 20 cards
            return True

    # Standard Win Check: Must have >= 3 primary sets
    if not can_see_maal(hand):
        return False

    # If Seen, wildcards can substitute for any card in impure sequences and sets
    # Fast combinatorial validation using recursive branch-and-bound
    return _can_partition_into_7_melds(hand, tiplu)


def _can_partition_into_7_melds(hand: List[Card], tiplu: Optional[Card]) -> bool:
    """
    Solves the 21-card hand partition problem into 7 valid melds of size 3.
    """
    # Separate wildcards and natural cards
    wildcards = []
    naturals = []
    for c in hand:
        if c.is_wildcard(tiplu):
            wildcards.append(c)
        else:
            naturals.append(c)

    # Count natural card frequencies
    nat_counts = Counter(c.type_id for c in naturals)
    num_wildcards = len(wildcards)

    # State representation for memoization: (tuple of (type_id, count), wildcards_remaining, primary_count)
    memo = {}

    def search(state_counts: Tuple[Tuple[int, int], ...], wild_avail: int, primaries: int) -> bool:
        state_key = (state_counts, wild_avail, min(primaries, 3))
        if state_key in memo:
            return memo[state_key]

        # Base case: All natural cards exhausted
        if not state_counts:
            # Remaining wildcards must be in multiples of 3 (forms wildcard sets)
            result = (wild_avail % 3 == 0) and (primaries >= 3)
            memo[state_key] = result
            return result

        # Pick the lowest type_id card to eliminate
        first_type, count = state_counts[0]
        suit = first_type // 13
        rank = (first_type % 13) + 1

        counts_dict = dict(state_counts)

        # 1. Try forming Tunella (if count >= 3) -> Primary set!
        if count >= 3:
            counts_dict[first_type] -= 3
            if counts_dict[first_type] == 0:
                del counts_dict[first_type]
            next_state = tuple(sorted(counts_dict.items()))
            if search(next_state, wild_avail, primaries + 1):
                memo[state_key] = True
                return True
            counts_dict = dict(state_counts)

        # 2. Try forming Pure Sequence (rank, rank+1, rank+2 in same suit) -> Primary set!
        if rank <= 11:
            t2 = suit * 13 + (rank)      # rank + 1
            t3 = suit * 13 + (rank + 1)  # rank + 2
            if counts_dict.get(t2, 0) > 0 and counts_dict.get(t3, 0) > 0:
                counts_dict[first_type] -= 1
                counts_dict[t2] -= 1
                counts_dict[t3] -= 1
                clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
                next_state = tuple(sorted(clean_dict.items()))
                if search(next_state, wild_avail, primaries + 1):
                    memo[state_key] = True
                    return True
                counts_dict = dict(state_counts)

        # High Ace Pure Sequence: 10(9), J(10), Q(11), K(12), A(0) -> Q, K, A
        if rank == 1:
            # Ace can also match Q(12) and K(13)
            t_q = suit * 13 + 11  # Q
            t_k = suit * 13 + 12  # K
            if counts_dict.get(t_q, 0) > 0 and counts_dict.get(t_k, 0) > 0:
                counts_dict[first_type] -= 1
                counts_dict[t_q] -= 1
                counts_dict[t_k] -= 1
                clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
                next_state = tuple(sorted(clean_dict.items()))
                if search(next_state, wild_avail, primaries + 1):
                    memo[state_key] = True
                    return True
                counts_dict = dict(state_counts)

        # 3. Try forming Natural Set (same rank, different suits)
        other_suits = [s for s in range(4) if s != suit]
        matchable_types = [s * 13 + (rank - 1) for s in other_suits if counts_dict.get(s * 13 + (rank - 1), 0) > 0]
        if len(matchable_types) >= 2:
            # Pick first 2 available
            s1_type, s2_type = matchable_types[0], matchable_types[1]
            counts_dict[first_type] -= 1
            counts_dict[s1_type] -= 1
            counts_dict[s2_type] -= 1
            clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
            next_state = tuple(sorted(clean_dict.items()))
            if search(next_state, wild_avail, primaries):
                memo[state_key] = True
                return True
            counts_dict = dict(state_counts)

        # 4. Try using Wildcards (Impure Sequence or Impure Set)
        if wild_avail >= 1:
            # Impure sequence with 1 wildcard: (r, r+1) + Wildcard
            if rank <= 12:
                t2 = suit * 13 + rank  # rank + 1
                if counts_dict.get(t2, 0) > 0:
                    counts_dict[first_type] -= 1
                    counts_dict[t2] -= 1
                    clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
                    next_state = tuple(sorted(clean_dict.items()))
                    if search(next_state, wild_avail - 1, primaries):
                        memo[state_key] = True
                        return True
                    counts_dict = dict(state_counts)

            # Impure set with 1 wildcard: (r, r) + Wildcard (two of same rank, different suits or same)
            for m_type in matchable_types:
                counts_dict[first_type] -= 1
                counts_dict[m_type] -= 1
                clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
                next_state = tuple(sorted(clean_dict.items()))
                if search(next_state, wild_avail - 1, primaries):
                    memo[state_key] = True
                    return True
                counts_dict = dict(state_counts)

            # Natural pair + Wildcard:
            if count >= 2:
                counts_dict[first_type] -= 2
                clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
                next_state = tuple(sorted(clean_dict.items()))
                if search(next_state, wild_avail - 1, primaries):
                    memo[state_key] = True
                    return True
                counts_dict = dict(state_counts)

        if wild_avail >= 2:
            # Single card + 2 Wildcards
            counts_dict[first_type] -= 1
            clean_dict = {k: v for k, v in counts_dict.items() if v > 0}
            next_state = tuple(sorted(clean_dict.items()))
            if search(next_state, wild_avail - 2, primaries):
                memo[state_key] = True
                return True

        memo[state_key] = False
        return False

    initial_state = tuple(sorted(nat_counts.items()))
    return search(initial_state, num_wildcards, 0)
