"""
tests/test_equivariance.py
Verifies mathematical permutation invariance/equivariance:
Shuffling the order of cards in the agent's hand produces identical representation embeddings.
"""

import sys
import os
import random
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marriage.core.engine import MarriageEngine
from marriage.models.deep_sets import DeepSets
from marriage.models.r_gcn import RelationalGNN
from marriage.models.flat_mlp import FlatMLP

def test_permutation_invariance():
    engine = MarriageEngine(max_turns=30)
    engine.reset(seed=777)

    # Hand for player 0
    hand_orig = engine.hands[0][:]

    # Model instances
    mlp = FlatMLP()
    deep_sets = DeepSets()
    rgcn = RelationalGNN()
    device = torch.device("cpu")

    # 1. Evaluate with original hand ordering
    obs_orig = engine.get_observation(0)
    
    with torch.no_grad():
        # MLP
        x_mlp1 = mlp.extract_features_from_obs(obs_orig, device)
        logits_mlp1, val_mlp1 = mlp(x_mlp1)

        # DeepSets
        cards_ds1, ctx_ds1 = deep_sets.extract_features_from_obs(obs_orig, device)
        logits_ds1, val_ds1 = deep_sets(cards_ds1, ctx_ds1)

        # RGCN
        nodes_g1, wild_g1, ctx_g1 = rgcn.extract_features_from_obs(obs_orig, device)
        logits_g1, val_g1 = rgcn(nodes_g1, wild_g1, ctx_g1)

    # 2. Shuffle cards in hand
    rng = random.Random(999)
    hand_shuffled = hand_orig[:]
    rng.shuffle(hand_shuffled)
    engine.hands[0] = hand_shuffled

    obs_shuffled = engine.get_observation(0)

    with torch.no_grad():
        # MLP (count histogram is naturally order-invariant)
        x_mlp2 = mlp.extract_features_from_obs(obs_shuffled, device)
        logits_mlp2, val_mlp2 = mlp(x_mlp2)

        # DeepSets
        cards_ds2, ctx_ds2 = deep_sets.extract_features_from_obs(obs_shuffled, device)
        logits_ds2, val_ds2 = deep_sets(cards_ds2, ctx_ds2)

        # RGCN
        nodes_g2, wild_g2, ctx_g2 = rgcn.extract_features_from_obs(obs_shuffled, device)
        logits_g2, val_g2 = rgcn(nodes_g2, wild_g2, ctx_g2)

    # Assert exact invariance
    assert torch.allclose(logits_mlp1["discard"], logits_mlp2["discard"], atol=1e-6), "FlatMLP failed invariance!"
    assert torch.allclose(logits_ds1["discard"], logits_ds2["discard"], atol=1e-6), "DeepSets failed permutation invariance!"
    assert torch.allclose(logits_g1["discard"], logits_g2["discard"], atol=1e-6), "RGCN failed permutation invariance!"

    print("PERMUTATION INVARIANCE TEST PASSED PERFECTLY FOR ALL THREE ARCHITECTURES!")


if __name__ == "__main__":
    test_permutation_invariance()
