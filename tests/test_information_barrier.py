"""
tests/test_information_barrier.py
Automated Leakage Test verifying enforcement of the Information Barrier:
Perturbing hidden environmental state (opponents' hands, unrevealed Tiplu, stock order)
must not alter the deployed unprivileged actor's observation tensor or decision logits.
"""

import sys
import os
import copy
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marriage.core.engine import MarriageEngine, TurnPhase
from marriage.core.card import Card, Suit, Rank
from marriage.agents.ppo_agent import MaskedPPOAgent

def test_information_barrier_leakage():
    engine = MarriageEngine(max_turns=50)
    agent = MaskedPPOAgent(name="Audit-PPO", backbone_type="flat")

    # 1. Base observation for player 0 (Unseen)
    obs_orig = engine.get_observation(0)
    assert not obs_orig["context"]["is_seen"], "Player 0 must begin as Unseen"
    # Verify wildcard flags in observation are strictly zero
    assert sum(obs_orig["wildcard_vec"]) == 0, "Wildcard vector must be strictly masked for Unseen player"

    # Forward pass on original observation
    with torch.no_grad():
        x_orig = agent.actor.extract_features_from_obs(obs_orig, agent.device)
        logits_orig, _ = agent.actor(x_orig)

    # 2. Perturbation A: Swap opponent hands entirely
    engine_pert_a = copy.deepcopy(engine)
    temp_hand = engine_pert_a.hands[1]
    engine_pert_a.hands[1] = engine_pert_a.hands[2]
    engine_pert_a.hands[2] = temp_hand

    obs_pert_a = engine_pert_a.get_observation(0)
    with torch.no_grad():
        x_pert_a = agent.actor.extract_features_from_obs(obs_pert_a, agent.device)
        logits_pert_a, _ = agent.actor(x_pert_a)

    assert torch.allclose(x_orig, x_pert_a, atol=1e-7), "Leakage detected: observation changed after opponent hand swap!"
    assert torch.allclose(logits_orig["discard"], logits_pert_a["discard"], atol=1e-7), "Leakage detected: actor discard logits changed!"

    # 3. Perturbation B: Mutate unrevealed Tiplu card
    engine_pert_b = copy.deepcopy(engine)
    # Change Tiplu to King of Spades
    engine_pert_b.tiplu = Card(Suit.SPADES, Rank.KING)
    obs_pert_b = engine_pert_b.get_observation(0)
    with torch.no_grad():
        x_pert_b = agent.actor.extract_features_from_obs(obs_pert_b, agent.device)
        logits_pert_b, _ = agent.actor(x_pert_b)

    assert torch.allclose(x_orig, x_pert_b, atol=1e-7), "Leakage detected: observation changed after unrevealed Tiplu change!"
    assert torch.allclose(logits_orig["discard"], logits_pert_b["discard"], atol=1e-7), "Leakage detected: actor output changed after Tiplu mutation!"

    # 4. Perturbation C: Shuffle hidden stock deck
    engine_pert_c = copy.deepcopy(engine)
    engine_pert_c.stock.reverse()
    obs_pert_c = engine_pert_c.get_observation(0)
    with torch.no_grad():
        x_pert_c = agent.actor.extract_features_from_obs(obs_pert_c, agent.device)
        logits_pert_c, _ = agent.actor(x_pert_c)

    assert torch.allclose(x_orig, x_pert_c, atol=1e-7), "Leakage detected: observation changed after stock shuffle!"
    assert torch.allclose(logits_orig["discard"], logits_pert_c["discard"], atol=1e-7), "Leakage detected: actor output changed after stock shuffle!"

    print("INFORMATION BARRIER LEAKAGE TEST PASSED PERFECTLY!")


if __name__ == "__main__":
    test_information_barrier_leakage()
