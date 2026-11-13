"""
tests/test_action_masking.py
Validates that action masking strictly prevents any illegal actions.
Simulates multiple rounds across different agents and asserts illegal action count is identically 0.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marriage.core.engine import MarriageEngine, TurnPhase
from marriage.agents.random_agent import RandomAgent
from marriage.agents.heuristic_tuned import HeuristicTunedAgent
from marriage.agents.heuristic_expert import HeuristicExpertAgent
from marriage.agents.ppo_agent import MaskedPPOAgent

def test_strict_action_masking():
    agents = [
        RandomAgent(name="Random-0", seed=101),
        HeuristicTunedAgent(name="Tuned-1"),
        HeuristicExpertAgent(name="Expert-2"),
        MaskedPPOAgent(name="PPO-3", backbone_type="flat")
    ]

    total_turns_tested = 0
    total_rounds = 5

    for r in range(total_rounds):
        engine = MarriageEngine(max_turns=60)
        engine.reset(seed=2000 + r)

        for a in agents:
            a.reset_round()

        while not engine.done:
            p = engine.current_player
            agent = agents[p]
            phase = engine.turn_phase
            obs = engine.get_observation(p)
            mask = obs["action_mask"]

            # Must have at least 1 legal action
            assert sum(mask) > 0, f"Action mask is completely zero in phase {phase} for player {p}!"

            if phase == TurnPhase.DRAW:
                action = agent.act_draw(obs, mask)
                assert mask[action] == 1, f"Draw action {action} violates mask {mask}"
                engine.step_draw(action)
            elif phase == TurnPhase.SEE_MAAL:
                action = agent.act_see(obs, mask)
                assert mask[action] == 1, f"See action {action} violates mask {mask}"
                engine.step_see_maal(action)
            elif phase == TurnPhase.DISCARD:
                action = agent.act_discard(obs, mask)
                assert mask[action] == 1, f"Discard action {action} violates mask {mask}"
                engine.step_discard(action)
            elif phase == TurnPhase.DECLARE_WIN:
                action = agent.act_win(obs, mask)
                assert mask[action] == 1, f"Win action {action} violates mask {mask}"
                engine.step_declare_win(action)

            total_turns_tested += 1

        # Check engine illegal action count
        assert engine.illegal_action_count == 0, f"Engine logged {engine.illegal_action_count} illegal actions!"

    print(f"STRICT ACTION MASKING TEST PASSED! Verified {total_turns_tested} actions across {total_rounds} rounds with 0 illegal actions.")


if __name__ == "__main__":
    test_strict_action_masking()
