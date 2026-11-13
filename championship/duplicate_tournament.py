"""
marriage/championship/duplicate_tournament.py
Executes duplicate deal benchmarks with full 4-seat cyclic rotation.
Treats the deal seed as the fundamental unit of analysis.
Eliminates deal-luck variance and validates zero illegal actions.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np

from marriage.core.engine import MarriageEngine, TurnPhase, NUM_PLAYERS
from marriage.agents.base_agent import BaseAgent

class DuplicateTournament:
    """
    Tournament engine executing cyclic duplicate deals.
    Every deal seed is played 4 times with seats cyclically permuted.
    """
    def __init__(
        self,
        agents: List[BaseAgent],
        deal_seeds: List[int],
        max_turns_per_round: int = 120,
    ):
        if len(agents) != NUM_PLAYERS:
            raise ValueError(f"DuplicateTournament requires exactly {NUM_PLAYERS} agents, got {len(agents)}")
        self.agents = agents
        self.deal_seeds = deal_seeds
        self.max_turns_per_round = max_turns_per_round
        self.engine = MarriageEngine(max_turns=max_turns_per_round)

    def play_single_round(self, round_agents: List[BaseAgent], seed: int) -> Tuple[List[float], Dict[str, Any]]:
        """
        Executes a single match of 4 players on a given seed.
        Returns:
            payoffs: [R_0, R_1, R_2, R_3] strictly summing to 0.
            info: Metadata including winner, turns, player_seen, illegal actions.
        """
        obs = self.engine.reset(seed=seed)
        for a in round_agents:
            a.reset_round()

        while not self.engine.done:
            p = self.engine.current_player
            agent = round_agents[p]
            phase = self.engine.turn_phase
            obs = self.engine.get_observation(p)
            mask = obs["action_mask"]

            if phase == TurnPhase.DRAW:
                action = agent.act_draw(obs, mask)
                self.engine.step_draw(action)
            elif phase == TurnPhase.SEE_MAAL:
                action = agent.act_see(obs, mask)
                self.engine.step_see_maal(action)
            elif phase == TurnPhase.DISCARD:
                action = agent.act_discard(obs, mask)
                self.engine.step_discard(action)
            elif phase == TurnPhase.DECLARE_WIN:
                action = agent.act_win(obs, mask)
                self.engine.step_declare_win(action)

        info = {
            "winner": self.engine.winner,
            "turns": self.engine.turn_count,
            "player_seen": self.engine.player_seen[:],
            "raw_maal": self.engine.raw_maal_held[:],
            "illegal_actions": self.engine.illegal_action_count,
            "is_timeout": self.engine.is_timeout
        }
        return self.engine.terminal_payoffs[:], info

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Runs the full cyclic duplicate tournament across all deal seeds.
        For each deal seed:
            Rotates seating 4 times (0, 1, 2, 3).
        Records deal-level aggregated metrics.
        """
        num_deals = len(self.deal_seeds)
        num_agents = NUM_PLAYERS
        agent_names = [a.name for a in self.agents]

        # Deal-level score accumulator: shape [num_deals, num_agents]
        deal_payoffs = np.zeros((num_deals, num_agents), dtype=np.float64)
        deal_wins = np.zeros((num_deals, num_agents), dtype=np.float64)
        deal_seen = np.zeros((num_deals, num_agents), dtype=np.float64)

        total_rounds = num_deals * 4
        total_illegal_actions = 0
        total_timeouts = 0
        round_lengths = []

        start_time = time.time()

        for d_idx, seed in enumerate(self.deal_seeds):
            # Play 4 cyclic seat rotations
            for rotation in range(4):
                # Seating assignment: at rotation r, agent i is seated at position (i + r) % 4
                # round_agents[seat] is the agent seated at that seat
                round_agents = [None] * 4
                for agent_idx in range(4):
                    seat = (agent_idx + rotation) % 4
                    round_agents[seat] = self.agents[agent_idx]

                payoffs, info = self.play_single_round(round_agents, seed=seed)

                total_illegal_actions += info["illegal_actions"]
                if info["is_timeout"]:
                    total_timeouts += 1
                round_lengths.append(info["turns"])

                # Map seat payoffs back to original agent indices
                for agent_idx in range(4):
                    seat = (agent_idx + rotation) % 4
                    agent_payoff = payoffs[seat]
                    deal_payoffs[d_idx, agent_idx] += agent_payoff / 4.0

                    if info["winner"] == seat:
                        deal_wins[d_idx, agent_idx] += 1.0 / 4.0
                    if info["player_seen"][seat]:
                        deal_seen[d_idx, agent_idx] += 1.0 / 4.0

            if verbose and (d_idx + 1) % max(1, (num_deals // 5)) == 0:
                print(f"Completed {d_idx + 1}/{num_deals} duplicate deals ({(d_idx + 1)*4} rounds)...")

        elapsed = time.time() - start_time

        # Compute summary statistics
        mean_payoffs = np.mean(deal_payoffs, axis=0)
        mean_win_rates = np.mean(deal_wins, axis=0)
        mean_seen_rates = np.mean(deal_seen, axis=0)

        results = {
            "agent_names": agent_names,
            "num_deals": num_deals,
            "total_rounds": total_rounds,
            "elapsed_seconds": elapsed,
            "total_illegal_actions": total_illegal_actions,
            "timeout_rate": total_timeouts / total_rounds,
            "mean_round_length": float(np.mean(round_lengths)),
            "deal_payoffs": deal_payoffs,
            "deal_wins": deal_wins,
            "deal_seen": deal_seen,
            "summary": {
                name: {
                    "mean_coins_per_round": float(mean_payoffs[i]),
                    "win_rate": float(mean_win_rates[i]),
                    "seen_rate": float(mean_seen_rates[i]),
                }
                for i, name in enumerate(agent_names)
            }
        }
        return results
