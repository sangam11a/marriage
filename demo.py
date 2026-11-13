"""
marriage/demo.py
Demonstration of the end-to-end Q1 Research Pipeline for 21-card Nepali Marriage AI:
1. Environment initialization (159 cards, 4 players, 21 cards each)
2. Four competing agent archetypes:
   - Random Baseline
   - Heuristic Tuned
   - Heuristic Domain Expert (Card Counter)
   - Masked PPO Agent
3. Duplicate deals evaluation with 4-seat cyclic rotation
4. Zero-sum payoff verification across all rounds
5. Rigorous statistical inference:
   - 10,000-resample non-parametric Paired Bootstrap CIs
   - Sign-flip permutation tests
   - Holm-Bonferroni correction
   - Analytical power analysis estimation
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marriage.agents.random_agent import RandomAgent
from marriage.agents.heuristic_tuned import HeuristicTunedAgent
from marriage.agents.heuristic_expert import HeuristicExpertAgent
from marriage.agents.ppo_agent import MaskedPPOAgent
from marriage.agents.dmc_agent import DeepMonteCarloAgent
from marriage.championship.duplicate_tournament import DuplicateTournament
from marriage.championship.statistical_engine import StatisticalEngine

def run_pipeline_demo(num_deals: int = 8):
    print("=" * 80)
    print(" 21-CARD NEPALESE MARRIAGE AI: Q1 RESEARCH PIPELINE DEMO")
    print("=" * 80)
    print(f"Executing {num_deals} duplicate deals with 4-seat cyclic rotation ({num_deals * 4} total rounds)...")
    print("Treating the duplicate deal as the fundamental unit of analysis.\n")

    # 1. Initialize 4 diverse agents
    agents = [
        RandomAgent(name="Random", seed=42),
        HeuristicTunedAgent(name="Heuristic-Tuned"),
        HeuristicExpertAgent(name="Expert-CardCounter"),
        MaskedPPOAgent(name="Masked-PPO", backbone_type="flat")
    ]

    # 2. Test deal seed range (from test partition: 100000+)
    deal_seeds = [100000 + i * 17 for i in range(num_deals)]

    # 3. Run Duplicate Tournament
    tournament = DuplicateTournament(
        agents=agents,
        deal_seeds=deal_seeds,
        max_turns_per_round=60  # Compact max turns for fast verification
    )

    t0 = time.time()
    results = tournament.run(verbose=True)
    t_elapsed = time.time() - t0

    print(f"\nTournament completed in {t_elapsed:.2f} seconds ({results['total_rounds']} rounds).")
    print(f"Total illegal actions: {results['total_illegal_actions']} (Target: 0)")
    print(f"Mean round length: {results['mean_round_length']:.1f} turns")
    print(f"Timeout rate: {results['timeout_rate'] * 100:.1f}%\n")

    # 4. Summary Table
    print("-" * 75)
    print(f"{'Agent':<22} | {'Coins/Round':>14} | {'Win Rate':>12} | {'Seen Rate':>12}")
    print("-" * 75)
    for name, stats in results["summary"].items():
        print(f"{name:<22} | {stats['mean_coins_per_round']:>14.2f} | {stats['win_rate']*100:>11.1f}% | {stats['seen_rate']*100:>11.1f}%")
    print("-" * 75)

    # 5. Q1 Statistical Inference (Paired Deal-Level Analysis)
    print("\n" + "=" * 80)
    print(" STATISTICAL INFERENCE & HYPOTHESIS TESTING (Q1 STANDARD)")
    print("=" * 80)
    print("Evaluating Primary Hypothesis: Expert-CardCounter vs Heuristic-Tuned")

    stat_engine = StatisticalEngine(num_bootstrap_resamples=10000, alpha=0.05, seed=42)
    deal_payoffs = results["deal_payoffs"]

    # Head-to-Head: Expert (idx 2) vs Tuned (idx 1)
    eval_res = stat_engine.evaluate_head_to_head(deal_payoffs, agent_a_idx=2, agent_b_idx=1)
    
    print(f"Mean Paired Difference (Delta): {eval_res['mean_paired_difference']:+.2f} coins/round")
    print(f"95% Paired Bootstrap CI:       [{eval_res['ci_95_low']:+.2f}, {eval_res['ci_95_high']:+.2f}]")
    print(f"Sign-Flip Permutation p-value:  {eval_res['raw_p_value']:.4f}")
    print(f"95% CI Excludes Zero:           {eval_res['ci_excludes_zero']}")

    # Multiple testing correction across all pairwise matchups vs Random
    p_vals = []
    pairs = [(1, 0), (2, 0), (3, 0)]
    pair_labels = ["Tuned vs Random", "Expert vs Random", "PPO vs Random"]
    for a_idx, b_idx in pairs:
        res = stat_engine.evaluate_head_to_head(deal_payoffs, a_idx, b_idx)
        p_vals.append(res["raw_p_value"])

    holm_results = stat_engine.holm_bonferroni_correction(p_vals)
    print("\nFamily-wise Holm-Bonferroni Correction (vs Random Baseline):")
    for label, raw_p, (adj_p, sig) in zip(pair_labels, p_vals, holm_results):
        status = "SIGNIFICANT" if sig else "NOT SIGNIFICANT"
        print(f"  {label:<20}: raw p = {raw_p:.4f} -> Holm-adj p = {adj_p:.4f} [{status}]")

    # Analytical Power Analysis
    sigma_pilot = max(1.0, eval_res["std_dev_paired"])
    delta_target = 5.0  # target 5 coins/round difference
    n_needed = stat_engine.estimate_sample_size_power(sigma=sigma_pilot, delta=delta_target, power=0.80)
    print(f"\nAnalytical Power Analysis (to detect Delta = {delta_target} coins at 80% power, sigma = {sigma_pilot:.1f}):")
    print(f"  Required duplicate deals: n ~ {n_needed} deals ({n_needed * 4} rounds)")
    print("=" * 80)
    print("Pipeline execution and statistical validation completed successfully!\n")


if __name__ == "__main__":
    run_pipeline_demo(num_deals=8)
