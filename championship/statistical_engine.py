"""
marriage/championship/statistical_engine.py
Q1 Statistical Inference Engine:
- Paired deal-level non-parametric Bootstrap Confidence Intervals (10,000 resamples)
- Two-tailed Sign-Flip Permutation Tests on paired deal differences
- Family-wise Holm-Bonferroni correction
- Analytical power analysis estimation
- Strictly rejects pooled Welch t-tests and Cohen's d on rotated rounds per Q1 guidelines
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class StatisticalEngine:
    """
    Implements the rigorous statistical inference standards mandated for Q1 journal submissions.
    Treats the duplicate deal as the paired unit of analysis.
    """
    def __init__(self, num_bootstrap_resamples: int = 10000, alpha: float = 0.05, seed: int = 42):
        self.num_resamples = num_bootstrap_resamples
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def paired_bootstrap_ci(self, diffs: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes 95% non-parametric bootstrap confidence interval on paired deal differences.
        
        Args:
            diffs: 1D array of paired deal differences Delta_k = Score_A,k - Score_B,k.
            
        Returns:
            mean_diff: Sample mean difference.
            ci_low: 2.5th percentile of bootstrap distribution.
            ci_high: 97.5th percentile of bootstrap distribution.
        """
        n = len(diffs)
        if n == 0:
            return 0.0, 0.0, 0.0
            
        mean_diff = float(np.mean(diffs))
        # Draw bootstrap sample means
        indices = self.rng.integers(0, n, size=(self.num_resamples, n))
        boot_means = np.mean(diffs[indices], axis=1)

        ci_low = float(np.percentile(boot_means, 100 * (self.alpha / 2.0)))
        ci_high = float(np.percentile(boot_means, 100 * (1.0 - self.alpha / 2.0)))

        return mean_diff, ci_low, ci_high

    def sign_flip_permutation_test(self, diffs: np.ndarray, num_permutations: int = 10000) -> float:
        """
        Exact / Monte Carlo sign-flip permutation test for paired deal differences.
        Tests H_0: The symmetric distribution of Delta_k is centered at 0.
        """
        n = len(diffs)
        if n == 0:
            return 1.0

        observed_t = np.abs(np.mean(diffs))
        
        # Randomly flip signs (+1 or -1)
        flips = self.rng.choice([-1.0, 1.0], size=(num_permutations, n))
        perm_means = np.abs(np.mean(diffs * flips, axis=1))

        # Two-tailed p-value with Laplace continuity correction
        p_val = float((np.sum(perm_means >= observed_t) + 1.0) / (num_permutations + 1.0))
        return p_val

    def holm_bonferroni_correction(self, p_values: List[float]) -> List[Tuple[float, bool]]:
        """
        Applies family-wise Holm-Bonferroni correction to a list of p-values.
        Returns:
            List of tuples (adjusted_p_value, is_significant_at_alpha).
        """
        m = len(p_values)
        if m == 0:
            return []

        # Sort indices by p-value
        sorted_indices = np.argsort(p_values)
        adjusted = [0.0] * m
        significant = [False] * m

        running_max = 0.0
        for rank, idx in enumerate(sorted_indices):
            raw_p = p_values[idx]
            multiplier = m - rank
            p_adj = min(1.0, raw_p * multiplier)
            # Monotonicity adjustment
            p_adj = max(running_max, p_adj)
            running_max = p_adj

            adjusted[idx] = p_adj
            significant[idx] = p_adj < self.alpha

        return list(zip(adjusted, significant))

    @staticmethod
    def estimate_sample_size_power(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
        """
        Calculates required duplicate deal count using analytical power formula:
        n approx ((z_{1 - alpha/2} + z_{1 - beta}) * sigma / delta)^2
        """
        # Standard normal quantiles
        # For alpha = 0.05 (two-tailed), z = 1.95996
        # For power = 0.80, z = 0.84162; for power = 0.90, z = 1.28155
        z_alpha = 1.95996
        z_beta = 0.84162 if power <= 0.85 else 1.28155
        
        if abs(delta) < 1e-6:
            return 100000
        n = ((z_alpha + z_beta) * sigma / delta) ** 2
        return int(np.ceil(n))

    def evaluate_head_to_head(self, deal_payoffs: np.ndarray, agent_a_idx: int, agent_b_idx: int) -> Dict[str, Any]:
        """
        Performs complete paired deal-level inference between Agent A and Agent B.
        """
        diffs = deal_payoffs[:, agent_a_idx] - deal_payoffs[:, agent_b_idx]
        mean_diff, ci_low, ci_high = self.paired_bootstrap_ci(diffs)
        p_val = self.sign_flip_permutation_test(diffs)

        # Significant if 95% CI excludes 0
        excludes_zero = (ci_low > 0) or (ci_high < 0)

        return {
            "mean_paired_difference": mean_diff,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "raw_p_value": p_val,
            "ci_excludes_zero": excludes_zero,
            "std_dev_paired": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        }
