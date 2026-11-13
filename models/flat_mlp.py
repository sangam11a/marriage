"""
marriage/models/flat_mlp.py
Flat MLP representation baseline (~350k parameters).
Processes count histograms and game context via dense feedforward layers with LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class FlatMLP(nn.Module):
    """
    Capacity-matched Flat MLP model (~350k parameters).
    Accepts flattened 275-dim observation tensor and outputs latent features,
    action logits (draw, see, discard, win), and state value.
    """
    def __init__(self, input_dim: int = 275, hidden_dim: int = 290, latent_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

        # Policy Heads
        self.draw_head = nn.Linear(latent_dim, 2)
        self.see_head = nn.Linear(latent_dim, 2)
        self.discard_head = nn.Linear(latent_dim, 53)
        self.win_head = nn.Linear(latent_dim, 2)

        # Value Head
        self.value_head = nn.Linear(latent_dim, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features_from_obs(self, obs: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Converts raw dictionary observation into flattened input tensor."""
        hand = obs["hand_vec"]
        wild = obs["wildcard_vec"]
        disc = obs["discard_vec"]
        resid = obs["residual_vec"]
        opp_pick = obs["opp_pickups_vec"]
        ctx = obs["context"]

        ctx_features = [
            float(ctx["is_seen"]),
            ctx["turn_count"] / 120.0,
            ctx["stock_size"] / 159.0,
            ctx["top_discard_type"] / 53.0 if ctx["top_discard_type"] >= 0 else 0.0,
            float(ctx["opponents_seen"][0]),
            float(ctx["opponents_seen"][1]),
            float(ctx["opponents_seen"][2]),
            1.0 if obs["turn_phase"] == 0 else 0.0,
            1.0 if obs["turn_phase"] == 1 else 0.0,
            1.0 if obs["turn_phase"] == 2 else 0.0,
        ]

        raw = hand + wild + disc + resid + opp_pick + ctx_features
        return torch.tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        Returns:
            logits_dict: Dictionary with 'draw', 'see', 'discard', 'win' logits.
            value: Scalar state value prediction.
        """
        latent = self.net(x)
        logits = {
            "draw": self.draw_head(latent),
            "see": self.see_head(latent),
            "discard": self.discard_head(latent),
            "win": self.win_head(latent)
        }
        val = self.value_head(latent)
        return logits, val
