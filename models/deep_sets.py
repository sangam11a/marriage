"""
marriage/models/deep_sets.py
Permutation-Invariant DeepSets representation model (~350k parameters).
Encodes cards as an unordered multiset via element-wise mapping and symmetric pooling:
rho( sum_i phi(card_i) ) concatenated with public context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class DeepSets(nn.Module):
    """
    Capacity-matched DeepSets architecture (~350k parameters).
    Ensures mathematical permutation invariance over the ordering of cards in hand.
    """
    def __init__(self, card_feat_dim: int = 16, card_embed_dim: int = 128, hidden_dim: int = 280, latent_dim: int = 256):
        super().__init__()
        self.card_feat_dim = card_feat_dim
        self.card_embed_dim = card_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Per-card encoder phi: maps card features (type, suit, rank, wildcard, residual) to embedding
        # Input features per card: type_onehot (53) or compact features: suit (4), rank (13), wildcard (1), residual (1) = 19
        self.phi = nn.Sequential(
            nn.Linear(19, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, card_embed_dim),
            nn.LayerNorm(card_embed_dim),
            nn.ReLU()
        )

        # Context encoder
        # Public discard histogram (53) + opponent pickups (53) + context (10) = 116
        self.context_encoder = nn.Sequential(
            nn.Linear(116, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Aggregation network rho: takes pooled card embedding (card_embed_dim * 2: sum + max) + context (128) = 384
        self.rho = nn.Sequential(
            nn.Linear(card_embed_dim * 2 + 128, hidden_dim),
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

    def extract_features_from_obs(self, obs: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts:
        1. Card set tensor [B, M, 19]
        2. Context vector [B, 116]
        """
        hand_counts = obs["hand_vec"]
        wild = obs["wildcard_vec"]
        resid = obs["residual_vec"]

        # Build list of cards in hand
        card_features = []
        for t in range(53):
            count = hand_counts[t]
            if count > 0:
                # 19-dim feature: suit onehot (4), rank onehot (13), wildcard (1), residual / 3.0 (1)
                feat = [0.0] * 19
                if t < 52:
                    suit = t // 13
                    rank = t % 13
                    feat[suit] = 1.0
                    feat[4 + rank] = 1.0
                else:
                    feat[0] = 0.25
                    feat[1] = 0.25
                    feat[2] = 0.25
                    feat[3] = 0.25
                feat[17] = float(wild[t])
                feat[18] = resid[t] / 3.0

                for _ in range(count):
                    card_features.append(feat)

        if not card_features:
            card_features.append([0.0] * 19)

        cards_tensor = torch.tensor(card_features, dtype=torch.float32, device=device).unsqueeze(0)

        # Context vector
        disc = [c / 3.0 for c in obs["discard_vec"]]
        opp_pick = [c / 3.0 for c in obs["opp_pickups_vec"]]
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
        context_tensor = torch.tensor(disc + opp_pick + ctx_features, dtype=torch.float32, device=device).unsqueeze(0)

        return cards_tensor, context_tensor

    def forward(self, cards: torch.Tensor, context: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # cards: [B, M, 19]
        # phi element-wise: [B, M, card_embed_dim]
        phi_out = self.phi(cards)

        # Permutation-invariant symmetric pooling
        sum_pooled = torch.sum(phi_out, dim=1)  # [B, card_embed_dim]
        max_pooled, _ = torch.max(phi_out, dim=1)  # [B, card_embed_dim]
        card_rep = torch.cat([sum_pooled, max_pooled], dim=-1)

        # Context representation
        ctx_rep = self.context_encoder(context)

        # Combined representation through rho
        combined = torch.cat([card_rep, ctx_rep], dim=-1)
        latent = self.rho(combined)

        logits = {
            "draw": self.draw_head(latent),
            "see": self.see_head(latent),
            "discard": self.discard_head(latent),
            "win": self.win_head(latent)
        }
        val = self.value_head(latent)
        return logits, val
