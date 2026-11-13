"""
marriage/models/r_gcn.py
Relational Graph Convolutional Network (R-GCN) (~350k parameters).
Models card interactions via relational typed edges:
- e_seq: consecutive ranks in identical suit (pure/impure sequences)
- e_set: identical ranks across suits (triplets/quads)
- e_wild: wildcard/synergy relations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

class RelationalGNN(nn.Module):
    """
    Capacity-matched Relational Graph Convolutional Network (~350k parameters).
    Embeds cards as nodes in a graph with game-theoretic relational edges.
    """
    def __init__(self, node_in_dim: int = 5, hidden_dim: int = 112, num_relations: int = 4, latent_dim: int = 256):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.latent_dim = latent_dim

        # Input projection
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # 3 Relational GCN Layers
        # Each layer has weight matrices for each relation r in {0: seq, 1: set, 2: wild, 3: self}
        self.w_rel1 = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.b_rel1 = nn.Parameter(torch.zeros(hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.w_rel2 = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.b_rel2 = nn.Parameter(torch.zeros(hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.w_rel3 = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.b_rel3 = nn.Parameter(torch.zeros(hidden_dim))
        self.norm3 = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.w_rel1)
        nn.init.xavier_uniform_(self.w_rel2)
        nn.init.xavier_uniform_(self.w_rel3)

        # Context encoder: 10 scalar game context features
        self.context_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Pooling & Readout
        # Graph pooling: sum + mean of 53 nodes (hidden_dim * 2) + context (64) = 96 * 2 + 64 = 256
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

        # Heads
        self.draw_head = nn.Linear(latent_dim, 2)
        self.see_head = nn.Linear(latent_dim, 2)
        self.discard_head = nn.Linear(latent_dim, 53)
        self.win_head = nn.Linear(latent_dim, 2)
        self.value_head = nn.Linear(latent_dim, 1)

        # Precompute static adjacency matrices for relations 0 (seq) and 1 (set)
        self.register_buffer("adj_seq", self._build_seq_adj())
        self.register_buffer("adj_set", self._build_set_adj())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _build_seq_adj() -> torch.Tensor:
        """Adjacency for consecutive ranks in same suit (normalized)."""
        adj = torch.zeros(53, 53, dtype=torch.float32)
        for s in range(4):
            for r in range(1, 13):
                u = s * 13 + (r - 1)
                v = s * 13 + r
                adj[u, v] = 1.0
                adj[v, u] = 1.0
            # High ace: Q-K-A (Ace connects to King)
            u_ace = s * 13 + 0
            v_king = s * 13 + 12
            adj[u_ace, v_king] = 1.0
            adj[v_king, u_ace] = 1.0

        deg = torch.clamp(adj.sum(dim=-1, keepdim=True), min=1.0)
        return adj / deg

    @staticmethod
    def _build_set_adj() -> torch.Tensor:
        """Adjacency for identical ranks in different suits."""
        adj = torch.zeros(53, 53, dtype=torch.float32)
        for r in range(13):
            suits_for_r = [s * 13 + r for s in range(4)]
            for u in suits_for_r:
                for v in suits_for_r:
                    if u != v:
                        adj[u, v] = 1.0

        deg = torch.clamp(adj.sum(dim=-1, keepdim=True), min=1.0)
        return adj / deg

    def extract_features_from_obs(self, obs: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts:
        - node_features: [1, 53, 5]
        - wild_adj: [1, 53, 53]
        - context: [1, 10]
        """
        hand = obs["hand_vec"]
        wild = obs["wildcard_vec"]
        disc = obs["discard_vec"]
        resid = obs["residual_vec"]
        opp_pick = obs["opp_pickups_vec"]

        nodes = []
        for t in range(53):
            nodes.append([
                hand[t] / 3.0,
                float(wild[t]),
                disc[t] / 3.0,
                resid[t] / 3.0,
                opp_pick[t] / 3.0
            ])

        nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=device).unsqueeze(0)

        # Wildcard adjacency: fully connects wildcards to all other cards in hand
        wild_adj = torch.zeros(53, 53, dtype=torch.float32, device=device)
        wild_indices = [i for i, w in enumerate(wild) if w == 1]
        for w_idx in wild_indices:
            for t in range(53):
                wild_adj[w_idx, t] = 1.0
                wild_adj[t, w_idx] = 1.0
        deg = torch.clamp(wild_adj.sum(dim=-1, keepdim=True), min=1.0)
        wild_adj = (wild_adj / deg).unsqueeze(0)

        # Context
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
        context_tensor = torch.tensor(ctx_features, dtype=torch.float32, device=device).unsqueeze(0)

        return nodes_tensor, wild_adj, context_tensor

    def _rgcn_layer(self, h: torch.Tensor, w_rel: torch.Tensor, b_rel: torch.Tensor, norm: nn.LayerNorm, wild_adj: torch.Tensor) -> torch.Tensor:
        # h: [B, 53, hidden_dim]
        # Relation 0: Sequence
        out0 = torch.bmm(self.adj_seq.unsqueeze(0).expand(h.size(0), -1, -1), h) @ w_rel[0]
        # Relation 1: Set
        out1 = torch.bmm(self.adj_set.unsqueeze(0).expand(h.size(0), -1, -1), h) @ w_rel[1]
        # Relation 2: Wildcard
        out2 = torch.bmm(wild_adj, h) @ w_rel[2]
        # Relation 3: Self loop
        out3 = h @ w_rel[3]

        out = out0 + out1 + out2 + out3 + b_rel
        return F.relu(norm(out))

    def forward(self, nodes: torch.Tensor, wild_adj: torch.Tensor, context: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        h = F.relu(self.node_encoder(nodes))

        # 3 Relational message passing steps
        h = self._rgcn_layer(h, self.w_rel1, self.b_rel1, self.norm1, wild_adj)
        h = self._rgcn_layer(h, self.w_rel2, self.b_rel2, self.norm2, wild_adj)
        h = self._rgcn_layer(h, self.w_rel3, self.b_rel3, self.norm3, wild_adj)

        # Graph pooling: sum + mean
        sum_pool = torch.sum(h, dim=1)  # [B, hidden_dim]
        mean_pool = torch.mean(h, dim=1)  # [B, hidden_dim]

        ctx_emb = self.context_encoder(context)

        combined = torch.cat([sum_pool, mean_pool, ctx_emb], dim=-1)
        latent = self.readout(combined)

        logits = {
            "draw": self.draw_head(latent),
            "see": self.see_head(latent),
            "discard": self.discard_head(latent),
            "win": self.win_head(latent)
        }
        val = self.value_head(latent)
        return logits, val
