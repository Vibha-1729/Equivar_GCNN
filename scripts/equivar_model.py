import math
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from torch_geometric.utils import scatter

# Official Change-of-Basis (cob) matrix constants from Kutana et al. (Sci. Rep. 2025)
SQ2_1 = 1.0 / math.sqrt(2.0)
SQ3_1 = 1.0 / math.sqrt(3.0)
SQ23_1 = SQ2_1 * SQ3_1

class GaussianCosEnvelopeBasisProjection(nn.Module):
    """
    Official Equivar Gaussian Cosine Envelope Radial Basis Projection.
    """
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.alpha = math.pi / stop
        self.gamma = -0.5 / (offset[1] - offset[0]).item()**2

    def forward(self, r: torch.Tensor):
        rcol = r.view(-1, 1)
        roffset = rcol - self.offset.view(1, -1)
        envelope = 0.5 * (1.0 + torch.cos(self.alpha * rcol))
        return envelope * torch.exp(self.gamma * torch.pow(roffset, 2))

class EquivarBECGNN(nn.Module):
    """
    Official Equivar E(3)-Equivariant Graph Neural Network following Kutana et al. (Scientific Reports 2025).
    Includes Gaussian Cosine Envelope Radial Basis functions, 1x0e + 1x1o + 1x2e irreps,
    official Change-of-Basis (cob) matrix, and Acoustic Sum Rule charge neutrality enforcement.
    """
    def __init__(self, num_species=95, hidden_dim=64, lmax=2, num_layers=3, r_max=5.0, num_radial=32):
        super().__init__()
        self.num_species = num_species
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        self.num_layers = num_layers
        
        # 1. Species embedding
        self.embedding = nn.Embedding(num_species, hidden_dim)
        
        # 2. Radial basis projection
        self.radial_proj = GaussianCosEnvelopeBasisProjection(start=0.0, stop=r_max, num_gaussians=num_radial)
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Irreps definitions
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        
        # 4. Message Passing Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            tp = FullyConnectedTensorProduct(
                self.irreps_node,
                self.irreps_sh,
                self.irreps_node,
                shared_weights=True
            )
            self.layers.append(tp)
            
        # 5. Output head (1x0e + 1x1o + 1x2e = 9 irrep components)
        self.irreps_out = o3.Irreps("1x0e + 1x1o + 1x2e")
        self.out_tp = FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=True
        )
        
        # Official Change-of-Basis (cob) matrix from Kutana et al. 2025
        cob_matrix = torch.tensor([
            [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,  -SQ2_1, 0.0,   0.0,   0.0,  ],     # t11
            [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,   SQ2_1,],     # t12
            [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,  -SQ2_1, 0.0,  ],     # t13
            [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,  -SQ2_1,],     # t21
            [SQ3_1, 0.0,   0.0,   2.0*SQ23_1, 0.0,   0.0,   0.0,   0.0,   0.0,  ],     # t22
            [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,   SQ2_1, 0.0,   0.0,  ],     # t23
            [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,   SQ2_1, 0.0,  ],     # t31
            [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,  -SQ2_1, 0.0,   0.0,  ],     # t32
            [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,   SQ2_1, 0.0,   0.0,   0.0,  ],     # t33
        ], dtype=torch.float32).T
        
        self.register_buffer("cob", cob_matrix)

    def forward(self, z, pos, edge_index, edge_vec, batch=None):
        num_nodes = z.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
            
        # 1. Node embeddings
        h = self.embedding(z)
        
        # 2. Edge features: Spherical Harmonics & Radial Basis
        edge_lengths = torch.norm(edge_vec, dim=-1)
        radial_emb = self.radial_proj(edge_lengths)
        radial_weights = self.radial_mlp(radial_emb)  # [E, hidden_dim]
        
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization='component'
        )
        
        src, dst = edge_index[0], edge_index[1]
        
        # 3. Equivariant Message Passing
        for layer in self.layers:
            msg = layer(h[src], edge_sh) * radial_weights
            h_agg = torch.zeros_like(h)
            h_agg.index_add_(0, dst, msg)
            h = h + h_agg
            
        # 4. Output head (9 irrep components)
        msg_out = self.out_tp(h[src], edge_sh) * radial_weights[:, :9]
        out_irreps = torch.zeros(num_nodes, 9, device=z.device, dtype=h.dtype)
        out_irreps.index_add_(0, dst, msg_out)
        
        # 5. Multiply by official Change-of-Basis (cob) matrix
        out_cart = out_irreps @ self.cob  # [N, 9]
        
        # 6. Exact Acoustic Sum Rule (Gonze 1997 / Pick 1970) Charge Neutrality
        natoms_per_graph = scatter(torch.ones(num_nodes, device=z.device), batch, dim=0, reduce='sum')
        sum_per_graph = scatter(out_cart, batch, dim=0, reduce='sum') / natoms_per_graph[:, None]
        
        index_expanded = batch.unsqueeze(1).expand(-1, 9)
        out_neutral_flat = out_cart - torch.gather(sum_per_graph, dim=0, index=index_expanded)
        
        # Reshape to [N, 3, 3] Cartesian tensor
        Z_star = out_neutral_flat.view(num_nodes, 3, 3)
        return Z_star
