import torch
import torch.nn as nn

# From https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
def build_alibi_slopes(num_heads, alibi_bias_max=8, device=None):
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64, device=device).float()
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)[:, :num_heads, ...]

    return slopes.view(num_heads)

# From the CoPE paper: https://arxiv.org/pdf/2405.18719
# experiments for testing CoPE and FIRE with ILR were incomplete due to time and resource constraints
class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)

        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)

# From the FIRE paper: https://arxiv.org/abs/2310.04418
class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6):
        super(FIRE, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads)
        )

        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        return fire_bias
