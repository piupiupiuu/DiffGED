import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool,global_add_pool
from torch_geometric.nn.norm import GraphNorm
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ScalarEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        
        pos_x = (x_embed[:, None] / dim_t).squeeze(1)
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x

class AGNN(nn.Module):
    def __init__(self,hidden_dim,time_emb_dim,noise_dim):
        super(AGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_transform = nn.Linear(noise_dim,hidden_dim)
        self.P = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.Q = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.R = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.U = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.bn_bip_h = GraphNorm(self.hidden_dim)
        self.bn_bip_e = GraphNorm(self.hidden_dim)
        self.time_emb_layer = torch.nn.Sequential(torch.nn.ReLU(),torch.nn.Linear(time_emb_dim,hidden_dim))
        self.out_layer = nn.Sequential(nn.LayerNorm(hidden_dim, elementwise_affine=True),nn.SiLU(),nn.Linear(hidden_dim, hidden_dim))

    def forward(self,features,edge_mapping_idx,noise_mapping_emb,time_emb,batch):
        noise_mapping_emb = self.edge_transform(noise_mapping_emb)
        Q_h = self.Q(features)
        R_h = self.R(features)
        mapping_e_hat = self.P(noise_mapping_emb) + Q_h[edge_mapping_idx[0]] + R_h[edge_mapping_idx[1]]
        gates = torch.sigmoid(mapping_e_hat)

        U_h = self.U(features)
        V_h = self.V(features)
        aggr = global_add_pool(V_h[edge_mapping_idx[1]] * gates,edge_mapping_idx[0])
        h = U_h + aggr

        h = self.bn_bip_h(h,batch)
        e = self.bn_bip_e(mapping_e_hat,batch[edge_mapping_idx[0]])
        
        h = nn.functional.relu(h)
        e = nn.functional.relu(e)
        
        e = e + self.time_emb_layer(time_emb)[batch[edge_mapping_idx[0]]]
        h = features + h
        #e = self.out_layer(noise_mapping_emb + e)
        e = noise_mapping_emb + self.out_layer(e)
        return h,e