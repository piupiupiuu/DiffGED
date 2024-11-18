import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from layers import AGNN, timestep_embedding, ScalarEmbeddingSine
from math import exp
from torch_geometric.utils import dense_to_sparse,to_undirected
import math
import torch_geometric as pyg
from torch_geometric.nn.pool import global_add_pool
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils



class DiffMatch(torch.nn.Module):
    def __init__(self,args,number_of_labels):
        super(DiffMatch, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        self.hidden_dims = self.args.hidden_dim
        self.num_layers = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for l in range(self.num_layers):
            if l == 0:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                agnn = AGNN(self.hidden_dims[l],self.hidden_dims[0]//2,self.hidden_dims[l])
            else:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[l-1], self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                agnn = AGNN(self.hidden_dims[l],self.hidden_dims[0]//2,self.hidden_dims[l-1])
            self.conv_layers.append(GINConv(nn, train_eps=True))
            self.agnn_layers.append(agnn)
            self.gns.append(GraphNorm(self.hidden_dims[l]))
        
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0]//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[0]//2, self.hidden_dims[0]//2),
        )
        
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0], normalize=False)
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        
        self.mapMatrix = torch.nn.Sequential(torch.nn.Linear(self.hidden_dims[-1],self.hidden_dims[-1]*2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.hidden_dims[-1]*2,self.hidden_dims[-1]),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.hidden_dims[-1],1))
        

    def convolutional_pass(self, features,graph_edge_index,edge_mapping_idx,noise_mapping_emb,time_emb,batch,graph_2):
        bn_batch = batch * 2
        bn_batch[graph_2] += 1

        for l in range(self.num_layers):
            features = torch.nn.functional.relu(self.gns[l](self.conv_layers[l](features,graph_edge_index),batch=bn_batch))
            features, noise_mapping_emb = self.agnn_layers[l](features,edge_mapping_idx,noise_mapping_emb,time_emb,batch)
        
        return features,noise_mapping_emb
    
    def forward(self, data,noise_mapping_attr,t):
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch = data.batch
        edge_mapping_idx = data.edge_index_mapping

        undirected_edge_mapping_idx,undirected_noise_mapping_attr = to_undirected(edge_mapping_idx,noise_mapping_attr)
        pair_indicator = data.x_indicator   
        graph_1 = (pair_indicator==0).squeeze(1)
        graph_2 = (pair_indicator==1).squeeze(1)
        graph_batch_1 = batch[graph_1]
        graph_batch_2 = batch[graph_2]

        # initialize the embeddings of time step t, node matching matrix M, and transpose(M)
        time_emb =self.time_embed(timestep_embedding(t, self.hidden_dims[0]))       
        undirected_noise_mapping_emb = self.edge_embed(self.edge_pos_embed(undirected_noise_mapping_attr))
        
        abstract_features,noise_mapping_emb = self.convolutional_pass(graph_x,graph_edge_index,undirected_edge_mapping_idx,undirected_noise_mapping_emb,time_emb,batch,graph_2)

        map_matrix = self.mapMatrix(noise_mapping_emb)

        # sum the values for node matching matrix M, and transpose(M)
        _,map_matrix = to_undirected(undirected_edge_mapping_idx,map_matrix)
        map_matrix = map_matrix[(pair_indicator[undirected_edge_mapping_idx[0]]==0).squeeze(1)]

        return map_matrix