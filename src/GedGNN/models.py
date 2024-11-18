import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from layers import AttentionModule, TensorNetworkModule, GedMatrixModule
from torch_geometric.utils import to_undirected,to_dense_adj
import math
import torch_geometric as pyg
from torch_geometric.nn.pool import global_add_pool
from torch import nn


class GedGNN(torch.nn.Module):
    
    def __init__(self,args,number_of_labels):
        super(GedGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        self.hidden_dims = self.args.hidden_dim
        self.num_layers_conv = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()
        self.fc_dim = self.args.bottle_neck_neurons
        self.num_layers_fc = len(self.fc_dim)
        self.fcs = torch.nn.ModuleList()
        for l in range(self.num_layers_conv):
            if l == 0:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
            else:
                nn = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[l-1], self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
            self.conv_layers.append(GINConv(nn, train_eps=True))
            self.gns.append(GraphNorm(self.hidden_dims[l]))

        self.mapMatrix = GedMatrixModule(self.hidden_dims[-1], self.args.weight_matrix_dim)
        self.costMatrix = GedMatrixModule(self.hidden_dims[-1], self.args.weight_matrix_dim)

        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)

        for l in range(self.num_layers_fc):
            if l == 0:
                self.fcs.append(torch.nn.Linear(self.args.tensor_neurons,self.fc_dim[l]))
            else:
                self.fcs.append(torch.nn.Linear(self.fc_dim[l-1],self.fc_dim[l]))
        
        self.scoring_layer = torch.nn.Linear(self.fc_dim[-1], 1)
    
    def convolutional_pass(self, features,graph_edge_index,batch,graph_2):
        bn_batch = batch * 2
        bn_batch[graph_2] += 1

        for l in range(self.num_layers_conv):
            features = self.gns[l](self.conv_layers[l](features,graph_edge_index),batch=bn_batch)
            if l != self.num_layers_conv - 1:
                features = torch.nn.functional.dropout(torch.nn.functional.relu(features),p=self.args.dropout,training=self.training)
        return features
            
    def get_bias_value(self, abstract_features_1, abstract_features_2,graph_batch_1,graph_batch_2):
        pooled_features_1 = self.attention(abstract_features_1,graph_batch_1)
        pooled_features_2 = self.attention(abstract_features_2,graph_batch_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)

        for l in range(self.num_layers_fc):
            scores = torch.nn.functional.relu(self.fcs[l](scores))
        scores = self.scoring_layer(scores)
        return scores

    def forward(self, data):
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch = data.batch
        edge_mapping_idx = data.edge_index_mapping
        undirected_edge_mapping_idx,_ = to_undirected(edge_mapping_idx)
        pair_indicator = data.x_indicator   
        graph_1 = (pair_indicator==0).squeeze(1)
        graph_2 = (pair_indicator==1).squeeze(1)
        graph_batch_1 = batch[graph_1]
        graph_batch_2 = batch[graph_2]

        abstract_features = self.convolutional_pass(graph_x,graph_edge_index,batch,graph_2)
        abstract_features_1 = abstract_features[edge_mapping_idx[0]]
        abstract_features_2 = abstract_features[edge_mapping_idx[1]]

        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)

        prob = pyg.utils.softmax(map_matrix.squeeze(-1),edge_mapping_idx[0]).unsqueeze(-1)
        soft_matrix = prob * cost_matrix

        abstract_features_1 = abstract_features[graph_1]
        abstract_features_2 = abstract_features[graph_2]
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2,graph_batch_1,graph_batch_2)

        score = torch.sigmoid(global_add_pool(soft_matrix,batch[edge_mapping_idx[0]]) + bias_value)
        
        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data.avg_v.unsqueeze(-1)
        elif self.args.target_mode == "linear":
            
            pre_ged = score * data.hb.unsqueeze(-1)
        else:
            assert False
        
        return score, pre_ged, map_matrix


        


