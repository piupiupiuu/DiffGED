from layers import AttentionModule, TensorNetworkModule, Block, DiffPool
from hungarian_ged import hungarian_ged
from torch_geometric.nn import GCNConv, GINConv, SplineConv
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree, dense_to_sparse, subgraph
import torch
import torch.nn.functional as F
from a_star import a_star
import numpy as np
from itertools import combinations
import time
VERY_LARGE_INT = 65536
PRINT_TIMING = False
class GENN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GENN, self).__init__()
        
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
        self.enable_a_star = self.args.enable_astar
        if self.enable_a_star:
            self.gnn_1_cache = dict()
            self.gnn_2_cache = dict()
            self.heuristic_cache = dict()

    def register_additional_layer(self):
        self.cross_graph = torch.nn.Linear(self.args.filters_3 * 2, self.args.filters_3)

    def reset_cache(self):
        self.gnn_1_cache = dict()
        self.gnn_2_cache = dict()
        self.heuristic_cache = dict()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'spline':
            self.convolution_1 = SplineConv(self.number_labels, self.args.filters_1, 1, 16)
            self.convolution_2 = SplineConv(self.args.filters_1, self.args.filters_2, 1, 16)
            self.convolution_3 = SplineConv(self.args.filters_2, self.args.filters_3, 1, 16)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
            )
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
            )
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
            )
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        if self.args.diffpool:
            self.attention = DiffPool(self.args)
        else:
            self.attention = AttentionModule(self.args)

        self.tensor_network = TensorNetworkModule(self.args)
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_count, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def calculate_histogram(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(abstract_features_1, abstract_features_2.permute([0, 2, 1])).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[:num_nodes[i], :num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist / torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)

        return torch.stack(hist_list).view(-1, self.args.bins)

    def convolutional_pass(self, edge_index, features, edge_weight=None):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        if edge_weight is not None:
            if self.args.gnn_operator == 'spline':
                edge_weight = edge_weight.unsqueeze(-1)

        features = self.convolution_1(features, edge_index, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index, edge_weight)
        return features

    def diffpool(self, abstract_features, edge_index, batch):
        """
        Making differentiable pooling.
        :param abstract_features: Node feature matrix.
        :param batch: Batch vector, which assigns each node to a specific example
        :return pooled_features: Graph feature matrix.
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        device = next(self.parameters()).device

        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        if hasattr(data["g1"], 'edge_attr') and hasattr(data["g2"], 'edge_attr'):
            edge_attr_1 = data["g1"].edge_attr
            edge_attr_2 = data["g2"].edge_attr
        else:
            edge_attr_1 = None
            edge_attr_2 = None
        node_1 = data["g1"].x
        node_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        batch_num = data["g1"].num_graphs

        ns_1 = torch.bincount(data["g1"].batch)
        ns_2 = torch.bincount(data["g2"].batch)

        adj_1 = to_dense_adj(edge_index_1, batch=batch_1, edge_attr=edge_attr_1)
        dummy_adj_1 = torch.zeros(adj_1.shape[0], adj_1.shape[1] + 1, adj_1.shape[2] + 1, device=device)
        dummy_adj_1[:, :-1, :-1] = adj_1
        adj_2 = to_dense_adj(edge_index_2, batch=batch_2, edge_attr=edge_attr_2)
        dummy_adj_2 = torch.zeros(adj_2.shape[0], adj_2.shape[1] + 1, adj_2.shape[2] + 1, device=device)
        dummy_adj_2[:, :-1, :-1] = adj_2

        node_1, _ = to_dense_batch(node_1, batch=batch_1)
        node_2, _ = to_dense_batch(node_2, batch=batch_2)

        dummy_node_1 = torch.zeros(adj_1.shape[0], node_1.shape[1] + 1, node_1.shape[-1], device=device)
        dummy_node_1[:, :-1, :] = node_1
        dummy_node_2 = torch.zeros(adj_2.shape[0], node_2.shape[1] + 1, node_2.shape[-1], device=device)
        dummy_node_2[:, :-1, :] = node_2

        k_diag = self.node_metric(dummy_node_1, dummy_node_2)

        mask_1 = torch.zeros_like(dummy_adj_1)
        mask_2 = torch.zeros_like(dummy_adj_2)
        for b in range(batch_num):
            mask_1[b, :ns_1[b] + 1, :ns_1[b] + 1] = 1
            mask_1[b, :ns_1[b], :ns_1[b]] -= torch.eye(ns_1[b], device=mask_1.device)
            mask_2[b, :ns_2[b] + 1, :ns_2[b] + 1] = 1
            mask_2[b, :ns_2[b], :ns_2[b]] -= torch.eye(ns_2[b], device=mask_2.device)

        a1 = dummy_adj_1.reshape(batch_num, -1, 1)
        a2 = dummy_adj_2.reshape(batch_num, 1, -1)
        m1 = mask_1.reshape(batch_num, -1, 1)
        m2 = mask_2.reshape(batch_num, 1, -1)
      
        k = torch.abs(a1 - a2) * torch.bmm(m1, m2)
       
        k[torch.logical_not(torch.bmm(m1, m2).to(dtype=torch.bool))] = VERY_LARGE_INT
        k = k.reshape(batch_num, dummy_adj_1.shape[1], dummy_adj_1.shape[2], dummy_adj_2.shape[1], dummy_adj_2.shape[2])
        k = k.permute([0, 1, 3, 2, 4])
        k = k.reshape(batch_num, dummy_adj_1.shape[1] * dummy_adj_2.shape[1], dummy_adj_1.shape[2] * dummy_adj_2.shape[2])
        k = k / 2

        for b in range(batch_num):
            k_diag_view = torch.diagonal(k[b])
            k_diag_view[:] = k_diag[b].reshape(-1)

        if self.enable_a_star:
            self.reset_cache()
            start = time.process_time()
            x_pred, tree_size = a_star(
                data, k, ns_1.cpu().numpy(), ns_2.cpu().numpy(),
                self.net_prediction_cache,
                self.heuristic_prediction_hun,
                net_pred=self.args.astar_use_net,
                beam_width=self.args.astar_beamwidth,
                trust_fact=self.args.astar_trustfact,
                no_pred_size=self.args.astar_nopred,
            )
            time_spent = time.process_time() - start
           
            # x_pred = self.a_star(data, k, ns_1.numpy(), ns_2.numpy(), beam_width=0, trust_fact=1.)
            ged = self.comp_ged(x_pred, k)

            if self.training:
                scores, sup_scores = [], []
                x_pred = x_pred[0]
                for matched_len in range(1, torch.sum(x_pred).to(torch.long)+1):
                    for matched_pairs in combinations(torch.nonzero(x_pred, as_tuple=False), matched_len):
                        partial_x = x_pred.clone()
                        for r, c in matched_pairs:
                            partial_x[r, c] = 0
                        if partial_x[:-1, :].sum() == ns_1[0] or partial_x[:, :-1].sum() == ns_2[0]:
                            continue
                        score = self.net_prediction_cache(data, partial_pmat=partial_x, return_ged_norm=True)
                        g_p = self.comp_ged(partial_x, k[0])
                        h_p = ged - g_p
                        n1plsn2 = ns_1[0] + ns_2[0] - partial_x[:-1, :].sum() - partial_x[:, :-1].sum()
                        sup_score = torch.exp(- h_p * 2 / n1plsn2)
                        scores.append(score)
                        sup_scores.append(sup_score)
                return torch.cat(scores), torch.cat(sup_scores)
            else:
                norm_ged = ged * 2 / (ns_1 + ns_2).to(device)
                
                return torch.exp(-norm_ged), torch.tensor(tree_size), time_spent
        else:
            return self.net_prediction(data, return_ged_norm=True)

    def node_metric(self, node1, node2):
        if 'AIDS' in self.args.dataset:
            encoding = torch.sum(torch.abs(node1[:, :, :29].unsqueeze(2) - node2[:, :, :29].unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.tensor([0, 1, 1],device=encoding.device)
        elif self.args.dataset in ['Willow']:
            encoding = torch.sum(torch.abs(node1[:, :, :].unsqueeze(2) - node2[:, :, :].unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.tensor([0, VERY_LARGE_INT, 0],device=encoding.device)
        else:
            encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.tensor([0, 1, 0],device=encoding.device)
        return mapping[encoding]

    def net_prediction(self, data, batch_idx=None, partial_pmat=None, cur_idx=None, return_ged_norm=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        start000 = time.process_time()
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        if hasattr(data["g1"], 'edge_attr') and hasattr(data["g2"], 'edge_attr'):
            edge_attr_1 = data["g1"].edge_attr
            edge_attr_2 = data["g2"].edge_attr
        else:
            edge_attr_1 = None
            edge_attr_2 = None
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        if self.enable_a_star:
            assert partial_pmat is not None
            assert torch.all(batch_1 == 0)
            assert torch.all(batch_2 == 0)

            start = time.process_time()
            graph_1_mask = torch.ones_like(batch_1)
            graph_2_mask = torch.ones_like(batch_2)
            if PRINT_TIMING: print('create graph_mask', time.process_time() - start)

            start = time.process_time()
            graph_1_matched = partial_pmat.sum(dim=-1).to(dtype=torch.bool)[:graph_1_mask.shape[0]]
            graph_2_matched = partial_pmat.sum(dim=-2).to(dtype=torch.bool)[:graph_2_mask.shape[0]]
            if PRINT_TIMING: print('graph_matched', time.process_time() - start)

            start = time.process_time()
            graph_1_mask = torch.logical_not(graph_1_matched)
            graph_2_mask = torch.logical_not(graph_2_matched)
            if PRINT_TIMING: print('graph_mask', time.process_time() - start)

            start = time.process_time()
            edge_index_1, edge_attr_1 = subgraph(graph_1_mask, edge_index_1, edge_attr_1, relabel_nodes=True)
            edge_index_2, edge_attr_2 = subgraph(graph_2_mask, edge_index_2, edge_attr_2, relabel_nodes=True)
            if PRINT_TIMING: print('subgraph', time.process_time() - start)

            start = time.process_time()
            features_1 = features_1[graph_1_mask]
            features_2 = features_2[graph_2_mask]

            batch_1 = batch_1[graph_1_mask]
            batch_2 = batch_2[graph_2_mask]
            if PRINT_TIMING: print('features[mask], batch[mask]', time.process_time() - start)

        start = time.process_time()
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1, edge_attr_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2, edge_attr_2)
        if PRINT_TIMING: print('convolutional pass', time.process_time() - start)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)

        if self.args.diffpool:
            pooled_features_1 = self.diffpool(abstract_features_1, edge_index_1, batch_1)
            pooled_features_2 = self.diffpool(abstract_features_2, edge_index_2, batch_2)
        else:
            start = time.process_time()
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)
            if PRINT_TIMING: print('attention', time.process_time() - start)

        start = time.process_time()
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        if PRINT_TIMING: print('tensor network', time.process_time() - start)

        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1)

        start = time.process_time()
        score = self.scoring_layer(scores).view(-1)
        if PRINT_TIMING: print('scoring layer', time.process_time() - start)

        if PRINT_TIMING: print('total time', time.process_time() - start000)
        if PRINT_TIMING: print('-' * 10)

        if return_ged_norm:
            return score
        else:
            ged = - torch.log(score) * (batch_1.shape[0] + batch_2.shape[0]) / 2
            return ged


    def net_prediction_cache(self, data, g_p=None, partial_pmat=None, cur_idx=None, return_ged_norm=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        start000 = time.process_time()
        start = time.process_time()
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        if hasattr(data["g1"], 'edge_attr') and hasattr(data["g2"], 'edge_attr'):
            edge_attr_1 = data["g1"].edge_attr
            edge_attr_2 = data["g2"].edge_attr
        else:
            edge_attr_1 = None
            edge_attr_2 = None
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        if PRINT_TIMING: print('prepare', time.process_time() - start)

        start = time.process_time()
        assert self.enable_a_star
        if 'gnn_feat' not in self.gnn_1_cache:
            abstract_features_1 = self.convolutional_pass(edge_index_1, features_1, edge_attr_1)
            self.gnn_1_cache['gnn_feat'] = abstract_features_1
        else:
            abstract_features_1 = self.gnn_1_cache['gnn_feat']
        if 'gnn_feat' not in self.gnn_2_cache:
            abstract_features_2 = self.convolutional_pass(edge_index_2, features_2, edge_attr_2)
            self.gnn_2_cache['gnn_feat'] = abstract_features_2
        else:
            abstract_features_2 = self.gnn_2_cache['gnn_feat']
        if PRINT_TIMING: print('convolutional pass', time.process_time() - start)

        start = time.process_time()
        graph_1_mask = torch.ones_like(batch_1)
        graph_2_mask = torch.ones_like(batch_2)
        if PRINT_TIMING: print('create graph_mask', time.process_time() - start)

        start = time.process_time()
        graph_1_matched = partial_pmat.sum(dim=-1).to(dtype=torch.bool)[:graph_1_mask.shape[0]]
        graph_2_matched = partial_pmat.sum(dim=-2).to(dtype=torch.bool)[:graph_2_mask.shape[0]]
        if PRINT_TIMING: print('graph_matched', time.process_time() - start)

        start = time.process_time()
        graph_1_mask = torch.logical_not(graph_1_matched)
        graph_2_mask = torch.logical_not(graph_2_matched)
        if PRINT_TIMING: print('graph_mask', time.process_time() - start)

        start = time.process_time()
        abstract_features_1 = abstract_features_1[graph_1_mask]
        abstract_features_2 = abstract_features_2[graph_2_mask]

        batch_1 = batch_1[graph_1_mask]
        batch_2 = batch_2[graph_2_mask]
        if PRINT_TIMING: print('features[mask], batch[mask]', time.process_time() - start)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)

        if self.args.diffpool:
            pooled_features_1 = self.diffpool(abstract_features_1, edge_index_1, batch_1)
            pooled_features_2 = self.diffpool(abstract_features_2, edge_index_2, batch_2)
        else:
            start = time.process_time()
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)
            if PRINT_TIMING: print('attention', time.process_time() - start)

        start = time.process_time()
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        if PRINT_TIMING: print('tesnsor network', time.process_time() - start)

        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1)

        start = time.process_time()
        score = self.scoring_layer(scores).view(-1)
        if PRINT_TIMING: print('scoring layer', time.process_time() - start)

        if PRINT_TIMING: print('size', batch_1.shape[0], batch_2.shape[0])

        if PRINT_TIMING: print('total time', time.process_time() - start000)
        if PRINT_TIMING: print('-' * 10)

        if return_ged_norm:
            return score
        else:
            ged = - torch.log(score) * (batch_1.shape[0] + batch_2.shape[0]) / 2
            return ged

    def heuristic_prediction_hun(self, k, n1, n2, partial_pmat):
        start = time.process_time()
        if 'node_cost' in self.heuristic_cache:
            node_cost_mat = self.heuristic_cache['node_cost']
        else:
            k_prime = k.reshape(-1, n1+1, n2+1)
            node_costs = torch.empty(k_prime.shape[0])
            for i in range(k_prime.shape[0]):
                _, node_costs[i] = hungarian_ged(k_prime[i], n1, n2)
            node_cost_mat = node_costs.reshape(n1+1, n2+1)
            self.heuristic_cache['node_cost'] = node_cost_mat

        graph_1_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
        graph_2_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
        graph_1_mask[-1] = 1
        graph_2_mask[-1] = 1
        node_cost_mat = node_cost_mat.to(graph_1_mask.device)
        node_cost_mat = node_cost_mat[graph_1_mask, :]
        node_cost_mat = node_cost_mat[:, graph_2_mask]

        pred_x, ged = hungarian_ged(node_cost_mat, torch.sum(graph_1_mask[:-1]), torch.sum(graph_2_mask[:-1]))
        if PRINT_TIMING: print('size',  torch.sum(graph_1_mask[:-1]), torch.sum(graph_2_mask[:-1]))
        if PRINT_TIMING: print('hung', time.process_time() - start)
        if PRINT_TIMING: print('-' * 10)
        return ged

    def a_star(self, data, k, ns_1, ns_2, beam_width=0, trust_fact=1.):
        batch_num = k.shape[0]
        x_size = torch.Size([ns_1.max()+1, ns_2.max()+1])
        open_set = [[{
                        'x_idx': torch.LongTensor(size=(0, 2)).to(device=k.device),
                        'g_p': 0.,
                        'h_p': float('inf'),
                        'g+h': float('inf'),
                        'idx': 0
                    }] for b in range(batch_num)]
        ret_x = torch.zeros(batch_num, ns_1.max()+1, ns_2.max()+1, device=k.device)
        stop_flags = torch.zeros(batch_num, dtype=torch.bool)
        while not torch.all(stop_flags):
            for b in range(batch_num):
                if stop_flags[b] == 1:
                    continue

                selected = open_set[b].pop(0)
                if selected['idx'] == ns_1[b]:
                    stop_flags[b] = 1
                    indices = selected['x_idx']
                    v = torch.ones(indices.shape[0], device=k.device)
                    x = torch.sparse.FloatTensor(indices.t(), v, x_size).to_dense()
                    ret_x[b] = x
                    continue

                cur_set = []
                for n2 in range(ns_2[b] + 1):
                    if n2 in selected['x_idx'][:, 1] and n2 != ns_2[b]:
                        continue
                    if selected['idx'] + 1 == ns_1[b]:
                        x_idx = [selected['x_idx'], torch.LongTensor([[selected['idx'], n2]]).to(device=k.device)]
                        for _n2 in range(ns_2[b]):
                            if _n2 not in selected['x_idx'][:, 1] and _n2 != n2:
                                x_idx.append(torch.LongTensor([[ns_1[b], _n2]]).to(device=k.device))
                        indices = torch.cat(x_idx, dim=0)
                    else:
                        indices = torch.cat((selected['x_idx'], torch.LongTensor([[selected['idx'], n2]]).to(device=k.device)), dim=0)
                    v = torch.ones(indices.shape[0], device=k.device)
                    x = torch.sparse.FloatTensor(indices.t(), v, x_size)
                    x_dense = x.to_dense()

                    g_p = self.comp_ged(x_dense, k[b])

                    if selected['idx'] + 1 == ns_1[b]:
                        h_p = 0
                    else:
                        h_p = self.net_prediction(data, b, x_dense)

                    cur_set.append(
                        {
                            'x_idx': indices,
                            'g_p': g_p,
                            'h_p': h_p * trust_fact,
                            'g+h': g_p + h_p * trust_fact,
                            'idx': selected['idx'] + 1
                        }
                    )
                if beam_width > 0:
                    cur_set.sort(key=lambda ele: (ele['g+h'], -ele['idx']))
                    cur_set = cur_set[:min(beam_width, len(cur_set))]
                open_set[b] += cur_set

                open_set[b].sort(key=lambda ele: (ele['g+h'], -ele['idx']))

        return ret_x

    @staticmethod
    def comp_ged(_x, _k):
        if len(_x.shape) == 3 and len(_k.shape) == 3:
            _batch = _x.shape[0]
            return torch.bmm(torch.bmm(_x.reshape(_batch, 1, -1), _k), _x.reshape(_batch, -1, 1)).view(_batch)
        elif len(_x.shape) == 2 and len(_k.shape) == 2:
            return torch.mm(torch.mm(_x.reshape( 1, -1), _k), _x.reshape( -1, 1)).view(1)
        else:
            raise ValueError('Input dimensions not supported.')

    @property
    def device(self):
        return next(self.parameters()).device