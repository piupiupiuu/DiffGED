from lapjv import lapjv
from scipy.optimize import linear_sum_assignment
import sys
import time
from typing import List

import dgl
import torch

import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt

from math import exp
from scipy.stats import spearmanr, kendalltau

from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse,to_undirected,sort_edge_index,coalesce,to_dense_adj,remove_self_loops,to_dense_batch,group_argsort,to_networkx
import torch_geometric as pyg
from torch_geometric.nn.pool import global_add_pool,global_mean_pool
import networkx as nx
import operator
import json

class Trainer(object):
    """
    A general model trainer.
    """


    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []

        self.use_gpu = torch.cuda.is_available()
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        # generate synthetic graphs for large graphs (if any)
        self.gen_delta_graphs()
        self.init_graph_pairs()

        self.testing_data_loader = DataLoader(self.testing_graphs,batch_size=1,shuffle=False)
        self.testing_data_small_loader = DataLoader(self.testing_graphs_small,batch_size=1,shuffle=False)
        self.testing_data_large_loader = DataLoader(self.testing_graphs_large,batch_size=1,shuffle=False)
    
    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        
        ged_dict = dict()
        load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
        t2 = time.time()
        self.load_data_time = t2 - t1
    
    def transfer_data_to_torch(self):
        t1 = time.time()

        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)
        
        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        # number of nodes
        self.gn = [g['n'] for g in self.graphs]
        # number of edges
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    gt_mapping = gt_mappings[0]
                    for x, y in enumerate(gt_mapping):
                        mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float()
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        
        self.ged = ged
        self.mapping = mapping
        
        t2 = time.time()
        self.to_torch_time = t2 - t1
    
    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense()
       
        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long()

        feature2 = torch.zeros(f.shape)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data
    
    def gen_delta_graphs(self):
        random.seed(0)
        k = self.args.num_delta_graphs
        for i, g in enumerate(self.graphs):
            # Do not generate delta graphs for small graphs.
            if g['n'] <= 10:
                continue
            # gen k delta graphs
            f = self.features[i]
            self.delta_graphs[i] = [self.delta_graph(g, f, self.device) for j in range(k)]
    
    def pack_graph_pair(self,pair):
        new_data = Data()
        (pair_type, id_1, id_2) = pair
        if pair_type == 0:
            new_data.i_j = torch.tensor([[id_1,id_2]])
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])

            real_ged = self.ged[id_1][id_2][0]

            n1 = self.gn[id_1]
            n2 = self.gn[id_2]

            new_data.n = torch.tensor([[n1,n2]])
            new_data.x = torch.cat([self.features[id_1],self.features[id_2]],dim=0)
            new_data.edge_index = torch.cat([self.edge_index[id_1],self.edge_index[id_2]+n1],dim=1)
            # (G,G'): If G, then x_indicator=0. If G', x_indicator=1
            new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

            # transfer mapping to edge index between G and G'
            mapping = self.mapping[id_1][id_2] + 0.1
            mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
            mapping_edge_index[1] += n1
            new_data.edge_index_mapping = mapping_edge_index
            new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)

            new_data.ged = real_ged
        
        else:
            # synthetic graph
            new_data.i_j = torch.tensor([[id_1,id_2]])
            dg: dict = self.delta_graphs[id_1][id_2]
            real_ged = dg["ta_ged"][0]
            n1 = self.gn[id_1]
            n2 = dg["n"]
            new_data.n = torch.tensor([[n1,n2]])
            new_data.x = torch.cat([self.features[id_1],dg["features"]],dim=0)
            new_data.edge_index = torch.cat([self.edge_index[id_1],dg["edge_index"]+n1],dim=1)
            new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

            mapping =  dg["mapping"] + 0.1
            mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
            mapping_edge_index[1] += n1
            new_data.edge_index_mapping = mapping_edge_index
            new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)
            new_data.ged = real_ged
        return new_data
    
    def check_pair(self, i, j):
        if i == j:
            return (0, i, j)
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return (0, i, j)
        elif (id2, id1) in self.ged_dict:
            return (0, j, i)
        else:
            return None
    
    def init_graph_pairs(self):
        start = time.time()
        random.seed(1)
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing_graphs_small = []
        self.testing_graphs_large = []

        dg = self.delta_graphs

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        # each training graph is paired with all other training graphs or 100 synthetic graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        pair = self.pack_graph_pair(tmp)
                        self.training_graphs.append(pair)
                    
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    pair = self.pack_graph_pair((1, i, j))
                    self.training_graphs.append(pair)
        
        # each val / testing graph is pair with 100 training graphs or synthetic graphs
        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)
        
        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j))
                    self.val_graphs.append(pair)
               
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j))
                    self.val_graphs.append(pair)
                   
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j))
                    self.testing_graphs.append(pair)
                    self.testing_graphs_small.append(pair)
                    
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j))
                    self.testing_graphs.append(pair)
                    self.testing_graphs_large.append(pair)

        end = time.time()
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} val graph pairs.".format(len(self.val_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generate {} small testing graph pairs.".format(len(self.testing_graphs_small)))
        print("Generate {} large testing graph pairs.".format(len(self.testing_graphs_large)))
        print("Generation time:",end-start)

    def to_nx(self,data):
        x1 = torch.argmax(data.x[(data.x_indicator==0).squeeze(1)],dim=-1)
        x2 = torch.argmax(data.x[(data.x_indicator==1).squeeze(1)],dim=-1)
        edge1 = data.edge_index[:,(data.x_indicator[data.edge_index[0]]==0).squeeze(1)]
        edge1 = remove_self_loops(edge1)[0]
        edge2 = data.edge_index[:,(data.x_indicator[data.edge_index[0]]==1).squeeze(1)] - data.n[0,0].item()
        edge2 = remove_self_loops(edge2)[0]
        g1 = to_networkx(Data(x=x1,edge_index=edge1),to_undirected=True,node_attrs='x')
        g2 = to_networkx(Data(x=x2,edge_index=edge2),to_undirected=True,node_attrs='x')
        
        return g1,g2

    def cost_matrix_construction(self,G1, G2, dname:str):
        INF = G1.number_of_nodes() + G1.number_of_edges() + G2.number_of_nodes() + G2.number_of_edges() + 1
        ns1 = G1.number_of_nodes()
        ns2 = G2.number_of_nodes()
        cost_matrix = np.zeros((ns1 + ns2, ns1 + ns2), dtype=float)
        if dname == 'AIDS':
            node_label = {i: G1.nodes[i]['x'] for i in G1.nodes}
            node_label = sorted(node_label.items(), key=operator.itemgetter(0))
            g1_labels = np.array([k[1] for k in node_label])    
            node_label = {i: G2.nodes[i]['x'] for i in G2.nodes}
            node_label = sorted(node_label.items(), key=operator.itemgetter(0))
            g2_labels = np.array([k[1] for k in node_label]) 
            g1_labels = np.expand_dims(g1_labels, axis=1)
            g2_labels = np.expand_dims(g2_labels, axis=0)
            label_substitution_cost = np.abs(g1_labels - g2_labels)
            label_substitution_cost[np.nonzero(label_substitution_cost)] = 1
            cost_matrix[0:ns1, 0:ns2] = label_substitution_cost

        cost_matrix[0:ns1, ns2:ns1+ns2] = np.array([1 if i == j else INF for i in range(ns1) for j in range(ns1) ]).reshape(ns1, ns1)
        cost_matrix[ns1:ns1+ns2, 0:ns2] = np.array([1 if i == j else INF for i in range(ns2) for j in range(ns2) ]).reshape(ns2, ns2)


        # do not consider node and edge labels, i.e., the cost of edge Eui equals to the degree difference
        g1_degree = np.array([G1.degree(n) for n in range(ns1)], dtype=int)
        g2_degree = np.array([G2.degree(n) for n in range(ns2)], dtype=int)
        g1_degree = np.expand_dims(g1_degree, axis=1)
        g2_degree = np.expand_dims(g2_degree, axis=0)
        degree_substitution_cost = np.abs(g1_degree - g2_degree)
        cost_matrix[0:ns1, 0:ns2] += degree_substitution_cost
        return cost_matrix

    def bipartite_for_cost_matrix(self,G1, G2, cost_matrix, alg_type:str, dname:str):
        if G1.number_of_nodes() == G2.number_of_nodes():
            cost_matrix = cost_matrix[0:G1.number_of_nodes(), 0:G1.number_of_nodes()]
        mapping_str = ""
        can_used_for_AStar = True
        if alg_type == 'hungarian':
            row, col = linear_sum_assignment(cost_matrix)
        elif alg_type == 'vj':
            row, col, _ = lapjv(cost_matrix)
        node_match = {}
        cost = 0
        common = 0
        for i, n in enumerate(row):
            if n < G1.number_of_nodes():
                if col[i] < G2.number_of_nodes():
                    node_match[n] = col[i]
                    if G1.nodes[n]['x'] != G2.nodes[col[i]]['x'] and dname in ['AIDS']:
                        cost += 1
                    mapping_str += "{}|{} ".format(n, col[i])
                else:                      
                    node_match[n] = None
                    cost +=1
                    can_used_for_AStar = False

        for n in G2.nodes:
            if n not in node_match.values(): cost += 1

        for edge in G1.edges():
            (p, q) = (node_match[edge[0]], node_match[edge[1]])
            if (p, q) in G2.edges():
                common += 1
        cost = cost + G1.number_of_edges() + G2.number_of_edges() - 2 * common
        # generate mapping string
        return cost, can_used_for_AStar, mapping_str

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def score(self,testing_graph_set='test', algo='hungarian'):
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nEvalute traditional {} on {} set.\n".format(algo,testing_graph_set))
        num = 0  # total testing number
        time_usage = 0
        
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        pres = {}
        gts = {}

        for batch in tqdm(loader,file=sys.stdout):
            g1,g2 = self.to_nx(batch)
            start_time = time.time()
            cost_matrix = self.cost_matrix_construction(g1,g2,self.args.dataset)
            pred_ged,valid,mapping = self.bipartite_for_cost_matrix(g1,g2,cost_matrix,algo,self.args.dataset)
            end_time = time.time()
            num += 1
            gt_ged = batch.ged.item()
            i_j = batch.i_j
            i = i_j[0][0].item()
            if i in pres:
                pres[i].append(pred_ged)
                gts[i].append(gt_ged)
            else:
                pres[i]=[pred_ged]
                gts[i]=[gt_ged]
            mae.append(abs(pred_ged-gt_ged))
            if pred_ged == gt_ged:
                num_acc +=1
                num_fea +=1
            elif pred_ged > gt_ged:
                num_fea +=1
            time_usage += (end_time-start_time)
        
        for i in pres:
            rho.append(spearmanr(pres[i],gts[i])[0])
            tau.append(kendalltau(pres[i],gts[i])[0])
            pk10.append(self.cal_pk(10, pres[i],gts[i]))
            pk20.append(self.cal_pk(20, pres[i],gts[i]))


        time_usage = round(time_usage / num, 5)
        mae = round(np.mean(mae), 3)
        
        acc = round(num_acc / num, 3)
        
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)' , 'mae', 'acc',
                            'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

        with open(self.args.abs_path + self.args.result_path + f'result_traditional_{algo}_{self.args.dataset}_{testing_graph_set}.json','w') as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
