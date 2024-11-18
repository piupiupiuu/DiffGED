import sys
import time
from typing import List

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from math import exp
from scipy.stats import spearmanr, kendalltau

from models import GPN
from noah import graph_edit_distance
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
        self.setup_model()
        # generate synthetic graphs for large graphs (if any)
        self.gen_delta_graphs()
        self.init_graph_pairs()

        self.training_data_loader = DataLoader(self.training_graphs,batch_size=self.args.batch_size,shuffle=True)  
        self.testing_data_loader = DataLoader(self.testing_graphs,batch_size=1,shuffle=False)
        self.testing_data_small_loader = DataLoader(self.testing_graphs_small,batch_size=1,shuffle=False)
        self.testing_data_large_loader = DataLoader(self.testing_graphs_large,batch_size=1,shuffle=False)
    
    def setup_model(self):
        self.model = GPN(self.args, self.number_of_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
    
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

            n1,m1 = self.gn[id_1],self.gm[id_1]
            n2,m2 = self.gn[id_2],self.gm[id_2]

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

            if self.args.target_mode == "exp":
                avg_v = (n1 + n2) / 2.0
                new_data.avg_v = avg_v
                new_data.target = torch.exp(torch.tensor([-real_ged / avg_v]).float())
            elif self.args.target_mode == "linear":
                higher_bound = max(n1, n2) + max(m1, m2)
                new_data.hb = higher_bound
                new_data.target = torch.tensor([real_ged / higher_bound]).float()
        
        else:
            # synthetic graph
            new_data.i_j = torch.tensor([[id_1,id_2]])
            dg: dict = self.delta_graphs[id_1][id_2]
            real_ged = dg["ta_ged"][0]
            n1,m1 = self.gn[id_1],self.gm[id_1]
            n2,m2 = dg["n"],dg["m"]
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

            if self.args.target_mode == "exp":
                avg_v = (n1 + n2) / 2.0
                new_data.avg_v = avg_v
                new_data.target = torch.exp(torch.tensor([-real_ged / avg_v]).float())

            elif self.args.target_mode == "linear":
                higher_bound = max(n1, n2) + max(m1, m2)
                new_data.hb = higher_bound
                new_data.target = torch.tensor([real_ged / higher_bound]).float()
               
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
    
    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()

        self.model.train()
    
        with tqdm(total=len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",file=sys.stdout) as pbar:
            
            loss_sum = 0
            main_index = 0
            index = 0
            for batch in self.training_data_loader:
                batch.to(self.device)
                batch_total_loss = self.process_batch(batch)
                
                loss_sum += batch_total_loss
                main_index += (torch.max(batch.batch)+1).item()
                loss = loss_sum / main_index
                pbar.update(len(batch))
                pbar.set_description(
                    "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),
                                                                    index,
                                                                    round(1000 * batch_total_loss / len(batch), 3)))
                index += 1
            tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
            training_loss = round(1000 * loss, 3)
        t2 = time.time()
        training_time = t2 - t1

        self.results.append(
            ('model_name', 'dataset', 'graph_set', "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
    
    def process_batch(self,batch):
        self.optimizer.zero_grad()
        
        target = batch.target

        prediction, pred_ged = self.model(batch)
        losses = F.mse_loss(target.unsqueeze(-1),prediction,reduction='sum')
        
        losses.backward()
        self.optimizer.step()
        return losses.item()
    
    def data_to_nx(self,edges, features):
        edges = edges.t().tolist()

        nx_g = nx.Graph()
        n, num_label = features.shape

        if num_label == 1:
            labels = [-1 for i in range(n)]
        else:
            labels = [-1] * n
            for i in range(n):
                for j in range(num_label):
                    if features[i][j] > 0.5:
                        labels[i] = j
                        break

        for i, label in enumerate(labels):
            nx_g.add_node(i, label=label)

        for u, v in edges:
            if u < v:
                nx_g.add_edge(u, v)
        return nx_g

    def run_noah(self,batch,beam_size):
        start = time.time()

        n1 = batch.n[0,0].item()
        n2 = batch.n[0,1].item()
        x1 = batch.x[:n1]
        x2 = batch.x[n1:]

        x1_edge = batch.edge_index[:,batch.edge_index[0]<n1]
        x2_edge = batch.edge_index[:,batch.edge_index[0]>=n1] - n1

        g1 = self.data_to_nx(x1_edge, x1)
        g2 = self.data_to_nx(x2_edge, x2)

        lower_bound = 'Noah'
        min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(self.model, g1, g2, lower_bound,beam_size)
        end = time.time()
        '''
        # compute mapping
        permute = [-1] * n1
        used = [False] * n2
        for u, v in min_path1:
            if u is not None and v is not None:
                assert 0 <= u < n1 and 0 <= v < n2 and not used[v]
                permute[u] = v
                used[v] = True
        for u in range(n1):
            if permute[u] == -1:
                for v in range(n2):
                    if not used[v]:
                        permute[u] = v
                        used[v] = True
                        break
        '''
        
        return cost1,end-start
    
    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch) + '_' + self.args.model_name +'.pt')

    def load(self, epoch):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch) + '_' + self.args.model_name +'.pt'))
    
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
    
    def score(self,testing_graph_set='test', beam_size=100):
        assert beam_size > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nEvalute Noah with beamsize {} on {} set.\n".format(beam_size,testing_graph_set))
        self.model.eval()
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
            batch.to(self.device)
            gt_ged = batch.ged
            gt = gt_ged.item()
            model_out = self.run_noah(batch,beam_size)
            pre_ged,running_time = model_out[0],model_out[1]
            num += 1
            time_usage += running_time
  
            i_j = batch.i_j
            i = i_j[0][0].item()
            
            if i in pres:
                pres[i].append(pre_ged)
                gts[i].append(gt)
            else:
                pres[i] = [pre_ged]
                gts[i] = [gt]
            mae.append(abs(pre_ged-gt))
            if pre_ged== gt:
                num_acc += 1
                num_fea += 1
            elif pre_ged> gt:
                num_fea += 1
            
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

        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'mae', 'acc',
                            'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

        with open(self.args.abs_path + self.args.result_path + f'result_Noah_{self.args.dataset}_{testing_graph_set}_{beam_size}.json','w') as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
        