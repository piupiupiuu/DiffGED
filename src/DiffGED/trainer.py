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
from gedgnn_kbest import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau

from models import DiffMatch
from loss_fn import mapping_loss
from diffusion_schedulers import CategoricalDiffusion,InferenceSchedule
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
        self.model = DiffMatch(self.args, self.number_of_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)
    
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
    
    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()

        self.model.train()
    
        with tqdm(total= len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",file=sys.stdout) as pbar:
            
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
        
        batch_size = (torch.max(batch.batch) + 1).item()
        gt_mapping_idx,gt_mapping_label = batch.edge_index_mapping,batch.edge_attr_mapping
        # sample random time steps t
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        # one-hot encoding of ground-truth matching matrix
        gt_mapping_onehot = torch.nn.functional.one_hot(gt_mapping_label.long(), num_classes=2).float()
        mapping_batch = batch.batch[gt_mapping_idx[0]]
        # sample noisy matching matrix
        diffused_mapping = self.diffusion.sample(gt_mapping_onehot, t,mapping_batch)
        t = torch.from_numpy(t).float()
        # predict matching matrix
        pred_mapping_label = self.model(batch,diffused_mapping.to(self.device),t.to(self.device))
        losses = mapping_loss(pred_mapping_label,batch)
       
        losses.backward()
        self.optimizer.step()
        return losses.item()
        
    def diffusion_ged_parallel(self,batch,test_k=100):
        # generate k node matching matrices
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
        new_batch = Batch().from_data_list([data for i in range(num_parallel_sampling)])
        gt_mapping_label = new_batch.edge_attr_mapping
        # sample random node matching matrix
        mapping_t = torch.randn_like(gt_mapping_label,device=self.device)
        mapping_t = (mapping_t>0).long()

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        # diffusion
        for s in range(steps):
            t1,t2 = time_schedule(s)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(new_batch,mapping_t,t1,t2)

        n1 = batch.n[0,0].item()
        n2 = batch.n[0,1].item()

        pred_matching_matrix = torch.zeros((num_parallel_sampling,n1,n2),device=self.device)
        mapping_edge_idx = new_batch.edge_index_mapping
        graph_edge_idx = new_batch.edge_index
        batch_mapping_edge_idx = mapping_edge_idx - new_batch.batch[mapping_edge_idx[0]] * (n1+n2)
        batch_mapping_edge_idx[1] -= n1

        pred_matching_matrix[new_batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]] = mapping_t.squeeze(-1)
        batch_idx = torch.arange(num_parallel_sampling,device=pred_matching_matrix.device)

        # extract node mappings
        greedy_mask = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
        solution = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)

        for s in range(min(n1,n2)):
            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling,-1)
            argmax_result = torch.argmax(pred_matching_matrix,dim=-1)
            rows = argmax_result // n2
            columns = argmax_result % n2
            
            solution[batch_idx,rows,columns] = True
            greedy_mask[batch_idx,rows,:] = True
            greedy_mask[batch_idx,:,columns] = True

            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling,n1,n2)
            pred_matching_matrix[greedy_mask] = float('-inf')
        
        zeros_column = torch.where(~torch.any(solution == 1, dim=1))
        # if |V| < |V'|, add nodes to V with empty labels, and map each to an unmatched nodes in V'
        solution = torch.cat([solution,torch.zeros(num_parallel_sampling,n2-n1,n2,device=solution.device)],dim=1)
        solution[zeros_column[0],torch.arange(n1,n2,device=solution.device).repeat(num_parallel_sampling),zeros_column[1]] = 1
        extracted_mapping = torch.nonzero(solution)

        x1 = new_batch.x[(new_batch.x_indicator==0).squeeze(1)]
        x2 = new_batch.x[(new_batch.x_indicator==1).squeeze(1)]
        dense_x1 = x1.view(num_parallel_sampling,n1,-1)
        dense_x2 = x2.view(num_parallel_sampling,n2,-1)

        x1 = new_batch.x[(new_batch.x_indicator==0).squeeze(1)]
        x2 = new_batch.x[(new_batch.x_indicator==1).squeeze(1)]
        dense_x1 = x1.view(num_parallel_sampling,n1,-1)
        dense_x2 = x2.view(num_parallel_sampling,n2,-1)

        # permute G' according to the extracted mapping
        permuted_x2 = (dense_x2[extracted_mapping[:,0],extracted_mapping[:,2]]).view(num_parallel_sampling,n2,-1)
        dense_x1 = torch.cat([dense_x1,torch.zeros(num_parallel_sampling,n2-n1,dense_x1.shape[-1],device=dense_x1.device)],dim=1)
        edge1 = new_batch.edge_index[:,(new_batch.x_indicator[new_batch.edge_index[0]]==0).squeeze(1)]
        edge1 = remove_self_loops(edge1)[0]
        edge1_batch = new_batch.batch[edge1[0]]
        edge1 = edge1 - edge1_batch * n1 
        dense_adj_1 = to_dense_adj(edge_index=edge1,batch=new_batch.batch[(new_batch.x_indicator==1).squeeze(1)],max_num_nodes=n2)
        reversed_mapping = torch.tensor(sorted(extracted_mapping.tolist(),key=lambda x:(x[0],x[2])),device=extracted_mapping.device)
        edge2 = new_batch.edge_index[:,(new_batch.x_indicator[new_batch.edge_index[0]]==1).squeeze(1)]
        edge2_batch = new_batch.batch[edge2[0]]
        edge2 = edge2 - (edge2_batch+1) * n1
        reversed_mapping[:,2] += reversed_mapping[:,0] * n2
        reversed_mapping[:,1] += reversed_mapping[:,0] * n2
        edge2[0] = reversed_mapping[edge2[0],1]
        edge2[1] = reversed_mapping[edge2[1],1]
        dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0],batch=new_batch.batch[(new_batch.x_indicator==1).squeeze(1)],max_num_nodes=n2)

        # ged = difference in G and G' (after permuting G')
        adj_diff = torch.abs(dense_adj_1-dense_adj_2).view(num_parallel_sampling,-1).sum(dim=-1) // 2
        feat_diff = torch.sum(~torch.all(dense_x1 == permuted_x2,dim=-1),dim=-1)
        ged = adj_diff + feat_diff

        min_ged_idx = torch.argmin(ged)
        min_ged = ged[min_ged_idx].item()
        end_time = time.time()
        min_mapping = solution[min_ged == ged,:n1]
        return min_ged,min_mapping,end_time-start_time
    
    def diffusion_ged_sequential(self,batch,test_k=100,k_range=None,return_mapping=False):
        start_time = time.time()
        gt_mapping_idx,gt_mapping_label = batch.edge_index_mapping,batch.edge_attr_mapping
        mapping_t = torch.randn_like(gt_mapping_label,device=self.device)
        mapping_t = (mapping_t>0).long()
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        # diffusion
        for s in range(steps):
            t1,t2 = time_schedule(s)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(batch,mapping_t,t1,t2)
        
        mapping_t = pyg.utils.softmax(mapping_t.squeeze(-1), gt_mapping_idx[0]).unsqueeze(-1)
        mapping_t = (mapping_t * 1e9 + 1).round()

        n1 = batch.n[0,0].item()
        n2 = batch.n[0,1].item()
        x1 = batch.x[:n1]
        x2 = batch.x[n1:]

        x1_edge = batch.edge_index[:,batch.edge_index[0]<n1]
        x2_edge = batch.edge_index[:,batch.edge_index[0]>=n1] - n1

        g1 = dgl.graph((x1_edge[0],x1_edge[1]),num_nodes=n1)
        g2 = dgl.graph((x2_edge[0],x2_edge[1]),num_nodes=n2)
        g1.ndata['f'] = x1
        g2.ndata['f'] = x2
        pred_matching_matrix = torch.zeros((n1,n2),device=self.device)
        pred_matching_matrix[batch.edge_index_mapping[0],batch.edge_index_mapping[1]-n1] = mapping_t.squeeze(-1)
        
        # GEDGNN topk
        solver = KBestMSolver(pred_matching_matrix, g1, g2)
        if k_range == None:
            solver.get_matching(test_k)
            min_ged = solver.min_ged
            min_mappings = []
            end_time = time.time()
            if return_mapping:
                for sp in solver.subspaces:
                    if sp.ged == solver.min_ged:
                        min_mappings.append(sp.best_matching)
                    elif sp.ged2 == solver.min_ged:
                        min_mappings.append(sp.second_matching)
            return min_ged,min_mappings,end_time-start_time
        
        else:
            k_running_time = {}
            pre_geds = {}
            for k in k_range:
                solver.get_matching(k)
                min_ged = solver.min_ged
                end_time = time.time()
                k_running_time[k] = end_time-start_time
                pre_geds[k] = min_ged
            return pre_geds,None,k_running_time



    def categorical_denoise_step(self,data,mapping_t,t1,t2):      
        batch_size = torch.max(data.batch) + 1
        t1 = torch.from_numpy(t1).repeat(batch_size)

        # predict node matching matrix
        with torch.no_grad():
            pred_mapping_label = self.model(data,mapping_t,t1.float().to(self.device))
        prob_mapping = torch.nn.functional.sigmoid(pred_mapping_label)

        # compute posterior
        prob_mapping = torch.cat([1-prob_mapping,prob_mapping],dim=-1)
        mapping_t = self.categorical_posterior(t2,t1,prob_mapping,mapping_t,data.batch[data.edge_index_mapping[0]])
       
        return mapping_t
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt, mapping_batch):
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
        target_t = target_t.repeat(t.shape[0])

        Q_t = (np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t])
        Q_t = Q_t.reshape(t.shape[0],2,2)
        
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device).reshape(t.shape[0],2,2)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device).reshape(t.shape[0],2,2)

        x0_pred_prob = x0_pred_prob.unsqueeze(1)
        xt = F.one_hot(xt.long(), num_classes=2).float()
        
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t[mapping_batch].permute((0,2, 1)).contiguous())
        
        x_t_target_prob_part_2 = Q_bar_t_target[:,0]
        
        x_t_target_prob_part_3 = (Q_bar_t_source[:,0][mapping_batch].unsqueeze(1) * xt).sum(dim=-1,keepdim=True)
        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3
        
        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        x_t_target_prob_part_2_new = Q_bar_t_target[:,1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[:,1][mapping_batch].unsqueeze(1) * xt).sum(dim=-1,keepdim=True)
        x_t_target_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3_new
        sum_x_t_target_prob += x_t_target_prob_new[..., 1] * x0_pred_prob[..., 1]
        
        if target_t[0] > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
        return xt
    
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
    
    def score(self,testing_graph_set='test', test_k=100, top_k_approach='parallel'):
        assert test_k > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
    
        print("\n\nEvalute DiffGED with {} topk {} on {} set.\n".format(top_k_approach,test_k,testing_graph_set))
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
            if top_k_approach == 'parallel':
                model_out = self.diffusion_ged_parallel(batch,test_k)
            else:
                model_out = self.diffusion_ged_sequential(batch,test_k)
            
            pre_ged,running_time = model_out[0],model_out[2]
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

        self.results.append(('model_name', 'topk_approach' 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'mae', 'acc',
                            'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

        with open(self.args.abs_path + self.args.result_path + f'result_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}_{test_k}.json','w') as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
    
    def analyze_topk(self,testing_graph_set = 'test',k_range=[1,10,20,30,40,50,60,70,80,90,100],top_k_approach='parallel'):
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nAnalyze {} topk approach on {} set.\n".format(top_k_approach,testing_graph_set))
        self.model.eval()

        num = 0  # total testing number
        time_usage = {i:0 for i in k_range}
        
          # ged mae
        mae = {i:[] for i in k_range}
        num_acc = {i:0 for i in k_range}  
        acc = {}
       
        if top_k_approach == 'sequential':
            for batch in tqdm(loader,file=sys.stdout):
                batch.to(self.device)
                gt_ged = batch.ged
                gt = gt_ged.item()
                model_out = self.diffusion_ged_sequential(batch,k_range=k_range)
                pre_ged,running_time = model_out[0],model_out[2]
                num +=1
                for k in k_range:
                    mae[k].append(abs(pre_ged[k]-gt))
                    time_usage[k] += running_time[k]
                    if pre_ged[k] == gt:
                        num_acc[k] += 1
                  
            for k in k_range:
                time_usage[k] = round(time_usage[k]/num,5)
                mae[k] = round(np.mean(mae[k]),3)
                acc[k] = round(num_acc[k]/num,3)
                print(f'dataset: {self.args.dataset}, topk_approach: {top_k_approach}, k: {k}, avg time: {time_usage[k]}, mae: {mae[k]}, acc: {acc[k]}')
        
        else:
            for k in k_range:
                num = 0
                for batch in tqdm(loader,file=sys.stdout):
                    batch.to(self.device)
                    gt_ged = batch.ged
                    gt = gt_ged.item()
                    model_out = self.diffusion_ged_parallel(batch,k)
                    pre_ged,running_time = model_out[0],model_out[2]
                    num += 1
                    time_usage[k] += running_time
                    mae[k].append(abs(pre_ged-gt))
                    if pre_ged == gt:
                        num_acc[k] += 1
                   
                time_usage[k] = round(time_usage[k]/num,5)
                mae[k] = round(np.mean(mae[k]),3)
                acc[k] = round(num_acc[k]/num,3)
              
                print(f'dataset: {self.args.dataset}, topk_approach: {top_k_approach}, k: {k}, avg time: {time_usage[k]}, mae: {mae[k]}, acc: {acc[k]}')
            
        with open(self.args.abs_path + self.args.result_path + f'topk_analysis_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}.json','w') as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc},f)
    
    def analyze_solution_diversity(self,testing_graph_set = 'test',top_k_approach='parallel',test_k=100):
        assert test_k > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nAnalyze Solution Diversity with {} topk {} on {} set.\n".format(top_k_approach,test_k,testing_graph_set))
        self.model.eval()

        diversity_pred_ged = []
        diversity_gt_ged = []
        
        for batch in tqdm(loader,file=sys.stdout):
            batch.to(self.device)
            gt_ged = batch.ged
            if top_k_approach == 'parallel':
                model_out = self.diffusion_ged_parallel(batch,test_k)
                pre_ged, pre_mappings = model_out[0], model_out[1]
                num_distinct = torch.unique(torch.argmax(pre_mappings,dim=-1),dim=0).shape[0]
               
            else:
                model_out = self.diffusion_ged_sequential(batch,test_k,return_mapping=True)
                pre_ged, pre_mappings = model_out[0], model_out[1]
                num_distinct = torch.unique(torch.tensor(pre_mappings),dim=0).shape[0]
                
            diversity_pred_ged.append(num_distinct)
            diversity_gt_ged.append(num_distinct * int(int(pre_ged) == gt_ged.item()))
       
        print(f'dataset: {self.args.dataset}, topk_approach: {top_k_approach}, k: {test_k}, diversity_pred_ged: {np.mean(diversity_pred_ged)}, diversity_gt_ged: {np.mean(diversity_gt_ged)}')

        with open(self.args.abs_path + self.args.result_path + f'diversity_analysis_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}_{test_k}.json','w') as f:
            json.dump({'diversity_pred_ged':np.mean(diversity_pred_ged),' diversity_gt_ged':np.mean(diversity_gt_ged)},f)





           
