import glob
import os.path
import copy

import torch_geometric.nn
from tqdm import tqdm, trange
import torch.nn
from torch_scatter import scatter
import time
import pickle
from torch_geometric.nn import GCNConv, GINConv, SplineConv, GraphConv
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import Batch,Data
import torch_geometric.transforms as T
import random
import networkx as nx
from myloss_func import *
from randomWalk import AddRandomWalkPE
from mydegree import MyDegree
from myConstant import MyConstanct
from utils import load_all_graphs, load_labels, load_ged
from torch_geometric.utils import to_networkx
import sys
from models import Mata
import json
class Trainer(object):

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
    
        self.results = []
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.gen_delta_graphs()
        self.init_graph_pairs()
        
        self.init_astar()
        self.setup_model()
    
    def setup_model(self):
        self.model = Mata(self.args, self.num_labels, self.app_astar).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    
    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)
        # print(self.global_labels)

        ged_dict = dict()
        # We could load ged info from several files.
        # load_ged(ged_dict, self.args.abs_path, dataset_name, 'xxx.json')
        load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
        # print(self.ged['2050']['30'])
        t2 = time.time()
        self.load_data_time = t2 - t1
    
    def transfer_data_to_torch(self):
        """
        Transfer loaded data to torch.
        """
        t1 = time.time()

        self.edge_index = []
        # self.A = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            #edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)
            # A = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (g['n'], g['n'])).to_dense().to(self.device)
            # self.A.append(A)
        
        if 'AIDS' in self.args.dataset:
            self.features = [torch.tensor(x).float() for x in self.features]

        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]

        self.real_graphs = []

        n = len(self.graphs)
        for g in range(n):
            if 'AIDS' in self.args.dataset:
                curr_g = Data(x=self.features[g],edge_index=self.edge_index[g])
                self.real_graphs.append([curr_g,self.nx2txt(to_networkx(Data(x=torch.argmax(self.features[g],dim=-1),edge_index =self.edge_index[g]),node_attrs='x',to_undirected=True),str(g),self.args.dataset)])
            else:
                curr_g = Data(edge_index=self.edge_index[g],num_nodes=self.gn[g])
                self.real_graphs.append([curr_g,self.nx2txt(to_networkx(curr_g,to_undirected=True),str(g),self.args.dataset)])
        
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        
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
        new_data = Data()

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
        new_data.n = n
        new_data.m = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        #new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long()

        if f is not None:
            feature2 = torch.zeros(f.shape)
            for x, y in enumerate(permute):
                feature2[y] = f[x]
            new_data.x = feature2
        else:
            new_data.num_nodes = n
        new_data.permute = permute
        new_data.mapping = mapping
        ged = del_num + add_num
        new_data.ta_ged = (ged, 0, 0, ged)
        new_data.edge_index = new_edge
        
        return new_data

    def gen_delta_graphs(self):
        random.seed(0)
        k = self.args.num_delta_graphs
        
        for i, g in enumerate(self.graphs):
            # Do not generate delta graphs for small graphs.
            if g['n'] <= 10:
                continue
            
            # gen k delta graphs
            if 'AIDS' in self.args.dataset:
                f = self.features[i]
            else:
                f = None
            self.delta_graphs[i] = []
            for j in range(k):
                dg = self.delta_graph(g, f, self.device)
                if 'AIDS' in self.args.dataset:
                    nx_dg = self.nx2txt(to_networkx(Data(x=torch.argmax(dg.x,dim=-1),edge_index =dg.edge_index),node_attrs='x',to_undirected=True),str(j),self.args.dataset)
                else:
                    nx_dg = self.nx2txt(to_networkx(Data(edge_index=dg.edge_index,num_nodes = dg.num_nodes),to_undirected=True),str(j),self.args.dataset)
                self.delta_graphs[i].append([dg,nx_dg])
        

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
    
    def nx2txt(self,G: networkx.Graph, id:str, alg:str):  #
        if alg in ['AIDS']:
            line = "t " + "# " + id + "\n"
            for id, label in G.nodes(data=True):
                line += "v " + str(id) + " " + str(label['x']) + "\n"
            for (u, v) in G.edges():
                line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
            return line
        elif alg in ['IMDB','Linux']:
            line = "t " + "# " + id + "\n"
            for id, label in G.nodes(data=True):
                line += "v " + str(id) + " " + str(1) + "\n"
            for (u, v) in G.edges():
                line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
            return line
        else:
            return ""


    def pack_graph_pair(self,pair,transform):
        
        (pair_type, id_1, id_2) = pair
        if pair_type == 0:
            gid_pair = (self.gid[id_1], self.gid[id_2])
            
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])
            n1, m1 = (self.gn[id_1], self.gm[id_1])
            n2, m2 = (self.gn[id_2], self.gm[id_2]) 
            
            g1= self.real_graphs[id_1][0]
            nx_g1_str = self.real_graphs[id_1][1]
            g2 = self.real_graphs[id_2][0]
            nx_g2_str = self.real_graphs[id_2][1]
            real_ged = self.ged[id_1][id_2][0]
            avg_v = (n1 + n2) / 2.0
            norm_ged = torch.exp(torch.tensor([-real_ged / avg_v]).float())

            if gid_pair not in self.ged_dict and id_1==id_2:
                map_order = list(range(self.gn[id_1]))
            else:
                map_order = self.ged_dict[gid_pair][1][0]

            return (g1,g2,nx_g1_str,nx_g2_str,real_ged,norm_ged,map_order,pair[1],pair[2])

        else:
            dg: dict = self.delta_graphs[id_1][id_2]
            real_ged = dg[0].ta_ged[0]
            n1, m1 = (self.gn[id_1], self.gm[id_1])
            n2, m2 = (dg[0].n, dg[0].m)

            g1 = self.real_graphs[id_1][0]
            nx_g1_str = self.real_graphs[id_1][1]
            g2 = Data(x = dg[0].x,edge_index=dg[0].edge_index,cent_pe = dg[0].cent_pe,rw_pe=dg[0].rw_pe,num_nodes = dg[0].num_nodes)
            nx_g2_str = dg[1]

            
            avg_v = (n1 + n2) / 2.0
            norm_ged = torch.exp(torch.tensor([-real_ged / avg_v]).float())
            map_order = dg[0].permute
            return (g1,g2,nx_g1_str,nx_g2_str,real_ged,norm_ged,map_order,pair[1],pair[2])

    def init_graph_pairs(self):
        print("start augmentation")
        # add features
        if self.args.nonstruc:
            transform = MyConstanct(1.0,cat='AIDS' in self.args.dataset)
        else:
            const = MyConstanct(1.0,cat='AIDS' in self.args.dataset)
            max_degree = self.args.max_degree
            myDeg = MyDegree(max_degree)
            randwalk = AddRandomWalkPE(self.args.random_walk_step)
            transform = T.Compose([const, myDeg, randwalk])
        for g in range(len(self.real_graphs)):
            self.real_graphs[g][0] = transform(self.real_graphs[g][0])
        for dgs in range(len(self.delta_graphs)):
            
            if self.delta_graphs[dgs] is not None:
                
                for g in range(len(self.delta_graphs[dgs])):
                    self.delta_graphs[dgs][g][0] = transform(self.delta_graphs[dgs][g][0])
        print("finish augmentation")

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

        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        pair = self.pack_graph_pair(tmp,transform)
                        self.training_graphs.append(pair)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    pair = self.pack_graph_pair((1, i, j),transform)
                    self.training_graphs.append(pair)
        
        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)
        
        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j),transform)
                    self.val_graphs.append(pair)
            
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j),transform)
                    self.val_graphs.append(pair)
                   
        
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j),transform)
                    
                    self.testing_graphs.append(pair)
                    self.testing_graphs_small.append(pair)
                    
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j),transform)
                    self.testing_graphs.append(pair)
                    self.testing_graphs_large.append(pair)
        
        end = time.time()
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} val graph pairs.".format(len(self.val_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generate {} small testing graph pairs.".format(len(self.testing_graphs_small)))
        print("Generate {} large testing graph pairs.".format(len(self.testing_graphs_large)))
        print("Generation time:",end-start)
        self.num_labels = self.training_graphs[0][0].num_features + self.args.max_degree + self.args.random_walk_step
        print("Feature shape of 1st graph:", self.num_labels)
    
    def init_astar(self):
        so_path = os.path.join('mata/Astar', 'mata.so')
        app_astar = ctypes.cdll.LoadLibrary(so_path)  # app_astar: approximate astar
        app_astar.ged.restype = ctypes.c_char_p
        app_astar.mapping_ed.restype = ctypes.c_int
        self.app_astar = app_astar
    
    def create_batches(self):
        """
        Creating batches from the training graph list.
        """
        
        new_data = {"g1":[],"g2":[],"map":[],"nor_ged":[]}
        pair = random.sample(self.training_graphs,self.args.batch_size)
        list_map = []
        max_n1 = 0
        max_n2 = 0
        for p in pair:
            new_data["g1"].append(p[0])
            new_data["g2"].append(p[1])
            new_data["nor_ged"].append(p[5])
            if p[0].x.shape[0] > max_n1:
                max_n1 = p[0].x.shape[0]
            if p[1].x.shape[0] > max_n2:
                max_n2 = p[1].x.shape[0]
        for p in pair:
            map_order = p[-3]
            map_idx = torch.zeros((max_n1, max_n2), dtype=int)
            row = np.arange(0, len(map_order), dtype=int)
            col = np.array(map_order, dtype=int)
            map_idx[[row, col]] = 1
            list_map.append(map_idx)
        batch_map = torch.stack(list_map, dim=0).to(self.model.device)
        new_data["map"] = batch_map
        new_data["g1"] = Batch.from_data_list(new_data["g1"]).to(self.model.device)
        new_data["g2"] = Batch.from_data_list(new_data["g2"]).to(self.model.device)
        new_data["nor_ged"] = torch.cat(new_data["nor_ged"]).to(self.model.device)
        return new_data
    
    def process_batches(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor(0.0, requires_grad=True)
        ged_score, simmat1, simmat2 = self.model(batch)
        if self.args.loss_type == 1:
            criterion = BCELoss()
        elif self.args.loss_type == 2:
            criterion = MultiMatchingLoss()
        elif self.args.loss_type == 3:
            criterion = GEDLoss(self.app_astar, self.nx_graphs, batch)
        else:
            print("Unknown loss type")
        rows = torch.bincount(batch["g1"].batch).to(self.model.device)
        cols = torch.bincount(batch["g2"].batch).to(self.model.device)

        if self.args.tasks == 1:
            losses = losses + F.mse_loss(ged_score, batch["nor_ged"], reduction="mean")
        elif self.args.tasks == 2:
            losses = losses + criterion(simmat2, batch["map"], rows, cols)
        else:
            losses = losses + 50 * F.mse_loss(ged_score, batch["nor_ged"], reduction="mean")
            losses = losses + criterion(simmat1, batch["map"], rows, cols)
            losses = losses + criterion(simmat2, batch["map"], rows, cols)

        losses.backward(retain_graph=True)
        self.optimizer.step()
        return losses.item()
    
    def fit(self):
        print('\nmodel training \n')
        self.model.train()
        epochs = trange(self.args.epochs, ascii=True, leave=True, desc="Epoch", position=0)
        best_sim_score = float("inf")           # best_sim_score 
        for epoch in epochs:
            batches = self.create_batches()
            loss_score = self.process_batches(batches)
            loss = loss_score
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
        
            if epoch > 0 and epoch % (self.args.val_epochs) == 0:
                cur_sim_score = self.score(testing_graph_set='val')     
                if cur_sim_score < best_sim_score:
                    print("update the model. The average similarity score of epoch({}):{}".format(epoch, cur_sim_score))
                    self.save()
                    best_sim_score = cur_sim_score
    
    def do_astar(self, sim_mat, sim_mat2, new_data):
        if len(sim_mat.shape) == 2:
            sim_mat = sim_mat.unsqueeze(0)
        if len(sim_mat2.shape) == 2:
            sim_mat2 = sim_mat2.unsqueeze(0)
        batch_num = sim_mat.shape[0]
        ged_prediction, search_space = [], []
        for b in range(batch_num):
            nx_g1 = new_data["nx_g1"][b]
            nx_g2 = new_data["nx_g2"][b]
            
            n1 = new_data['g1'][b].num_nodes
            n2 = new_data['g2'][b].num_nodes
            e1 = new_data['g1'][b].num_edges
            e2 = new_data['g2'][b].num_edges
            beam_size = get_beam_size(n1, n2, e1/2, e2/2, self.args.dataset)
            topk = min(self.args.topk, n1, n2)
            if topk == 0: topk = 1
            matching_nodes, matching_order = self.find_topk_hun(sim_mat[b, :n1, :n2].detach(), sim_mat2[b, :n1, :n2].detach(),  topk, n1, n2)
            
            matching_order[0], matching_order[1] = 0, 0
            astar_out = self.app_astar.ged(CT(nx_g1), CT(nx_g2), int1ArrayToPointer(matching_order),
                                           int1ArrayToPointer(matching_order),int2ArrayToPointer(matching_nodes), CT(2 * topk), CT(beam_size))
            astar_out = astar_out.decode('ascii').split()
            
            pred = normalize_ged(n1, n2, int(astar_out[0]))
            ged_prediction.append(pred)
            search_space.append(int(astar_out[1]))
        return ged_prediction, search_space
 

    def find_topk_hun(self, sim_matrix, sim_matrix2, topk, n1 = None, n2 = None):
        if n1 is None and n2 is None:
            n1 = sim_matrix.shape[0]
            n2 = sim_matrix.shape[1]
        matching_nodes, matching_order = [], [n for n in range( n1 )]
        mink = min(n2, topk)

        col_inds,col_inds2 = [], []
        for i in range(mink):
            row_ind, col_ind = linear_sum_assignment(cost_matrix=abs(sim_matrix[:, :]), maximize=True)
            sim_matrix[row_ind, col_ind] = 0
            col_inds.append(col_ind)

        for i in range(mink):
            row_ind2, col_ind2 = linear_sum_assignment(cost_matrix=abs(sim_matrix2[:, :]), maximize=True)
            sim_matrix2[row_ind2, col_ind2] = 0
            col_inds2.append(col_ind2)
        
        for i in range(len(row_ind)):
            t = []
            for j in range(len(col_inds)):
                t.append(col_inds[j][i])
                t.append(col_inds2[j][i])
            matching_nodes.append(t)
        
        return np.array(matching_nodes, dtype=int), np.array(matching_order, dtype=int)
    
    def save(self):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name +'.pt')

    def load(self):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + self.args.model_name +'.pt'))
    
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
    
    def score(self, testing_graph_set='test'):
        self.model.eval()
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        
        ret_score = []
        prediction_mat = np.full(len(testing_graphs),1e-10)
        num = 0  # total testing number
        time_usage = 0
        mae = []
        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk10 = []
        pk20 = []
        pres = {}
        gts = {}

        for i,pair in enumerate(tqdm(testing_graphs,file=sys.stdout)):
            data = {"g1":[],"g2":[],"nor_ged":[],"nx_g1":[],"nx_g2":[]}
            data["g1"].append(pair[0])
            data["g2"].append(pair[1])
            data["nor_ged"].append(pair[5])
            data["nx_g1"].append(pair[2])
            data["nx_g2"].append(pair[3])
            data["g1"] = Batch.from_data_list(data["g1"]).to(self.model.device)
            data["g2"] = Batch.from_data_list(data["g2"]).to(self.model.device)
            data["nor_ged"] = torch.cat(data["nor_ged"]).to(self.model.device)
            target = data["nor_ged"].cpu()

            start = time.time()
            pred_ged, simmat1, simmat2  = self.model(data)
            astar_prediction, search_space = self.do_astar(simmat1.detach().cpu(), simmat2.detach().cpu(), data)
            astar_prediction = astar_prediction[0]
            
            end = time.time()
            
            num+=1
            id_1,id_2 = pair[-2],pair[-1]
            gt_ged = pair[4]
            pre_ged = (-torch.log(astar_prediction) * 0.5 * (pair[0].x.shape[0] + pair[1].x.shape[0]))
            rounded_ged = torch.round(pre_ged).item()
            pre_ged = pre_ged.item()
            
            if id_1 in pres:
                pres[id_1].append(pre_ged)
                gts[id_1].append(gt_ged)
            else:
                pres[id_1] = [pre_ged]
                gts[id_1] = [gt_ged]
            mae.append(abs(rounded_ged-gt_ged))
            if rounded_ged == gt_ged:
                num_acc += 1
                num_fea += 1
            elif rounded_ged > gt_ged:
                num_fea += 1
            

            time_usage += (end-start)
        
        
        for id1 in pres:
            ret_score.append(np.mean(pres[id1]))
            rho.append(spearmanr(pres[id1],gts[id1])[0])
            tau.append(kendalltau(pres[id1],gts[id1])[0])
            pk10.append(self.cal_pk(10, pres[id1],gts[id1]))
            pk20.append(self.cal_pk(20, pres[id1],gts[id1]))
        time_usage = round(time_usage / num, 5)
        
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        #fea = round(num_fea / num, 3)
        fea = num_fea / num
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(('model_name', 'dataset', 'graph_set','#testing_pairs', 'time_usage(s/100p)', 'mae', 'acc',
                            'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        self.model.train()
        if testing_graph_set != 'val':
            with open(self.args.abs_path + self.args.result_path + f'result_MATA*_{self.args.dataset}_{testing_graph_set}_{self.args.topk}.json','w') as f:
                json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
        return np.mean(np.array(ret_score))
    

        