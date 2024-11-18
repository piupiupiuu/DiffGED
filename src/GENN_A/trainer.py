import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau
from itertools import combinations

from models import GENN

from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree, dense_to_sparse, subgraph
from torch_geometric.transforms import OneHotDegree, Constant
from utils import load_all_graphs, load_labels, load_ged
import random
import sys
import json
class Trainer(object):
    """
    GENN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.results = []
        self.device = 'cuda:0'
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.gen_delta_graphs()
        self.init_graph_pairs()
        self.setup_model()
    
    def setup_model(self):
        self.model = GENN(self.args, self.number_of_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    
    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        
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
            

        n = len(self.graphs)
        
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
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
        #new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long()

        if f is not None:
            feature2 = torch.zeros(f.shape)
            for x, y in enumerate(permute):
                feature2[y] = f[x]
            new_data["features"] = feature2
        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        
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
            self.delta_graphs[i] = [self.delta_graph(g, f, self.device) for j in range(k)]
    
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

    def pack_graph_pair(self,pair,transform):
        g1 = Data()
        g2 = Data()
        (pair_type, id_1, id_2) = pair
        if pair_type == 0:
            gid_pair = (self.gid[id_1], self.gid[id_2])
            
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])
            n1, m1 = (self.gn[id_1], self.gm[id_1])
            n2, m2 = (self.gn[id_2], self.gm[id_2]) 
            if 'AIDS' in self.args.dataset:    
                g1.x = self.features[id_1]
                g2.x = self.features[id_2]
            else:
                g1.num_nodes = n1
                g2.num_nodes = n2
            g1.edge_index = self.edge_index[id_1]
            g2.edge_index = self.edge_index[id_2]
           
            g1 = transform(g1)
            g2 = transform(g2)
            
            
            real_ged = self.ged[id_1][id_2][0]
            avg_v = (n1 + n2) / 2.0
            norm_ged = torch.exp(torch.tensor([-real_ged / avg_v]).float())
            return (g1,g2,real_ged,norm_ged,pair[1],pair[2])

        else:
            dg: dict = self.delta_graphs[id_1][id_2]
            real_ged = dg["ta_ged"][0]
            n1, m1 = (self.gn[id_1], self.gm[id_1])
            n2, m2 = (dg["n"], dg["m"])
            if 'AIDS' in self.args.dataset:
                g1.x = self.features[id_1]
                g2.x = dg["features"]
            else:
                g1.num_nodes = n1
                g2.num_nodes = n2
            g1.edge_index = self.edge_index[id_1]
            g2.edge_index = dg["edge_index"]
            g1 = transform(g1)
            g2 = transform(g2)
           
            avg_v = (n1 + n2) / 2.0
            norm_ged = torch.exp(torch.tensor([-real_ged / avg_v]).float())
            return (g1,g2,real_ged,norm_ged,pair[1],pair[2])

    def init_graph_pairs(self):
        # add features
        max_degree = 0
        for g in self.edge_index:
            if g.size(1)>0:
                max_degree = max(max_degree, int(degree(g[0]).max().item()))
        for dgs in self.delta_graphs:
            if dgs is not None:
                for g in dgs:
                    if g["edge_index"].size(1)>0:
                        max_degree = max(max_degree, int(degree(g["edge_index"][0]).max().item()))
        if 'AIDS' in self.args.dataset:
            assert max_degree <= 6
            max_degree = 6
        if 'AIDS' in self.args.dataset:
            transform = OneHotDegree(max_degree, cat=True)
        else:
            transform = OneHotDegree(max_degree, cat=False)
        

        start = time.time()
        random.seed(1)
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing_graphs_small = []
        self.testing_graphs_large = []
        self.testing2_graphs = []

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
        self.number_of_labels = self.training_graphs[0][0].num_features
        print("Feature shape of 1st graph:", self.number_of_labels)
    
    def create_batches(self):
        """
        Creating batches from the training graph list.
        """

        new_data = {"g1":[],"g2":[],"target":[]}
        pair = random.sample(self.training_graphs,self.args.batch_size)
        for p in pair:
            new_data["g1"].append(p[0])
            new_data["g2"].append(p[1])
            new_data["target"].append(p[3])
        new_data["g1"] = Batch.from_data_list(new_data["g1"]).to(self.model.device)
        new_data["g2"] = Batch.from_data_list(new_data["g2"]).to(self.model.device)
        new_data["target"] = torch.cat(new_data["target"]).to(self.model.device)
        return new_data
    
    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        self.optimizer.zero_grad()
        #data = self.transform(data)
        if not self.model.enable_a_star:
            target = data["target"]
            prediction = self.model(data)
            loss = F.mse_loss(prediction, target, reduction='sum')
        else:
            prediction, target = self.model(data)
            loss = F.mse_loss(prediction, target, reduction='mean')
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.model.train()

        epochs = trange(self.args.epochs, ascii=True, leave=True, desc="Epoch", position=0)
        best_mse = float('inf')
        for epoch in epochs:
            batch_pair = self.create_batches()
            main_index = 0
            loss_sum = 0
           
     
            loss_score = self.process_batch(batch_pair)
          
            main_index = main_index + batch_pair['g1'].num_graphs
            loss_sum = loss_sum + loss_score

            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
           
            if epoch > 0 and (epoch + 1) % 1000 == 0 and not self.model.enable_a_star:
                valid_mse = self.score(testing_graph_set='val')
                if best_mse > valid_mse:
                    self.save(self.model.enable_a_star)
                    best_mse = valid_mse

        if self.model.enable_a_star:
            self.save(self.model.enable_a_star)
    
    def save(self,enable_a_star):
        if enable_a_star:
            torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name + '_' + 'astar.pt')
        else:
            torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name + '.pt')
    
    def load(self,enable_astar,astar_use_net,eval):
        if eval:
            if enable_astar:
                if astar_use_net:
                    self.model.load_state_dict(torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name + '_' + 'astar.pt'))
            else:
                self.model.load_state_dict(torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name + '.pt'))

        else:
            if enable_astar:
                self.model.load_state_dict(torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_'  + self.args.model_name + '.pt'))



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
        
        scores = np.zeros(len(testing_graphs))
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
            data = {"g1":[],"g2":[],"target":[]}
            data["g1"].append(pair[0])
            data["g2"].append(pair[1])
            data["target"].append(pair[3])
            data["g1"] = Batch.from_data_list(data["g1"]).to(self.model.device)
            data["g2"] = Batch.from_data_list(data["g2"]).to(self.model.device)
            data["target"] = torch.cat(data["target"]).to(self.model.device)
            target = data["target"].cpu()
            
            start = time.time()
            model_return = self.model(data)
            end = time.time()
            if self.model.enable_a_star:
                prediction, tree_size, time_spent = model_return
                prediction = prediction.detach().cpu()
                
            else:
                prediction = model_return.detach().cpu()
           
            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()
            if not testing_graph_set == 'val':
                num+=1
                id_1,id_2 = pair[-2],pair[-1]
                gt_ged = pair[2]
                pre_ged = (-torch.log(prediction) * 0.5 * (pair[0].x.shape[0] + pair[1].x.shape[0]))
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
            

        if testing_graph_set == 'val':
            model_error = np.mean(scores)
            self.model.train()
            return model_error
        else:
            for id1 in pres:
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

            self.results.append(( 'model_name','dataset', 'graph_set','#testing_pairs', 'enable_astar', 'time_usage(s/100p)', 'mae', 'acc',
                                'fea', 'rho', 'tau', 'pk10', 'pk20'))
            self.results.append(( self.args.model_name, self.args.dataset, testing_graph_set, num, self.args.enable_astar, time_usage, mae, acc,
                                fea, rho, tau, pk10, pk20))

            print(*self.results[-2], sep='\t')
            print(*self.results[-1], sep='\t')
            if self.args.enable_astar:
                save_path = f'result_GENNA*_{self.args.dataset}_{testing_graph_set}_enable_astar.json'
            else:
                save_path = f'result_GENNA*_{self.args.dataset}_{testing_graph_set}.json'
            with open(self.args.abs_path + self.args.result_path + save_path,'w') as f:
                json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
    
                    