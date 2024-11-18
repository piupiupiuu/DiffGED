import torch.nn
import torch_geometric.nn
from torch_geometric.nn import GCNConv, GINConv, SplineConv, GraphConv
from layers import AttentionModule, NeuralTensorNetwork, Affinity,  Sinkhorn, NorSim, soft_topk, greedy_perm
import torch.nn.functional as F
SK_ITER_NUM = 6
SK_EPSILON = 1.0e-4
SK_TAU = 1.0
class Mata(torch.nn.Module):
    def __init__(self, args, num_of_labels, app_astar):
        super(Mata, self).__init__()

        self.app_astar = app_astar
        self.args = args
        self.num_labels = num_of_labels
        self.setup_layers()

    # the number of bottleneck features
    def calculate_bottleneck_features(self):
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        if self.args.gnn_operator == 'gc':
            self.convolution_1 = GraphConv(self.args.filter_1, self.args.filter_1)
            self.convolution_1 = GraphConv(self.args.filter_1, self.args.filter_2)
            self.convolution_1 = GraphConv(self.args.filter_2, self.args.filter_3)

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.args.filter_1, self.args.filter_1)        #  self.num_labels
            self.convolution_2 = GCNConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = GCNConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'spline':
            self.convolution_1 = SplineConv(self.args.filter_1, self.args.filter_1)    # #  self.num_labels
            self.convolution_2 = SplineConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = SplineConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_1, self.args.filter_1),
                torch.nn.ReLU(),
            )
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_1, self.args.filter_2),
                torch.nn.ReLU(),
            )
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_2, self.args.filter_3),
                torch.nn.ReLU(),
            )
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        # self.bi_conv_1 = GraphConv(in_channels=(self.args.filter_3, self.args.filter_3), out_channels=self.args.filter_3, aggr="mean", bias=True)
        # self.bi_conv_2 = GraphConv(in_channels=(self.args.filter_3, self.args.filter_3), out_channels=self.args.filter_3, aggr="mean", bias=True)

        self.bi_conv_1 = GraphConv(self.args.filter_3, self.args.filter_3)
        self.bi_conv_2 = GraphConv(self.args.filter_3, self.args.filter_3*2)

        self.init_layer = torch.nn.Sequential(
            torch.nn.Linear(self.num_labels, self.args.filter_1),
            torch.nn.ReLU()
        )

        self.degree_emb = torch.nn.Parameter(torch.Tensor(self.args.max_degree, self.args.max_degree))
        torch.nn.init.xavier_uniform_(self.degree_emb)
        self.attention = AttentionModule(self.args.filter_3)
        self.attention2 = AttentionModule(self.args.filter_3*2)

        # self.tensor_network = NeuralTensorNetwork(self.args)
        self.affinity = Affinity(self.args.filter_3)
        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, epsilon=SK_EPSILON, tau=SK_TAU)
        self.nor_sim = NorSim()
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(8 * self.args.filter_3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1),
            torch.nn.Sigmoid()
        )


    def calculate_similarity(self, abstract_feature_1, abstract_feature_2):
        sim_matrix = torch.mm(abstract_feature_1, abstract_feature_2)
        sim_matrix = torch.softmax(sim_matrix, dim=-1)
        return sim_matrix


    def convolutional_pass(self, edge_index, features, edge_weight=None):
        """
        Making convolutional pass
        :param edge_index: Edge indices
        :param features: Feature matrix
        :return: Abstract feature matrix
        """
        features_1 = self.convolution_1(features, edge_index)
        features = F.relu(features_1)
        features = F.dropout(features, p = self.args.dropout, training=self.training)
        features_2 = self.convolution_2(features, edge_index)
        features = F.relu(features_2)
        features = F.dropout(features, p = self.args.dropout, training=self.training)
        features_3 = self.convolution_3(features, edge_index)
        return features_1, features_2, features_3

    def bi_conv_pass(self, edge_index, features_in, features_out, edge_weight=None):
        edge_weight = None
        # edge_index_reverse = torch.stack((edge_index[1], edge_index[0]),dim=0)
        features =  torch.cat((features_in, features_out), dim=0)
        features = self.bi_conv_1(features, edge_index, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p = self.args.dropout, training=self.training)
        features = self.bi_conv_2(features, edge_index, edge_weight)
        return features


    def forward(self, data):
        g1, g2 = data["g1"], data["g2"]
        batch_1, batch_2 = g1.batch, g2.batch
        device = next(self.parameters()).device
        edge_index_1 = g1.edge_index.to(device)
        edge_index_2 = g2.edge_index.to(device)

        if self.args.nonstruc:
            feature_1 = g1.x
            feature_2 = g2.x
        else:
            feature_1 = torch.cat([g1.x, self.degree_emb[g1.cent_pe.squeeze(1)], g1.rw_pe], dim=1).to(device)
            feature_2 = torch.cat([g2.x, self.degree_emb[g2.cent_pe.squeeze(1)], g2.rw_pe], dim=1).to(device)

        feature_1 = self.init_layer(feature_1)
        feature_2 = self.init_layer(feature_2)


        g1_af_1, g1_af_2, g1_af_3 = self.convolutional_pass(edge_index_1, feature_1) 
        g2_af_1, g2_af_2, g2_af_3 = self.convolutional_pass(edge_index_2, feature_2)


        rows = torch.bincount(g1.batch).to(device)
        cols = torch.bincount(g2.batch).to(device)
        sim_mat1 = self.affinity(feature_1, feature_2, batch_1, batch_2)
        gt_ks = torch.full([sim_mat1.shape[0]], self.args.topk / 2, device=device, dtype=torch.float)
        _, sim_mat1 = soft_topk(sim_mat1, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)

        sim_mat2 = self.affinity(g1_af_3, g2_af_3, batch_1, batch_2)
        _, sim_mat2 = soft_topk(sim_mat2, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)
 

        global_feature1 = torch.cat((feature_1, g1_af_1), dim=1)
        global_feature1 = torch.cat((global_feature1, g1_af_2), dim=1)
        global_feature1 = torch.cat((global_feature1, g1_af_3), dim=1)
        global_feature1 = torch_geometric.nn.global_add_pool(global_feature1, batch_1)

        global_feature2 = torch.cat((feature_2, g2_af_1), dim=1)
        global_feature2 = torch.cat((global_feature2, g2_af_2), dim=1)
        global_feature2 = torch.cat((global_feature2, g2_af_3), dim=1)
        global_feature2 = torch_geometric.nn.global_max_pool(global_feature2, batch_2)

        scores = torch.cat((global_feature1, global_feature2), dim=1)
        ged_score = self.scoring_layer(scores).view(-1)

        return ged_score, sim_mat1, sim_mat2
    
    @property
    def device(self):
        return next(self.parameters()).device
