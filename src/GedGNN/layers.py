import torch
import torch.nn
from torch_geometric.nn.pool import global_add_pool,global_mean_pool

class GedMatrixModule(torch.nn.Module):
    def __init__(self, d, k):
        """
        :param args: Arguments object.
        """
        super(GedMatrixModule, self).__init__()

        self.d = d
        self.k = k
        self.init_weight_matrix()
        self.init_mlp()

    def init_weight_matrix(self):
        """
        Define and initilize a weight matrix of size (k, d, d).
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.k, self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def init_mlp(self):
        k = self.k
        layers = []

        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        # layers.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similar matrix.
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2
        matrix = torch.bmm(torch.matmul(embedding_1, self.weight_matrix).permute(1,0,2),embedding_2.unsqueeze(-1)).squeeze(-1)
        matrix = self.mlp(matrix)

        return matrix

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.hidden_dim[-1],
                                                             self.args.hidden_dim[-1]))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding,batch):
        """
        Making a forward propagation pass to create a graph level representation.
        """
        global_context = global_mean_pool(torch.matmul(embedding, self.weight_matrix),batch)
        transformed_global = torch.tanh(global_context)
        sigmod_scores = torch.sigmoid((embedding * transformed_global[batch]).sum(dim=-1))
        representation = sigmod_scores.unsqueeze(-1) * embedding
       
        representation = global_add_pool(representation,batch)
        
        return representation

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args, input_dim=None):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = self.args.hidden_dim[-1] if (input_dim is None) else input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim,
                                                             self.input_dim,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(batch_size, self.input_dim, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.input_dim, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.relu(scoring + block_scoring + self.bias.view(-1))

        return scores