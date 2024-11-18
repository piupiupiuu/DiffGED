import os.path as osp
import random

import networkx
import numpy as np
import math
from texttable import Texttable
import networkx as nx
import torch.nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import ctypes
import numpy as np
from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json


INT = ctypes.c_int
PINT = ctypes.POINTER(ctypes.c_int)
PPINT = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

types = ['Ac', 'Ag', 'Al', 'As', 'Au', 'B', 'Bi', 'Br', 'C', 'Ca', 'Cd',
         'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Er', 'F', 'Fe', 'Ga', 'Gd', 'Ge',
         'Hg', 'Ho', 'I', 'Ir', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na',
         'Nb', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Re',
         'Rh', 'Ru', 'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn', 'Tb', 'Te', 'Ti',
         'Tl', 'U', 'V', 'W', 'Yb', 'Zn', 'Zr']

if __name__ == '__main__':
    types.sort()
    print(types)

def read_gexf_data(graphname, new_types):
    # new_types = set()
    G = nx.read_gexf(graphname)
    # TODO: Mapping of nodes in `G` to a contiguous number: AIDS数据集已经确定是连续的整数了，这个去掉。
    # mapping = {name: j for j, name in enumerate(G.nodes())}
    # G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=G.number_of_nodes())

    x = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    for node, info in G.nodes(data=True):
         x[int(node)] = types.index(info['type'])
    x = F.one_hot(x, num_classes=len(types)).to(torch.float)

    return edge_index, x, G.number_of_nodes()

def get_from_all_graphs(all_graphs, id):
    G = all_graphs[id]
    # Mapping of nodes in `G` to a contiguous number:
    mapping = {name: j for j, name in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=G.number_of_nodes())

    x = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    for node, info in G.nodes(data=True):
        x[int(node)] = types.index(info['type'])

    x = F.one_hot(x, num_classes=len(types)).to(torch.float)
    return edge_index, x, G.number_of_nodes()


def denormalize_ged(g1_nodes, g2_nodes, sim_score):
    """
    Converts normalized similar into ged.
    """
    nged = -math.log(sim_score, math.e)
    return round(nged * (g1_nodes + g2_nodes) / 2)

def normalize_ged(g1_nodes, g2_nodes, ged):
    """
    Converts ged into normalized ged.
    """
    return torch.exp(-1 * torch.tensor (2 * ged/ ( g1_nodes+ g2_nodes)))

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def random_assign(row_num):
    res= [[],[]]
    res[0] = [i for i in range(row_num)]
    res[1] = [i for i in range(row_num)]
    random.shuffle(res[1])
    return res



def calculate_prec_at_k(k, prediction, groundtruth):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[-k:]
    best_k_gt = groundtruth.argsort()[-k:]

    return len(set(best_k_pred).intersection(set(best_k_gt))) / k


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation

def int1ArrayToPointer(arr): #Converts a 2D numpy to ctypes 2D array.
    # Init needed data types
    ARR_DIMX = INT * arr.shape[0]
    arr_ptr = ARR_DIMX()
    for i, val in enumerate(arr):
        arr_ptr[i] = val
    return arr_ptr

def int2ArrayToPointer(arr): #Converts a 2D numpy to ctypes 2D array.
    # Init needed data types
    ARR_DIMX = INT * arr.shape[1]
    ARR_DIMY = PINT * arr.shape[0]
    # Init pointer
    arr_ptr = ARR_DIMY()
    # Fill the 2D ctypes array with values
    for i, row in enumerate(arr):
        arr_ptr[i] = ARR_DIMX()
        for j, val in enumerate(row):
            arr_ptr[i][j] = val
    return arr_ptr



def CT(input): # convert type
    ctypes_map = {int: ctypes.c_int, float: ctypes.c_double, str: ctypes.c_char_p}
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is " + input)
            return None
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is " + input)
            return None

def nx2txt(G: networkx.Graph, id:str, alg:str):  #
    if alg in ['AIDS700nef', 'CANCER']:
        line = "t " + "# " + id + "\n"
        for id, label in G.nodes(data=True):
            line += "v " + str(id) + " " + label['type'] + "\n"
        for (u, v) in G.edges():
            line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
        return line
    elif alg in ['IMDBMulti']:
        line = "t " + "# " + id + "\n"
        for id, label in G.nodes(data=True):
            line += "v " + str(id) + " " + str(1) + "\n"
        for (u, v) in G.edges():
            line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
        return line
    else:
        return ""

def get_beam_size(n1, n2, e1, e2, dataset:str):
    beam_size = -1
    if dataset in ['AIDS700nef']:
        beam_size = -1
    elif dataset in ['IMDBMulti']:
        if n1+e1 < 40 and n2+e2 < 40 and n1 <= 10 and n2 <= 10:     # 2kb
            beam_size = -1
        elif n1+e1 < 65 and n2+e2 < 65 and n1 <= 13 and n2 <= 13:   # 3kb
            beam_size = 1000
        elif n1+e1 < 200 and n2+e2 < 200 and n1 <= 21 and n2 <= 21: #10kb
            beam_size = 100
        elif e1 > 300 or e2 > 300:
            beam_size = 2
        else:
            beam_size = 10
    elif dataset in ['CANCER']:
        if n1 + e1 < 65 and n2 + e2 < 65 and n1 <= 25 and n2 <= 25:
            beam_size = 1000
        else:
            beam_size = 300

    return beam_size

if __name__ == '__main__':
    t = torch.Tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    c = torch.Tensor([[0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1035, 0.0999, 0.1018, 0.1042,
         0.0000],
        [0.1003, 0.0000, 0.1019, 0.0987, 0.0997, 0.1044, 0.1000, 0.1021, 0.1052,
         0.0000],
        [0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1037, 0.1000, 0.1018, 0.1044,
         0.0000],
        [0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1037, 0.1000, 0.1018, 0.1044,
         0.0000],
        [0.1004, 0.0000, 0.1019, 0.0985, 0.0998, 0.1043, 0.1000, 0.1020, 0.1047,
         0.0000],
        [0.1004, 0.0000, 0.1022, 0.0983, 0.0997, 0.1050, 0.1000, 0.1023, 0.1055,
         0.0000],
        [0.1003, 0.0000, 0.1020, 0.0987, 0.0997, 0.1042, 0.0999, 0.1021, 0.1049,
         0.0000],
        [0.1003, 0.0000, 0.1019, 0.0987, 0.0997, 0.1044, 0.1000, 0.1021, 0.1052,
         0.0000],
        [0.1000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.0000],
        [0.1000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.0000]])
    c.unsqueeze_(0)
    t.unsqueeze_(0)

    y = torch.sum(t, dim=2)
    print(y)

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs

def load_all_graphs(data_location, dataset_name):
    graphs = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "json")
    train_num = len(graphs)
    graphs += iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "json")
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    return train_num, val_num, test_num, graphs

def load_labels(data_location, dataset_name):
    path = data_location + "json_data/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "onehot") \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features

def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)

def load_features(data_location, dataset_name, feature_name):
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", feature_name) \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", feature_name)
    feature_dim = len(features[0][0])
    print('Load {} features (dim = {}) of {}.'.format(feature_name, feature_dim, dataset_name))
    return feature_dim, features