import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from lapsolver import solve_dense
#from models import Overall

class Graph(Data):  
    """
    Wrapper on torch_geometric data class. This has convenience methods to convert tensors of the class in one go.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_vals(self, fn):
        attr_names = ['x', # Feature vector of node. NXD
                           'edge_attr', # Feature vector of Edges MXD
                           'edge_index', # 2XM, where edge enpoints index is stored
                        ]   
        for attr_name in attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_val = getattr(self, attr_name)
                    setattr(self, attr_name, fn(old_val))

    def tensor(self):
        self.change_vals(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self.change_vals(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self.change_vals(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        self.change_vals(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        self.change_vals(attr_change_fn=lambda x: x.cuda())
        return self
    def to(self, device):
        self.change_vals(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device

        return torch.device('cpu')

class HungarianDataset:
    def __init__(self, num_samples, mode="train"):
        super(HungarianDataset,self).__init__()
        self.mode = mode
        self.num_samples = num_samples
        self.min_nodes = 10
        self.max_nodes = 20
        if mode == "test":
            self.all_graphs = []
            for i in range(num_samples):
                self.all_graphs.append(self.get_random_graph())
    def get_random_graph(self):
        left_num = np.random.randint(self.min_nodes, self.max_nodes) ##Num of nodes on left side
        right_num = np.random.randint(self.min_nodes, self.max_nodes)
        weights = np.random.rand(left_num,right_num) ##Random weight matrix
        # weights_row = weights/(weights.sum(axis=0).reshape([1,-1]))
        # weights_col = weights/(weights.sum(axis=1).reshape([-1,1]))
        # weights = (weights_row+weights_col)/2.0
        ##Prepare edge matrix (2Xnum_edges)
        x, y = np.indices((left_num,right_num))  ##Make all the pairs
        edges = np.stack([x.flatten(),y.flatten()]) ##Make edges 
        edges_weights = weights[edges[0,:], edges[1,:]] ##Take corresponding weights
        edges[1,:] += left_num ##Make numbering of right nodes start from where left side nodes ends
        node_feat = torch.zeros(left_num + right_num, 8) + 0.001 ##Will be changed later
        costs = 1 - weights ###lapsolver solves in costs
        rids, cids = solve_dense(costs)
        cids += left_num ##Again index of nodes at right side starts from 0, Clubbing both side nodes into one list
        labels = np.zeros([left_num+right_num, left_num + right_num]).astype(np.int32) ##GT label
        labels[rids, cids] = 1
        labels = labels[edges[0,:], edges[1,:]] ##Slicing the matrix so that final gt has correspondence with edges formed

        labels = torch.Tensor(labels.flatten()).float()
        
        edges = torch.LongTensor(edges)
        edges_weights = torch.FloatTensor(1.0 - edges_weights.reshape([-1,1])) ##using distance rather than weights, Does not matter
        graph = Graph(x = node_feat, 
                    edge_attr = edges_weights, 
                    edge_index = edges,
                    labels = labels)
        graph.labels = labels
        return graph

    def __getitem__(self,index):
        if self.mode == "test":
            graph = self.all_graphs[index]
            new_graph = Graph(x = graph.x.clone(), 
                    edge_attr = graph.edge_attr.clone(), 
                    edge_index = graph.edge_index.clone(),
                    labels = graph.labels.clone())
            return new_graph
        else:
            return self.get_random_graph()
        

    def __len__(self):
        return self.num_samples

if __name__ == '__main__':

    model = Overall()
    dataset = HungarianDataset()
    loader = DataLoader(
        dataset, batch_size=4,
        num_workers=1, pin_memory=True, drop_last=True)
    tot_files = 0
    for i, elem in enumerate(loader):
        
        print(i, elem)
