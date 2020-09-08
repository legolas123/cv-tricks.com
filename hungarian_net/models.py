import torch
import torch.nn as nn
import torch_scatter
from torch.nn import functional as F
class FCStack(nn.Module):
    def __init__(self, in_channels, list_out_channels, batch_norm=True):
        super(FCStack, self).__init__()

        layers = []
        for out_channels in list_out_channels:
            layers.append(nn.Linear(in_channels, out_channels))
            if out_channels != 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.fc = nn.Sequential(*layers)
    def forward(self,x):
        return self.fc(x)

class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.edge_model = FCStack(1, [8,8,8,8])
        self.node_model = FCStack(8,[8,8,8,8])

        self.edge_mpn = FCStack(2*8 + 2*8, [16,8,8,8])
        self.node_mpn = FCStack(3*8, [8,8,8,8])

        self.classifier = FCStack(8, [8,8,1])

        self.num_steps = 7

        #self.node_init = nn.Parameter(torch.rand(1,8)-0.5).cuda()

    def forward(self, graph):
        edges_feature = graph.edge_attr
        edges = graph.edge_index
        edge_encoded = self.edge_model(edges_feature)
        #node_feat = self.node_init.expand(graph.num_nodes, 8).contiguous()
        #node_feat = torch.cat([edges_feature, edges_feature], dim = 0)
        #node_feat = torch_scatter.scatter_mean(node_feat, edges.flatten(),dim=0)
        #node_feat = torch.zeros(graph.num_nodes, 8) + 0.001
        # node_encoded_left = torch_scatter.scatter_mean(edges_feature, edges[0,:],dim=0) ##TODO
        # node_encoded_right = torch_scatter.scatter_mean(edges_feature, edges[1,:],dim=0)
        # node_feat = torch.cat([node_encoded_left, node_encoded_right],dim=0)
        node_feat = graph.x
        node_encoded = self.node_model(node_feat)

        edge_mpn_output = edge_encoded
        node_mpn_output = node_encoded
        outputs = []
        for i in range(self.num_steps):
            edge_mpn_input = torch.cat([edge_encoded, edge_mpn_output], dim=1)
            corr_node_left = node_encoded[edges[0,:]]
            corr_node_right = node_encoded[edges[1,:]]
            edge_mpn_input = torch.cat([corr_node_left, edge_mpn_input, corr_node_right], dim=1)

            edge_mpn_output = self.edge_mpn(edge_mpn_input)

            node_left_mpn_input = torch.cat([node_mpn_output[edges[1,:]], node_mpn_output[edges[0,:]], edge_mpn_output], dim = 1) ##TODO
            #import pdb; pdb.set_trace()
            node_left_mpn_output = self.node_mpn(node_left_mpn_input)
            node_right_mpn_input = torch.cat([node_mpn_output[edges[0,:]], node_mpn_output[edges[1,:]], edge_mpn_output], dim = 1)
            node_right_mpn_output = self.node_mpn(node_right_mpn_input)
            node_mpn_output = torch.cat([node_left_mpn_output, node_right_mpn_output], dim=0)
            node_mpn_output = torch_scatter.scatter_mean(node_mpn_output, edges.flatten(), dim = 0)
            if i > 1:
                edge_output = self.classifier(edge_mpn_output)
                outputs.append(edge_output)
        #import pdb; pdb.set_trace()
        return outputs

def edge_loss(preds, labels):
    #import pdb; pdb.set_trace()
    num_pos = labels.sum()
    num_neg = labels.shape[0] - num_pos
    pos_weight = num_neg.float()/num_pos
    loss = 0
    for pred in preds:
        loss += F.binary_cross_entropy_with_logits(pred.view(-1), labels.view(-1), pos_weight = pos_weight)
    return loss