# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from Graph Contrastive Learning with Augmentations
#   (https://github.com/Shen-Lab/GraphCL)
# Copyright (c) 2020 Shen Lab at Texas A&M University
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# The date of modifications: July, 2021

import torch
import numpy as np
import random
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def random_view(candidates):
    def views_fn(batch_data):
        data_list = batch_data.to_data_list()
        transformed_list = []
        for data in data_list:
            view_fn = random.choice(candidates)
            transformed = view_fn(data)
            transformed_list.append(transformed)

        return Batch.from_data_list(transformed_list)

    return views_fn

def edge_perturbation(add=True, drop=False, ratio=0.1):
    '''
    Args:
        add (bool): Set True if randomly add edges in a given graph.
        drop (bool): Set True if randomly drop edges in a given graph.
        ratio: Percentage of edges to add or drop.
    '''

    def do_trans(data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        perturb_num = int(edge_num * ratio)

        edge_index = data.edge_index.detach().clone()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2,-1)

        if drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num - perturb_num, replace=False)]

        if add:
            idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain.to(edge_index), idx_add.to(edge_index)), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return Data(x=data.x, edge_index=new_edge_index)

    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    original edge_index tensor with shape [2, num_edges].
        Returns:
            x tensor with shape [num_nodes, num_node_features];
            edge_index tensor with shape [2, num_perturb_edges];
            batch tensor with shape [num_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def node_attr_mask(mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5):
    '''
    Args:
        mode: Masking mode with three options:
                'whole': mask all feature dimensions of the selected node with a Gaussian distribution;
                'partial': mask only selected feature dimensions with a Gaussian distribution;
                'onehot': mask all feature dimensions of the selected node with a one-hot vector.
        mask_ratio: Percentage of masking feature dimensions.
        mask_mean: Mean of the Gaussian distribution.
        mask_std: Standard deviation of the distribution. Must be non-negative.
    '''
    def do_trans(data):
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()
        mask = torch.zeros(node_num)

        if mode == 'whole':
            mask_num = int(node_num * mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std, size=(mask_num, feat_dim)),
                                       dtype=torch.float32).to(x)

        elif mode == 'partial':
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std), dtype=torch.float32).to(x)

        elif mode == 'onehot':
            mask_num = int(node_num * mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32).to(x)

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

        return Data(x=x, edge_index=data.edge_index)

    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    original x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    edge_index tensor with shape [2, num_edges].
        Returns:
            x tensor with shape [num_nodes, num_node_features];
            edge_index tensor with shape [2, num_edges];
            batch tensor with shape [num_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def uniform_sample(ratio=0.1):
    '''
    Args:
        ratio: Percentage of nodes to drop.
    '''

    def do_trans(data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * ratio)

        idx_drop = np.random.choice(node_num, drop_num, replace=False)
        idx_nondrop = [n for n in range(node_num) if not n in idx_drop]


        # data.x = data.x[idx_nondrop]
        edge_index = data.edge_index.cpu().numpy()

        adj = torch.zeros((node_num, node_num)).to(data.x)
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = adj.nonzero().t()

        return Data(x=data.x, edge_index=edge_index)

    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    edge_index tensor with shape [2, num_edges].
        Returns:
            x tensor with shape [num_nondrop_nodes, num_node_features];
            edge_index tensor with shape [2, num_nondrop_edges];
            batch tensor with shape [num_nondrop_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn


def RW_sample(ratio=0.1, add_self_loop=True):
    '''
    Args:
        ratio: Percentage of nodes to sample from the graph.
        add_self_loop (bool): Set True if add self-loop in edge_index.
    '''

    def do_trans(data):
        node_num, _ = data.x.size()
        sub_num = int(node_num * ratio)

        if add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t().to(data.edge_index)
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        # edge_index = edge_index.numpy()
        idx_sub = [np.random.randint(node_num, size=1)[0]]
        # idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0] == idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            # idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_sampled = idx_sub
        adj = to_dense_adj(edge_index)[0]
        adj = adj[idx_sampled, :][:, idx_sampled]

        return Data(x=data.x[idx_sampled], edge_index=dense_to_sparse(adj)[0])

    def views_fn(data):
        '''
        Args:
            data: A graph data object containing:
                    batch tensor with shape [num_nodes];
                    x tensor with shape [num_nodes, num_node_features];
                    y tensor with arbitrary shape;
                    edge_attr tensor with shape [num_edges, num_edge_features];
                    edge_index tensor with shape [2, num_edges].
        Returns:
            x tensor with shape [num_sampled_nodes, num_node_features];
            edge_index tensor with shape [2, num_sampled_edges];
            batch tensor with shape [num_sampled_nodes].
        '''
        if isinstance(data, Batch):
            dlist = [do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return do_trans(data)

    return views_fn