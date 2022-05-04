# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
#
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

import torch.nn as nn
from .GraphNets import GIN
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool
from utils import compute_pre_recall_f1

class Pooling(nn.Module):

    def __init__(self,pool_cls):
        '''
        dim: Integer. Embedding dimension.
        aug1, aug2: String. Should be in ['dropN', 'permE', 'subgraph',
                    'maskN', 'random2', 'random3', 'random4'].
        aug_ratio: Float between (0,1).
        '''
        super(Pooling, self).__init__()
        if pool_cls == 'add':
            self.net = global_add_pool
        elif pool_cls == 'mean':
            self.net = global_mean_pool
        elif pool_cls == 'max':
            self.net = global_max_pool
    def forward(self,graph):

        z = self.net(graph.x,graph.batch)
        return z

class OCPool:
    def __init__(self, config):

        self.embedder = Pooling(config['pool'])
        self.detector =  OneClassSVM(nu=0.1)

    def __call__(self, train_loader,cls,val_loader=None,test_loader=None):
        # for inference, output anomaly score
        val_auc,test_auc,test_ap,test_f1,test_score,test_label = None,None,None,None,None,None

        train_embeddings = []
        with torch.no_grad():
            for data in train_loader:
                data = data
                z = self.embedder(data)
                train_embeddings.append(z)
            train_embeddings = torch.cat(train_embeddings, 0).numpy()

        self.detector.fit(train_embeddings)
        if val_loader is not None:
            val_embeddings = []
            val_label = []

            with torch.no_grad():
                for data in val_loader:
                    data = data
                    z = self.embedder(data)
                    val_embeddings.append(z)
                    val_label.append(data.y!=cls)
                val_embeddings = torch.cat(val_embeddings, 0).numpy()
                val_label = torch.cat(val_label).numpy()
            val_score = -self.detector.decision_function(val_embeddings)

            val_auc = roc_auc_score(val_label,val_score)

        if test_loader is not None:
            test_embeddings = []
            test_label = []
            with torch.no_grad():
                for data in test_loader:
                    data = data
                    z = self.embedder(data)
                    test_embeddings.append(z)
                    test_label.append(data.y!=cls)
                test_embeddings = torch.cat(test_embeddings, 0).numpy()
                test_label = torch.cat(test_label).numpy()
            test_score = -self.detector.decision_function(test_embeddings)

            test_auc = roc_auc_score(test_label,test_score)
            test_ap = average_precision_score(test_label, test_score)
            test_f1= compute_pre_recall_f1(test_label, test_score)
        return val_auc,test_auc,test_ap,test_f1,test_score,test_label


class OCGTL(nn.Module):
    def __init__(self, dim_features,config):
        super(OCGTL, self).__init__()

        num_trans = config['num_trans']
        dim_targets = config['hidden_dim']
        num_layers = config['num_layers']
        self.device = config['device']
        self.gins = []
        for _ in range(num_trans):
            self.gins.append(GIN(dim_features,dim_targets,config))
        self.gins = nn.ModuleList(self.gins)
        self.center = nn.Parameter(torch.empty(1, 1,dim_targets*num_layers), requires_grad=True)
        self.reset_parameters()
    def forward(self,data):
        data = data.to(self.device)
        z_cat = []
        for i,model in enumerate(self.gins):
            z = model(data)
            z_cat.append(z.unsqueeze(1))
        z_cat = torch.cat(z_cat,1)
        z_cat[:,0] = z_cat[:,0]+self.center[:,0]
        return [z_cat,self.center]

    def reset_parameters(self):
        init.normal_(self.center)
        for nn in self.gins:
            nn.reset_parameters()

class GTL(nn.Module):
    def __init__(self, dim_features,config):
        super(GTL, self).__init__()

        num_trans = config['num_trans']
        dim_targets = config['hidden_dim']
        self.device = config['device']
        self.gins = []
        for _ in range(num_trans):
            self.gins.append(GIN(dim_features,dim_targets,config))
        self.gins = nn.ModuleList(self.gins)
        self.reset_parameters()
    def forward(self,data):
        data = data.to(self.device)
        z_cat = []
        for i,model in enumerate(self.gins):
            z = model(data)
            z_cat.append(z.unsqueeze(1))

        return torch.cat(z_cat,1)

    def reset_parameters(self):

        for nn in self.gins:
            nn.reset_parameters()

class OCGIN(nn.Module):
    def __init__(self, dim_features, config):
        super(OCGIN, self).__init__()

        self.dim_targets = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.net = GIN(dim_features, self.dim_targets, config)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cuda')
        self.reset_parameters()
    def forward(self, data):
        data = data.to(self.device)
        z = self.net(data)
        return [z, self.center]

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

from .GraphNets import GIN_classifier
from .GraphTransform import *

class GTP(nn.Module):

    def __init__(self,dim_features,config):
        super(GTP, self).__init__()
        aug_ratio = 0.1
        self.views_fn = []
        self.views_fn.append(lambda x: x)
        self.views_fn.append(uniform_sample(ratio=aug_ratio))
        self.views_fn.append(edge_perturbation(add=True,drop=False,ratio=aug_ratio))
        self.views_fn.append(edge_perturbation(add=False, drop=True, ratio=aug_ratio))
        self.views_fn.append(RW_sample(ratio=aug_ratio))
        self.views_fn.append(node_attr_mask(mask_ratio=aug_ratio))

        dim_targets = len(self.views_fn)
        self.net = GIN_classifier(dim_features, dim_targets, config)
        self.reset_parameters()
    def forward(self,graph):

        y_pred_all = []
        for i,v_fn in enumerate(self.views_fn):
            view = v_fn(graph)
            y_pred = self.net(view)
            y_pred_all.append(y_pred.unsqueeze(1))

        y_pred_all = torch.cat(y_pred_all,1)
        return y_pred_all

    def reset_parameters(self):
        self.net.reset_parameters()
