"""
Class defining the GCNN used to predict the wavelengths of
emission and absorption for chromophores in various solvents
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNChem(nn.Module):
    def __init__(self):
        super(GCNChem, self).__init__()
        self.gcn = nn.Linear(30, 30)
        self.fc1 = nn.Linear(30, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_int = nn.Linear(128, 64)
        self.fc_int2 = nn.Linear(64, 2)
        self.bngcn = nn.BatchNorm1d(num_features=30, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(num_features=512, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=256, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=128, track_running_stats=True)
        self.accuracy = None

    def forward(self, chromo_f, chromo_a, solv_f, solv_a, gcn_layers=2):
        chromo_x_resid = chromo_f.copy()
        solv_x_resid = solv_f.copy()
        
        # Graph convolutional layers 
        for i in range(gcn_layers):
            if i == 0:
                pass
            else:
                chromo_x_resid = chromo_x_resid.cpu().detach().numpy()
                solv_x_resid = solv_x_resid.cpu().detach().numpy()
            matmul = torch.tensor(np.transpose(np.einsum('mnr,ndr->mdr', chromo_a, chromo_x_resid), (0, 2, 1))).to("cuda")
            chromo_x = self.gcn(matmul.float())
            chromo_x = torch.tensor(np.transpose(chromo_x.cpu().detach().numpy(), (0, 2, 1))).float().to("cuda")
            chromo_x_resid = F.relu(chromo_x)
            chromo_x_resid = self.bngcn(chromo_x_resid)

            matmul = torch.tensor(np.transpose(np.einsum('mnr,ndr->mdr', solv_a, solv_x_resid), (0, 2, 1))).to("cuda")
            solv_x = self.gcn(matmul.float())
            solv_x = torch.tensor(np.transpose(solv_x.cpu().detach().numpy(), (0, 2, 1))).float().to("cuda")
            solv_x_resid = F.relu(solv_x)
            solv_x_resid = self.bngcn(solv_x_resid)
        
        # "Chemical space layers"
        z_0_chromo = torch.sum(chromo_x_resid, dim=0)
        z_1_chromo = self.fc1(torch.t(z_0_chromo))
        z_1_chromo = F.relu(z_1_chromo)
        z_1_chromo = self.bn1(z_1_chromo)
        z_2_chromo = self.fc2(z_1_chromo)
        z_2_chromo = F.relu(z_2_chromo)
        z_2_chromo = self.bn2(z_2_chromo)
        z_3_chromo = self.fc3(z_2_chromo)
        z_3_chromo = F.relu(z_3_chromo)
        z_3_chromo = self.bn3(z_3_chromo)
        
        z_0_solv = torch.sum(solv_x_resid, dim=0)
        z_1_solv = self.fc1(torch.t(z_0_solv))
        z_1_solv = F.relu(z_1_solv)
        z_1_solv = self.bn1(z_1_solv)
        z_2_solv = self.fc2(z_1_solv)
        z_2_solv = F.relu(z_2_solv)
        z_2_solv = self.bn2(z_2_solv)
        z_3_solv = self.fc3(z_2_solv)
        z_3_solv = F.relu(z_3_solv)
        z_3_solv = self.bn3(z_3_solv)
        
        # "Interaction layers"
        int_x = torch.cat((z_3_chromo, z_3_solv), dim=1)
        int_x = self.fc_int(int_x)
        int_x = F.relu(int_x)
        int_x = self.bn3(int_x)
        int_x = self.fc_int2(int_x)
        return int_x

    def loss(self, prediction, y_data, reduction='mean'):
        loss_val = F.mse_loss(prediction,torch.tensor(y_data).to("cuda"), reduction=reduction)
        return loss_val

