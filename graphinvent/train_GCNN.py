"""
Originally used in the form of a juptyer notebook but converted
to .py file.

Read in and cleans data, contains functions for producing the
feature and adjacency matrices and for training and testing.
Also, shows the execution of these functions to train and
save the GCNN model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem

from PhotochemModel import GCNChem

def get_adj_matrix(data, max_atoms=150):
    """
    Get adjacency matrix given a SMILES string
    -Magnitude of edges propertional to bond order
    """
    adj_mat = np.zeros([max_atoms,max_atoms,len(data.index)])
    for i in range(len(data.index)):
        mol = Chem.MolFromSmiles(data[i])
        mol_adj = Chem.GetAdjacencyMatrix(mol, useBO=True)
        adj_mat[0:np.size(mol_adj, axis=0), 0:np.size(mol_adj, axis=1), i] = mol_adj[:, :]
        adj_mat[0:np.size(mol_adj, axis=0), 0:np.size(mol_adj, axis=1), i] += np.identity(np.size(mol_adj, axis=1))
    return adj_mat

def one_hot_features(smiles_data):
    """
    One-hot encode features of each node of a molecule from the SMILES string
    element C,N,O,S,H,F,Cl,Br,I,Se,Te,Si,P,B,Sn,Ge --> added Na
    number of hydrogen atoms bonded 0-4
    number of atoms bonded 0-5
    aromaticity? T/F
    hybridization? 5 options
    formal charge? -1 to 1
    """
    atom_dict = dict([('C',0), ('N',1), ('O',2), ('S',3), ('Cl',4), ('F',5), ('H',6)])#, ('H',4), ('F',5), ('Cl',6), ('Br',7),
                    #('I',8), ('Se',9), ('Te',10), ('Si',11), ('P',12), ('B',13), ('Sn',14), ('Ge',15), ('Na',16)])
    hybrid_dict = dict([('S',0), ('SP',1), ('SP2',2), ('SP3',3), ('SP3D',4), ('SP3D2',5)])
    num_features = 30 
    feature_mat = np.zeros([150,num_features, len(smiles_data)]) #num possible atoms x features
    for i in range(len(smiles_data)):
        mol = Chem.MolFromSmiles(smiles_data[i])
        count_j = 0
        for atom in mol.GetAtoms():
            get_value = atom_dict[atom.GetSymbol()]
            feature_mat[count_j, get_value, i] = 1
            count_j += 1
        count_k = 0
        for atom in mol.GetAtoms():
            get_value = atom.GetTotalNumHs()
            feature_mat[count_k, len(atom_dict)+get_value, i] = 1
            count_k += 1
        count_l = 0
        for atom in mol.GetAtoms():
            get_value = atom.GetIsAromatic()
            if get_value == True:
                feature_mat[count_l, len(atom_dict)+5, i] = 1
            count_l += 1
        count_m = 0
        for atom in mol.GetAtoms():
            get_value = atom.GetDegree()
            feature_mat[count_m, len(atom_dict)+5+1+get_value, i] = 1
            count_m += 1
        count_n = 0
        for atom in mol.GetAtoms():
            get_value = hybrid_dict[str(atom.GetHybridization())]
            feature_mat[count_n, len(atom_dict) + 5+ 1 + 6 + get_value, i] = 1
            count_n += 1
        count_o = 0
        for atom in mol.GetAtoms():
            get_value = atom.GetFormalCharge()
            feature_mat[count_o, len(atom_dict) + 5 + 1 + 6 + len(hybrid_dict) + 1 + get_value] = 1
            count_o += 1
    
    return feature_mat

def train(model, device, chromo_f, chromo_a, solv_f, solv_a, train_y, optimizer, epoch, batch_size=1000):#12817):
    model.train()
    losses = []
    shuffle_batch = torch.randperm(chromo_f.shape[2])
    for i in range(int(chromo_f.shape[2]/batch_size)):
        chromo_f_batch = chromo_f[:, :, shuffle_batch[i*batch_size:(i+1)*batch_size]]
        chromo_a_batch = chromo_a[:, :, shuffle_batch[i*batch_size:(i+1)*batch_size]]
        solv_f_batch = solv_f[:, :, shuffle_batch[i*batch_size:(i+1)*batch_size]]
        solv_a_batch = solv_a[:, :, shuffle_batch[i*batch_size:(i+1)*batch_size]]
        train_y_batch = train_y[shuffle_batch[i*batch_size:(i+1)*batch_size], :]
        optimizer.zero_grad()
        output = model(chromo_f_batch, chromo_a_batch, solv_f_batch, solv_a_batch)
        loss = model.loss(output, train_y_batch)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def test(model, device, chromo_f, chromo_a, solv_f, solv_a, test_y):
    model.eval()
    with torch.no_grad():
        output = model(chromo_f, chromo_a, solv_f, solv_a)
        test_loss = model.loss(output, test_y).item()
    return test_loss, output

# Read data
full_data = pd.read_csv('./full_chromophore_db.csv')
data = full_data[['Chromophore', 'Solvent', 'Absorption max (nm)', 'Emission max (nm)']].copy()
data_nonan = data.dropna()
data_nonan = data_nonan.reset_index(drop = True)
data_nonan['Molecular relaxation (nm)'] = (data_nonan['Emission max (nm)'] - data_nonan['Absorption max (nm)'])/2

# Clean data by removing entries with certain atoms
smi_data = data_nonan[['Chromophore', 'Solvent', 'Absorption max (nm)', 'Emission max (nm)']].copy()
smi_data['Solvent'][4197-113] = 'O'
smi_data['Solvent'][13850-113] = 'O'
smi_data['Solvent'][14383-113:14394-113] = 'O'
smi_data['Solvent'][14513-113] = 'O'
smi_data['Solvent'][14519-113] = 'O'
smi_data['Solvent'][14528-113:14530-113] = 'O'
smi_data['Solvent'][14533-113] = 'O'
smi_data['Solvent'][14535-113] = 'O'
smi_data = smi_data[~smi_data['Chromophore'].str.contains("B", na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains("b", na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains("Na", na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Si', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('P', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('p', na=False)]
smi_data = smi_data[~smi_data['Solvent'].str.contains('P', na=False)]
smi_data = smi_data[~smi_data['Solvent'].str.contains('p', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Ge', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Se', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Br', na=False)]
smi_data = smi_data[~smi_data['Solvent'].str.contains('Br', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Sn', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('Te', na=False)]
smi_data = smi_data[~smi_data['Chromophore'].str.contains('I', na=False)]
smi_data = smi_data.reset_index()
nha_chromo = smi_data['Chromophore'].copy()
nha_solvent = smi_data['Solvent'].copy()
y_data = smi_data[['Absorption max (nm)', 'Emission max (nm)']].copy()
y_data = y_data.to_numpy()

# Use functions to produce adjacency and feature matrices for the solvent and chromophore
chromo_adj = get_adj_matrix(nha_chromo)
solv_adj = get_adj_matrix(nha_solvent)
chromo_f = one_hot_features(nha_chromo)
solv_f = one_hot_features(nha_solvent)

# Shuffle dataset and split into test (20%) and train (80%) dataset
indices = torch.randperm(np.size(chromo_f, axis=2))
train_chromo_f = chromo_f[:,:,indices[0:int(0.8*np.size(chromo_f, axis=2))]].copy()
train_solv_f = solv_f[:,:,indices[0:int(0.8*np.size(chromo_f, axis=2))]].copy()
train_chromo_a = chromo_adj[:,:,indices[0:int(0.8*np.size(chromo_f, axis=2))]].copy()
train_solv_a = solv_adj[:,:,indices[0:int(0.8*np.size(chromo_f, axis=2))]].copy()
train_y = y_data[indices[0:int(0.8*np.size(chromo_f, axis=2))], :].copy()
test_chromo_f = chromo_f[:,:,indices[int(0.8*np.size(chromo_f, axis=2)):int(np.size(chromo_f, axis=2))]].copy()
test_solv_f = solv_f[:,:,indices[int(0.8*np.size(chromo_f, axis=2)):int(np.size(chromo_f, axis=2))]].copy()
test_chromo_a = chromo_adj[:,:,indices[int(0.8*np.size(chromo_f, axis=2)):int(np.size(chromo_f, axis=2))]].copy()
test_solv_a = solv_adj[:,:,indices[int(0.8*np.size(chromo_f, axis=2)):int(np.size(chromo_f, axis=2))]].copy()
test_y = y_data[indices[int(0.8*np.size(chromo_f, axis=2)):], :].copy()

# Standardize output/y values
mean_s1 = np.average(train_y[:, 0])
var_s1 = np.var(train_y[:, 0])
train_y[:, 0] = (train_y[:, 0] - mean_s1)/np.sqrt(var_s1)
test_y[:, 0] = (test_y[:, 0] - mean_s1)/np.sqrt(var_s1)
mean_lambda = np.average(train_y[:, 1])
var_lambda = np.var(train_y[:, 1])
train_y[:, 1] = (train_y[:, 1]-mean_lambda)/np.sqrt(var_lambda)
test_y[:, 1] = (test_y[:, 1]-mean_lambda)/np.sqrt(var_lambda)

# Train model and save final model as a .pt file
EPOCHS = 100
LEARNING_RATE = 0.01
USE_CUDA = True
WEIGHT_DECAY = 0.0005

use_cuda = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

model = GCNChem().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
start_epoch = 0

train_losses = []
test_losses = []

test_loss, output = test(model, device, test_chromo_f, test_chromo_a, test_solv_f, test_solv_a, torch.tensor(test_y).float())
print(test_loss)
test_losses.append((start_epoch, test_loss))

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        print(epoch)
        train_loss = train(model, device, train_chromo_f, train_chromo_a, train_solv_f, train_solv_a, torch.tensor(train_y).float(), optimizer, epoch)
        test_loss, output = test(model, device, test_chromo_f, test_chromo_a, test_solv_f, test_solv_a, torch.tensor(test_y).float())
        print(test_loss)
        print('RMSE $\lambda$$_{abs}$: ' + str(np.sqrt((((output[:, 0].cpu()*np.sqrt(var_s1)+mean_s1) - (test_y[:, 0]*np.sqrt(var_s1)+mean_s1)) ** 2).mean())))
        print('RMSE $\lambda$$_{em}$: ' + str(np.sqrt((((output[:, 1].cpu()*np.sqrt(var_lambda)+mean_lambda) - (test_y[:, 1]*np.sqrt(var_lambda)+mean_lambda)) ** 2).mean())))
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))


except KeyboardInterrupt as ke:
    print('Interrupted')
except:
    import traceback
    traceback.print_exc()

finally:
    ep, val = zip(*train_losses)
    plt.plot(ep, val)
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.show()
    ep, val = zip(*test_losses)
    plt.plot(ep, val)
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.show()
    #torch.save(model, "best_model_gpu.pt")
    final_model = model
    torch.save(final_model.state_dict(), 'model_noheavyatoms_gpu.pt')


