import pandas as pd
import numpy as np
from rdkit import Chem


def get_adj_matrix(data, max_atoms=150):
    #adj_mat = np.zeros([max_atoms,max_atoms,len(data.index)])
    #for i in range(len(data.index)):
    adj_mat = np.zeros([max_atoms,max_atoms,len(data)])
    for i in range(len(data)):
        mol = Chem.MolFromSmiles(data[i])
        if mol == None:
            pass
        else:
            mol_adj = Chem.GetAdjacencyMatrix(mol, useBO=True)
            adj_mat[0:np.size(mol_adj, axis=0), 0:np.size(mol_adj, axis=1), i] = mol_adj[:, :]
            adj_mat[0:np.size(mol_adj, axis=0), 0:np.size(mol_adj, axis=1), i] += np.identity(np.size(mol_adj, axis=1))
    return adj_mat

def one_hot_features(smiles_data):
    """
    #element C,N,O,S,H,F,Cl,Br,I,Se,Te,Si,P,B,Sn,Ge --> added Na
    #number of hydrogen atoms bonded 0-4
    #number of atoms bonded 0-5
    #aromaticity? T/F
    #hybridization? 5 options
    #formal charge? -4 to 4
    """
    atom_dict = dict([('C',0), ('N',1), ('O',2), ('S',3), ('Cl',4), ('F',5), ('H',6)])#, ('H',4), ('F',5), ('Cl',6), ('Br',7),
                    #('I',8), ('Se',9), ('Te',10), ('Si',11), ('P',12), ('B',13), ('Sn',14), ('Ge',15), ('Na',16)])
    hybrid_dict = dict([('S',0), ('SP',1), ('SP2',2), ('SP3',3), ('SP3D',4), ('SP3D2',5)])
    num_features = 30 #43 
    feature_mat = np.zeros([150,num_features, len(smiles_data)]) #num possible atoms x features
    for i in range(len(smiles_data)):
        mol = Chem.MolFromSmiles(smiles_data[i])
        if mol == None:
            pass
        else:
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
                #feature_mat[count_o, len(atom_dict) + 5 + 1 + 6 + len(hybrid_dict) + 4 + get_value] = 1
                feature_mat[count_o, len(atom_dict) + 5 + 1 + 6 + len(hybrid_dict) + 1 + get_value] = 1
                count_o += 1
    
    return feature_mat
