# ChromoOpt: Final project for CSE490G1

As the world turns to clean energy alternatives, many scientists look to the sun to harvest energy. The sun emits a wide range of wavelengths but not all wavelengths are being absorbed by chromophores, meaning that we are not harvesting all the sun’s energy. The goal of my final project was to generate new, viable molecules with specific photochemical properties. To accomplish this, I started with the GraphINVENT codebase (https://github.com/MolecularAI/GraphINVENT) which was used by the authors to design drugs with certain properties.<sup>1</sup> I re-trained the generator using a database of chromophores, redefined the scoring function to include important photochemical properties, and built and trained a graph convolutional neural network (GCNN) to predict the photochemical properties<sup>2</sup> defined in the scoring function.

## Introduction
A wide range of electromagnetic waves are emitted from the sun, ranging from ~250 nm to ~2500 nm. Much progress in solar cells has been focused on the visible range of the electromagnetic spectrum. However, as is clear from the figure below, this comprises a relatively small fraction of the total emissions from the sun. To better utilize the entire spectrum of light from the sun, we need chromophores (molecules that absorb light) that absorb in the near IR and IR range of the electromagnetic series. However, finding molecules with specific qualities is a well-established problem in chemistry. In recent years, the application of deep learning to this problem have proven fruitful in drug design (CITE). These methods have not yet been widely used in photochemistry and chromophore design. To tackle this problem, I have used an existing codebase, GraphINVENT (), to generate chromophore-like molecules and used a graph convolutional neural network to provide a score for these generated molecules.

<a href="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS"><img src="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS" style="width: 500px; max-width: 100%; height: auto" /></a>

## Related Work
The use of deep learning for drug design is prevalent in the literature. These have included reinforcement learning methods (Actor-critic, Q-learning, DQN) and a combination of generative adversarial networks (GANs) with RL methods. On major hurdle that must be overcome is

The major advantage that GANs provide is that the generator can learn to produce molecules that are chemically-viable (acceptable number of bonds, higher order bonds, etc.). These methods have been shown to produce numerous drug candidates with specific features such as activity, lipophilicity, and molecules size, for example. However, few people have utilized a similar approach to design chromophores. By simply changing the scoring function of these reinforcement learning (RL) methods, we can bias the generation of molecules towards those with desired photochemical properties.

For photochemical systems, a recent study demonstrated the ability of a GCNN in predicting photochemical properties………………


## Approach
### Dataset:
The database that I used is comprised of >30,000 chromophore/solvent pairings and 7 of their experimental photochemical properties (I only focused on the absorption and emission wavelengths). (CITE) Because we are interested in heavy-atom-free chromophores, I removed all compounds that contained elements other than H, O, C, N, S, and Cl. By removing these entries and also entries that contained “NaN”, the final dataset was reduced to ~12,000 entries. Here is an example showing 5 lines of the dataset:

<a href="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs"><img src="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs" style="width: 500px; max-width: 100%; height: auto" /></a>
  
The chromophore and solvent are represented by SMILES strings. This notation is used to denote the chemical structure in a compact way. Here is an example of a SMILES string and the corresponding chemical structure.

<a href="https://drive.google.com/uc?export=view&id=1r-ApHNsG6ty2tvOK39cF4_WhVIhiQMfW"><img src="https://drive.google.com/uc?export=view&id=1r-ApHNsG6ty2tvOK39cF4_WhVIhiQMfW" style="width: 300px; max-width: 100%; height: auto" /></a>
  
From the SMILES strings, I used RDkit (CITE) to extract chemical features at each atom site (i.e., atom name, hybridization, number of atoms bonded to it, aromaticity, etc.). This information was then one-hot encoded to yield a feature matrix. The adjacency matrix was also produced using RDkit with the magnitude edges being proportional to the bond order (1 for single bond, 1.5 for aromatic systems, 2 for double bonds, 3 for triple bonds).
  
I shuffled this dataset and split the data into a test (20%) and train (80%) dataset.

### Network design:

I built a graph convolutional neural network (GCNN) to predict the wavelengths of absorption and emission from the SMILES strings of the chromophore and solvent. The general set-up of this GCNN follows that published by Joung et. al 2021. The diagram below shows the outline of the designed network. The features of the chromophore and solvent independently pass through two graph convolutional layers (message-passing layers). The arrays on the structures represent the passing of information from the neighboring atoms (nodes). The resulting matrix is summed over all atoms and passed through three fully-connected layers. The resulting vectors of the chromophore and solvent and then concatenated together and passed through two additional fully-connected layers, resulting in the two predicted values. ReLU activations were used between each layer, residual connections were implemented between each message-passing layer, and batch normalization was also implemented between each layer in the network.
  
<a href="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K"><img src="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K" style="width: 700px; max-width: 100%; height: auto" /></a>

### Use of existing codebase: GraphINVENT
Lastly, I used existing code to generate new chromophore-like structures. To do this, I used the SMILES strings from the dataset of chromophores to train the GraphINVENT network on. From my understanding, they implemented a gated graph neural network to "build" molecules atom-by-atom. I pre-trained the GraphINVENT GGNN for 30 epochs. With this pre-trained GGNN, I generated 5,000 new chromophore-like structures. My original plan was to then use the GraphINVENT code to "fine-tune" the network using a scoring function. I had re-defined the scoring function to use the GCNN that I built to give rewards to the network for structures that are expected to have a large wavelength of absorption. However, while running the fine-tuning stage, I repeatedly ran into CUDA memory issues. (After many hours of troubleshooting, I found that the only way to get around this was to decrease the number of molecules produced by the agent to only 4. With this limitation, I was not able to succeed in this goal of the project). However, I used the GCNN that I trained to evaluate the 5,000 structures generated with the pre-trained GGNN.
  
### Organization of Github repo
Because I used an existing codebase, GraphINVENT, for a portion of this project, I wanted to give an outline of the Github repo highlighting all of my contributions to the uploaded code. The files in italics were changed by me for use in this project. The files in bold were new files containing code that I wrote.


## Results
### GCNN to predict absorption and emission wavelengths
I trained the GCNN for 60 epochs with a learning rate of 0.01. Because I implemented batch normalization, I found that the optimization was way more efficient. The optimization converged smoothly with little overfitting of the model.

<a href="https://drive.google.com/uc?export=view&id=1wiy6vI7bJ9sb3QVCJoxv_abfpByHGPP_"><img src="https://drive.google.com/uc?export=view&id=1wiy6vI7bJ9sb3QVCJoxv_abfpByHGPP_" style="width: 350px; max-width: 100%; height: auto" /></a>
<a href="https://drive.google.com/uc?export=view&id=1GHIAiT43C79gYF_0JCM4FWaDbKVzkg01"><img src="https://drive.google.com/uc?export=view&id=1GHIAiT43C79gYF_0JCM4FWaDbKVzkg01" style="width: 350px; max-width: 100%; height: auto" /></a> 

The final model yielded a RMSE of 33.8 nm and 44.1 nm on the absorption and emission wavelengths, respectively. Below are the correlation plots exhibiting the predicted absorption and emission energies against the experimental absorption and emission energies for the test dataset.
  
<a href="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI"><img src="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI" style="width: 350px; max-width: 100%; height: auto" /></a>
<a href="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_"><img src="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_" style="width: 350px; max-width: 100%; height: auto" /></a> 
  
### Pre-training GGNN in GraphINVENT on chromophore database
I pre-trained the GGNN in the GraphINVENT codebase for 30 epochs total. 

### Using pre-trained GGNN to generate new chromophore-like structures
  
### Analysis of ~5,000 generated structures using the trained GCNN
  
## Discussion
  
The accuracy of the GCNN on predicting photochemical properties of the test dataset was quite good. An RMSE of 30-40 nm is comparable (and sometimes better) than common theoretical methods in chemistry, for example TD-DFT. However, the maximum predicted wavelength of absorbance on the new generated molecules was < 700 nm which makes me question the tranferability of this model to molecules with very low energy (high wavelength) excited state energies. In the future, augmenting the dataset with chromophores with absorption wavelengths between 700-1200 nm would be important to ensure the transferability of this model.

I was impressed with the performance of the GraphINVENT GGNN in generating valid molecules. By the end of pre-training, >..................were chemically-viable structures. That said, this method of building molecules from an empty graph was very successful. However, in the future, I would like to implement more constraints on the molecule generation so that molecules are also more easily synthesizable. For example, starting with part of an existing structure instead of an empty graph or constraining the generation so that the molecules will be build symmetrically.
  
Using the "fine-tuning" component of GraphINVENT required A LOT of GPU memory, making this part of my goal infeasible. However, I was able to successfully define my own component of the scoring function and use the GCNN to give feedback to the agent. 

Using code written by others on Github is a research skill that I wanted more experience with. However, this, of course, came with its own set of challenges. At many points, I had to debug the GraphINVENT code. Also, when I was running into CUDA memory issues in the fine-tuning stage, it was difficult to determine how to alter their code to remedy this issue. If it was code that I wrote, I think this process would have been easier. However, I think I learned a lot from this entire project and hope to continue working on this scientific problem in my own research.
  
## References:
1. Romeo Atance S, Viguera Diez J, Engkvist O, Olsson S, Mercado R. De novo drug design using reinforcement learning with graph-based deep generative models. ChemRxiv. Cambridge: Cambridge Open Engage; 2021
2. Joonyoung F. Joung, Minhi Han, Jinhyo Hwang, Minseok Jeong, Dong Hoon Choi, and Sungnam Park JACS Au 2021 1 (4), 427-438.

