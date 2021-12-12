# ChromoOpt: Final project for CSE490G1

As the world turns to clean energy alternatives, many scientists look to the sun to harvest energy. The sun emits a wide range of wavelengths but not all wavelengths are being absorbed by chromophores, meaning that we are not harvesting all the sun’s energy. The goal of my final project was to generate new, viable molecules with specific photochemical properties. To accomplish this, I started with the GraphINVENT codebase (https://github.com/MolecularAI/GraphINVENT) which was used by the authors to design drugs with certain properties. I re-trained the generator using a database of chromophores, redefined the scoring function to include important photochemical properties, and built and trained a graph convolutional neural network (GCNN) to predict the photochemical properties defined in the scoring function.

## Introduction
A wide range of electromagnetic waves are emitted from the sun, ranging from ~250 nm to ~2500 nm. Much progress in solar cells has been focused on the visible range of the electromagnetic spectrum. However, as is clear from the figure below, this comprises a relatively small fraction of the total emissions from the sun. To better utilize the entire spectrum of light from the sun, we need chromophores (molecules that absorb light) that absorb in the near IR and IR range of the electromagnetic series. However, finding molecules with specific qualities is a well-established problem in chemistry. In recent years, the application of deep learning to this problem have proven fruitful in drug design (CITE). These methods have not yet been widely used in photochemistry and chromophore design. To tackle this problem, I have used an existing codebase, GraphINVENT (), to generate chromophore-like molecules and used a graph convolutional neural network to provide a score for these generated molecules.

<a href="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS"><img src="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS" style="width: 500px; max-width: 100%; height: auto" />

## Related Work

## Approach
### Dataset:
The database that I used is comprised of >30,000 chromophore/solvent pairings and 7 of their experimental photochemical properties (I only focused on the absorption and emission wavelengths). (CITE) Because we are interested in heavy-atom-free chromophores, I removed all compounds that contained elements other than H, O, C, N, S, and Cl. By removing these entries and also entries that contained “NaN”, the final dataset was reduced to ~12,000 entries. Here is an example showing 5 lines of the dataset:

<a href="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs"><img src="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs" style="width: 500px; max-width: 100%; height: auto" />
  
The chromophore and solvent are represented by SMILES strings. This notation is used to denote the chemical structure in a compact way. From the SMILES strings, I used RDkit (CITE) to compute features at each atom site (i.e., atom name, hybridization, number of atoms bonded to it, aromaticity, etc.). This information was then one-hot encoded to yield a feature matrix. The adjacency matrix was also produced using RDkit with the edges being proportional to the bond order.
  
I shuffled this dataset and split the data into a test (20%) and train (80%) dataset.

### Network design:

I built a graph convolutional neural network (GCNN) to predict the wavelengths of absorption and emission from the SMILES strings of the chromophore and solvent. The general set-up of this GCNN follows that published by Joung et. al 2021. The diagram below shows the outline of the designed network. The features of the chromophore and solvent independently pass through two graph convolutional layers (message-passing layers). The arrays on the structures represent the passing of information from the neighboring atoms (nodes). The resulting matrix is summed over all atoms and passed through three fully-connected layers. The resulting vectors of the chromophore and solvent and then concatenated together and passed through two additional fully-connected layers, resulting in the two predicted values. ReLU activations were used between each layer and batch normalization was also implemented in the network.
  
<a href="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K"><img src="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K" style="width: 700px; max-width: 100%; height: auto" />

### Use of existing codebase: GraphINVENT
Lastly, I used existing code to generate new chromophore-like structures. To do this, I used the SMILES strings from the dataset of chromophores to train the GraphINVENT network on. From my understanding, they implemented a gated graph neural network to "build" molecules atom-by-atom. I pre-trained the GraphINVENT GGNN for 30 epochs. With this pre-trained GGNN, I generated 5,000 new chromophore-like structures. My original plan was to then use the GraphINVENT code to "fine-tune" the network using a scoring function. I had re-defined the scoring function to use the GCNN that I built to give rewards to the network for structures that are expected to have a large wavelength of absorption. However, while running the fine-tuning stage, I repeatedly ran into CUDA memory issues. (After many hours of troubleshooting, I found that the only way to get around this was to decrease the number of molecules produced by the agent to only 4. With this limitation, I was not able to succeed in this goal of the project). However, I used the GCNN that I trained to evaluate the structures generated with the pre-trained GGNN.

## Results
Below are the correlation plots exhibiting the predicted absorption and emission energies against the experimental absorption and emission energies for the test dataset.
  
<a href="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI"><img src="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI" style="width: 350px; max-width: 100%; height: auto" />  
<a href="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_"><img src="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_" style="width: 350px; max-width: 100%; height: auto" />  

Using batch normalization greatly improved the optimization process, shown by the train and test loss curves.
  
Results from GraphINVENT
  
Analysis of generated structures
  
## Discussion
  
  
  
