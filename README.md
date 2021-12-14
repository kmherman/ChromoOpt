# ChromoOpt: Final project for CSE490G1
### *Using DL to generate new chromophore structures with specific photochemical properties*

As the world turns to clean energy alternatives, many scientists look to the sun to harvest energy. The sun emits a wide range of wavelengths but not all wavelengths are being absorbed by chromophores. An overlooked area is the near IR and IR region which comprises the largest area of the emitted spectrum from the sun (despite being the lowest energy photons within the spectrum). In order to absorb this energy, we must first design new chromophores with specific photochemical properties in mind (i.e., wavelength of absorbance in the near IR region). The goal of my final project was chip away at this scientific problem. The goal of my final project was to generate new, viable molecules with specific photochemical properties. To accomplish this, I started with the GraphINVENT codebase (https://github.com/MolecularAI/GraphINVENT) which was used by the authors to design drugs with certain properties.<sup>1</sup> I re-trained the generator using a database of chromophores, redefined the scoring function to include important photochemical properties, and built and trained a graph convolutional neural network (GCNN) to predict the photochemical properties<sup>2</sup> defined in the scoring function. Due to GPU memory constraints, I was not able to successfully train the agent within GraphINVENT. Despite this roadblock, I used the pre-trained network to generate thousands of new chromophore structures and rank them based on their predicted wavelength of absorption.

[![Watch video](https://img.youtube.com/vi/1PUtflaZROY/0.jpg)](https://www.youtube.com/watch?v=1PUtflaZROY)

(This is a little long but wanted to get the relevant info in)

## Introduction
A wide range of electromagnetic waves are emitted from the sun, ranging from ~250 nm to ~2500 nm. Much progress in solar cells has been focused on the visible range of the electromagnetic spectrum. However, as is clear from the figure below, this comprises a relatively small fraction of the total emissions from the sun. To better utilize the entire spectrum of light from the sun, we need chromophores (molecules that absorb light) that absorb in the near IR and IR range of the electromagnetic series. However, finding molecules with specific qualities is a well-established problem in chemistry. In recent years, the application of deep learning to this problem have proven fruitful in drug design. Surprisingly, these methods have not yet been widely used in photochemistry and chromophore design. One reason for this is likely due to the difficulty of predicting photochemical properties of molecules. Joung et. al 2021<sup>2</sup> recently showed the efficacy of graph convolutional neural networks (GCNN) in doing that. To tackle this problem, I have used an existing codebase, GraphINVENT (https://github.com/MolecularAI/GraphINVENT), to generate chromophore-like molecules and built and trained my own graph convolutional neural network to provide a score for these generated molecules based on their predicted wavelengths of absorption.

<a href="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS"><img src="https://drive.google.com/uc?export=view&id=1BwN8vKHTFqkVwXcM4eweX1lQSIjTs9IS" style="width: 500px; max-width: 100%; height: auto" /></a>

## Related Work
The use of deep learning for drug design is prevalent in the literature. These have included reinforcement learning (RL) methods (Actor-critic, Q-learning, DQN) and a combination of generative adversarial networks (GANs) with RL methods.<sup>7</sup> These methods have been shown to produce numerous drug candidates with specific features such as activity, lipophilicity, and molecules size, for example.<sup>1,8</sup> One major hurdle that must be overcome is ensuring that the generation of molecules yields chemically-viable structures.

Many researchers have turned to generative adversarial networks (GANs) to address this issue, leading to the development of networks such as MolGAN<sup>5</sup> and ORGAN<sup>7</sup>. The challenge of training GANs is the issue of mode collapse and unstable training<sup>6</sup> (as we discussed in class). This often limits the generation to relatively small, simple molecules. A recently developed codebase, GraphINVENT, has shown the success of gated graph neural networks (GGNNs) in building molecules up bond-by-bond and *implicitly* learning chemistry bonding rules by pre-training the deep generative model on a database of SMILES strings of molecules. Conversely, others have carefully and explicitly defined alterations that can be made to a molecule in order to ensure chemical validity. By using one of these methods and implementing RL, researchers have successfully been able to bias the generation of molecules towards those with specific properties for use as drugs.

While these methods ahve been widely used in drug design, a similar approach has not yet been implemented to design chromophores. By simply changing the scoring function of these reinforcement learning (RL) methods, we can bias the generation of molecules towards those with desired photochemical properties.

## Approach
### Dataset:
The database is comprised of >30,000 chromophore/solvent pairings and 7 of their experimental photochemical properties (I only focused on the absorption and emission wavelengths).<sup>3</sup> Because we are interested in heavy-atom-free chromophores, I removed all compounds that contained elements other than H, O, C, N, S, and Cl. By removing these entries and also entries that contained “NaN”, the final dataset was reduced to ~12,000 entries. Here is an example showing 5 lines of the dataset:

<a href="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs"><img src="https://drive.google.com/uc?export=view&id=1oDzTtuT0LS0VVhXTSPjrxf7U5QoieDBs" style="width: 500px; max-width: 100%; height: auto" /></a>
  
The chromophore and solvent are represented by SMILES strings. This notation is used to denote the chemical structure in a compact way. Here is an example of a SMILES string and the corresponding chemical structure.

<a href="https://drive.google.com/uc?export=view&id=1r-ApHNsG6ty2tvOK39cF4_WhVIhiQMfW"><img src="https://drive.google.com/uc?export=view&id=1r-ApHNsG6ty2tvOK39cF4_WhVIhiQMfW" style="width: 300px; max-width: 100%; height: auto" /></a>
  
From the SMILES strings, I used RDkit<sup>4</sup> to extract chemical features at each atom site (i.e., atom name, hybridization, number of atoms bonded to it, aromaticity, etc.). This information was then one-hot encoded to yield a feature matrix. The adjacency matrix was also produced using RDkit with the magnitude edges being proportional to the bond order (1 for single bond, 1.5 for aromatic systems, 2 for double bonds, 3 for triple bonds).
  
I shuffled this dataset and split the data into a test (20%) and train (80%) dataset.

### Network design:

I built a graph convolutional neural network (GCNN) to predict the wavelengths of absorption and emission from the SMILES strings of the chromophore and solvent. The general set-up of this GCNN follows that published by Joung et. al 2021. The diagram below shows the outline of the designed network. The features of the chromophore and solvent independently pass through two graph convolutional layers (message-passing layers). The arrays on the structures represent the passing of information from the neighboring atoms (nodes). The resulting matrix is summed over all atoms and passed through three fully-connected layers. The resulting vectors of the chromophore and solvent and then concatenated together and passed through two additional fully-connected layers, resulting in the two predicted values. ReLU activations were used between each layer, residual connections were implemented between each message-passing layer, and batch normalization was also implemented between each layer in the network.
  
<a href="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K"><img src="https://drive.google.com/uc?export=view&id=1NbsMyc58zjQWEEe36BHf54aLdjpHrt9K" style="width: 700px; max-width: 100%; height: auto" /></a>

### Use of existing codebase: GraphINVENT
Lastly, I used existing code to generate new chromophore-like structures. To do this, I used the SMILES strings from the dataset of chromophores to train the GraphINVENT network on. From my understanding, they implemented a gated graph neural network to "build" molecules atom-by-atom. I pre-trained the GraphINVENT GGNN for 30 epochs. With this pre-trained GGNN, I generated 5,000 new chromophore-like structures. My original plan was to then use the GraphINVENT code to "fine-tune" the network using a scoring function. I had re-defined the scoring function to use the GCNN that I built to give rewards to the network for structures that are expected to have a large wavelength of absorption. However, while running the fine-tuning stage, I repeatedly ran into CUDA memory issues. (After many hours of troubleshooting, I found that the only way to get around this was to decrease the number of molecules produced by the agent to only 4. With this limitation, I was not able to succeed in this goal of the project). However, I used the GCNN that I trained to evaluate the 5,000 structures generated with the pre-trained GGNN.
  
### Organization of Github repo
Because I used an existing codebase, GraphINVENT, for a portion of this project, I wanted to give an outline of the Github repo highlighting all of my contributions to the uploaded code. The files in italics were altered by me for use in this project. The files in bold were new files containing code written solely by me.

<a href="https://drive.google.com/uc?export=view&id=16iyBwaSElXVbYu2RF5l3oMGu4q_bRsP2"><img src="https://drive.google.com/uc?export=view&id=16iyBwaSElXVbYu2RF5l3oMGu4q_bRsP2" style="width: 300px; max-width: 100%; height: auto" /></a>


## Results
### GCNN to predict absorption and emission wavelengths
I trained the GCNN for 60 epochs with a learning rate of 0.01. Because I implemented batch normalization, I found that the optimization was way more efficient. The optimization converged smoothly with little overfitting of the model.

<a href="https://drive.google.com/uc?export=view&id=1wiy6vI7bJ9sb3QVCJoxv_abfpByHGPP_"><img src="https://drive.google.com/uc?export=view&id=1wiy6vI7bJ9sb3QVCJoxv_abfpByHGPP_" style="width: 350px; max-width: 100%; height: auto" /></a>
<a href="https://drive.google.com/uc?export=view&id=1GHIAiT43C79gYF_0JCM4FWaDbKVzkg01"><img src="https://drive.google.com/uc?export=view&id=1GHIAiT43C79gYF_0JCM4FWaDbKVzkg01" style="width: 350px; max-width: 100%; height: auto" /></a> 

The final model yielded a RMSE of 33.8 nm and 44.1 nm on the absorption and emission wavelengths, respectively. Below are the correlation plots exhibiting the predicted absorption and emission energies against the experimental absorption and emission energies for the test dataset.
  
<a href="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI"><img src="https://drive.google.com/uc?export=view&id=1QySpwIftbtpF3Ibu1qLH57LlefyuV3DI" style="width: 350px; max-width: 100%; height: auto" /></a>
<a href="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_"><img src="https://drive.google.com/uc?export=view&id=1_21fVBflZjZ78bOWHA2oRouo4GqrpJu_" style="width: 350px; max-width: 100%; height: auto" /></a> 
  
### Pre-training GGNN in GraphINVENT on chromophore database
I pre-trained the GGNN in the GraphINVENT codebase on the SMILES strings from the chromophore database for 30 epochs total. By using this database, I hoped that the network would learn to produce chemically-viable structures and also structures that resembled that of other existing chromophores. The training curve (below) steadily decreases with increasing epochs whereas the validation loss stays roughly constant after ~15 epochs.

<a href="https://drive.google.com/uc?export=view&id=1uIQEFHawM-8Iq6H0dF8XhGcZZcMBz95k"><img src="https://drive.google.com/uc?export=view&id=1uIQEFHawM-8Iq6H0dF8XhGcZZcMBz95k" style="width: 350px; max-width: 100%; height: auto" /></a>

In the first few epochs, we have the smallest fraction of unique and valid molecules. However, we see that the fraction of unique and valid molecules gradually increases throughout training. During the last half of training (epochs 15-30), we see high values for the validity of the structures and the uniqueness of the generated structures.

<a href="https://drive.google.com/uc?export=view&id=1cXXF5MbU1N8he7OqkchByIn5to7CzV3r"><img src="https://drive.google.com/uc?export=view&id=1cXXF5MbU1N8he7OqkchByIn5to7CzV3r" style="width: 350px; max-width: 100%; height: auto" /></a>
<a href="https://drive.google.com/uc?export=view&id=1wX_IJLzE7pop-VZxccE5CQS3OIx6Ow5c"><img src="https://drive.google.com/uc?export=view&id=1wX_IJLzE7pop-VZxccE5CQS3OIx6Ow5c" style="width: 350px; max-width: 100%; height: auto" /></a>

As the training progresses, the network generates larger, more complex molecules as exhibited by the figure below. The number of nodes (heavy atoms) steadily increases during training.

<a href="https://drive.google.com/uc?export=view&id=1euQl6AhG_sNtoA_MBCS9BuIFy_6BFE7c"><img src="https://drive.google.com/uc?export=view&id=1euQl6AhG_sNtoA_MBCS9BuIFy_6BFE7c" style="width: 350px; max-width: 100%; height: auto" /></a>

In the first few epochs, most of the molecules were very simple. A few examples are shown below. These first structures are the "building blocks" for some larger molecules.

<a href="https://drive.google.com/uc?export=view&id=1Z6vlvvu59kB0F_Dza035XHdF_slHsQzq"><img src="https://drive.google.com/uc?export=view&id=1Z6vlvvu59kB0F_Dza035XHdF_slHsQzq" style="width: 400px; max-width: 100%; height: auto" /></a>

After 30 epochs, the GGNN was capable of producing complex and still chemically-valid structures. It was impressive to see certain functional groups (ethers, carboxylic acids, benzene rings, amides, etc.) appear by the end of the training. Some of these structures are technically valid but likely unstable. For example, 3-membered rings and very large rings would not be stable.

<a href="https://drive.google.com/uc?export=view&id=1lBAycOrur66k2ESqNsIV_qwO4m-RagO6"><img src="https://drive.google.com/uc?export=view&id=1lBAycOrur66k2ESqNsIV_qwO4m-RagO6" style="width: 550px; max-width: 100%; height: auto" /></a>

### Using pre-trained GGNN to generate new chromophore-like structures and analyzing them with trained GCNN
Using the pre-trained GGNN, I generated ~5,000 new chromophore-like structures. In an attempt to find new chromophore structures with high wavlengths of absorption, I used the trained GCNN to predict the wavelength of absorption for each of the molecules.

The top 5 chromophore structures are shown with their corresponding predicted excitation energies. The fact that many of these structures are large with a high amount of conjugation is what we would expect conceptually. However, the highest predicted wavelength is 608 nm which is not yet in the near IR or IR. Because the dataset that the GCNN and GGNN were trained on consisted of chromophores that absorb light between 300 nm-600 nm, this somewhat makes sense.

<a href="https://drive.google.com/uc?export=view&id=1T1f7vpG92930hUQmsLwNpw2IqEBQs3ab"><img src="https://drive.google.com/uc?export=view&id=1T1f7vpG92930hUQmsLwNpw2IqEBQs3ab" style="width: 700px; max-width: 100%; height: auto" /></a>

The 5 chromophores with the lowest predicted wavlength of excitation are shown below.

<a href="https://drive.google.com/uc?export=view&id=1j-VRmM-rm4G7GBH3H4HHM7AsPMwtzMb6"><img src="https://drive.google.com/uc?export=view&id=1j-VRmM-rm4G7GBH3H4HHM7AsPMwtzMb6" style="width: 600px; max-width: 100%; height: auto" /></a>

Overall, the molecules with the highest predicted wavelength of absorption tend to be larger molecules with larger amounts of conjugation. In general, this is expected. However, some of the molecules that are predicted to have a low wavelength of absorption are still relatively large. In addition, the range of predicted wavelengths is only ~200 nm. This is a relatively small range given that there were ~5,000 generated molecules.
  
  
## Discussion
  
The accuracy of the GCNN on predicting photochemical properties of the test dataset was quite good. An RMSE of 30-40 nm is comparable (and sometimes better) than common theoretical methods in chemistry, for example TD-DFT. However, the maximum predicted wavelength of absorbance on the new generated molecules was < 700 nm which makes me question the tranferability of this model to molecules with very low energy (high wavelength) excited state energies. In the future, augmenting the dataset with chromophores with absorption wavelengths between 700-1200 nm would be important to ensure the transferability of this model.

I was impressed with the performance of the GraphINVENT GGNN in generating valid molecules. By the end of pre-training, nearly all of the generated structures were chemically-viable structures. That said, this method of building molecules from an empty graph was very successful. However, in the future, I would like to implement more constraints on the molecule generation so that molecules are also more easily synthesizable. For example, starting with part of an existing structure instead of an empty graph or constraining the generation so that the molecules will be build symmetrically.
  
Using the "fine-tuning" component of GraphINVENT required A LOT of GPU memory, making this part of my goal infeasible. However, I was able to successfully define my own component of the scoring function and use the GCNN to give feedback to the agent. In the future, I think it would be helpful for me to experiment with different reinforcement learning algorithms (for example, DQN) to find an algorithm that may be more computationally feasible. Further, because I have a general chromophore structure in mind, I think explicitly defining allowed actions would make sense, given the constraints I would like to impose on the generation.

Using code written by others on Github is a research skill that I wanted more experience with. However, this, of course, came with its own set of challenges. At many points, I had to debug the GraphINVENT code. Also, when I was running into CUDA memory issues in the fine-tuning stage, it was difficult to determine how to alter their code to remedy this issue. If it was code that I wrote, I think this process would have been easier. Overvall, I think I learned a lot from this entire project and am really happy with the progress I made.
  
## References:
1. Romeo Atance S, Viguera Diez J, Engkvist O, Olsson S, Mercado R. De novo drug design using reinforcement learning with graph-based deep generative models. ChemRxiv. Cambridge: Cambridge Open Engage; 2021
2. Joonyoung F. Joung, Minhi Han, Jinhyo Hwang, Minseok Jeong, Dong Hoon Choi, and Sungnam Park JACS Au 2021 1 (4), 427-438.
3. Joung, J.F., Han, M., Jeong, M. et al. Experimental database of optical properties of organic compounds. Sci Data 7, 295 (2020).
4. RDKit: Open-source cheminformatics. https://www.rdkit.org
5. De Cao, N.; Kipf, T. MolGAN: An Implicit Generative Model for Small Molecular Graphs. arXiv:1805.11973 [cs, stat] 2018
6. Lin, E.; Lin, C.-H.; Lane, H.-Y. Relevant Applications of Generative Adversarial Networks in Drug Design and Discovery: Molecular De Novo Design, Dimensionality Reduction, and De Novo Peptide and Protein Design. Molecules 2020, 25, 3250
7. Guimaraes, G. L.; Sanchez-Lengeling, B.; Outeiral, C.; Farias, P. L. C.; Aspuru-Guzik, A. Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models. arXiv:1705.10843 [cs, stat] 2018.
8. Zhou, Z.; Kearnes, S.; Li, L.; Zare, R. N.; Riley, P. Optimization of Molecules via Deep Reinforcement Learning. Sci Rep 2019, 9 (1), 10752. https://doi.org/10.1038/s41598-019-47148-x.
