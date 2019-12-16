# Variational Auto-Encoder (VAE) model
## This is the Variational Auto-Encoder model implemented by Ben in Pytorch, transfering from previous tf version 
# Developer Log:

### RoadMap for this work:
1. Identify the difference between the VAE model with the forward model
2. Implement them

## 2019.12.04
Function completed:
1. Parameter.py commented

## 2019.12.15
1. Model maker finished with 3 structure
2. Parameter section for reading 

## 2019.12.16
 Function completed/Changed:
 1. Separate structure abandoned since there is no second step training, combining the model would give better and easier saving and loading behavior
 2. kl_loss monitoring added, bdy loss monitoring
 3. parameters updated with dim z and dim spectra encoder
4 . Tested on CPU



# To-do list
1. Get the latent variable strucutre
2. Modify the train 
3. Modify the eval
4. Get the params clear
