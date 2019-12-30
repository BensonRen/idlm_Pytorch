# Conditional Generative Adversarial Network (cGAN) model
## This is the cGAN model implemented by Ben in Pytorch, transfering from previous tf version 
# Developer Log:

### RoadMap for this work:
1. Implement from scratch the GAN model

## 2019.12.18
1. Ground work laid from Tandem model which also have 2 modules

## 2019.12.29
### CGAN model reworked. New model goes like follows:
#### Forward_model:
1. Pre-trained with high accuracy
2. Takes the Geometry and outputs spectra like simulator
3. This module has both linear and upconv modules
#### Discriminator:
1. Takes Geometry as input, Spectra as the "class label" for conditioning
2. Output the score which is given by the forward model MSE for spectra
3. Takes half ground truth pairs and half generated pairs during training
4. This module has only linear, no conv
#### Generator:
1. Takes random noise z as input, Spectra as the "class label" for conditioning information
2. Output generated geometry information
3. This module has only liner, no conv
#### Spectra_Encoder:
1. Encode the spectra 300 dimension into lower dimension using conv and linear
2. This module has both linear and upconv modules

## 2019.12.30


# To-do list
1. Finish the overall structure
2. Training function 