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
#### Discriminator:
1. Takes Geometry as input, Spectra as the "class label" for conditioning
2. Output the score which is given by the forward model MSE for spectra
3. Takes half ground truth pairs and half generated pairs during training
#### Generator:
1. Takes random noise z as input, Spectra as the "class label" for conditioning information
2. Output generated geometry information

# To-do list
1. 