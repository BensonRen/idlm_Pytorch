# Forward Model
## This is the forward model created by Chrisitan Nadell and Bohao Huang. This is merely an pytorch implementation of the same model (with some parameters slightly different)

# Developer Log:

### RoadMap for this work:
1. Transition the major infra-structure from tensorflow to pytorch
2. Implement a simple forward network for meta-material project
3. Implement the GAN model that was not available in tensorflow version

## 2019.11.20
Background: Start Pytorch transition. GAN model (not in previous tf repo) would be the first model to work on.

Function completed:
1. Construct the network main class wrapper for Forward
2. Handle the data reader transition to pytorch
3. Flag reader to pytorch version

## 2019.11.21

Function completed/Bug Fixed:
1. Forward training module done
2. Forward training tested
3. Bug Fixed for storing the parameters.txt file

## 2019.11.23

Function completed:
1. Forward inference module

## 2019.11.24

Function completed:
1. GPU training on Tesla
2. Create a new repo for pytorch version
3. comments added

## 2019.11.25
Function completed:
1. Learning rate scheduler (Decay when training error is plateaued)

## 2019.11.30
Function completed:
1. Pickle the flag object for further retrieval in flag_reader module
2. Comments furnished

## 2019.12.01
Function completed:
1. Raw model saving issue solved using torch
2. Evaluation part finished
3. Evaluation tested on GPU
4. Hyper swiping module finished and tested

## 2019.12.02
Function Added:
1. Omar added lorentzian module to model maker
2. Option created for training with lorentz or not, controlled by flag.use_lorentz
3. Omar code debugged with type error (torch.pow only works for tensors)
4. Plotting of the MSE loss distribution at the end of evaluation

## 2019.12.03
Function Added:
1. Parallelizing Omar's Lorentz oscillat
or, code finished but NAN Error now

## 2019.12.04
Function added:
1. NAN error bug fixed with adding sigmoid activation function

## 2019.12.05
Updated with parameter.py formatting with comments

## 2019.12.09
Function Added:
1. Intermediate plotting function output to tensorboard for training debugging. (Yet to test on GPU machine)
2. GPU test passed
3. histogram function done for lorentzian parameters

## 2019.12.10
Function added:
1. plot the individual Lorentzian oscillators
2. debugging the histogram setting
3. Set the range of Lorentzian parameters to be [0,5]
4. Add the epsilon_inf to the function and add to the E1 calculation (deactivated)

## 2019.12.12
Function changed:
1. Lorentzian calculation bug fixed for aggregating oscillators from eps step instead of Transmission step
2. Trainable parameter for epsilon_inf activated

## To dos:
1. Tune the model to Christian's accuracy (1.2e-3) using GPU


