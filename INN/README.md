# Invertible Neural Network model
## This is the Invertible Network model implemented by Ben in Pytorch, transfering from previous tf version 
## Reference and credit to original Paper: 
### Analyzing Inverse Problems with Invertible Neural Networks, Lynton Ardizzone et.al.
# Developer Log:

### RoadMap for this work:
1. Implement the Invertible Layers
2. Network Structure
3. Evaluation phase

## 2020.01.07
1. Directory created and code infrastructure copied from Back propagation

## 2020.01.09
1. Coupling layer finished
2. parameter flags and flag reader updated

## 2020.01.10
1. INN training structure outline finished. Loss left undefined

## 2020.01.13
1. Auto Encoder structure finished
2. Auto Encoder trianing module finihsed
3. Auto Encoder tunning on GPU machine

## To-do list
1. Implement the maximum mean discrepancy for loss
2. INN model connection using coupling layer
3. Dealing with convolution w.r.t. coupling layer
