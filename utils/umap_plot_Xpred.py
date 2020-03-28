#!/usr/bin/env python
# coding: utf-8

# In[1]:


import umap as umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# ### Read the Xpred files from a folder

# In[2]:


def extract_data(data_dir='.'):
    # Extract the data
    Xpred_dict = {}
    label_dict = {}
    current_label = 0
    label_dictionary  = []
    for dirs in os.listdir(data_dir):
        print(dirs)
        print("Is this is directory?:", os.path.isdir(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)) or '.' in dirs:
            print("This is not a directory")
            continue
        Xpred_comma = pd.read_csv(os.path.join(data_dir, dirs,'Xpred.csv'), 
                           header=None, delimiter=',').values
        Xpred_space = pd.read_csv(os.path.join(data_dir, dirs,'Xpred.csv'), 
                           header=None, delimiter=' ').values
        if np.shape(Xpred_comma)[1] > np.shape(Xpred_space)[1]:
            Xpred = Xpred_comma
        else:
            Xpred = Xpred_space
        l,w = np.shape(Xpred)
        label = current_label * np.ones([l,])
        label_dictionary.append(dirs)
        Xpred_dict[dirs] = Xpred
        label_dict[dirs] = label
        current_label += 1
    return Xpred_dict, label_dict, label_dictionary


# ### Put all the labels and Xpred into a huge numpy array

# In[3]:


def get_all_Xpred_label(Xpred_dict, label_dict):
    all_Xpred = None
    for data in Xpred_dict.values():
        print(np.shape(data))
        if all_Xpred is None:
            all_Xpred = data
        else:
            all_Xpred = np.concatenate([all_Xpred, data], axis=0)
    print("They shape of overall data", np.shape(all_Xpred))

    all_label = None
    for data in label_dict.values():
        print(np.shape(data))
        if all_label is None:
            all_label = data
        else:
            all_label = np.concatenate([all_label, data], axis=0)
    print("They shape of overall data", np.shape(all_label))
    return all_Xpred, all_label


# ### do Umap clustering

# In[4]:


def Umap(all_Xpred):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_Xpred)
    embedding.shape
    return embedding


# ### Plotting of the different clusters

# In[5]:


def plotall(embedding, all_label, label_dictionary, plot_name):
    input_x = np.arange(len(label_dictionary))[label_dictionary == 'Input'] # get the label of the 
    # Draw each vs all other classes
    for i in range(len(label_dictionary)):
        f = plt.figure(figsize=[20,20])
        plt.scatter(embedding[all_label==i, 0], embedding[all_label==i, 1], 
                    label='{}'.format(label_dictionary[i]),s=10)
        plt.scatter(embedding[all_label==i, 0], embedding[all_label==i, 1], 
                    label='other points',s=1)
        plt.legend()
        plt.title(plot_name + '{} scatter plot after umap for all'.format(label_dictionary[i]))
        plt.savefig(plot_name + label_dictionary[i] + 'scatter.png')
    
    for i in range(len(label_dictionary)):
        f = plt.figure(figsize=[20,20])
        plt.scatter(embedding[all_label==i, 0], embedding[all_label==i, 1], 
                    label='{}'.format(label_dictionary[i]),s=10)
        plt.scatter(embedding[all_label==input_x, 0], embedding[all_label==input_x, 1], 
                    label='input distribution',s=1)
        plt.legend()
        plt.title(plot_name + '{} scatter plot after umap for all vs input distribution'.format(label_dictionary[i]))
        plt.savefig(plot_name + label_dictionary[i] + 'vs input scatter.png')

    # Plot the total plot
    f = plt.figure(figsize=[20,20])
    for i in range(len(label_dictionary)):
        plt.scatter(embedding[all_label==i, 0], 
                    embedding[all_label==i, 1], 
                    label='{}'.format(label_dictionary[i]),s=1)
    plt.legend()
    plt.title(plot_name + "scatter plot for all")
    plt.savefig(plot_name + 'all scatter.png')


# ### Main function to run

# In[ ]:


datalist = ['sine_wave', 'robotic_arm','meta_material']
for dataset in datalist:
    Xpred_dict, label_dict, label_dictionary = extract_data(os.path.join('/work/sr365/Diversity',dataset))
    all_Xpred, all_label = get_all_Xpred_label(Xpred_dict, label_dict)
    embedding = Umap(all_Xpred)
    plotall(embedding, all_label, label_dictionary, dataset)


# In[ ]:




