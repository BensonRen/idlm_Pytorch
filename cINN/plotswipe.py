import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['sine_wave']
    for pathname in pathnamelist:
        plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','couple_layer_num Heat Map',save_name=pathname + '_heatmap.png',
                                HeatMap_dir='models/'+pathname,feature_1_name='couple_layer_num',feature_2_name='dim_z')
