import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['robotic_kl_lr']
    for pathname in pathnamelist:
        plotsAnalysis.HeatMapBVL(plot_x_name='learn_rate', plot_y_name='kl_coefficient', title='kl_coeff vs lr Heat Map',
                                 save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,
                                 feature_1_name='learn_rate', feature_2_name='kl_coeff')
        #plotsAnalysis.HeatMapBVL(plot_x_name='linear_d_layer', plot_y_name='linear_unit', title='linear_d_layer vs linear_unit Heat Map',
        #                         save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,
        #                         feature_1_name='linear_d', feature_2_name='linear_unit')
