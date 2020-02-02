import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['robotic_arm']
    for pathname in pathnamelist:
        plotsAnalysis.HeatMapBVL(plot_x_name='dim_z', plot_y_name='linear_d_layer', title='dim_z vs linear_d_layer Heat Map',
                                 save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,
                                 feature_1_name='dim_z', feature_2_name='linear_d')
        #plotsAnalysis.HeatMapBVL(plot_x_name='linear_d_layer', plot_y_name='linear_unit', title='linear_d_layer vs linear_unit Heat Map',
        #                         save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,
        #                         feature_1_name='linear_d', feature_2_name='linear_unit')
