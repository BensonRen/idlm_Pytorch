import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['']
    for pathname in pathnamelist:
        plotsAnalysis.HeatMapBVL(plot_x_name='dim_z', plot_y_name='dim_spec', title='dim_z vs dim_spec Heat Map',
                                 save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,
                                 feature_1_name='dim_z', feature_2_name='dim_spec')