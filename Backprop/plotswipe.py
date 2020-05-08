import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['meta_kernel_swipe']
    for pathname in pathnamelist:
        plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
                                HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
