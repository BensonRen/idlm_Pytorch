import torch
import sys
sys.path.append('../utils/')
from utils import plotsAnalysis
if __name__ == '__main__':
    pathname = ''
    plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
                                HeatMap_dir='models/'+pathname,feature_1_name='linear_b',feature_2_name='linear_unit')
