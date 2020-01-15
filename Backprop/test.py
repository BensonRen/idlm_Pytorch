import torch
import plotsAnalysis
if __name__ == '__main__':
    plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name='linear_complexity_heatmap.png',
                                HeatMap_dir='models/',feature_1_name='linear',feature_2_name='linear_unit')
