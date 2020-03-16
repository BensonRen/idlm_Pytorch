import torch
from utils import plotsAnalysis
if __name__ == '__main__':
    #plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name='linear_complexity_heatmap.png',
    #                            HeatMap_dir='models/',feature_1_name='linear_b',feature_2_name='linear_unit')
    #plotsAnalysis.MeanAvgnMinMSEvsTry_all('/work/sr365/multi_eval')
    #plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='sine_wave')
    #plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='robotic_arm')
    plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='meta_material')

