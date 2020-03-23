import torch
from utils import plotsAnalysis
if __name__ == '__main__':
    #plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name='linear_complexity_heatmap.png',
    #                            HeatMap_dir='models/',feature_1_name='linear_b',feature_2_name='linear_unit')
    plotsAnalysis.MeanAvgnMinMSEvsTry_all('/work/sr365/multi_eval')
    plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='sine_wave')
    #plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='robotic_arm')
    #plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='meta_material')



     #plotsAnalysis.PlotPossibleGeoSpace("Diversity_meta_material_backprop", "/work/sr365/Diversity/Backprop/", compare_original=True, calculate_diversity='AREA') 
     #plotsAnalysis.PlotPossibleGeoSpace("Diversity_meta_material_Tandem", "/work/sr365/Diversity/Tandem/", compare_original=True, calculate_diversity='AREA') 
     #plotsAnalysis.PlotPossibleGeoSpace("Diversity_meta_material_cINN", "/work/sr365/Diversity/cINN/", compare_original=True, calculate_diversity='AREA') 
     #plotsAnalysis.PlotPossibleGeoSpace("Diversity_meta_material_VAE", "/work/sr365/Diversity/VAE/", compare_original=True, calculate_diversity='AREA') 

    #plotsAnalysis.DrawEvaluationTime('/work/sr365/time_evaluation/', data_name='sine_wave')
    #plotsAnalysis.DrawEvaluationTime('/work/sr365/time_evaluation/', data_name='gaussian')
    #plotsAnalysis.DrawEvaluationTime('/work/sr365/time_evaluation/', data_name='robotic_arm')
    #plotsAnalysis.DrawEvaluationTime('/work/sr365/time_evaluation/', data_name='meta_material')
