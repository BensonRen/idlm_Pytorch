import torch
from utils import plotsAnalysis

if __name__ == '__main__':
    #plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name='linear_complexity_heatmap.png',
    #                            HeatMap_dir='models/',feature_1_name='linear_b',feature_2_name='linear_unit')
    plotsAnalysis.MeanAvgnMinMSEvsTry_all('/work/sr365/multi_eval')
    plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='ballistics')
    #plotsAnalysis.DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval', data_name='sine_wave')
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





"""
    # Plotting the gaussian plots
    import pandas as pd
    from Simulated_DataSets.Gaussian_Mixture.generate_Gaussian import plotData
    import numpy as np
    import os
    data_dir = '/work/sr365/multi_eval'
    for dirs in os.listdir(data_dir):
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)):
            continue
        for subdirs in os.listdir(os.path.join(data_dir, dirs)):
            if 'gaussian' in subdirs:                          
                for subfiles in os.listdir(os.path.join(data_dir, dirs, subdirs)):
                    if 'inference0' not in subfiles:
                        continue;
                    if 'Ypred' in subfiles:
                        filename = os.path.join(data_dir, dirs, subdirs, subfiles)
                        data_y = pd.read_csv(filename, header=None, sep=' ').values.astype('float')
                        if 'Backprop' in dirs:
                            data_y = np.argmax(data_y,axis=1)
                        data_y = np.ravel(data_y)
                        print("shape of data_y", np.shape(data_y))
                    if 'Xpred' in subfiles:
                        filename = os.path.join(data_dir, dirs, subdirs, subfiles)
                        data_x = pd.read_csv(filename, header=None, sep=' ').values
                        print("shape of data_x", np.shape(data_x))
                plotData(data_x, data_y, save_dir=dirs+'generated_gaussian_inference0.png',eval_mode=True)
"""
