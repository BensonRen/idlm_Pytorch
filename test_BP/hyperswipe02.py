"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    #linear_unit_list = [ 50, 100, 200, 500]
    linear_unit_list = [500]
    #linear_unit_list = [300, 150, 50]
    #reg_scale_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 5, 10]
    reg_scale_list = [20, 50, 100, 200, 1000]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(5,6):
            flags = flag_reader.read_flag()  	#setting the base case
            linear = [linear_unit for j in range(i)]        #Set the linear units
            linear[0] = 1                   # The start of linear
            linear[-1] = 1                # The end of linear
            flags.linear = linear
            for reg_scale in reg_scale_list:
                flags.reg_scale = reg_scale
                for j in range(1):
                        flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_complexity_swipe_layer" + str(linear_unit) + "_num" + str(i)
                        train.training_from_flag(flags)

