"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    linear_unit_list = [5, 10, 15, 20, 25]
    #linear_unit_list = [1000, 500]
    #linear_unit_list = [1000, 500, 300, 150]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    reg_scale_list = [5e-3, 5e-4]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(4, 10):
            flags = flag_reader.read_flag()  	# setting the base case
            linear = [linear_unit for j in range(i)]        # Set the linear units
            linear[0] =  2                   # The start of linear_B
            linear[-1] = 3                # The end of linear_B
            flags.linear_b = linear
            for reg_scale in reg_scale_list:
                flags.reg_scale = reg_scale
                for j in range(3):
                        flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j)+"_tandem_linear_b_complexity_swipe_layer" + str(linear_unit) + "_num" + str(i)
                        train.training_from_flag(flags)

