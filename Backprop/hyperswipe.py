"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    linear_unit_list = [1000, 500, 300, 150]
    for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for i in range(4,10):
            flags = flag_reader.read_flag()  	#setting the base case
            linear = [linear_unit for j in range(i)]        #Set the linear units
            linear[0] = 8                   # The start of linear
            linear[-1] = 150                # The end of linear
            flags.linear_b = linear
            for j in range(3):
                flags.model_name = "trail_"+str(j)+"_backward_model_complexity_swipe_layer" + str(linear_unit) + "_num" + str(i)
                train.training_from_flag(flags)

