"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    # Setting the loop for setting the parameter
    for i in range(3,6):
        flags = flag_reader.read_flag()  	#setting the base case
        linear = [1000 for j in range(i)]        #Set the linear units
        linear[0] = 8                   # The start of linear
        linear[-1] = 12                # The end of linear
        flags.linear = linear
        for j in range(1):
            flags.model_name = "trail_"+str(j)+"_layer_num" + str(i) + "unit_1000"
            train.training_from_flag(flags)

