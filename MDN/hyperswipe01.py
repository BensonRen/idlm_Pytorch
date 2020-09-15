"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    #linear_unit_list = [ 50, 100, 200, 500]
    #linear_unit_list = [1000, 500]
    linear_unit_list = [1000]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    #reg_scale_list = [5e-4]
    for num_gaussian in range(3,15):
        for linear_unit in linear_unit_list:
        #for kernel_first in conv_kernel_size_first_list:
        #    for kernel_second in conv_kernel_size_second_list:
                # Setting the loop for setting the parameter
            for i in range(5, 8):
                flags = flag_reader.read_flag()  	#setting the base case
                flags.data_set = 'sine_wave'
                flags.ckpt_dir = '/work/sr365/MDN_results/' + flags.data_set
                flags.num_gaussian = num_gaussian   # Setting the number of gaussians
                linear = [linear_unit for j in range(i)]        #Set the linear units
                linear[0] = 1                   # The start of linear
                linear[-1] = 2                # The end of linear
                flags.linear = linear
                #for reg_scale in reg_scale_list:
                #    flags.reg_scale = reg_scale
                for j in range(1):
                    flags.model_name = 'Gaussian_' + str(num_gaussian) + '/' + flags.data_set + "_linear_" + str(linear_unit) + "_layer_" + str(i) + "trail_"+str(j)
                    train.training_from_flag(flags)

