"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    # linear_unit_list = [150, 300]
    # linear_unit_list = [1000, 500]
    dim_z_list = [30, 35]
    dim_spec_list = [15, 20, 30, 35] 
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    for dim_z in dim_z_list:
    #for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for dim_spec in dim_spec_list:
        #for i in range(7,10):
            flags = flag_reader.read_flag()  	#setting the base case
            #linear = [linear_unit for j in range(i)]        #Set the linear units
            #linear[0] = 150                   # The start of linear
            #linear[-1] = 8                # The end of linear
            #flags.linear_b = linear
            flags.dim_z = dim_z
            flags.dim_spec = dim_spec
            # FOR MODIFY THE DEPENDENT PARAMS
            flags.linear_D[0] = dim_spec + dim_z
            flags.linear_E[0] = 8 + dim_spec
            flags.linear_SE[-1] = dim_spec
            #for reg_scale in reg_scale_list:
                #flags.reg_scale = reg_scale
            for j in range(3):
                flags.model_name = "trail_"+str(j)+"_VAE_dim_z_" + str(dim_z) + "_dim_spec_" + str(dim_spec)
                train.training_from_flag(flags)

