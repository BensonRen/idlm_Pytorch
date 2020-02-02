"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    # linear_unit_list = [20, 30, 40, 50]
    dim_spec_list = [10, 15, 20]
    dim_z_list = [20, 30, 40, 50, 60]
    #linear_unit_list = [1000, 500]
    #linear_unit_list = [1000, 500, 300, 150]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    reg_scale_list = [5e-4]
    for dim_spec in dim_spec_list:
    #for linear_unit in linear_unit_list:
        # Setting the loop for setting the parameter
        for dim_z in dim_z_list:
            
        #for i in range(4, 10):
            flags = flag_reader.read_flag()  	            # setting the base case
            #linear = [linear_unit for j in range(i)]        # Set the linear units
            #linear[0] = flags.dim_y + flags.dim_z           # The start of linear
            #linear[-1] = flags.dim_x                        # The end of linear
            #flags.linear_d = linear
            flags.linear_se = dim_spec
            flags.linear_d = flags.dim_x + dim_z
            flags.linear_g = dim_spec + dim_z
            for reg_scale in reg_scale_list:
                flags.reg_scale = reg_scale
                #for dim_z in dim_z_list:
                #    flags.dim_z = dim_z
                #    flags.linear_d[0] = flags.dim_y + flags.dim_z
                    # print("flags.linear_d", flags.linear_d)
                    # print("flags.linear_e", flags.linear_e)
                for j in range(3):
                    flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_GAN_dim_spec_" + str(dim_spec) + "_dim_z_" + str(dim_z)
                    train.training_from_flag(flags)

