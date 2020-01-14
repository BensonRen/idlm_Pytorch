"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    # Setting the loop for setting the parameter
    for code in range(3,15):
        flags = flag_reader.read_flag()  	#setting the base case
        # linear = [500 for j in range(i)]        #Set the linear units
        # linear[0] = 8                   # The start of linear
        # linear[-1] = 150                # The end of linear
        # flags.linear = linear
        flags.dim_code = code
        """
        Calculation based hyper-parameter, no need to change
        """
        flags.encoder_linear[-1] = code
        flags.decoder_linear[0] = code
        """
        Calculation based hyper-parameter block end
        """

        for j in range(3):
            flags.model_name = "trail_"+str(j)+"_dim_code_swipe" + str(code)
            train.training_from_flag(flags)

