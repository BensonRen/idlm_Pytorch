"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    num_lorentz_oscilator_list = np.arange(3, 31, 3)
    for fix in [True, False]:
        # Setting the loop for setting the parameter
        for num_lor in num_lorentz_oscilator_list:
            flags = flag_reader.read_flag()  	#setting the base case
            flags.linear[-1] = num_lor * 3
            flags.fix_w0 = fix
            if flags.fix_w0:
                fix_or_free = "fix"
            else:
                fix_or_free = "free"
            flags.model_name = "3-layer-Adam-" + fix_or_free + "-no-" + str(num_lor)
            train.training_from_flag(flags)

