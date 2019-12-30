"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
# Own module
import train
# os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import flag_reader
import numpy as np


if __name__ == '__main__':

    # Learning rate hyperswiping
    # lr_list = [0.01, 0.005, 0.001]
    num_lorentz_oscilator = np.arange(1, 50, 1)

    # Setting the loop for setting the parameter
    for nlo in num_lorentz_oscilator:
        # setting the base case
        flags = flag_reader.read_flag()
        # Set the current learning rate
        flags.linear[-1] = nlo * 3
        fix_or_free = None
        if flags.fix_w0:
            fix_or_free = "fix-"
        else:
            fix_or_free = "free-"
        flags.model_name = "6layer-Adam-Lor-" + fix_or_free + str(nlo)
        # Train from the current learning rate
        train.training_from_flag(flags)


    #   flags.backward_fc_filters = (100,300,300,300,300,100,8)
    #    backward_fc_filters = backward_fc_filters_front
    #    for j in range(i):
    #        backward_fc_filters += (added_layer_size,)
    #    backward_fc_filters += backward_fc_filters_back
    #    flags.backward_fc_filters = backward_fc_filters
    #    print(flags.backward_fc_filters)



