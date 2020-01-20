"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil

# Torch

# Own
import flag_reader
import data_reader
from class_wrapper import Network
from model_maker import Forward, Discriminator, Generator, Spectra_encoder
from utils.helper_functions import put_param_into_folder

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(x_range=flags.x_range,
                                                      y_range=flags.y_range,
                                                      geoboundary=flags.geoboundary,
                                                      batch_size=flags.batch_size,
                                                      normalize_input=flags.normalize_input,
                                                      data_dir=flags.data_dir)
    # Reset the boundary is normalized
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]

    print("Boundary is set at:", flags.geoboundary)
    print("Making network now")

    # Make Network
    ntwk = Network(Discriminator, Generator, Spectra_encoder, Forward, flags, train_loader, test_loader)

    # Training process
    ntwk.train_forward()
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    flag_reader.write_flags_and_BVE(flags, ntwk.best_validation_loss)
    put_param_into_folder(ntwk.ckpt_dir)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    # Call the train from flag function
    training_from_flag(flags)



