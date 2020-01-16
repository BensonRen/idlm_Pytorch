"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import Backprop


def put_param_into_folder():
    """
    Put the parameter.txt into the folder and the flags.obj as well
    :return: None
    """
    list_of_files = glob.glob('models/*')                           # Use glob to list the dirs in models/
    latest_file = max(list_of_files, key=os.path.getctime)          # Find the latest file (just trained)
    print("The parameter.txt is put into folder " + latest_file)    # Print to confirm the filename
    # Move the parameters.txt
    destination = os.path.join(latest_file, "parameters.txt");
    shutil.move("parameters.txt", destination)
    # Move the flags.obj
    destination = os.path.join(latest_file, "flags.obj");
    shutil.move("flags.obj", destination)


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
    ntwk = Network(Backprop, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    flag_reader.write_flags_and_BVE(flags, ntwk.best_validation_loss)
    put_param_into_folder()


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    # Call the train from flag function
    training_from_flag(flags)



