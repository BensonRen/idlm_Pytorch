"""
This is the helper functions for various functions
1-4: retrieving the prediction or truth files in data/
5: Put flags.obj and parameters.txt into the folder
"""
import os
import shutil


def get_Xpred(path):
    for filename in os.listdir(path):
        if ("Xpred" in filename):
            out_file = filename
            print("Xpred File found", filename)
            break
    return os.path.join(path,out_file)


def get_Ypred(path):
    for filename in os.listdir(path):
        if ("Ypred" in filename):
            out_file = filename
            print("Ypred File found", filename)
            break;
    return os.path.join(path,out_file)


def get_Xtruth(path):
    for filename in os.listdir(path):
        if ("Xtruth" in filename):
            out_file = filename
            print("Xtruth File found", filename)
            break;
    return os.path.join(path,out_file)


def get_Ytruth(path):
    for filename in os.listdir(path):
        if ("Ytruth" in filename):
            out_file = filename
            print("Ytruth File found", filename)
            break;
    return os.path.join(path,out_file)

def put_param_into_folder(ckpt_dir):
    """
    Put the parameter.txt into the folder and the flags.obj as well
    :return: None
    """
    """
    Old version of finding the latest changing file, deprecated
    # list_of_files = glob.glob('models/*')                           # Use glob to list the dirs in models/
    # latest_file = max(list_of_files, key=os.path.getctime)          # Find the latest file (just trained)
    # print("The parameter.txt is put into folder " + latest_file)    # Print to confirm the filename
    """
    # Move the parameters.txt
    destination = os.path.join(ckpt_dir, "parameters.txt");
    shutil.move("parameters.txt", destination)
    # Move the flags.obj
    destination = os.path.join(ckpt_dir, "flags.obj");
    shutil.move("flags.obj", destination)

