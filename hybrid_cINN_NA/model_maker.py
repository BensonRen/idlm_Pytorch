# Import from cINN lib
from cINN.model_maker import cINN

#from cINN.parameters import *
# Import from NA lib
from Backprop.model_maker import Backprop

#if __name__ == '__main__':
def make_cINN_and_NA(flags):
    #print(LINEAR)
    #flags = flag_reader.read_flag()
    model_cINN = cINN(flags)
    model_NA = Backprop(flags)
    return model_cINN, model_NA
