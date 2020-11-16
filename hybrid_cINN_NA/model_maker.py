# Import from cINN lib
from cINN.model_maker import cINN
import cINN.flag_reader as flag_reader
#from cINN.parameters import *
# Import from NA lib
from Backprop.model_maker import Backprop

if __name__ == '__main__':
    #print(LINEAR)
    flags = flag_reader.read_flag()
    model_cINN = cINN(flags)
    model_NA = Backprop(flags)
