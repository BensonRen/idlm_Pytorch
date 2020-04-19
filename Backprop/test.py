from utils.evaluation_helper import eval_from_simulator
import flag_reader
from utils.evaluation_helper import plotMSELossDistrib

flags = flag_reader.read_flag()
#plotMSELossDistrib('data/test_Ypred_sine_wavereg0.005trail_1_complexity_swipe_layer1000_num8.csv',
#                   'data/test_Ytruth_sine_wavereg0.005trail_1_complexity_swipe_layer1000_num8.csv',
#                        flags)
Xpred_file = "data/test_Xpred_sine_wavereg0.005trail_1_complexity_swipe_layer1000_num8.csv"
eval_from_simulator(Xpred_file, flags)



