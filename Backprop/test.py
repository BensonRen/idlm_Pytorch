from utils.evaluation_helper import eval_from_simulator
import flag_reader

flags = flag_reader.read_flag()
Xpred_file = "data/test_Xpred_ballisticsreg0.0005trail_0_complexity_swipe_layer500_num5.csv"
eval_from_simulator(Xpred_file, flags)
