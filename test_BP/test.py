from utils.evaluation_helper import eval_from_simulator
import flag_reader

#flags = flag_reader.read_flag()
#Xpred_file = "data/test_Xpred_ballisticsreg0.0005trail_0_complexity_swipe_layer500_num5.csv"
#eval_from_simulator(Xpred_file, flags)

import numpy as np
sample_points = np.linspace(-3, 3, 600)
np.savetxt('range_3_full_Xpred.csv', sample_points)
