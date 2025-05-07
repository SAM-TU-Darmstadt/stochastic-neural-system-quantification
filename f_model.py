
from files import feed_forward_nn as ffnn
from files import util as ut
import config_2 as config

f_config = config.f_config()
rnn_config = config.rnn_config()
rnn_config["verbose"] = 0
rnn_model = ffnn.feed_forward_nn(rnn_config)
rnn_model.load(model_name='trial_no_time_18')

f_config['rand_or_lhs'] = True

# Initiate class instances
uti_obj = ut.util(f_config)
f_model = ffnn.feed_forward_nn(f_config)

# Fetch data, if you use rnn_model, it will take some time for the first time or try_loadin=False
[inp, out] = uti_obj.get_f_model_training_data( rnn_model=rnn_model)

f_model.train(inp,out)


# Save model
f_model.save_as()

