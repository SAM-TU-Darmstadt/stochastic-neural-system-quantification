from files import feed_forward_nn as ffnn
from files import util as ut
import numpy as np
import pandas as pd
import config_2 as config
import f_prediction_data

config_rnn = config.rnn_config()
config_f = config.f_config()
# Initiate class instances
uti_obj_rnn = ut.util(config_rnn)
rnn_model = ffnn.feed_forward_nn(config_rnn)
uti_obj_f = ut.util(config_f)
f_model = ffnn.feed_forward_nn(config_f)
rnn_model.load()
f_model.load()

# data for the rnn to estimate behavioral vector
[inp_rnn, out_rnn] = f_prediction_data.load_prediction_data(0)

# data for prediction
[inp_f, out_f] = f_prediction_data.load_prediction_data(1) #hier kenne ich out_f nicht
new_inp_rnn = np.concatenate((inp_rnn, out_rnn[:, np.newaxis]),1)
new_inp_rnn = new_inp_rnn[:,np.newaxis,:]

# estimate behavioral vector using [inp_rnn, out_rnn]
pred_b = rnn_model.pred_model(new_inp_rnn)

#print(pred_b)
pred_b = pred_b[-1,:]
#print(pred_b)
print(inp_f)
replicated_pred_b = np.tile(pred_b, (inp_f.shape[0], 1))
combined_matrix = np.concatenate((replicated_pred_b,inp_f), axis=1)
#print(combined_matrix)

#make prediction using the estmated behavioral vector
pred = f_model.pred_model(combined_matrix) #hier auch b eingeben

print(np.vstack((pred[:,0],out_f)),0)
mse = np.mean((out_f - pred[:,0]) ** 2)

print("Mean Squared Error:", mse)

#reverse normalization
lower_bound = np.log(1)
upper_bound = np.log(31.0)
norm_log_trans_dens = 1 - pred
log_trans_dens = norm_log_trans_dens * (upper_bound - lower_bound) + lower_bound

# Reverse the logarithmic transformation
pred_dens = 101 - np.exp(log_trans_dens)
#print(pred_dens)
#evaluate model
#eva = f_model.model.evaluate(inp_f, out_f) #???


