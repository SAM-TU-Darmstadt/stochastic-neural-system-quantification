
from files import feed_forward_nn as ffnn
import config_2 as conf_file
from files import experimental_data
from files import util as ut
import f_prediction_data_3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
np.random.seed(123)

#%%


config = conf_file.pinn_config()

    

# welche daten hier überhaupt geladen werden
#exp = experimental_data.experimental_data(config)
#[inp, outp] = exp.generate_physical_data()
#print([inp, outp])
config['verbose'] = 1
# Initiate model


f_model = ffnn.feed_forward_nn(config)
# Train model with physical data
f_model.model_fit_with_physical_data(0)
# Train model with mess data

#f_model.validation_split = 0
hist = f_model.model_fit_with_mess_data()
#save model in Model folder
f_model.save_as()


#%% plot von wissenstransfer




# uti_obj = ut.util(config)
# x_lhs = uti_obj.get_x_lhs()
# f_model.pred_model(x_lhs, 1)
# print(f_model.pred_model(x_lhs, 1))

# [inp_f, out_f] = f_prediction_data_3.load_prediction_data(1)
# pred = f_model.pred_model(inp_f, 1)

# print(np.vstack((pred[:,0],out_f)),1)
# mse = np.mean((out_f - pred[:,0]) ** 2)
# print(mse)

# # Assuming y_data is the transformed output and x_data is the input matrix
# # Reverse the normalization
# lower_bound = np.log(1)
# upper_bound = np.log(31.0)
# norm_log_trans_dens = 1 - pred
# log_trans_dens = norm_log_trans_dens * (upper_bound - lower_bound) + lower_bound

# # Reverse the logarithmic transformation
# pred_dens = 101 - np.exp(log_trans_dens)
# print(pred_dens)