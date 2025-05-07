from files import feed_forward_nn as ffnn
from files import util as ut
import numpy as np
import pandas as pd
import config as config
import f_prediction_data
from files import experimental_data as exd
import matplotlib.pyplot as plt
#import copy

config_rnn = config.rnn_config()
config_f = config.f_config()
# Initiate class instances
uti_obj_rnn = ut.util(config_rnn)
rnn_model = ffnn.feed_forward_nn(config_rnn)
uti_obj_f = ut.util(config_f)
f_model = ffnn.feed_forward_nn(config_f)
rnn_model.load(model_name='trial_no_time_45')
f_model.load(model_name='trial35')


ex_d = exd.experimental_data(config_rnn)
[train_x,train_y] = ex_d.load_messurement_data()
train_x = train_x[:,config_rnn['size_b']:]
train_y = train_y[:, np.newaxis]
n = int(train_x.shape[0]*(config_rnn['test_split']))
train_x = train_x[:n,:]
train_y = train_y[:n,:]

new_inp_rnn = np.concatenate((train_x, train_y),1)
new_inp_rnn = new_inp_rnn[np.newaxis,:,:]

# estimate behavioral vector using [inp_rnn, out_rnn]
rmse_single = np.zeros((n))
rmse_full = np.zeros((n))
rang_zahl = 11
rmse_range = np.zeros((n,rang_zahl))
for ii in range(1,n):
    
    inp_x = train_x[:ii,:]
    inp_y = train_y[:ii,:]
    
    new_inp_rnn = np.concatenate((inp_x, inp_y),1)
    new_inp_rnn = new_inp_rnn[np.newaxis,:,:]
    pred_b = rnn_model.pred_model(new_inp_rnn)
    
    
    replicated_pred_b = np.tile(pred_b, (ii, 1))
    combined_matrix = np.concatenate((replicated_pred_b,inp_x), axis=1)
    pred_y = f_model.pred_model(combined_matrix)
    rmse_single[ii] = np.sqrt(np.mean(np.square(pred_y-inp_y)))
    
    replicated_pred_b = np.tile(pred_b, (n, 1))
    combined_matrix = np.concatenate((replicated_pred_b,train_x), axis=1)
    pred_y = f_model.pred_model(combined_matrix)
    rmse_full[ii] = np.sqrt(np.mean(np.square(pred_y-train_y)))
    

    for jj in np.array([1,3,5,10]):
        #inp_rang = np.arange(np.max(ii-jj,1),ii)
        out_rang = np.arange(ii,np.min((ii+jj+1,n)))
        size_out_rang = len(out_rang)
    
        replicated_pred_b = np.tile(pred_b, (size_out_rang, 1))
        combined_matrix = np.concatenate((replicated_pred_b,train_x[out_rang,:]), axis=1)
        pred_y = f_model.pred_model(combined_matrix)
        rmse_range[ii,jj] = np.sqrt(np.mean(np.square(pred_y-train_y[out_rang,:])))


#%%    
tu_colors_fixed = [
    "#DDDF48",  # 379
    "#E9503E",  # 1795
    "#C9308E",  # 247
    "#5D85C3",  # 2727
    "#50B695",  # 338
    "#AFCC50",  # 381
    "#804597",  # 2685
    "#FFE05C",  # 113
    "#F8BA3C",  # 129
    "#EE7A34",  # 158
    "#009CDA"   # 299
]

def setup_plot(x_label='X', y_label='Z', width=6, height=4):
    plt.figure(figsize=(width, height))  # Plotgröße
    
    plt.rcParams.update({
        'font.size': 12,
        'legend.fontsize': 9,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.usetex': True,  # Optional LaTeX-Stil,
    })
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    

setup_plot(x_label = 'Anzahl Beobachtungen', y_label='RMSE')

plt.plot( np.arange(1,n),rmse_single[1:], label='nur vorhandene Beobachtungen', color = tu_colors_fixed[0])
plt.plot(np.arange(1,n),rmse_full[1:],  label='alle Beobachtungen', color = tu_colors_fixed[1])
    
    

plt.legend(loc='upper left')
plt.savefig('fff_plot_rmse.png',dpi=300)
plt.show()


setup_plot(x_label = 'Anzahl Beobachtungen', y_label='RMSE')

plt.plot(np.arange(1,n),rmse_range[1:,1], label='Vorhersage der nächsten Beobachtung', color = tu_colors_fixed[0])
plt.plot(np.arange(1,n),rmse_range[1:,3], label='Vorhersage der nächsten 3 Beobachtungen', color = tu_colors_fixed[1])
plt.plot(np.arange(1,n),rmse_range[1:,5], label='Vorhersage der nächsten 5 Beobachtung', color = tu_colors_fixed[2])
plt.plot(np.arange(1,n),rmse_range[1:,10], label='Vorhersage der nächsten 10 Beobachtung', color = tu_colors_fixed[3])
    
    

plt.legend(loc='upper left')
plt.savefig('fff_plot_rmse_rang.png',dpi=300)
plt.show()



#%%


mean_rmse_single = np.mean(rmse_single,axis=0)
mean_rmse_full = np.mean(rmse_full,axis=0)
mean_rmse_range = np.mean(rmse_range,axis=0)

vec = rmse_range[1:,1]

filtered_vec = [vec[0]] + [vec[i] for i in range(2, len(vec)) if vec[i] <= vec[i-1] + 0.05 and vec[i-1] <= vec[i-2] + 0.05]
new_rmse_mean_single = np.mean(filtered_vec)
    



# mean_rmse_single = np.sqrt(np.mean(np.square(rmse_single),axis=0))
# mean_rmse_full = np.sqrt(np.mean(np.square(rmse_full),axis=0))
# mean_rmse_range = np.sqrt(np.mean(np.square(rmse_range),axis=0))
#print(pred_b)
# pred_b = pred_b[-1,:]
# #print(pred_b)
# print(inp_f)
# replicated_pred_b = np.tile(pred_b, (inp_f.shape[0], 1))
# combined_matrix = np.concatenate((replicated_pred_b,inp_f), axis=1)
# #print(combined_matrix)

# #make prediction using the estmated behavioral vector
# pred = f_model.pred_model(combined_matrix) #hier auch b eingeben

# print(np.vstack((pred[:,0],out_f)),0)
# mse = np.mean((out_f - pred[:,0]) ** 2)

# print("Mean Squared Error:", mse)

# #reverse normalization
# lower_bound = np.log(1)
# upper_bound = np.log(31.0)
# norm_log_trans_dens = 1 - pred
# log_trans_dens = norm_log_trans_dens * (upper_bound - lower_bound) + lower_bound

# # Reverse the logarithmic transformation
# pred_dens = 101 - np.exp(log_trans_dens)
# #print(pred_dens)
# #evaluate model
# #eva = f_model.model.evaluate(inp_f, out_f) #???


