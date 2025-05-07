
from files import feed_forward_nn as ffnn
from files import util as ut
import config_rf as config_file
import numpy as np
import matplotlib.pyplot as plt



#%%load best rnn model
rnn_config = config_file.rnn_config()

# Initiate class instances
uti_obj = ut.util(rnn_config)
rnn_model = ffnn.feed_forward_nn(rnn_config)

rnn_model.load(model_name="trial_no_time_18")


#%%load ae model
ae_config = config_file.ae_config()

# Initiate class instances
uti_obj = ut.util(ae_config)
ae_model = ffnn.feed_forward_nn(ae_config)
#load best ae model
ae_model.load(model_name="trial49")


#%%load f model
f_config = config_file.f_config()

# Initiate class instances
uti_obj = ut.util(f_config)
f_model = ffnn.feed_forward_nn(f_config)
#load best ae model
f_model.load(model_name="trial44")



def S(ii,x_0):
    if ii == 0:
        return x_0**3 + 0.4 * x_0**2 - 0.5 * x_0 + 0.1
    if ii == 1:
        return 0.5 * x_0**3 + 0.4 * x_0**2 - 0.5 * x_0 + 0.2
                
    if ii == 2:
        return 0.25 * x_0**3 + 0.6 * x_0**2 - 0.3 * x_0 + 0.2
    if ii == 3:
        return 1-np.cos(x_0)


x_lhs = uti_obj.get_x_lhs()
max_anz_stutz = 11

rmse_stutz = 100
x_rmse = np.arange(0,1,1/rmse_stutz)
x_rmse = x_rmse[:,np.newaxis]

x_lhs = x_lhs[np.newaxis,:]
bvs = []
rnn_bvs = []
y_lhs_s = []
y_lhs_prog = []
real_b = []
rmse_ylhs = []
num_features = ae_config['size_b']
rmse_full_fs = []
rmse_bs = []
for ii in range(4):
    y_lhs = S(ii,x_lhs)
    y_rmse = S(ii,x_rmse)
    #y_lhs = y_lhs[np.newaxis,:]
    y_lhs_s.append(y_lhs)
    tmp = ae_model.model.predict(y_lhs)
    y_lhs_prog.append(tmp)
    bv = ae_model.encode(y_lhs)    
    bvs.append(bv)    
    rmse_ylhs.append(np.sqrt(np.mean(np.square(y_lhs-tmp))))
    
    rnn_bv = []
    f_out_stutzs = []
    rmse_full_f = []
    rmse_b = []
    
    for anz_stutz in range(1,max_anz_stutz):
        interv = 1/anz_stutz
        x = np.arange(1/2*interv,1,interv)
        x = np.random.permutation(x)
        x = x[:,np.newaxis]
        y = S(ii,x)
        inp = np.concatenate((x,y),axis=1)
        inp = inp[np.newaxis, :]
        rmse_b_ = np.ones(5)
        for jj in range(5):
            b = rnn_model.pred_model(inp)
            inp = np.concatenate((inp,inp),axis=1)
            rmse_b_[jj] = np.sqrt(np.mean(np.square(b-bv)))
        rnn_bv.append(b)
        rmse_b.append(np.min(rmse_b_))
        bv_full = np.full((anz_stutz, num_features), b)
        inp_f = np.concatenate((bv_full,x),axis=1)
        
        y_p = f_model.pred_model(inp_f)
        f_out_stutz = np.concatenate((x,y_p),axis=1)
        f_out_stutzs.append(f_out_stutz)
        
        
        bv_full = np.full((rmse_stutz, num_features), b)
        inp_f = np.concatenate((bv_full,x_rmse),axis=1)
        
        y_full = f_model.pred_model(inp_f)
        
        rmse_full_f.append(np.sqrt(np.mean(np.square(y_rmse-y_full))))
        
        #f_out_stutz = np.concatenate((x,y_p),axis=1)
        
        
    rnn_bvs.append(rnn_bv)
    
    rmse_bs.append(rmse_b)
    rmse_full_fs.append(rmse_full_f)
    
    
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
    
    

setup_plot(x_label = 'Anzahl Stützpunkte', y_label='RMSE')
for ii in range(2,4):    
    x = np.arange(1,11,1)
    y = np.array(rmse_full_fs[ii])
    plt.plot(x, y, label='$S_' + str(ii+1) + '$', color = tu_colors_fixed[ii])
    
    

plt.legend(loc='upper left')
plt.savefig('rs_plot_rmse_full.png',dpi=300)
plt.show()

setup_plot(x_label = 'Anzahl Stützpunkte', y_label='RMSE')
for ii in range(2,4):    
    x = np.arange(1,11,1)
    y = np.array(rmse_bs[ii])
    plt.plot(x, y, label='$S_' + str(ii+1) + '$', color = tu_colors_fixed[ii])
    
    

plt.legend(loc='upper left')
plt.savefig('rs_plot_rmse_b.png',dpi=300)
plt.show()


