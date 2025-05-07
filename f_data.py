


import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from csv import writer
from fred import feed_forward_nn as ffnn
from fred import util as ut


config = config.f_config()
uti_obj = ut.util(config)
ffn_object = ffn(config)
# In[3]:


x_file = 'variables/sorted_inp_data.txt'
y_file = 'variables/sorted_bedadhes.txt'
rnn_model_path = 'AE_Model/rnn.h5'


# In[4]:


x_data = np.genfromtxt(x_file)
y_data = np.genfromtxt(y_file)
rnn_model = ffn_object.load(rnn_model_path)


y_data = np.expand_dims(y_data, axis=-1)
x_data = x_data[:, 1:]
pred_data = np.concatenate([x_data, y_data], axis=1)
pred_data = np.reshape(pred_data, (pred_data.shape[0], pred_data.shape[1], 1))
pred_data.shape



prediction = rnn_model.predict(pred_data)


np.savetxt('variables/bv.txt', prediction)

