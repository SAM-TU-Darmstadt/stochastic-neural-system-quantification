
from files import feed_forward_nn as ffnn
from files import util as ut
import config as config_file
import numpy as np


if False: #Do this while sdnn is running
    config = config_file.ae_config()
    #config["verbose"] = 0
    print("Start AE training...")
    
    # Initiate class instances
    uti_obj = ut.util(config)
    ae_model = ffnn.feed_forward_nn(config)

    # Fetch data from csv file
    data = uti_obj.get_ae_train_data(try_loadin=False)

    history_ae = ae_model.train(data,data)

    #print(f_model.get_summary())
    # Save model
    ae_model.save_as()

    #get the behavioral Vector
    BV = ae_model.encode(data)

    #Save the behavioral Vector
    uti_obj.save_array(BV, 'bv')
    print('finish AE training, saved new BV')
    

config = config_file.rnn_config()

# Initiate class instances
uti_obj = ut.util(config)
rnn_model = ffnn.feed_forward_nn(config)

# Fetch physical data from csv file
[inp,out] = uti_obj.get_rnn_training_data()

out = out[:,0,:]
inp = np.asarray(inp, dtype=np.float32)
out = np.asarray(out, dtype=np.float32)

print("Input shape:", inp.shape)  # Sollte (64, 92, 3) sein
print("Output shape:", out.shape)  # Sollte (64, 2) sein

hist = rnn_model.train(inp,out)
#rnn_model.load()

#test_loss = rnn_model.evaluate(inp,out)

rnn_model.save_as()

# test_inp = inp[42,:,:]
# test_out = out[42,:,:]
# test_inp = np.expand_dims(test_inp, axis=0)
# outi = rnn_model.model.predict(test_inp)
