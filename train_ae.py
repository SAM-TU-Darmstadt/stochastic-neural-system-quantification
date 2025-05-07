

from files import feed_forward_nn as ffnn
from files import util as ut
import config_2 as config

config = config.ae_config()

# Initiate class instances
uti_obj = ut.util(config)



f_model = ffnn.feed_forward_nn(config)

# Fetch data from csv file
data = uti_obj.get_ae_train_data()

f_model.train(data,data)

print(f_model.get_summary())
# Save model
f_model.save_as()

#get the behavioral Vector
BV = f_model.encode(data)

#Save the behavioral Vector
uti_obj.save_array(BV, 'bv')


