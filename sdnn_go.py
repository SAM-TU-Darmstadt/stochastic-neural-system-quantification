from files import feed_forward_nn as ffnn
import config_rf as config
from files import util as ut


config = config.sdnn_config()

uti_obj = ut.util(config)
        
x_lhs = uti_obj.get_x_lhs()
count = len(uti_obj.get_load_list(config['save_to_sdnn'],'sdnn','json'))

print('Writing y_lhs values into CSV file...')
try:
    while True:
        # Break condition
        if(count>=config['sdnn_samples']):
            break

        # Initiate model
        f_model = ffnn.feed_forward_nn(config)
        # Train model with physical data
        hist = f_model.model_fit_with_physical_data()
        #print("mess data:")
        f_model.validation_split = 0
        f_model.model_fit_with_mess_data()
        val_loss = hist.history['val_loss'][-1]
        print('val_loss: ' + str(val_loss))
        if val_loss < 0.02:
            count = count + f_model.sdnn(x_lhs)
        print('count: '+ str(count))
    
         

except KeyboardInterrupt:
    # Press Ctrl+C to stop the program
    print('Brother, I stop')
    pass

