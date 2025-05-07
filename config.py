from tensorflow import keras
import numpy as np
import scipy.interpolate as inter
import os
import pandas as pd
from regressionmodel import regression


def global_config():
    #Feel free to change here
    config = {
        # Number of total cases and cases for pre-training:
        # For example, if anz_b = 6 and anz_b_phy = 2, it means (2, 3, 4, 5)
        'anz_b': 6,  
        
        # Number of cases for pre-training:
        # For example, if anz_b_phy = 2, it means (0, 1)
        'anz_b_phy': 2,  
        
        # simple p&f for pinn training
        'pf': 1,
        
        # Size of compressed behaviour vector
        'size_b': 12,  
        
        # Total number of input parameters for your problem
        'anz_inp_param': 125 ,  
        
        # Total number of output parameters for your problem
        'anz_out_param': 1, 
        
        #Mainpath where everything is stored, change this, if you change something 
        'save_to': 'bedadhes_pinn_opti', 
        
        # Define the proportion of data not used in training. The first 1 - test_split data is used for training.
        #Set this to zero, if you have seperate data for testing purpose
        'test_split': 0.15 
    }
    
    #________________you might not want to change below_________________

    config['len_lhs'] = 256  # Number of internal samples which are used to sample the function space. Should be from 2n to n

    config['rand_samples'] = 500 #How many random points are used 
    config['name'] = 'model'

    config = error_correct_config(config)

    
    
    
    
    
    
    
    def set_architecture(config):
        
        #__________________Change or add architectures for various NN___________
        
        # Architype:
        # 0..5: all variables in config, no change in this function
        # 6..19: custom models for Pinn and F
        
        
        if config['archi_type'] == 0:
            config['inp_shape'] = config['anz_inp_param'] + config['size_b']
            inp_shape = (config['inp_shape'],)
            
            dropout = config['dropout']
            size_1st_layer = config['size_1st_layer']
            size_last_layer = config['size_last_layer']
            anz_layer = config['anz_layer']
            interpol = config['interpol']
            act_func = config['activation_function']
            max_expon = config['max_exponent']
            logexp = max_expon+1
            
            
            #interpol = np.round(interpol)
            interpol = np.max((interpol,-logexp))
            interpol = np.min((interpol,logexp))
            
            if interpol < 1:
                interpol = interpol -2
            interpol = np.max((interpol,-logexp))
            interpol = np.min((interpol,logexp))
            
            if interpol< 0 and interpol>-logexp:
                interpol = -1/interpol
            elif interpol == -logexp:
                interpol = 0
                
                
            if interpol == 0:
                size_1st_layer = np.exp(size_1st_layer)
                size_last_layer = np.exp(size_last_layer)
            elif interpol == logexp:
                size_1st_layer = np.log(size_1st_layer)
                size_last_layer = np.log(size_last_layer)
            else:
                size_1st_layer = pow(size_1st_layer,interpol)
                size_last_layer = pow(size_last_layer,interpol)
                
                
            
            model = keras.Sequential()
            model.add(keras.Input(shape=inp_shape))
            for ii in range(anz_layer):
                size_dis_layer = size_1st_layer + (size_last_layer-size_1st_layer)*(ii/(anz_layer-1))
                if interpol == 0:
                    size_dis_layer = np.log(size_dis_layer)
                elif interpol == logexp:
                    size_dis_layer = np.exp(size_dis_layer)
                else:
                    size_dis_layer = pow(size_dis_layer,1/interpol)
                size_dis_layer = int(np.round(size_dis_layer))
                model.add(keras.layers.Dense(size_dis_layer, activation=act_func))
                if dropout > 0:        
                    model.add(keras.layers.Dropout(dropout))
        
            model.add(keras.layers.Dense(int(config['anz_out_param']), activation=act_func))

        
            model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error']) 
# =============================================================================
#             # Informationen zu den Schichten ausgeben
#             for i, layer in enumerate(model.layers):
#                 print(f"Layer {i}: {layer.name}")
#                 print(f"  Layer Build Config: {layer.get_build_config()}")
#                 print(f"  Layer Config: {layer.get_config()}")
#                 print("")
#             
#             # Überprüfe, ob das Modell kompiliert ist
#             if model.optimizer is None:
#                 print("Fehler: Modell ist nicht kompiliert!")
#             else:
#                 print(f"Optimizer: {model.optimizer.get_config()}")
# =============================================================================
        
            return model
        """
        Architecture type 20 for auto encoder
        """
        if config['archi_type'] == 20:
            
            #___________________Rather change in ae_config below_______________________
            model = keras.Sequential()
            size_b = config['size_b']
            len_lhs = config['len_lhs']
            if 'rand_or_lhs' in config and config['rand_or_lhs']:
                inp_shape = (config['anz_inp_param'] + config['anz_out_param']) * config['rand_samples']
            else:
                inp_shape = len_lhs
            inp_shape = int(inp_shape)
            inp_shape = (inp_shape,)
            
            ae_dimi = config['ae_dimi']
            dropout = config['dropout']
            ae_start_layer = config['ae_start_layer']
            
            n = int(np.floor(np.log(len_lhs/size_b)/np.log(ae_dimi)))
            
            model.add(keras.Input(shape=inp_shape))
        
            model.add(keras.layers.Dense(np.floor(len_lhs/ae_dimi), activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            
            for ii in range(1,n):
                if ii == 1:
                    continue
                model.add(keras.layers.Dense(np.floor(len_lhs/(ae_dimi ** ii)), activation='tanh')) 
                model.add(keras.layers.Dropout(dropout))
                
            
            model.add(keras.layers.Dense(size_b, activation='tanh', name='compressed_layer')) #name is important!
            model.add(keras.layers.Dropout(dropout))   
            
            for ii in range(n-1,-1,-1):
                model.add(keras.layers.Dense(np.floor(len_lhs/(ae_dimi ** ii)), activation='tanh'))
                model.add(keras.layers.Dropout(dropout))
            
            
            model.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                           loss=keras.metrics.mean_squared_error)
            return model
        
        if config['archi_type'] == 21:
            
            #___________________Rather change in ae_config below_______________________
            model = keras.Sequential()
            size_b = config['size_b']
            len_lhs = config['len_lhs']
            if 'rand_or_lhs' in config and config['rand_or_lhs']:
                inp_shape = (config['anz_inp_param'] + config['anz_out_param']) * config['rand_samples']
            else:
                inp_shape = len_lhs
            inp_shape = int(inp_shape)
            inp_shape = (inp_shape,)
            
            ae_dimi = config['ae_dimi']
            dropout = config['dropout']
            ae_activation = config['ae_activation']
            ae_start_layer = config['ae_start_layer']
            ae_last_layer = config['ae_last_layer']
            
            n = int(np.floor(np.log(ae_start_layer/ae_last_layer)/np.log(ae_dimi)))
        
            
            model.add(keras.Input(shape=inp_shape))
            model.add(keras.layers.Dense(ae_start_layer, activation=ae_activation))
            model.add(keras.layers.Dropout(dropout))
            
            for ii in range(1,n):
                model.add(keras.layers.Dense(int(np.floor(ae_start_layer/(ae_dimi ** ii))), activation=ae_activation)) 
                model.add(keras.layers.Dropout(dropout))
                
            model.add(keras.layers.Dense(ae_last_layer, activation=ae_activation))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(size_b, activation=ae_activation, name='compressed_layer')) #name is important!
            model.add(keras.layers.Dense(ae_last_layer, activation=ae_activation))
            model.add(keras.layers.Dropout(dropout))
            
            for ii in range(n-1,-1,-1):
                model.add(keras.layers.Dense(int(np.floor(ae_start_layer/(ae_dimi ** ii))), activation=ae_activation))
                model.add(keras.layers.Dropout(dropout))
            
            model.add(keras.layers.Dense(inp_shape[0], activation=ae_activation))
            model.add(keras.layers.Dropout(dropout))
            model.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                           loss=keras.metrics.MSE)
            return model
        """
        Architecture type 30 for behaviour vector guesser
        """
        if config['archi_type'] == 30:
            
            #input_shape = (config['anz_inp_param']+config['anz_out_param'],1)
            
            model = keras.Sequential()
            #________________________Change as needed____________________
            model.add(keras.layers.GRU(units=config['rnn_start_layer'], return_sequences=True))
            #model.add(keras.layers.Dense(units=64, activation='relu'))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=config['rnn_mid_layer'], activation=config['activation_function'])))
            model.add(keras.layers.Dropout(config['dropout']))
            model.add(keras.layers.LSTM(units=config['size_b'], return_sequences=True))
            #model.add(keras.layers.Dense(units=18, activation='relu'))

            model.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                           loss=keras.metrics.MSE)
            return model
        
        if config['archi_type'] == 31:
            
            #input_shape = (config['anz_inp_param']+config['anz_out_param'],1)
            
            dropout = config['dropout']
            size_1st_layer = config['size_1st_layer']
            size_last_layer = config['size_last_layer']
            anz_layer = config['anz_layer']
            interpol = config['interpol']
            act_func = config['activation_function']
            max_expon = config['max_exponent']
            logexp = max_expon+1
            
            
            #interpol = np.round(interpol)
            interpol = np.max((interpol,-logexp))
            interpol = np.min((interpol,logexp))
            
            if interpol < 1:
                interpol = interpol -2
            interpol = np.max((interpol,-logexp))
            interpol = np.min((interpol,logexp))
            
            if interpol< 0 and interpol>-logexp:
                interpol = -1/interpol
            elif interpol == -logexp:
                interpol = 0
                
                
            if interpol == 0:
                size_1st_layer = np.exp(size_1st_layer)
                size_last_layer = np.exp(size_last_layer)
            elif interpol == logexp:
                size_1st_layer = np.log(size_1st_layer)
                size_last_layer = np.log(size_last_layer)
            else:
                size_1st_layer = pow(size_1st_layer,interpol)
                size_last_layer = pow(size_last_layer,interpol)
                
            
            
            
            
            model = keras.Sequential()
            #________________________Change as needed____________________
            model.add(keras.layers.GRU(units=config['rnn_start_layer']))
            #model.add(keras.layers.Dense(units=64, activation='relu'))
            for ii in range(anz_layer):
                size_dis_layer = size_1st_layer + (size_last_layer-size_1st_layer)*(ii/(anz_layer-1))
                if interpol == 0:
                    size_dis_layer = np.log(size_dis_layer)
                elif interpol == logexp:
                    size_dis_layer = np.exp(size_dis_layer)
                else:
                    size_dis_layer = pow(size_dis_layer,1/interpol)
                size_dis_layer = int(np.round(size_dis_layer))
                model.add(keras.layers.Dense(units=size_dis_layer, activation=act_func))

                if dropout > 0:        
                    model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(units=config['size_b']))
            #model.add(keras.layers.Dense(units=18, activation='relu'))

            model.compile(optimizer="adam", 
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
            return model        
        
        """
        Architecture type 40 for PINN with functional behavioral vector
        """
        if config['archi_type'] == 40:
            
            inp_shape = (config['inp_shape'],)
            
            
            dropout = 0.15
            #Model should have some level of complexity, around 7 layers are fine
            model = keras.Sequential()
            model.add(keras.layers.Dense(62, activation='tanh',input_shape = inp_shape ))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(31, activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(16, activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(8, activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(4, activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(2, activation='tanh'))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(1, activation='tanh'))
            
            model.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                           loss=keras.metrics.mean_squared_error)
            return model
    
    def generate_physical_training_data(b,inp):
        x_anz = inp.shape[0]
        if b == 0: 
            
            #the influence of first layer parameters on adhesion between the 3D printer’s glass bed and ABS
            #1 round 21 measurments without initial line width
            #2 round 19 measurments
            
            #nozzle_temp
            nt = np.ones(40)*250 
            nt[0:5] = [235, 240, 245, 250, 255]   
            nt[21:40] = np.ones(19)*255
            nt[31:36] = [235,240,245,250,255]
            
            #bed_temp
            bt = np.ones(40)*100
            bt[5:9] = [90, 100, 110, 120]
            bt[21:40] = np.ones(19)*100
            bt[36:40] = [95,100,105,110]
            
            #first_layer_height
            fh = np.ones(40)*0.25
            fh[9:13] = [0.1, 0.25, 0.33, 0.4]
            fh[21:27] = [0.05,0.1,0.2,0.25,0.33,0.4]

            #first_layer_speed
            fs = np.ones(40)*20
            fs[13:17] = [10,20,30,40]
            fs[21:40] = np.ones(19)*30
            fs[27:31] = [10,20,30,40]
            
            
            #flow_rate
            fr = np.ones(40)*100
            fr[17:21] = [80,100,120,140]
            fr[21:40] = np.ones(19)*140
            
            
            force = [10, 10.5, 9.5, 10.5, 11.5,#nozzle_temp
                     4.7, 11, 11.7, 9, #bed_temp
                     9,11, 13.5, 13,#first_layer_height
                     12, 10, 13.5, 12,#first_layer_speed
                     4, 5.5, 8, 12,#flow_rate
                     14,17,20,21,17,17,#first_layer_height
                     21,21,25,21,#first_layer_speed
                     21,21,22,21.5,25,#nozzle_temp
                     8,20,28,18] #bed_temp
            
            #interpo = Rbf(nt,bt,fh,fs,fr,force, function='linear', neighbors=5)
            bed_temp = np.zeros(x_anz)
            first_layer_width = np.zeros(x_anz)
            first_layer_height = np.zeros(x_anz)
            normal_layer_width = np.zeros(x_anz)
            nozzle_temp = np.zeros(x_anz)
            first_layerspeed = np.zeros(x_anz)
            flow_rate = np.zeros(x_anz)
            for ii in range(0,x_anz):
                bed_temp[ii] = inp[ii,8]*(120-90)+90
                first_layer_width[ii] = inp[ii,58]*(0.7-0.4)+0.4
                first_layer_height[ii] = inp[ii,59]*(0.4-0.1)+0.1
                normal_layer_width[ii] = inp[ii,36]*(0.7-0.3)+0.3
                nozzle_temp[ii] = inp[ii,101]*(250-235)+235
                first_layerspeed[ii] = inp[ii,60]*(100-10)+10
                #cross_section = (first_layer_width-first_layer_height)*first_layer_height+np.pi*((first_layer_height/2)**2)
                #flow_rate = cross_section*first_layerspeed
                flow_rate[ii] = first_layer_width[ii]*100/normal_layer_width[ii]
            outp = inter.griddata((nt,bt,fh,fs,fr),force, (nozzle_temp,bed_temp,first_layer_height,first_layerspeed,flow_rate), method='linear')
               # if outp[ii] < 0:
               #     outp[ii] = 0
            
        
        if b == 1: 
            
            #Optimisation of the Adhesion of Polypropylene-Based Materials during Extrusion-Based Additive Manufacturing
                #1 round 9 measurments only bed temp
                #2 round 8 measurments DOE
                
            #round 2
            #nozzle_temp
            nt = [200,200,230,230,200,200,230,230]
                
            #first_layer_height
            fh = [1.6,1.6,1.6,1.6,2.4,2.4,2.4,2.4]
                
            #flow_rate
            fr = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2]
                
            force = [68,26.6,116.8,45.4,155.5,14.5,155.5,88.4]
                
            #interp = Rbf(nt,fh,fr,force, function='linear', neighbors=3)
            outp = np.zeros(x_anz)
            bed_temp = np.zeros(x_anz)
            first_layer_width = np.zeros(x_anz)
            first_layer_height = np.zeros(x_anz)
            normal_layer_width = np.zeros(x_anz)
            nozzle_temp = np.zeros(x_anz)
            first_layerspeed = np.zeros(x_anz)
            flow_rate = np.zeros(x_anz)
            cross_section = np.zeros(x_anz)
            flow_rate = np.zeros(x_anz)
            for ii in range(0,x_anz):
                bed_temp = inp[ii,8]*(120-90)+90
                first_layer_width = inp[ii,58]*(0.7-0.4)+0.4
                first_layer_height = inp[ii,59]*(0.4-0.1)+0.1
                normal_layer_width = inp[ii,36]*(0.7-0.3)+0.3
                nozzle_temp = inp[ii,101]*(250-235)+235
                first_layerspeed = inp[ii,60]*(100-10)+10
                cross_section = (first_layer_width-first_layer_height)*first_layer_height+np.pi*((first_layer_height/2)**2)
                flow_rate = cross_section*first_layerspeed
                    
                #flow_rate = first_layer_width*100/normal_layer_width
                outp[ii] = inter.griddata((nt,fh,fr),force, (nozzle_temp,first_layer_height,flow_rate), method='nearest')
               
                #outp[ii] = interp(nozzle_temp,first_layer_height,flow_rate)
                #1 round
                outp[ii] = outp[ii] -29 + np.interp(bed_temp,
                               (20,30,40,50,60,70,80,90,100),
                               (0,0,7,22,29,39,56,85,60))   #29 50° bed temp for all round 2 measurments
              

        if b < 2:
            for ii in range(x_anz-1,-1,-1):
                if np.isnan(outp[ii]):
                    outp = np.delete(outp,ii)
                    inp = np.delete(inp,ii, axis=0)
            outp = outp/200 # data must be between 0 and 1
        
        else:
            x_file = 'sorted_inp_data.txt'
            y_file = 'sorted_bedadhes.txt'


            inp = np.genfromtxt(x_file)
            outp = np.genfromtxt(y_file)
            # b_ax = (b-1)
            # print('b_ax: '+str(b_ax)+' '+str(inp[0,0]))
            # for ii in range(outp.shape[0]-1,-1,-1):
            #     if int(b_ax) != int(inp[ii,0]):
            #         outp = np.delete(outp,ii)
            #         inp = np.delete(inp,ii, axis=0)
                
        #print(inp.shape)
        #print(outp.shape)        
        return [inp, outp]
    def load_mess_data(b):
        return generate_physical_training_data(b,np.zeros([1,1]))

    config["set_architecture"] = set_architecture
    config["generate_physical_training_data"] = generate_physical_training_data
    config["load_mess_data"] = load_mess_data
    return config
    
def pinn_config():
    config = global_config() #global config for all configs
    config['save_to_model'] = config['save_to_pinn']
    config['name'] = 'pinn'
    config['inp_shape'] = config['anz_inp_param'] + config['size_b']
    config['pinn_samples'] = 50000
    config['pinn_gen_new'] = False #if new samples are to be generated. if there are no, new samples are always genereted
    config['pinn_overwrite_old'] = False #Wether new samples should be added to existing samples or overwritten
    
        #Hyperparameter for model creation
    config['dropout'] = 0.2
    config['size_1st_layer'] = 16
    config['size_last_layer'] = 28
    config['anz_layer'] = 3
    config['interpol'] = -4
    config['activation_function'] = 'tanh'
    config['max_exponent'] = 4
    
    #Hyperparameter for training from tensorflow.keras
    config['validation_split'] = 0.2
    config['batch_size'] = 96
    config['phy_epochs'] = 320
    config['epochs'] = 120
    config['verbose'] = 1
    
    #In this code training of a neural network using a multi-step approach is performed, 
    #where each step involves training the model with reduced batch size and epochs. 
    #This is a technique used for fine-tune and acceleration of the training process,
    #as well as to address issues like overfitting are addressed. 
    #The multistep_fit parameter in the configuration  controls 
    #the number of iterations of this training process. 
    #With each step, batch size is doubled and epochs are halved
    config['multistep_fit'] = 3
    config['archi_type'] = 0
    return config
    
    
def sdnn_config():
    config = pinn_config() #pinn networks are used for SDNN
    config['save_to_model'] = config['save_to_sdnn']
    config['name'] = 'sdnn'
    
    config['verbose'] = 0
    config['rand_train'] = 1 #How many random training points are used, can be set as config['anz_b_phy']
    config['rand_range'] = 0.2 #How far are random points allowed from the prelearned
    config['rand_distrib'] = 1 #What random distribution
    config['pinn_reuse'] = 5 #How often is a pinn retrained
    config['sdnn_samples'] = 20000 #How many data is generated
    return config


def ae_config():
    config = global_config() #global config for all configs
    config['save_to_model'] = config['save_to_ae']
    config['name'] = 'ae'
            
    config['archi_type'] = 21
    #With each training step, batch size is doubled and epochs are halved
    config['verbose'] = 1
    
    #Hyperparameter for training from tensorflow.keras
    config['validation_split'] = 0.3
    config['rand_or_lhs'] = False
    config['batch_size'] = 128
    config['epochs'] = 200
    #With each training step, batch size is doubled and epochs are halved
    config['multistep_fit'] = 7
    config['ae_dimi'] = 1.6 #Factor of which the AE is getting smaller/bigger from layer to layer    
    config['ae_start_layer'] = 58 #First Layer size
    config['ae_last_layer'] = 30 #Layer size around CFV
    config['ae_activation'] = 'tanh'
    config['dropout'] = 0 #Everywhere but CVF
    return config

def rnn_config():
    config = global_config() #global config for all configs
    config['save_to_model'] = config['save_to_rnn']
    config['name'] = 'rnn'
    
    config['validation_split'] = 0.2
    config['batch_size'] = 256
    config['epochs'] = 20
    config['verbose'] = 1
    config['rand_or_lhs'] = False
    
    
    config['archi_type'] = 31
    config['rnn_start_layer'] = 52
    config['rnn_mid_layer'] = 12
    
    
    config['dropout'] = 0
    config['size_1st_layer'] = 70
    config['size_last_layer'] = 3
    config['anz_layer'] = 5
    config['interpol'] = 0
    config['activation_function'] = 'tanh'
    config['max_exponent'] = 4   
    
    #With each training step, batch size is doubled and epochs are halved
    config['multistep_fit'] = 3
    
    #____________________ Dont edit below ____________________
    #config['idx_size'] = 3 #config['sdnn_samples']  How many random_points for training are used.
    
    config = error_correct_config(config)
    return config
            

def f_config():
    config = pinn_config() #of course options can be altered below
    config['save_to_model'] = config['save_to_f']
    config['name'] = 'f'
    
    #Hyperparameter for training from tensorflow.keras
    #config['validation_split'] = 0.2
    #config['batch_size'] = 128
    #config['epochs'] = 20
    #config['verbose'] = 1
    
    #In this code training of a neural network using a multi-step approach is performed, 
    #where each step involves training the model with reduced batch size and epochs. 
    #This is a technique used for fine-tune and acceleration of the training process,
    #as well as to address issues like overfitting are addressed. 
    #The multistep_fit parameter in the configuration  controls 
    #the number of iterations of this training process. 
    #With each step, batch size is doubled and epochs are halved
    #config['multistep_fit'] = 3
    
    config['archi_type'] = 0
    return config

def error_correct_config(config):
    
    # Set various paths for saving different types of files
    config['save_to_csv'] = os.path.join(config['save_to'], 'CSV')           # Path for CSV files
    config['save_to_pinn'] = os.path.join(config['save_to'], 'pinn_models')  # Path for PINN model files
    config['save_to_sdnn'] = os.path.join(config['save_to'], 'sdnn_model')   # Path for SDNN model files
    config['save_to_ae'] = os.path.join(config['save_to'], 'ae_models')      # Path for Autoencoder model files
    config['save_to_rnn'] = os.path.join(config['save_to'], 'rnn_models')    # Path for RNN model files
    config['save_to_f'] = os.path.join(config['save_to'], 'f_models')    # Path for RNN model files
    config['save_to_model'] = os.path.join(config['save_to'], 'models')     # Path for general model files
    
    config['save_to_model'] = config['save_to_'+config['name']]

    config['inp_shape'] = config['anz_inp_param'] + config['size_b']
    
    return config

