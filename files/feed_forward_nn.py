"""Define model for feed forward neural network with utilities"""

from tensorflow import keras
from keras import backend as K
from files import experimental_data as exd
import numpy as np
import matplotlib.pyplot as plt
from files import util
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import random
#from sdnn import config


class feed_forward_nn:
    #define the architecture and all parameters
    archi_type = None  
    #training parameters
    dropout = None
    validation_split = None    
    batch_size = None
    epochs = None
    verbose = None
    ae_dimi = None
    len_lhs = None
    anz_b = None
    anz_b_phy = None
    anz_param = None
    config = None
    #actual keras model
    model = 0
    encoder =None
    decoder =None
    
    """
    Initiate the init function that initiate model
    :param config: created with config file, containes a lot of information
    """
    def __init__(self, config, model = None ):
        
        self.archi_type = config["archi_type"]

        self.validation_split = config['validation_split'] if 'validation_split' in config else 0.2
        self.batch_size = config['batch_size'] if 'batch_size' in config else 256
        self.epochs = config['epochs'] if 'epochs' in config else 100
        self.verbose = config['verbose'] if 'verbose' in config else 0 
        self.ae_dimi = config['ae_dimi'] if 'ae_dimi' in config else 2
        self.len_lhs = config['len_lhs']
        self.anz_b = config['anz_b']
        self.anz_b_phy = config['anz_b_phy']
        self.anz_param = config['anz_inp_param'] 
        
        self.config = config
        if model is None:
            self.model = config["set_architecture"](config)
            
        if config['name']=='ae':
            self.encoder = self.get_compressed_layer_feature_extractor()
            self.decoder = self.get_decompressed_layer_feature_extractor()

    """
    Set the architecture of the model
    :return: model based on the architecture type
    """ 

    
    """
    Train the neural network with fit
    :param input_data: training input
    :param output_data: training output
    :return history of the training
    """
    def train(self, input_data, output_data, epochs = 0):
        
        if self.config['name'] == 'pinn' or self.config['name'] == 'sdnn':
            n = int(input_data.shape[0]*(1-self.config['test_split']))
            
            
            input_data = input_data[:n, :]
            
            if len(output_data.shape) == 1:
                output_data = output_data[:n, np.newaxis]
            else:
                output_data = output_data[:n, :]
        if False:
            # Generate a permutation index
            permutation_index = np.random.permutation(input_data.shape[0])
    
            # Shuffle both matrices using the same permutation index
            output_data = output_data[permutation_index]
            input_data = input_data[permutation_index]
       #print(input_data)
        if epochs == 0:
            epochs = self.epochs
        #print(output_data)
        for step in range(self.config['multistep_fit']):
            dimi = int(2**step)
            bt_s = int(self.batch_size*dimi)
            epo = int(epochs/dimi)
            vali_split = self.validation_split
            if input_data.shape[0] < 3:
                vali_split = 0
            hist = self.model.fit(input_data, output_data, validation_split = vali_split,
                 batch_size=bt_s, epochs=epo, verbose=self.verbose)
        return hist

    def pred_model(self, input_data, b=None, model=None):   
        if model is None:
            model = self.model
            
        if b is None:
            return model.predict(input_data, verbose=self.config["verbose"])
        else:
            ex = exd.experimental_data(self.config)
            
            input_data = ex.make_bin_data(input_data,b)
            input_tensor = tf.convert_to_tensor(input_data)
            try:
                return model.predict(input_tensor)
            except Exception:
                input_data = input_data[:, np.newaxis]
                input_tensor = tf.convert_to_tensor(input_data)
                return model.predict(input_tensor)

            

    """
    Get the summary of model
    :param model: model object
    :return: summary of the model
    """
    def get_summary(self, model=False):
        if not model:
            model = self.model
        return model.summary()
    
    """
    Get a model copy from another model data
    :param file_path: file path to hold model data
    :param model: model to be copied
    :return: copied model
    """
    def get_copy(self,model = None):
        if not model:            
            model = self.model
            
        # Find out Input shape    
        config = model.get_config() # Returns pretty much every information about your model
        input_shape = config["layers"][0]["config"]["batch_shape"]
            
        #Copy the model
        model_cp= keras.models.clone_model(model)        
        model_cp.build(input_shape) # replace 10 with number of variables in input layer
        model_cp.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model_cp.set_weights(model.get_weights())
        return model_cp


    """
    Draw model plot
    :param model: model to be copied
    :param lbl: label for the plot
    :param linstyl: linestyle for the plot
    """
    def draw_plot_model(self, model=False, lbl='',linstyl='solid'):
        if not model:
            model = self.model
        anzP = 1000
        x = np.arange(0,2,2/anzP)
        y_pred = model.predict(x)
        plt.plot(x,y_pred,label=lbl,linestyle = linstyl)

    """
    Save model json and  weight
    :param path: path to store, default Model/
    :return filename
    """
    def save_as(self, file_name=None):        
        
        utils = util.util(self.config)
        utils.save_model(self.model,model_name = file_name)


    """
    Load model from json and h5
    :param file_index: always takes the last saved file
    :param model_name: path to store, defaulted
    :return loaded_model
    """
    def load(self, file_index=-1, model_name=None):
        
        utils = util.util(self.config)
        self.model = utils.load_model(file_index = file_index,model_name = model_name)

    
    """
    Fit model with generated physical data
    :param anz_param: default 124
    :param anz_b_phy: default 4
    :return model
    """
    def model_fit_with_physical_data(self, pref_b = None):
        if self.config['pf'] == 1 and pref_b is None:
            pref_b = np.random.randint(0, self.config['anz_b_phy'])

        ex_d = exd.experimental_data(self.config)
        [pretrain_x,pretrain_y] = ex_d.generate_physical_data(b = pref_b)
        #print('pretrain_std', np.std(pretrain_y))
        #print('pretrain_shape', pretrain_x.shape, 'vs inp_shape', self.config['inp_shape'])

        return self.train(pretrain_x,pretrain_y,epochs = self.config['phy_epochs'])

    
    """
    Fit model with meassuremtn data
    :return model
    """
    def model_fit_with_mess_data(self, pref_b = None):


        ex_d = exd.experimental_data(self.config)
        [train_x,train_y] = ex_d.load_messurement_data(b =pref_b)
        #print("train_sdt" ,np.std(train_y))
        #print('train_shape', train_x.shape, 'vs inp_shape', self.config['inp_shape'])

                          
        return self.train(train_x, train_y)
    """
    Continous training model with generated physical data
    :param anz_param
    :param anz_b
    :return continously retrained model
    """
    def continuous_train_with_physical_data(self, anz_param=None, anz_b=None):

        from fred import experimental_data as exd

        anz_param = self.anz_param if anz_param is None  else anz_param
        anz_b = self.anz_b if anz_b is None  else anz_b

        print('Press interrupt key to break the continous traing, e.g. CTRL-C')
        model = self.model
        count = 0
        
        try:
            while True:
                count = count + 1
                print('Training iteration '+ str(count) + ' is starting...')
                ex_d = exd.experimental_data(self.anz_param, self.anz_b)
                [pretrain_x,pretrain_y] = ex_d.generate_physical_data(None, None, None, True, False)
                model.fit(pretrain_x, pretrain_y, validation_split=self.validation_split, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
                print('Training Count ' + str(count) + ' is completed!')

        except KeyboardInterrupt:
            print('Training terminated!')

        return model


    """
    Train model with random data
    :param model
    :param input_shape
    :param param **config: various config including parameters validation_split, batch_size, epochs, verbose
    :return continously retrained model
    """
    def train_random(self, model=None, b_max=None):
        if model is None:
            model = self.model
        else:
            model.compile(loss=keras.metrics.MSE, optimizer='adam')
            
        if b_max is None:
            b_max = self.config['anz_b']
        b_min = self.config['anz_b_phy']
        if b_max <= b_min:
            b_min = 0
        
        ex = exd.experimental_data(self.config)
        # inputs_random = 0
        # first_go = 1
        # for ii in range(self.config['rand_train']):
        input_random = np.random.rand(self.config['anz_inp_param'])
        b_random = np.random.randint(b_min,b_max)
        input_random = ex.make_bin_data(input_random,b_random)
            
        #     if self.config['rand_train'] >= b_max-b_min:
        #         b_s = np.mod(np.random.permutation(self.config['rand_train']),b_max-b_min)+b_min
        #     else:
        #         b_s = np.mod(np.random.permutation(b_max-b_min),b_max)+b_min
        #         b_s = b_s[0:self.config['rand_train']]
            
        
        # b_bin = np.array([ex.int_2_bin_arr(ii) for ii in b_s])
        # input_random = np.concatenate((b_bin,input_random),1)
        input_random = input_random[np.newaxis,:]
        output_true = model.predict(input_random)
        
        
        
        if self.config['rand_distrib'] == 0:
            output_random = np.random.rand(self.config['rand_train'],self.config['anz_out_param'])
            output_random = output_random*self.config['rand_range'] + output_true*(1-self.config['rand_range'])
        else:
            if self.config['rand_distrib'] == 1:
                output_random = np.random.normal(output_true, self.config['rand_range'])
        
       

        #print(b_bin.shape)
        #print(input_random.shape)
        #print(output_random.shape)

        model.fit(input_random, output_random, validation_split=0,
            batch_size=self.config['rand_train'], epochs=40, verbose=self.config['verbose'])

        return model


    def get_decompressed_layer_feature_extractor(self):
    #    feature_extractor = self.model(
    #        inputs=self.model.get_layer(name='compressed_layer').inputs,
    #        # outputs=[layer.output for layer in model.layers],
    #        outputs=self.model.output,
    #    )
        return None #feature_extractor
    
    def get_compressed_layer_feature_extractor(self):
        feature_extractor = keras.Model(
            inputs=self.model.inputs,
            # outputs=[layer.output for layer in model.layers],
            outputs=self.model.get_layer(name='compressed_layer').output,
        )
        feature_extractor.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                        loss=keras.metrics.MSE)
        return feature_extractor
    
    def encode(self,x):
        
        BV = self.encoder.predict(x)
        
        return BV
    
    def decode(self, x):
               
        y_lhs = self.decoder.predict(x)
        
        return y_lhs
    
    
    
    def sdnn(self, x_lhs = None):
        
        uti_obj = util.util(self.config)
        if x_lhs is None:
            x_lhs = uti_obj.get_x_lhs()
            
        count = 0
        for ii in range(self.config['pinn_reuse']):
            random_trained = self.train_random(self.get_copy())
            
            #save model in Model folder
            model_name = uti_obj.save_model(random_trained)
            if self.config['pf'] == 0:
                bs = range(self.config['anz_b_phy'],self.config['anz_b'])
            else:
                bs = [0]
            for jj in bs:
                y_lhs = self.pred_model(x_lhs,jj,random_trained)           
               
                y_lhs = y_lhs.reshape(-1)
                y_lhs_std = np.std(y_lhs)
                if y_lhs_std > 0.03:
                
                    array_name = "ylhs." + model_name + '.' + str(ii) + '.' + str(jj)
                    
                    uti_obj.save_array(y_lhs,'ylhs',array_name = array_name)  
                    #whot
                    x_rand = np.random.rand(self.config['rand_samples'],self.config['anz_inp_param'])  
                    y_rand = self.pred_model(x_rand,jj,random_trained)           
                   
                    y_rand = np.hstack((x_rand,y_rand)) 
                   
                    #y_rand = y_rand.reshape(-1)
        
                    array_name = "yrand." + model_name + '.' + str(ii) + '.' + str(jj)
                    
                    uti_obj.save_array(y_rand,'yrand',array_name = array_name)  
                    
                    count = count+1
            
        return count


    
    """
    Fetch intermidiate layer output
    Given the layer index
    Run lhs on each model
    :param model: original Model
    :param input_data: input data for the model as numpy array
    :layer_no: Index of targeted layer
    :return: layer output
    """
    def get_layer_output(self, input_data, layer_no, model=None):
        if model is None:
            model = self.model

        func = K.function([model.get_layer(index=0).input], model.get_layer(index=layer_no).output)
        layer_output = func([input_data])
        
        return layer_output

    
    def evaluate(self, input_data, output_data):
        
        n = int(input_data.shape[0]*(self.config['test_split']))
        input_data = input_data[:n,:]
        output_data = output_data[:n,:]
        return self.model.evaluate(input_data,output_data)
    