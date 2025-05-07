"""Different utility functions used in generic purpose"""
import os
import time
import numpy as np
from pyDOE import lhs
from tensorflow import keras
from files import feed_forward_nn as ffnn
from files import experimental_data as ed

class util:
    config = None

    def __init__(self, config):
        self.config = config
    """
    Load model from json
    :param keyword: keyword of the model
    :param path: path to store
    :file_index: which file to load, -1 for last (default) or 1 for first
    :return loaded_model
    """
    
    def load_model(self, keyword=None, path=None, file_index=-1, model_name=None):
        
        if keyword is None:
            keyword=self.config['name']
            
        if path is None:
            path=self.config['save_to_model']
                
        if model_name is None:
            model_name_list = self.get_load_list(path, keyword, 'json')
            model_name = model_name_list[file_index]
            
        model_name = os.path.join(path,model_name)
        json_file = open(model_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        
        #load weights into new model
        loaded_model.load_weights(model_name+'.weights.h5')
        return loaded_model
    
    
    """
    Save model json and  weight
    :param model: keras model to save
    :param path: path to store, defaulted
    :param model_name: model name, defaulted with timestamp
    :return filename
    """
    def save_model(self, model, path = None, model_name=None):
        
        
        if path is None:
            path=self.config['save_to_model']
        
        if model_name is None:
            model_name = self.config['name'] + "" + str(int(time.time()*1000))
         
        model_path_name = os.path.join(path, model_name)
        model_json = model.to_json()
        if not os.path.exists(path):
            os.makedirs(path)
        with open(model_path_name + '.json', 'w') as json_file:
            json_file.write(model_json)
        
        model.save_weights(model_path_name+'.weights.h5')
        print('Saved ' + self.config['name'] + '-model to disk successfully')
        return model_name
    
    """
    Load any array from csv-file
    :param keyword: keyword of the model
    :param path: path to store
    :file_index: which file to load, -1 for last (default) or 1 for first
    :return loaded_model
    """
    
    def load_array(self, keyword=None, path=None, file_index=-1, array_name=None):
        
        if path is None:
            path=self.config['save_to_csv']
                
        if array_name is None:
            if keyword is None:
                keyword='x_lhs'
            array_name_list = self.get_load_list(path, keyword, 'csv')
            
            if file_index < len(array_name_list) and array_name_list:
                array_name = array_name_list[file_index]
                array_name = os.path.join(path,array_name+'.csv')
                loaded_array = np.genfromtxt(array_name, delimiter=',')
                return loaded_array
            else:
                array_name_list = self.get_load_list(path, keyword, 'npy')
                if file_index < len(array_name_list) and array_name_list:
                    array_name = array_name_list[file_index]
                    array_name = os.path.join(path,array_name+'.npy')
                    loaded_array = np.load(array_name)
                    return loaded_array
                else:
                    return None
        else:
            array_name_tmp = os.path.join(path,array_name+'.csv')
            if os.path.exists(array_name_tmp):
                loaded_array = np.genfromtxt(array_name_tmp, delimiter=',')
            else:
                array_name = os.path.join(path,array_name+'.npy')
                loaded_array = np.load(array_name)
            return loaded_array    
            
    
    """
    Save model json and  weight
    :param model: keras model to save
    :param path: path to store, defaulted
    :param model_name: model name, defaulted with timestamp
    :return filename
    """
    def save_array(self,array, keyword, path=None , array_name=None):
        
        if array_name is None:
            array_name = keyword + "" + str(int(time.time()*1000))
        
        if path is None:
            path = self.config['save_to_csv']
        
        if not os.path.exists(path):
            os.makedirs(path)
        if len(array.shape)<3:
            array_path_name = os.path.join(path, array_name + '.csv')
            np.savetxt(array_path_name, array, delimiter=',', fmt='%e')
        else:
            array_path_name = os.path.join(path, array_name)
            np.save(array_path_name,array)
        #print('Saved ' + keyword + ' to disk successfully')
        return array_name
    
    
    """
    Fetch a list of all loadables from a directory
    :param path: path where loadables are stored
    :param keyword: what should be loaded
    :param file_type: what filetypes are looked at
    :return: list of loaded models
    """ 
    def get_load_list(self, path, keyword, file_type):        
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        files = [f for f in os.listdir(path) if f.endswith(file_type) and f.startswith(keyword)]
        files_sorted = sorted(files)
        files_without_extension = [os.path.splitext(file)[0] for file in files_sorted]
        
        return files_without_extension


    """
    Fetch multiple models from a directory
    Load models with archi 1
    Run lhs on each model
    With output run models with archi 2
    :param path: path where models are stored, default Model
    :return: FNN1(lhs) = out1; FNN2(out1)=out
    """ 
    def load_models_lhs(self, path=None, dropout=0.15, input_shape=(128,), **config):
        if path is None:
            path = self.config['models']
        models = self.get_saved_models(path)
        lhs_models = []
        

        archi_1 = ffnn.feed_forward_nn(0, dropout, input_shape, **config)
        archi_2 = ffnn.feed_forward_nn(20, dropout, input_shape, **config)
        exd = ed.experimental_data()
        for model in models:
            model = archi_1.model
            out_1 = exd.get_ylhs(model)
            lhs_models.append(out_1)
            model = archi_2.model
            out_2 = exd.get_ylhs(model)
            lhs_models.append(out_2)

        return lhs_models
    


    def get_ae_train_data(self, anz=None, try_loadin = False, not_for_rnn = True):
        
        rand_or_lhs = self.config['rand_or_lhs']
        
        ae_train_data = None
        
        for_rnn = ''
        if not_for_rnn:
            for_rnn = 'flatt'
        else:
            for_rnn = '_no_flatt'
        if try_loadin:
                    
                
            if rand_or_lhs:
                ae_train_data = self.load_array('ae_train_data_rand'+for_rnn)
            else:
                ae_train_data = self.load_array('ae_train_data_lhs'+for_rnn)
        
        # Get the list of file names to load
        if ae_train_data is None:
            if rand_or_lhs:
                load_list = self.get_load_list(self.config['save_to_csv'], 'yrand', '.csv')
            else:
                load_list = self.get_load_list(self.config['save_to_csv'], 'ylhs', '.csv')
            
            lst = len(load_list)
            
            # If no specific number of samples is provided, use all available samples
            if anz is None:
                anz = lst
            
            # Calculate the starting index to load data from
            frst = np.amax((lst - anz, 0))
            
            if frst == lst:
                print('Warning! No AE Training data is loaded!')
                
            # Load the train data arrays using a list comprehension
            train_data = [self.load_array(array_name=load_list[ii]) for ii in range(frst, lst)]
            
            # train_data enthält entweder Arrays mit dim 1 oder 2
            flattened_arrays = []
            
            for arr in train_data:
                # Prüfen, ob Array 2D ist (z.B. (10, 2))
                if arr.ndim == 2 and not_for_rnn:
                    flattened_arrays.append(arr.flatten())  # In ein flaches z.B. (20,) Array umwandeln
                # Prüfen, ob Array bereits flach ist (z.B. (20,))
                else:
                    flattened_arrays.append(arr)
                if arr.ndim < 1 or arr.ndim > 2:
                    raise ValueError(f"Unexpected array shape: {arr.shape}")
            
            # Die Arrays zu einem einzigen Array mit dynamischer Form umwandeln
            ae_train_data = np.array(flattened_arrays)
            
        
        
        if rand_or_lhs:
            self.save_array(ae_train_data,'ae_train_data_rand'+for_rnn)
        else:
            self.save_array(ae_train_data,'ae_train_data_lhs'+for_rnn)
    
        # Return the NumPy array containing the train data
        return ae_train_data

    """
    Get RNN Training data
    Tries to load RNN training data
    if unsuccessfull create trainingdata based on LHS
    With output run models with archi 2
    :param path: path where models are stored, default Model
    :return: FNN1(lhs) = out1; FNN2(out1)=out
    """ 
    
    def get_rnn_training_data(self, try_loadin = True):
        rand_or_lhs = self.config['rand_or_lhs']
        rnn_train_input = None
        if try_loadin:
            if rand_or_lhs:
                rnn_train_input = self.load_array('rnn_train_input_rand')
                rnn_train_output = self.load_array('rnn_train_output_rand')
            else:
                rnn_train_input = self.load_array('rnn_train_input_lhs')
                rnn_train_output = self.load_array('rnn_train_output_lhs')
                
                
        
        if rnn_train_input is None:
        
            bv = self.load_array('bv')
            self.config['rand_or_lhs'] = True
            yrand = self.get_ae_train_data(not_for_rnn = False)
            self.config['rand_or_lhs'] = False
            ylhs = self.get_ae_train_data()
            xlhs = self.get_x_lhs()
            self.config['rand_or_lhs'] = rand_or_lhs
            
            if ylhs.shape[0] > bv.shape[0]: #ongoing training
                ylhs = ylhs[0:bv.shape[0],:]
            
            ylhs_shape = ylhs.shape
            
            anz_cases = ylhs_shape[0]
            
            
            inp_cases_list = []
            out_cases_list = []
            
    
            # NEWLY ADDED: Create an array of indices and shuffle it
            indices = np.arange(anz_cases)
            np.random.shuffle(indices)
    
            for cases in indices:
            
                ylhs_tmp = ylhs[cases, :].T
                if self.config['anz_inp_param'] == 1:
                    xlhs_tmp = xlhs[:, np.newaxis]
                else:
                    xlhs_tmp = xlhs
                if self.config['anz_out_param'] == 1:
                    ylhs_tmp = ylhs_tmp[:, np.newaxis]
                inp_case = np.concatenate((xlhs_tmp, ylhs_tmp), 1)
                    
                    
                    
                inp_case_add = yrand[cases, :]
                inp_case = np.concatenate((inp_case, inp_case_add), 0)
                
                out_case = bv[cases]
                out_case = np.tile(out_case, (inp_case.shape[0], 1))
    
                perm = np.random.permutation(inp_case.shape[0])
                
    
                inp_cases_list.append(inp_case[perm,:])
                out_cases_list.append(out_case)
            
            # Convert the lists to 3D NumPy arrays
            rnn_train_input = np.array(inp_cases_list)
            rnn_train_output = np.array(out_cases_list)
            
            if rand_or_lhs:
                self.save_array(rnn_train_input,'rnn_train_input_rand')
                self.save_array(rnn_train_output,'rnn_train_output_rand')
            else:
                self.save_array(rnn_train_input,'rnn_train_input_lhs')
                self.save_array(rnn_train_output,'rnn_train_output_lhs')
            
                            
        return [rnn_train_input, rnn_train_output]
    

    def get_f_model_training_data(self, try_loadin = True, rnn_model = None):
        if rnn_model is None:
            methode = 0
        else:
            methode = 1
        f_train_input = None
        rand_or_lhs = self.config['rand_or_lhs']
        if try_loadin:
            if rand_or_lhs:
                f_train_input = self.load_array('f_train_input_rand_' + str(methode))
                f_train_output = self.load_array('f_train_output_rand_' + str(methode))
            else:
                f_train_input = self.load_array('f_train_input_lhs_' + str(methode))
                f_train_output = self.load_array('f_train_output_lhs_' + str(methode))
                
        if f_train_input is None:
            bv = self.load_array('bv')
            ylhs = self.get_ae_train_data(not_for_rnn=False)
            print(ylhs.shape)
            if rand_or_lhs:
                len_lhs = ylhs.shape[1]
            else:
                
                xlhs = self.get_x_lhs()
                
                len_lhs = xlhs.shape[0]
                
            num_features = self.config['size_b']   
            anz_inp = self.config['anz_inp_param']
            num_cases = ylhs.shape[0]
            inp_cases_list = []
            out_cases_list = []
            
            for case_idx in range(num_cases):
                ylhs_case = ylhs[case_idx, :]
                if (not rand_or_lhs) and self.config['anz_out_param'] == 1:
                        ylhs_case = ylhs_case[:, np.newaxis]
                if methode == 0:
                    bvs = np.full((len_lhs, num_features), bv[case_idx])
                else:
                    if not rand_or_lhs:
                        ylhs_tmp = ylhs[case_idx, :].T
                        xlhs_tmp = xlhs
                        if self.config['anz_inp_param'] == 1:
                            xlhs_tmp = xlhs[:, np.newaxis]
                        if self.config['anz_out_param'] == 1:
                            ylhs_tmp = ylhs_tmp[:, np.newaxis]
                        inp_rnn = np.concatenate((xlhs_tmp, ylhs_tmp), 1)
                    else:                    
                        inp_rnn = ylhs[case_idx, :]
                    inp_rnn = np.array(inp_rnn)
                    inp_rnn =inp_rnn[np.newaxis,:]
                    #print(inp_rnn.shape)                    
                    bvs = rnn_model.pred_model(inp_rnn)
                    #print(bvs)
                    bvs = np.squeeze(bvs)
#                    if self.config['rnn_bv_mean']:
                    #bvs = np.mean(bvs,0)
                    bvs = np.full((len_lhs, num_features), bvs)
                if rand_or_lhs:
                    inp_case = np.hstack((bvs, ylhs_case[:,0:anz_inp] ))
                    out_case = ylhs_case[:,anz_inp:]
                    
                else:                        
                
                    inp_case = np.hstack((bvs,xlhs_tmp))
                    out_case = ylhs_case
                
                inp_cases_list.append(inp_case)
                out_cases_list.append(out_case)           
            # Convert the lists to 3D NumPy arrays
            inp_cases_array = np.array(inp_cases_list)
            out_cases_array = np.array(out_cases_list)
            
            # Reshape to combine the first two dimensions
            f_train_input = inp_cases_array.reshape((-1, inp_cases_array.shape[-1]))
            f_train_output = out_cases_array.reshape((-1, out_cases_array.shape[-1]))
            
            #If rnn is present add physical data
            if methode == 2:
                ex_d = ed.experimental_data(self.config)
                [train_x,train_y] = ex_d.load_messurement_data()
                train_x = train_x[:,self.config["size_b"]:]
                if  self.config['anz_out_param'] == 1:
                        train_y = train_y[:, np.newaxis]
                inp_rnn = np.concatenate((train_x, train_y), 1)
                inp_rnn =inp_rnn[np.newaxis,:]
                print(inp_rnn.shape)                    
                bvs = rnn_model.pred_model(inp_rnn)
                #print(bvs)
                bvs = np.squeeze(bvs)
                if self.config['rnn_bv_mean']:
                    bvs = np.mean(bvs,0)
                    bvs = np.full((train_y.shape[0], num_features), bv[case_idx])
            
                inp_case = np.hstack((bvs,train_x))
                out_case = train_y
                for ii in range(self.config['exp_data_add']):
                    f_train_input = np.vstack((f_train_input,inp_case))
                    f_train_output = np.vstack((f_train_output,out_case))
                
                      
                

            # Generate a permutation index
            permutation_index = np.random.permutation(f_train_input.shape[0])
    
            # Shuffle both matrices using the same permutation index
            f_train_output = f_train_output[permutation_index]
            f_train_input = f_train_input[permutation_index]
            
            if rand_or_lhs:
                self.save_array(f_train_input,'f_train_input_rand_' + str(methode))
                self.save_array(f_train_output,'f_train_output_rand_' + str(methode))
            else:
                self.save_array(f_train_input,'f_train_input_lhs_' + str(methode))
                self.save_array(f_train_output,'f_train_output_lhs_' + str(methode))
                            
        return [f_train_input, f_train_output]

    
    
    def get_x_lhs(self):
        x_lhs = self.load_array('x_lhs')
        if x_lhs is None:
            # Generieren des Latin Hypercube Samplings
            random_seed = 123  # set the random seed for reproducebility of x_lhs
            np.random.seed(random_seed)
            x_lhs = lhs(self.config['anz_inp_param'], samples=self.config['len_lhs'])
            
            self.save_array(x_lhs, 'x_lhs')
                 
                
        return x_lhs

