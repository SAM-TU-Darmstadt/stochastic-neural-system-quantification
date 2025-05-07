"""Define model for training data and run training for recurrent neural network with utilities"""
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import genfromtxt
from csv import writer

class recurrent_nn:
    xlhs = None
    ylhs = None
    bv = None
    sys_range = None
    seq_range = None
    len_lhs = None
         

    """
    Create rnn model
    :return: Created RNN model
    """
    def create_model(self):
        model = keras.Sequential()

        model.add(layers.GRU(units=64, input_shape=(125,1), return_sequences=True))
        #model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dropout(.15))
        model.add(layers.LSTM(units=18))
        #model.add(layers.Dense(units=18, activation='relu'))

        model.compile(optimizer="adam", loss=keras.metrics.mean_squared_error)

        return model
    
    """
    Generate training input/output row
    :param sys_index
    :param seq_index
    :return: input, output row
    """
    def generate_ior(self, sys_index, seq_index):
        Y = self.ylhs[sys_index,2:][seq_index]
        X = self.xlhs[seq_index]
        inp_r = np.append(X, Y)
        out_r = self.bv[sys_index,2:]
        return inp_r, out_r


    """
    Generate training input/output csv file
    :param inp_path
    :param out_path
    :return: Generated csv files
    """
    def generate_io_csv(self, **config):
        
        self.xlhs = config['xlhs'] if 'xlhs' in config else genfromtxt('CSV_2/x_lhs_Model_2_.csv', delimiter=',')
        self.ylhs = config['ylhs'] if 'ylhs' in config else genfromtxt('CSV_2/y_lhs_Model_2_.csv', delimiter=',')
        self.bv = config['bv'] if 'bv' in config else genfromtxt('CSV_2/bv_Model_2_.csv', delimiter=',')
        self.sys_range = config['sys_range'] if 'sys_range' in config else self.ylhs.shape[0]
        self.seq_range = config['seq_range'] if 'seq_range' in config else 20
        self.len_lhs = config['len_lhs'] if 'len_lhs' in config else 128
        inp_path = config['inp_path'] if 'inp_path' in config else 'CSV_2/rnn_train_input.csv'
        out_path = config['out_path'] if 'out_path' in config else 'CSV_2/rnn_train_output.csv'
        # Create csv file if not already there
        if not os.path.exists(inp_path):
            open(inp_path, 'w').close()
        if not os.path.exists(out_path):
            open(out_path, 'w').close()

        s_idx = list(range(self.sys_range))
        random.shuffle(s_idx)

        for sys_index in s_idx:
            for i in range(self.seq_range):
                seq_index = random.randint(0, self.len_lhs - 1)
                t_in, t_out = self.generate_ior(sys_index, seq_index)

                with open(inp_path, 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(t_in)
                    f_object.close()

                with open(out_path, 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(t_out)
                    f_object.close()


    """
    Train RNN model with train data
    :param t_in_file
    :param t_out_file
    :param model_path
    :param fast_train
    :return: Generated csv files
    """
    def train(self, t_in_file = 'CSV_2/rnn_train_input_50.csv', t_out_file = 'CSV_2/rnn_train_output_50.csv', model_path = 'AE_Model/rnn_50.h5', fast_train = False):
        t_in = genfromtxt(t_in_file, delimiter=',')
        t_out = genfromtxt(t_out_file, delimiter=',')

        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
        else:
            model = self.create_model()

        if fast_train:
            n = 10000
            test_data_size = 1000
        else:
            n = int(t_out.shape[0]*0.85)
            test_data_size = t_out.shape[0] - n
        x_train = t_in[:n].reshape(t_in[:n].shape[0], t_in[:n].shape[1], 1)
        y_train = t_out[:n]

        model.fit(x_train, y_train, batch_size=256, epochs=20, validation_split=0.2)
        model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
        model.fit(x_train, y_train, batch_size=16, epochs=3, validation_split=0.2)

        if fast_train:
            m = n + test_data_size
            x_test = t_in[n:m].reshape(t_in[n:m].shape[0], t_in[n:m].shape[1], 1)
            y_test = t_out[n:m]
        else:
            x_test = t_in[n:].reshape(t_in[n:].shape[0], t_in[n:].shape[1], 1)
            y_test = t_out[n:]

        test_loss = model.evaluate(x_test, y_test)
        print(test_loss)

        model.save(model_path)
        