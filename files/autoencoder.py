"""Define model for feed forward neural network with utilities"""

from tensorflow import keras

class autoencoder(feed_forward_nn):
    #define the architecture and all parameters
    archi_type = None
    
    #training parameter
    dropout = None
    validation_split = None    
    batch_size = None
    epochs = None
    verbose = 0
    
    
    """
    Initiate the init function
    :param input: sample input
    """
    def __init__(self,archi_type):
        super(feed_forward_nn, self).__init__(name='my_model')
        self.archi_type = archi_type
        self.set_architecture()
        
        

    """
    Train the neural network
    :param input: training input
    :return: processed output
    """
    
    def set_architecture(self):
        if self.archi_type == 0:
            self.dropout = 0.15
            self.validation_split = 0.2
            self.batch_size = 256
            self.epochs = 100
            self.add(keras.layers.Dense(62, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(31, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(16, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(8, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(4, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(2, activation='tanh'))
            self.add(keras.layers.Dropout(self.dropout))
            self.add(keras.layers.Dense(1, activation='tanh'))
            
            self.compile(optimizer='adam',#tf.train.RMSPropOptimizer(0.001),
                           loss=keras.metrics.mean_squared_error)
    
    def train(self,input_data,output_data):
        
        self.fit(input_data, output_data, self.validation_split,
                 batch_size=self.batch_size, epochs=self.epochs, verbose=0)