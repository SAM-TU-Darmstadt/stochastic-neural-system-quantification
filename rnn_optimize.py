# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 08:32:26 2024

@author: wenzel
"""
from files import feed_forward_nn as ffnn
import config_2 as config_file
from files import experimental_data
from files import util as ut
import f_prediction_data_3
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
import os
import traceback
import shutil

def objective(trial):
    # Definieren Sie die Hyperparameter, die optimiert werden sollen
    config = config_file.rnn_config()
    config['verbose'] = 0
    dropout = trial.suggest_float('dropout', 0, 0.4, step=0.05)
    rnn_start_layer = trial.suggest_int('rnn_start_layer', 2, 128, step=2)
    size_1st_layer = trial.suggest_int('size_1st_layer', 2, 128, step=2)
    size_last_layer = trial.suggest_int('size_last_layer', 2, 30, step=1)
    anz_layer = trial.suggest_int('anz_layer',2, 8)
    interpol = trial.suggest_int('interpol', -4, 4)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'sigmoid'])
    batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
    epochs = 20
    multistep_fit = trial.suggest_int('multistep_fit', 1, 10)
    epochs = np.round(epochs/multistep_fit)
    config['rnn_start_layer'] = 24
    rand_or_lhs = trial.suggest_categorical('rand_or_lhs', [True, False])

    # Update config with hyperparameters
    config['dropout'] = dropout
    config['rnn_start_layer'] = rnn_start_layer
    config['size_1st_layer'] = size_1st_layer
    config['size_last_layer'] = size_last_layer
    config['anz_layer'] = anz_layer
    config['interpol'] = interpol
    config['activation_function'] = activation_function
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['multistep_fit'] = multistep_fit
    config['rand_or_lhs'] = rand_or_lhs



    # Create and train model
    hist = 5
    try:

        # Initiate class instances
        uti_obj = ut.util(config)
        rnn_model = ffnn.feed_forward_nn(config)

        # Fetch physical data from csv file
        [inp,out] = uti_obj.get_rnn_training_data()
        out = out[:,0,:]
        #print(inp)
        #print(out)
        hist =rnn_model.train(inp,out)
        #rnn_model.load()

        #test_loss = rnn_model.evaluate(inp,out)
        val_loss = 1
        # Return the validation loss for the last epoch
        if 'val_loss' in hist.history:
            val_loss = hist.history['val_loss'][-1]
        else:
            val_loss = hist.history['val_mean_squared_error'][-1]
                
        
        
        rnn_model.save_as('trial_2nd_'+str(trial.number))
    except Exception as e:
        val_loss = 1
        print("Error: ", e)
        traceback.print_exc()
        print(hist)
        try:
            if not isinstance(hist, int):
                print(hist.history)
        finally:
            print('no train, next time brother')
    return val_loss



config = config_file.rnn_config()
config['verbose'] = 0

# Erstellen Sie eine Optuna-Studie und optimieren Sie die Hyperparameter
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=5)

# Beste Hyperparameter abrufen
best_params = study.best_params
config.update(best_params)

print(f"Beste Hyperparameter: {best_params}")

#lpbf
#loss: 0.011140997521579266
#trial_no_time_16

#fff
#loss: 0.007500983309000730
#trial_no_time_45