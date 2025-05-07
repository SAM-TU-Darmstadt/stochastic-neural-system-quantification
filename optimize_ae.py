# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 08:32:26 2024

@author: wenzel
"""
from files import feed_forward_nn as ffnn
import config_2 as conf_file
from files import experimental_data
from files import util as ut
import f_prediction_data_3
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
import os
import shutil

def objective(trial):
    # Definieren Sie die Hyperparameter, die optimiert werden sollen
    config = conf_file.ae_config()
    config['verbose'] = 0
    #size_b = trial.suggest_int('size_b', 2, 10, step=1)
    ae_start_layer = trial.suggest_int('ae_start_layer', 6, 128, step=2)
    ae_last_layer = trial.suggest_int('ae_last_layer', 1, 30, step=1)
    ae_dimi = trial.suggest_float('ae_dimi',1.1, 2, step=0.05)
    dropout = trial.suggest_float('dropout', 0, 0.4, step=0.05)
    ae_activation = trial.suggest_categorical('ae_activation', ['relu', 'tanh', 'sigmoid'])
    batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
    rand_or_lhs = trial.suggest_categorical('rand_or_lhs', [True, False])
    epochs = 800
    multistep_fit = trial.suggest_int('multistep_fit', 1, 10)
    epochs = np.round(epochs/multistep_fit)

    # Update config with hyperparameters
    config['dropout'] = dropout
    config['ae_start_layer'] = ae_start_layer
    config['ae_last_layer'] = ae_last_layer
    config['ae_dimi'] = ae_dimi
    #config['size_b'] = size_b
    config['ae_activation'] = ae_activation
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['multistep_fit'] = multistep_fit
    config['rand_or_lhs'] = rand_or_lhs


    # Create and train model
    try:
        # Initiate class instances
        uti_obj = ut.util(config)
        f_model = ffnn.feed_forward_nn(config)
    
        # Fetch data from csv file
        data = uti_obj.get_ae_train_data()
    
        hist = f_model.train(data,data)

        # Return the validation loss for the last epoch
        val_loss = hist.history['val_loss'][-1]
        
        f_model.save_as('trial'+str(trial.number))
    except Exception as e:
        val_loss = 1
        print(e)
    return val_loss


config = conf_file.ae_config()
config['verbose'] = 0

    
# Erstellen Sie eine Optuna-Studie und optimieren Sie die Hyperparameter
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=5)

# Beste Hyperparameter abrufen
best_params = study.best_params
config.update(best_params)

print(f"Beste Hyperparameter: {best_params}")

#%%

# Initiate class instances
uti_obj = ut.util(config)



ae_model = ffnn.feed_forward_nn(config)

# Save model
ae_model.load(model_name = 'trial24')

data = uti_obj.get_ae_train_data()
#get the behavioral Vector
BV = ae_model.encode(data)

#Save the behavioral Vector
uti_obj.save_array(BV, 'bv')


