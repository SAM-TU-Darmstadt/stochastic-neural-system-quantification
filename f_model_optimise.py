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

def objective(trial = None):
    # Definieren Sie die Hyperparameter, die optimiert werden sollen
    config = conf_file.f_config()
    if not trial is None:
        config['verbose'] = 0
        dropout = trial.suggest_float('dropout', 0, 0.4, step=0.05)
        size_1st_layer = trial.suggest_int('size_1st_layer', 6, 128, step=2)
        size_last_layer = trial.suggest_int('size_last_layer', 1, 30, step=1)
        anz_layer = trial.suggest_int('anz_layer',3, 8)
        interpol = trial.suggest_int('interpol', -4, 4)
        activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'sigmoid'])
        rand_or_lhs = trial.suggest_categorical('rand_or_lhs',[True, False])
        batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
        phy_epochs = trial.suggest_int('phy_epochs', 0, 400, step=10)
        epochs = 400-phy_epochs
        multistep_fit = trial.suggest_int('multistep_fit', 1, 10)
        epochs = np.round(epochs/multistep_fit)
        phy_epochs = np.round(phy_epochs/multistep_fit)
        rand_or_lhs = trial.suggest_categorical('rand_or_lhs', [True, False])
    
        # Update config with hyperparameters
        config['dropout'] = dropout
        config['size_1st_layer'] = size_1st_layer
        config['size_last_layer'] = size_last_layer
        config['anz_layer'] = anz_layer
        config['interpol'] = interpol
        config['activation_function'] = activation_function
        config['batch_size'] = batch_size
        config['epochs'] = epochs
        config['multistep_fit'] = multistep_fit
        config['phy_epochs'] = phy_epochs
        config['rand_or_lhs'] = rand_or_lhs



# Create and train model
    
    
    [inp, out] = uti_obj.get_f_model_training_data(rnn_model=1)
    try:
        
        f_model = ffnn.feed_forward_nn(config)
        hist = f_model.train(inp,out)# Return the validation loss for the last epoch
        if 'val_loss' in hist.history:
            val_loss = hist.history['val_loss'][-1]
        else:
            val_loss = hist.history['val_mean_squared_error'][-1]
        f_model.save_as('trial'+str(trial.number))
    except Exception as e:
        print(e)
        print(str(e))
        val_loss =1
        
    
    # Return the validati
    
    return val_loss

f_config = conf_file.f_config()
rnn_config = conf_file.rnn_config()
rnn_config["verbose"] = 0
rnn_model = ffnn.feed_forward_nn(rnn_config)
rnn_model.load(model_name="trial_no_time_45")
# Initiate class instances
uti_obj = ut.util(f_config)
f_model = ffnn.feed_forward_nn(f_config)

# Fetch data, if you use rnn_model, it will take some time for the first time or try_loadin=False

uti_obj.config['rand_or_lhs'] = True
uti_obj.get_f_model_training_data(try_loadin=False, rnn_model=rnn_model)
uti_obj.config['rand_or_lhs'] = False
uti_obj.get_f_model_training_data(try_loadin=False, rnn_model=rnn_model)
    
# Erstellen Sie eine Optuna-Studie und optimieren Sie die Hyperparameter
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=5)

# Beste Hyperparameter abrufen
best_params = study.best_params
f_config.update(best_params)

print(f"Beste Hyperparameter: {best_params}")








