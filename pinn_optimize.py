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
    config = conf_file.pinn_config()
    if not trial is None:
        config['verbose'] = 0
        dropout = trial.suggest_float('dropout', 0, 0.4, step=0.05)
        size_1st_layer = trial.suggest_int('size_1st_layer', 6, 128, step=2)
        size_last_layer = trial.suggest_int('size_last_layer', 1, 30, step=1)
        anz_layer = trial.suggest_int('anz_layer',3, 8)
        interpol = trial.suggest_int('interpol', -4, 4)
        activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'sigmoid'])
        batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
        phy_epochs = trial.suggest_int('phy_epochs', 0, 400, step=10)
        epochs = 400-phy_epochs
        multistep_fit = trial.suggest_int('multistep_fit', 1, 10)
        epochs = np.round(epochs/multistep_fit)
        phy_epochs = np.round(phy_epochs/multistep_fit)
    
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



# Create and train model
    val_loss = np.ones(config['anz_b_phy'])
    for ii in range(config['anz_b_phy']):
        try:
            f_model = ffnn.feed_forward_nn(config)
            f_model.model_fit_with_physical_data(ii)
            hist = f_model.model_fit_with_mess_data()
            val_loss[ii] = hist.history['val_loss'][-1]
        except Exception as e:
            print(e)
            print(str(e))
            continue
        
        # Return the validation loss for the last epoch
        if not trial is None:
            f_model.save_as('trial'+str(trial.number)+'_b'+str(ii))
    val_loss = np.mean(val_loss)
    return val_loss

#objective()
config = conf_file.pinn_config()
config['verbose'] = 0
#$if os.path.exists(config['save_to']) and os.path.isdir(config['save_to']):
#    shutil.rmtree(config['save_to'])
    
# Erstellen Sie eine Optuna-Studie und optimieren Sie die Hyperparameter
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=5)

# Beste Hyperparameter abrufen
best_params = study.best_params
config.update(best_params)

print(f"Beste Hyperparameter: {best_params}")