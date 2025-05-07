# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:20:36 2024

@author: wenzel
"""
from files import feed_forward_nn as ffnn
import config as conf_file
from files import util as ut
import optuna
from optuna.integration import TFKerasPruningCallback
import shutil
import os

# Define the inner optimization function for the autoencoder
def optimize_autoencoder(trial, config):
    
    ae_config = conf_file.ae_config()
    
    ae_config['batch_size'] = trial.suggest_int('batch_size', 16, 128)
    ae_config['multistep_fit'] = trial.suggest_int('multistep_fit', 1, 4)
    #config['epoch'] = trial.suggest_int('epoch', 20, 40)
    ae_config['ae_dimi'] = trial.suggest_int('ae_dimi', 1, 3)
    ae_config['ae_activation'] = trial.suggest_categorical('ae_activation', ['relu', 'tanh', 'sigmoid'])
    ae_config['ae_start_layer'] = trial.suggest_int('ae_start_layer', 60, 240)
    ae_config['ae_last_layer'] = trial.suggest_int('ae_last_layer', 5, 20)
    ae_config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    

    ae_config = conf_file.ae_config()
    ae_config['save_to'] = config['save_to']
    ae_config = conf_file.error_correct_config(ae_config)
    

    # Train the autoencoder
    uti_obj = ut.util(ae_config)
    data = uti_obj.get_ae_train_data()
    f_model = ffnn.feed_forward_nn(ae_config)
    history = f_model.train(data, data)
    
    return min(history.history['val_loss'])

# Define the outer optimization function for curve generation
def optimize_curve_generation(trial):
    config = conf_file.sdnn_config()
    config['size_b'] = trial.suggest_int('size_b', 10, 15)
    config['rand_train'] = trial.suggest_int('rand_train', 1, 3)
    config['rand_range'] = trial.suggest_float('rand_range', 0.0, 1.0)
    config['rand_distrib'] = trial.suggest_categorical('rand_distrib', [0, 1])
    config['pinn_reuse'] = trial.suggest_int('pinn_reuse', 1, 5)
    config['rand_samples'] = trial.suggest_int('rand_samples', 20, 100)
    
    # Generate curves and train the model
    config['save_to'] = os.path.join('sdnn_optimise',str(trial.number))
    config = conf_file.error_correct_config(config)
    if os.path.exists(config['save_to']) and os.path.isdir(config['save_to']):
        shutil.rmtree(config['save_to'])
    uti_obj = ut.util(config)
    x_lhs = uti_obj.get_x_lhs()
    count = 0
    while count < config['sdnn_samples']:
        f_model = ffnn.feed_forward_nn(config)
        f_model.model_fit_with_physical_data()
        hist=f_model.model_fit_with_mess_data()
        val_loss = hist.history['val_loss'][-1]

        if val_loss < 0.02:
            count = count + f_model.sdnn(x_lhs)

    # Perform inner optimization for autoencoder
    inner_study = optuna.create_study(direction='minimize')
    inner_study.optimize(lambda t: optimize_autoencoder(t, config), n_trials=30)
    
    return inner_study.best_value

# Create a study for the outer optimization
outer_study = optuna.create_study(direction='minimize')

# Optimize the outer function
outer_study.optimize(optimize_curve_generation, n_trials=20, n_jobs=7)

# Get the best parameters
best_params = outer_study.best_params
print(f"Best parameters: {best_params}")

#%%

import optuna

# Name der Studie und Speicherort der SQLite-Datenbank
study_name = "sdnn_to_ann"
storage_name = "sqlite:///existing_study.db"

# Erstellen der Storage-Objekts und Zuweisen der Studie zu dieser Storage
study = optuna.create_study(study_name=study_name, storage=storage_name, direction=outer_study.direction, load_if_exists=True)

