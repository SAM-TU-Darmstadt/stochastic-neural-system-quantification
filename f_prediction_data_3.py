# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:46:32 2024

@author: 49157
"""

import numpy as np
import pandas as pd

excel_file_path1 = "Test_Data_Aconity.xlsx"
df = pd.read_excel(excel_file_path1)


def load_prediction_data(n):
    if n == 0:
        x = 12
        y = 51
    elif n == 1:
        x = 0
        y = 11

    # laser_power
    lp = df.loc[x:y, 'laser_power'].values
    # upper bound: 200, lower bound: 100 [W]
    norm_lp = (lp - 100) / (200 - 100)

    # scan_speed
    ss = df.loc[x:y, 'scan_speed'].values
    # upper bound: 1200, lower bound: 50 [mm/s]
    norm_ss = (ss - 50) / (1200 - 50)

    # hatch_dist
    hd = df.loc[x:y, 'hatch_dist'].values
    # upper bound: 70, lower bound: 30 [micro m]
    norm_hd = (hd - 30) / (70 - 30)

    # laser_sd
    lsd = df.loc[x:y, 'laser_sd'].values
    # upper bound: 80, lower bound: 40 [micro m]
    norm_lsd = (lsd - 40) / (80 - 40)

    # layer_thick
    lt = df.loc[x:y, 'layer_thick'].values
    # upper bound: 40, lower bound: 20 [micro m]
    norm_lt = (lt - 20) / (40 - 20)

    # powd_size
    ps = df.loc[x:y, 'powd_size'].values
    # upper bound: 30, lower bound: 20 [micro m]
    norm_ps = (ps - 20) / (30 - 20)

    # combine all normalized input parameters in one matrix
    inp_matrix = np.column_stack((norm_lp, norm_ss, norm_hd, norm_lsd, norm_lt, norm_ps))

    # goal value: rel_dens
    dens = df.loc[x:y, 'rel_dens'].values


    log_trans_dens = np.log(101 - dens.astype(float))

    lower_bound = np.log(1)
    upper_bound = np.log(31.0)

    inv_outp = (log_trans_dens - lower_bound) / (upper_bound - lower_bound)
    outp = 1 - inv_outp

    x_data = inp_matrix
    y_data = outp

    return [x_data, y_data]

# check if the data is properly normalized
#print(load_prediction_data(0))