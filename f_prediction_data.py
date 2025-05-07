import numpy as np
import pandas as pd

excel_file_path1 = "Data_Cases.xlsx"
df = pd.read_excel(excel_file_path1)


def load_prediction_data(n):
    if n == 0:
        x = 136
        y = 148
    elif n == 1:
        x = 149
        y = 159

    # laser_power
    lp = df.loc[x:y, 'laser_power'].values
    # upper bound: 1000, lower bound: 100 [W]
    norm_lp = (lp - 100) / (1200 - 100)

    # scan_speed
    ss = df.loc[x:y, 'scan_speed'].values
    # upper bound: 1200, lower bound: 50 [mm/s]
    norm_ss = (ss - 50) / (1200 - 50)

    # hatch_dist
    hd = df.loc[x:y, 'hatch_dist'].values
    # upper bound: 120, lower bound: 10 [micro m]
    norm_hd = (hd - 10) / (120 - 10)

    # laser_sd
    lsd = df.loc[x:y, 'laser_sd'].values
    # upper bound: 100, lower bound: 25 [micro m]
    norm_lsd = (lsd - 20) / (100 - 20)

    # layer_thick
    lt = df.loc[x:y, 'layer_thick'].values
    # upper bound: 55, lower bound: 10 [micro m]
    norm_lt = (lt - 10) / (55 - 10)

    # powd_size
    ps = df.loc[x:y, 'powd_size'].values
    # upper bound: 50, lower bound: 15 [micro m]
    norm_ps = (ps - 15) / (50 - 15)

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