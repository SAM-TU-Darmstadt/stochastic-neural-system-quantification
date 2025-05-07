# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:08:25 2024

@author: wenzel
"""

from files import feed_forward_nn as ffnn
from files import util as ut
import numpy as np
import pandas as pd
import config_2 as config

pinn_config = config.pinn_config()
pinn_config['archi_type'] = 0

for ii in range(-5,5):

    pinn_config['interpol'] = ii    
    pinn_model = ffnn.feed_forward_nn(pinn_config)
    print(pinn_model.get_summary())