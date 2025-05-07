"""Fetch and formulate experimental data"""
import numpy as np
import os
from collections.abc import Iterable

class experimental_data:
    anz_param = None
    anz_b = None
    len_lhs = None
    config = None
    
    """
    Initiate the constructor function
    :param anz_param
    :param anz_b
    """
    def __init__(self, config):
        self.anz_param = config['anz_inp_param']
        self.anz_b = config['anz_b']
        self.len_lhs = config['len_lhs']
        self.config = config

    """
    lhs value of x
    :return lhs value
    """


    
    """
    Generate lhs value
    :param model
    :param b
    :param b_range
    :return: lhs value
    """
    def get_ylhs(self, model, b=None, b_range=None):
        
        if b_range is None:
            b_range = self.config['anz_b_phy']
        
        if b is None:
            bin_bs = [self.int_2_bin_arr(c,0) for c in range(b_range)]
            b = np.array(bin_bs)


        shape_b = np.shape(b)
        if len(shape_b) == 2:
            ylhs = []
            for b_ii in range(shape_b[0]):
                ylhs.append(self.get_ylhs(model,b[b_ii,:]))
            ylhs = np.array(ylhs)
        else:
            inp_b = [b for ii in range(self.len_lhs)]
            inp_b = np.array(inp_b)
            inp = np.hstack((inp_b,self.get_xlhs()))
            ylhs = model.predict(inp)
        return ylhs

    
    """
    Generate physical data
    :param b
    :param inp
    :param x_anz
    :return: [input, output]
    """
    def generate_physical_data(self, b=None, inp = None, x_anz = None, try_loading = True):
        # Create parameters to teach
        # Number of values
        if try_loading:
            if b is None:
                inp_path = os.path.join(self.config['save_to_csv'], 'inp_phy.csv')
                out_path = os.path.join(self.config['save_to_csv'], 'out_phy.csv')
            else:
                inp_path = os.path.join(self.config['save_to_csv'], 'inp_phy_b_' + str(b) + '.csv')
                out_path = os.path.join(self.config['save_to_csv'], 'out_phy_b_' + str(b) + '.csv')
            
            gen_new = os.path.exists(inp_path) and os.path.exists(out_path)
            gen_new = not gen_new
            
            if self.config['pinn_gen_new'] or gen_new:
                inp_out = 0
                if b is None:
                    bs = range(self.config['anz_b_phy'])
                elif isinstance(b, Iterable):
                    bs = b
                else:
                    bs = [b]
                first_run = True
                for b in bs:
                    [inp_tmp,outp_tmp] = self.generate_physical_data(b,inp, x_anz,try_loading=False)
                    anz = inp_tmp.shape
                    #x_anz = anz[0]

                    if first_run:
                        inp_out = inp_tmp
                        outp_out= outp_tmp
                        first_run =False
                    else:
                        inp_out = np.concatenate((inp_out,inp_tmp))
                        outp_out = np.concatenate((outp_out,outp_tmp))
                if not (self.config["pinn_overwrite_old"] or gen_new): 
                    inp_phy = np.genfromtxt(inp_path, delimiter=',')
                    outp_phy = np.genfromtxt(out_path, delimiter=',')
                    inp_out = np.concatenate((inp_phy,inp_out))
                    outp_out = np.concatenate((outp_phy,outp_out))
                    
                
                        
                if not os.path.exists(self.config['save_to_csv']):
                    os.makedirs(self.config['save_to_csv'])
                np.savetxt(inp_path, inp_out, delimiter=',', fmt='%f')
                np.savetxt(out_path, outp_out, delimiter=',', fmt='%f')
                    
            else:
                inp_out = np.genfromtxt(inp_path, delimiter=',')
                outp_out = np.genfromtxt(out_path, delimiter=',')
            return [inp_out, outp_out]
        
        if inp is None: 
            if x_anz is None:
                x_anz = self.config["pinn_samples"]
            inp = np.random.rand(x_anz,self.anz_param)
            outp = np.zeros((x_anz,self.config['anz_out_param']))
        else:
            anz = inp.shape
            x_anz = anz[0]
            outp = np.zeros(x_anz)
            

        [inp,outp] = self.config["generate_physical_training_data"](b,inp)
        
        # inp_size = inp.shape
        # x_anz =inp_size[0]
        # if self.config['pf'] == 0:
        #     b_bin = self.int_2_bin_arr(b)
        #     b_bin = np.array([b_bin for tmp in range(x_anz)])
        # else:
        #     b_bin = np.zeros(((x_anz,self.config['size_b'])))
        # inp = np.hstack((b_bin,inp))
        inp = self.make_bin_data(inp, b)
        return [inp,outp]
        
    """
    Load meassurmente data
    :return: [input, output]
    """
    def load_messurement_data(self, b = None):
        if b is None:
            if self.config['pf'] == 0:
                b = range(self.config['anz_b_phy'], self.config['anz_b'])
            else:
                b = [np.random.randint(self.config['anz_b_phy'], self.config['anz_b'])]
        first_run = True
        for ii in b:
            [inp_tmp, outp_tmp] = self.config["load_mess_data"](ii)
            # inp_size = inp_tmp.shape
            # x_anz = inp_size[0]
            # if self.config['pf'] == 0:
            #     b_bin = self.int_2_bin_arr(ii)
            #     b_bin = np.array([b_bin for tmp in range(x_anz)])
            # else:
            #     b_bin = np.zeros(((x_anz,self.config['size_b'])))
            # inp_tmp = np.hstack((b_bin, inp_tmp))
            inp_tmp = self.make_bin_data(inp_tmp, ii)
            if first_run:
                inp_out = inp_tmp
                outp_out= outp_tmp
            else:
                inp_out = np.concatenate((inp_out,inp_tmp))
                outp_out = np.concatenate((outp_out,outp_tmp))
            first_run = False

        return [inp_out, outp_out]


    """
    Int to numpy array
    :param integ
    :param length
    :return: numpy array
    """
    def int_2_bin_arr(self, integ, length = None):
        if length is None:
            length = self.config['size_b']
        integ = int(integ)
        if self.config['bin_or_one_hot'] == 0:
            integ = integ+1
            tmp_str = format(integ, "b")
            bin_arr = [c for c in tmp_str]
            bin_arr = np.array(bin_arr)
            bin_arr = bin_arr.astype(np.float64)
    
            while len(bin_arr)<length:
                bin_arr = np.insert(bin_arr,0,0)
            while len(bin_arr) > length:
                bin_arr = np.delete(bin_arr,0)
            return bin_arr
        else:
            one_hot_array = np.zeros(length)
            if integ > -1:
            # Setze den Wert an der Stelle des Integers auf 1
                one_hot_array[integ] = 1
            
            return one_hot_array
        
    
    """
    Array to int
    :param bin_arr
    :return: int value
    """
    def bin_arr_2_int(self, bin_arr):
        integ = 0
        if self.config['bin_or_one_hot'] == 0:
            for bii in bin_arr:
                integ = integ*2
                integ += bii
            integ = integ-1
            return int(integ)
        else:
            for bii in bin_arr:
                if bii==1:
                    return int(integ)
                integ = integ+1
            return 0
    
    
    def make_bin_data(self, input_data, b):
        if self.config['anz_inp_param'] == 1 and input_data.ndim == 1:
            input_data = input_data[:, np.newaxis]
        
        if input_data.ndim == 1:
            if self.config['pf'] == 0:
                return np.concatenate((self.int_2_bin_arr(b),input_data))
            
            else:
                b_bin = np.zeros(((self.config['size_b'])))
                return np.concatenate((b_bin,input_data))
        else:
            
            inp_anz = input_data.shape[0]
            if self.config['pf'] == 0:
                b_bin = np.array([self.int_2_bin_arr(b) for ii in range(inp_anz)])
            else:
                b_bin = np.zeros((inp_anz,self.config['size_b']))
            return np.hstack((b_bin,input_data))

        