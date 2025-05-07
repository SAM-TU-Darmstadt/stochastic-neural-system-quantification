
from files import feed_forward_nn as ffnn
import config_rf as conf_file
from files import experimental_data
from files import util as ut
import f_prediction_data_3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

vers = 1
#%%
def S_1(x_0):
    return x_0**3 + 0.4 * x_0**2 - 0.5 * x_0 + 0.1

def S_2(x_0):
    return 0.5 * x_0**3 + 0.4 * x_0**2 - 0.5 * x_0 + 0.2

def S_3(x_0):
    return 0.25 * x_0**3 + 0.6 * x_0**2 - 0.3 * x_0 + 0.2

# Funktion zum Zeichnen des Plots
def draw_plot_sample(enum, lbl='S_1', col = ''):
    anzP = 1000  # Anzahl der Punkte
    x = np.arange(0, 1, 1 / anzP)  # x-Werte von 0 bis 1
    y = np.zeros(anzP)  # y-Werte initialisieren

    # Wähle die richtige Funktion basierend auf dem lbl
    if enum == 1:
        func = S_1
    elif enum == 2:
        func = S_2
    elif enum == 3:
        func = S_3
    else:
        raise ValueError("Ungültiges Label. Bitte 'S_V' oder 'S_R' verwenden.")

    # Berechne y-Werte basierend auf der gewählten Funktion
    for i in range(anzP):
        y[i] = func(x[i])

    # Plot
    plt.plot(x, y, label= lbl, color = col)


      
def draw_plot_model(model,config, b=0,lbl = '',linstyl = 'solid', col = '', noplot = 0):
    
    exp = experimental_data.experimental_data(config)
    anzP = 1000
    x = np.arange(0,1,1/anzP)   
    x = x.reshape(1000, 1) 
    y_pred = model.pred_model(x,b)
    if noplot == 0:
        plt.plot(x,y_pred,label=lbl,linestyle = linstyl,color = col) 
    return (x,y_pred)
    
def setup_plot(x_label='X', y_label='Z', label='Trainingspunkt', width=6, height=4):
    plt.figure(figsize=(width, height))  # Plotgröße
    
    plt.rcParams.update({
        'font.size': 12,
        'legend.fontsize': 9,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.usetex': True,  # Optional LaTeX-Stil,
    })
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    
    
tu_colors = [
    "#C9308E",  # 247
    "#DDDF48",  # 379
    "#E9503E",  # 1795
    "#5D85C3",  # 2727
    "#50B695",  # 338
    "#AFCC50",  # 381
    "#804597",  # 2685
    "#FFE05C",  # 113
    "#F8BA3C",  # 129
    "#EE7A34",  # 158
    "#009CDA"   # 299
]


    #%% plot von referenzsystem
setup_plot()

draw_plot_sample(1,'S_1',tu_colors[1])  
draw_plot_sample(2,'S_2',tu_colors[2])   
draw_plot_sample(3,'S_3',tu_colors[0])  

plt.legend(loc='upper left')
plt.savefig('plots/rs_multi_plot.png',dpi=300)
plt.show()
    

  #%% plot von wissenstransfer 

config = conf_file.pinn_config()
y_pred_all = []
y_pred_2_all = []
y_pred_all_s_3 = []
for seed in range(10):
    

    # welche daten hier überhaupt geladen werden
    exp = experimental_data.experimental_data(config)
    #[inp, outp] = exp.generate_physical_data()
    #print([inp, outp])
    config['verbose'] = 0
    # Initiate model

    np.random.seed(seed)
    f_model = ffnn.feed_forward_nn(config)
    # Train model with physical data
    f_model.model_fit_with_physical_data()
    
    
    x, y_pred = draw_plot_model(f_model,config, b = -1,lbl='Wissenstransfer',col=tu_colors[1],noplot = 1) 
    x, y_pred_2 = draw_plot_model(f_model,config, b = 2,lbl='Wissenstransfer',col=tu_colors[1],noplot = 1) 
    y = S_3(x)
    
    y_pred_all.append(y_pred.flatten())  # flach machen und zur Liste hinzufügen
    y_pred_2_all.append(y_pred_2.flatten())  # flach machen und zur Liste hinzufügen
    
    f_model.validation_split = 0
    
    hist = f_model.model_fit_with_mess_data()
    hist = f_model.model_fit_with_mess_data()
    hist = f_model.model_fit_with_mess_data()
    #x , y = config["load_mess_data"](0)
    #x = exp.make_bin_data(x,2)
    #y_pred = f_model.model.fit(x,y,validation_split=0,batch_size=5,epochs=200)
    
    
    x, y_pred = draw_plot_model(f_model,config, b = 2,lbl='Wissenstransfer',col=tu_colors[1],noplot = 1) 
    #find_y = y_pred[250,750,500]
    #rmse = np.sqrt(np.mean((y - find_y) ** 2))
    
    y_pred_all_s_3.append(y_pred.flatten())  # flach machen und zur Liste hinzufügen


# Umwandeln der Liste in ein 2D-Array
y_pred_all = np.array(y_pred_all)
# Umwandeln der Liste in ein 2D-Array
y_pred_2_all = np.array(y_pred_2_all)
# Umwandeln der Liste in ein 2D-Array
y_pred_all_s_3 = np.array(y_pred_all_s_3)


#%% Schritt 2: Erstelle ein 2D-Histogramm
bins = 200  # Anzahl der Bins in x- und y-Richtung
heatmap, xedges, yedges = np.histogram2d(np.tile(x.flatten(), 10), y_pred_all.flatten(), bins=bins)

# Erstellen der benutzerdefinierten Colormap: Transparentes Weiß -> #C9308E
colors = [(1, 1, 1, 0),  # Transparentes Weiß (RGBA)
          (201/255, 48/255, 142/255, 1)]  # Vollton "#C9308E" (RGBA)
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Schritt 3: Zeige das 2D-Histogramm als Bild an

setup_plot()

draw_plot_sample(1,'S_1',tu_colors[1])  
draw_plot_sample(2,'S_2',tu_colors[2])

plt.legend(loc='upper left')
plt.imshow(heatmap.T, origin='lower', aspect='auto', 
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)
plt.colorbar(label='Dichte der Wissenstransferkurven [\%]')
    
plt.ylim(-0.2, 1.1)
        
plt.savefig('plots/rs_multi_plot_bins_' +  str(bins) + '_'+ str(vers) + '.png',dpi=300)
plt.show()

#%% Schritt 2: Erstelle ein 2D-Histogramm
bins = 200  # Anzahl der Bins in x- und y-Richtung
heatmap, xedges, yedges = np.histogram2d(np.tile(x.flatten(), 10), y_pred_2_all.flatten(), bins=bins)

# Erstellen der benutzerdefinierten Colormap: Transparentes Weiß -> #C9308E
colors = [(1, 1, 1, 0),  # Transparentes Weiß (RGBA)
          (201/255, 48/255, 142/255, 1)]  # Vollton "#C9308E" (RGBA)
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Schritt 3: Zeige das 2D-Histogramm als Bild an

setup_plot()

draw_plot_sample(1,'S_1',tu_colors[1])  
draw_plot_sample(2,'S_2',tu_colors[2])

plt.legend(loc='upper left')
plt.imshow(heatmap.T, origin='lower', aspect='auto', 
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)
plt.colorbar(label='Dichte der Wissenstransferkurven [\%]')
    
plt.ylim(-0.2, 1.1)
        
plt.savefig('plots/rs_multi_plot_b_2_bins_' +  str(bins) + '_'+ str(vers) + '.png',dpi=300)
plt.show()


#%% Schritt 2: Erstelle ein 2D-Histogramm
bins = 100  # Anzahl der Bins in x- und y-Richtung
heatmap, xedges, yedges = np.histogram2d(np.tile(x.flatten(), 10), y_pred_all_s_3.flatten(), bins=bins)

# Erstellen der benutzerdefinierten Colormap: Transparentes Weiß -> #C9308E
colors = [(1, 1, 1, 0),  # Transparentes Weiß (RGBA)
          (201/255, 48/255, 142/255, 1)]  # Vollton "#C9308E" (RGBA)
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Schritt 3: Zeige das 2D-Histogramm als Bild an

setup_plot()

draw_plot_sample(1,'S_1',tu_colors[1])  
draw_plot_sample(2,'S_2',tu_colors[2])
draw_plot_sample(3,'S_3',tu_colors[0])  

plt.legend(loc='upper left')
plt.imshow(heatmap.T, origin='lower', aspect='auto', 
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)
plt.colorbar(label='Dichte der Prognosekurven für S_3 [\%]')
plt.ylim(-0.1, 1.1)
        
plt.savefig('plots/rs_multi_s_3_mit_bins_' +  str(bins) + '_'+ str(vers) + '.png',dpi=300)
plt.show()


# uti_obj = ut.util(config)
# x_lhs = uti_obj.get_x_lhs()
# f_model.pred_model(x_lhs, 1)
# print(f_model.pred_model(x_lhs, 1))

# [inp_f, out_f] = f_prediction_data_3.load_prediction_data(1)
# pred = f_model.pred_model(inp_f, 1)

# print(np.vstack((pred[:,0],out_f)),1)
# mse = np.mean((out_f - pred[:,0]) ** 2)
# print(mse)

# # Assuming y_data is the transformed output and x_data is the input matrix
# # Reverse the normalization
# lower_bound = np.log(1)
# upper_bound = np.log(31.0)
# norm_log_trans_dens = 1 - pred
# log_trans_dens = norm_log_trans_dens * (upper_bound - lower_bound) + lower_bound

# # Reverse the logarithmic transformation
# pred_dens = 101 - np.exp(log_trans_dens)
# print(pred_dens)