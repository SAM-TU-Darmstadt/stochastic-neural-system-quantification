from files import feed_forward_nn as ffnn
import config_rf as conf_file
from files import util as ut
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

#%% schritt 1 Hol die daten aus dem speicher
config = conf_file.sdnn_config()
config['rand_or_lhs'] = True

uti_obj = ut.util(config)


y_pred_all = []
anzmodels = 2000
count = 0
num = 0
maxnum = len(uti_obj.get_load_list(config['save_to_sdnn'],'sdnn','h5'))

while count<anzmodels and num<maxnum:
    f_model = ffnn.feed_forward_nn(config)
    try:
        f_model.load(num)
    except Exception:
        break
    num = num+1
    anzP = 1000
    x = np.arange(0,1,1/anzP)
    x = x.reshape(1000, 1) 
    if config['pf'] == 0:
        b = 2
    else:
        b = 0
    y_pred = f_model.pred_model(x,b)
    
    if np.max(y_pred)-np.min(y_pred) > 0.2:    
        y_pred_all.append(y_pred.flatten())  # flach machen und zur Liste hinzufügen
        count = count+1
    

a = uti_obj.get_ae_train_data()
    


# Umwandeln der Liste in ein 2D-Array
y_pred_all = np.array(y_pred_all)
anzmodels = count

#%% Schritt 2: Erstelle ein 2D-Histogramm
bins = 200  # Anzahl der Bins in x- und y-Richtung
heatmap, xedges, yedges = np.histogram2d(np.tile(x.flatten(), anzmodels), y_pred_all.flatten(), bins=bins)

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
plt.colorbar(label='Anzahl der generierten Kurven ')
    
plt.ylim(-0.2, 1.2)
        
plt.savefig('plots/sdnn_2dhistogram_' +  str(bins) + '_'+ str(anzmodels) + '.png',dpi=300)
plt.show()




