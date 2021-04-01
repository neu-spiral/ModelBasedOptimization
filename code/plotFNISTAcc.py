from plotter import linePlotter
import numpy as np
import matplotlib.pyplot as plt

ACCDICT_ell2 = {'SGD': {0.0: 0.754, 0.05: 0.770, 0.1: 0.806, 0.2: 0.754, 0.3: 0.757} }

ACCDICT_p2 = {'MBOOADM': {0.0:0.836, 0.05:0.835,  0.01:0.835, 0.2:0.826 , 0.3:0.817}  , 'MBOSGD': {0.0:0.824 , 0.05:0.832, 0.1:0.836 ,0.2:0.832 ,0.3:0.827 }, 'SGD': {0.0: 0.787, 0.05:0.806, 0.1:0.812 ,0.2:0.821 ,0.3:0.823 } }

ACCDICT_p1 = {'MBOOADM': {0.0:0.853, 0.05:0.850,  0.01:0.854, 0.2:0.849 , 0.3:0.846 } , 'MBOSGD': {0.0:0.853 , 0.05:0.847, 0.1:0.848 ,0.2:0.832 ,0.3:0.819 }, 'SGD': {0.0: 0.794, 0.05:0.814,  0.1:0.783,0.2:0.758 ,0.3:0.783 } }

ACCDICT_FashionMNIST = {'ell2SQ': ACCDICT_ell2, 2: ACCDICT_p2, 1: ACCDICT_p1}

for p in ACCDICT_FashionMNIST:


        if p == 'ell2SQ':
            line_styles = ['rD-','cX-','m*--','yH-', 'mv-']

            DICS_key_ord = ['SGD']

        else:
            DICS_key_ord = ['MBOOADM', 'MBOSGD', 'SGD']

            line_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']


        plt.yticks( np.arange( 75, 86, 1) / 100. )

        linePlotter(ACCDICT_FashionMNIST[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/acc_FashionMNIST_p{}".format(p), yaxis_label = "Test Accuracy", xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = False, line_styles=line_styles, leg_loc='top', leg_anchor=(0.2,0), ylim=[0.75,0.86] )

        linePlotter(ACCDICT_FashionMNIST[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/acc_FashionMNIST_no_lgd_p{}".format(p), yaxis_label = "Test Accuracy", xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = False, line_styles=line_styles, leg_loc=None, leg_anchor=(0.2,0), ylim=[0.75, 0.86]  )
    
