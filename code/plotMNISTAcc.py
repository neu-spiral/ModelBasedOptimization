from plotter import linePlotter
import numpy as np
import matplotlib.pyplot as plt

ACCDICT_ell2 = {'SGD': {0.0: 0.876, 0.05: 0.928, 0.1: 0.907, 0.2: 0.897, 0.3: 0.867} }

ACCDICT_p2 = {'MBOOADM': {0.0:0.926, 0.05:0.925,  0.01:0.925, 0.2:0.928 , 0.3:0.923 } , 'MBOSGD': {0.0:0.894 , 0.05:0.921, 0.1:0.924 ,0.2:0.856 ,0.3:0.925 }, 'SGD': {0.0: 0.868, 0.05:0.914, 0.1:0.916 ,0.2:0.914 ,0.3:0.913 } }

ACCDICT_p1 = {'MBOOADM': {0.0:0.936, 0.05:0.935,  0.01:0.935, 0.2:0.931 , 0.3:0.927 } , 'MBOSGD': {0.0:0.904 , 0.05:0.930, 0.1:0.926 ,0.2:0.931 ,0.3:0.922 }, 'SGD': {0.0: 0.861, 0.05:0.863, 0.1:0.873 ,0.2:0.894 ,0.3:0.922 } }

ACCDICT_MNIST = {'ell2SQ': ACCDICT_ell2, 2: ACCDICT_p2, 1: ACCDICT_p1}

for p in ACCDICT_MNIST:


        if p == 'ell2SQ':
            line_styles = ['rD-','cX-','m*--','yH-', 'mv-']

            DICS_key_ord = ['SGD']

        else:
            DICS_key_ord = ['MBOOADM', 'MBOSGD', 'SGD']

            line_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']


        plt.yticks( np.arange(85, 95, 1 ) / 100. )

        linePlotter(ACCDICT_MNIST[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/acc_MNIST_p{}".format(p), yaxis_label = "Test Accuracy", xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = False, line_styles=line_styles, leg_loc='top', leg_anchor=(0.2,0), ylim=[0.850, 0.94] )

        linePlotter(ACCDICT_MNIST[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/acc_MNIST_no_lgd_p{}".format(p), yaxis_label = "Test Accuracy", xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = False, line_styles=line_styles, leg_loc=None, leg_anchor=(0.2,0), ylim=[0.850, 0.94] )
    
