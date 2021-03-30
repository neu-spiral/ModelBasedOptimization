from plotter import linePlotter
from helpers import loadFile

DICS_NOUT = loadFile( 'data/stats/MNIST_NOUT')



for p in DICS_NOUT:


    if p == 'ell2Sq':
        DICS_key_ord = ['SGD']
        line_styles = ['rD-','cX-','m*--','yH-', 'mv-']


    else:
        DICS_key_ord = ['MBOOADM', 'MBOSGD', 'SGD']
        line_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']

    linePlotter(DICS_NOUT[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/linePlot_MNIST_p{}".format(p) + "_NOUT", yaxis_label = "$F_{NOUT}$", xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = True, line_styles=line_styles, leg_loc='upper left', leg_anchor=(0,1) )
