from plotter import linePlotter
from helpers import loadFile

DICS_NOUT = loadFile( 'data/stats/MNIST_NOUT')
DICS_TEST = loadFile( 'data/stats/MNIST_TEST')


for DIC_ind, DIC in enumerate( [DICS_NOUT, DICS_TEST] ):
    for p in DIC:


        if p == 'ell2Sq':
            DICS_key_ord = ['SGD']
            line_styles = ['rD-','cX-','m*--','yH-', 'mv-']


        else:
            DICS_key_ord = ['MBOOADM', 'MBOSGD', 'SGD']
            line_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']

        print(DIC[p])
        if DIC_ind == 0:
            line_type = "NOUT"

            yaxis_label = "$F_{NOUT}$"
        else:
            line_type = "TEST"

            yaxis_label = "$F_{TEST}$"

        linePlotter(DIC[p], outfile = "/scratch/armin_m/ModelFunctionFramework/plots/linePlot_no_lgd_MNIST_p{}_{}".format(p, line_type), yaxis_label = yaxis_label, xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord, normalize = True, line_styles=line_styles, leg_loc=None, leg_anchor=(0,1) )
