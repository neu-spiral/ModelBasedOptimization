from plotter import linePlotter
from helpers import loadFile

DICS_NOUT = loadFile( 'data/stats/MNIST_NOUT')
DICS_TEST = loadFile( 'data/stats/MNIST_TEST')



line_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']

DICS_key_ord = ['MBOOADM ($p=2$)', 'MBOSGD ($p=2$)', 'SGD (MSE)']

for ind, DIC in enumerate([DICS_NOUT, DICS_TEST]):
    select_DIC_NOUT = {}

    select_DIC_NOUT['SGD (MSE)'] = DIC['ell2Sq']['SGD']

    select_DIC_NOUT['MBOOADM ($p=2$)'] = DIC[2.0]['MBOOADM']

    select_DIC_NOUT['MBOSGD ($p=2$)'] = DIC[2.0]['MBOSGD']


    if ind == 0:
        line_type = 'NOUT'
        yaxis_label = "$F_{NOUT}$"

    else:
        line_type = 'TEST'
        yaxis_label = "$F_{TEST}$"



    for DICT_key in select_DIC_NOUT:
        break

        max_val = max( select_DIC_NOUT[DICT_key].values() )
      

        for key in select_DIC_NOUT[DICT_key]:

            print(key)
            select_DIC_NOUT[DICT_key][key] = select_DIC_NOUT[DICT_key][key] /  max_val


  #  print(select_DIC_NOUT)
        
        

    linePlotter(select_DIC_NOUT, outfile = "/scratch/armin_m/ModelFunctionFramework/plots/select_normalized_log_MNIST_{}".format( line_type ) , yaxis_label = yaxis_label, xaxis_label =  "$P_{OUT}$", x_scale='linear', y_scale='log', DICS_key_ord = DICS_key_ord, normalize = True, line_styles=line_styles, leg_loc='upper right', leg_anchor=(1,0.6) )
