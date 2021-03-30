
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
from matplotlib.transforms import Bbox
import numpy as np
from matplotlib.dates import date2num
import pickle
import re
import datetime

from plotter import linePlotter, whichKey


colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
hatches = ['////', '/', '\\', '\\\\', '-', '--', '+', '']
lin_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-'] * 2 #['r^', 'b-*', 'rD*', 'cX-', 'm*-','yH-']

Algorithms = {'admm':'ADMM', 'sgd3':'SGD (lr=1e-3)', 'sgd4':'SGD (lr=1e-4)', 'sgd5':'SGD (lr=1e-5)'}
batch_sizes = [100, 32, 16, 8,  4, 2, 1]





if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Generate bar plots comparing different algorithms.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='pickled files to be processed')
    parser.add_argument('--outfile', type=str, help="File to store the figure.")
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--yaxis',choices=['OBJ','time'], default='OBJ',help='Determine whether to plot gain or cost')
    parser.add_argument('--xaxis',choices=['iterations','time'], default='time',help='Iterations or time')
    parser.add_argument('--plot_type', choices=['bar', 'line'], default='bar')
    args = parser.parse_args()
    
    
    keywords = {}
    
  #  for BS in ['8', '32', '128']:
    for eps in ['0.1', "0.01", '0.001', '0.0001']:
            key = "adaptEps{}".format( eps )

            keywords[key] = "$\epsilon=${}".format( eps )


    kw_ord = keywords.keys()

    DICS_key_ord = keywords.values()
#####################################################
    
    DICS = {}

    for filename in args.filenames:

        #find out file corresponds to which alg.
        Alg  = whichKey(filename, keywords, kw_ord)


        with open(filename, 'rb') as current_file:
            trace  = pickle.load(current_file)

        if Alg not in DICS:
            if args.xaxis == 'iterations':
                DICS[Alg] = dict( [(ind + 1, val) for ind, val in enumerate( trace[ args.yaxis ] ) ] )
                x_axis_label = 'iterations'

            elif args.xaxis ==  'time':
                DICS[Alg] = dict( [ (trace['time'][ind]/3600, val) for ind, val in enumerate( trace[ args.yaxis ] ) ] )
                x_axis_label = 'time(h)' 
        
    #        continue
        #y-axis label
        if args.yaxis == 'OBJ':
            y_axis_label = 'Objective'
        else:
            y_axis_label = 'Time(s)'

    print(DICS.keys())
    #Plot 
    linePlotter(DICS, outfile=args.outfile, yaxis_label=y_axis_label, xaxis_label=x_axis_label, x_scale='linear', y_scale='linear', DICS_key_ord = DICS_key_ord) 
