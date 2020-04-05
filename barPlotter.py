

import argparse
from helpers import loadFile
from plotter import barPlotter, whichKey



colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
hatches = ['////', '/', '\\', '\\\\', '-', '--', '+', '']
lin_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-']

Algorithms = {'admm':'ADMM', 'sgd':'SGD'}
batch_sizes = [100, 32, 16, 8,  4, 2, 1]




if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Generate bar plots comparing different algorithms.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='pickled files to be processed')
    parser.add_argument('--outfile', type=str, help="File to store the figure.")
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--yaxis',choices=['OBJ','time', 'totalLoss', 'non_outliersLoss'], default='OBJ',help='Determine whether to plot gain or cost')
    parser.add_argument('--xaxis',choices=['iterations','time', 'outliers'], default='time',help='Iterations or time')
    parser.add_argument('--plot_type', choices=['bar', 'line'], default='bar')
    args = parser.parse_args()
    
    
    DICS = {}
    
    outliers={'outliers0.0':0, 'outliers0.1':0.1, 'outliers0.2':0.2}
    max_dict = {}
    for filename in args.filenames:

        #find out file corresponds to which alg.
        outlier  = whichKey(filename, keywords=outliers)
        stats = loadFile(filename)
    
        #make DICS
        for key in stats:
            if key not in DICS:
               DICS[key] = {}
            DICS[key][outlier] = stats[key][args.yaxis]
             

        #y-axis label
        if args.yaxis == 'OBJ':
            y_axis_label = 'Objective'
        else:
            y_axis_label = args.yaxis

    barPlotter(DICS=DICS, outfile=args.outfile, x_axis_label=args.xaxis,  y_axis_label = args.yaxis) 
