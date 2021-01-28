
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



colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
hatches = ['////', '/', '\\', '\\\\', '-', '--', '+', '']
lin_styles = ['b^-','g*-','rD-','cX-','m*--','yH-', 'mv-'] * 2 #['r^', 'b-*', 'rD*', 'cX-', 'm*-','yH-']

Algorithms = {'admm':'ADMM', 'sgd3':'SGD (lr=1e-3)', 'sgd4':'SGD (lr=1e-4)', 'sgd5':'SGD (lr=1e-5)'}
batch_sizes = [100, 32, 16, 8,  4, 2, 1]





def whichAlg( filename, keywords = {'admm':'admm'}):
    "Find filename corresponds to which algrotihm."
    if re.search('admm',  filename ):
        return 'admm'

    elif re.search('sgd',  filename ):
        for bsize in batch_sizes:
            if  re.search('sgd' + str(bsize),  filename ):

               return 'sgd (bsize {})'.format(bsize)
    return 'MBO'

def whichKey( filename, keywords = {'admm':'admm'}, keys_ordered=None):
    "Find filename corresponds to whcih key in keywords."

    if keys_ordered == None:
        keys_ordered = sorted( keywords.keys() )

    for key in keys_ordered:
        if re.search(key, filename):
            return keywords[key]
    
    

def barPlotter(DICS, outfile, x_axis_label, y_axis_label = 'Objective', normalize=False, lgd=True, log_bar=False, DICS_alg_keys_ordred = None):
    def formVals(DICS_alg, DICS_alg_keys_ordred = None):

        if DICS_alg_keys_ordred is None:
            DICS_alg_keys_ordred = sorted(DICS_alg.keys())

        out = [DICS_alg[key] for key in DICS_alg_keys_ordred if key in DICS_alg]

        labels =  [str(key) for key in DICS_alg_keys_ordred]

        return out, labels

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 4)
    width = 1


    N = len(DICS[ list( DICS.keys() )[0] ].keys()) 
    numb_bars = len(DICS.keys() ) + 1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    RECTS = []
    i = 0


    if normalize:
        plt.ylim([0,1.1])
        y_axis_label = "Normalized " + y_axis_label

    for i, key  in enumerate(DICS):
        values, labels = formVals(DICS[key], DICS_alg_keys_ordred)
        print(values, labels, key)
    #    ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=alg,log=True)

        RECTS+= ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=key,log=log_bar)

    #legend 
    if lgd:
        LGD = ax.legend(ncol=len(DICS.keys() ), borderaxespad=0.,loc=3, bbox_to_anchor=(0., 1.02, 1., .102),fontsize=15,mode='expand')    
    else:
        LGD = None
    
    ax.set_xticks(ind ) 
    ax.set_xticklabels(tuple(labels),fontsize = 18)
    ax.set_xlabel( x_axis_label, fontsize = 18 )
    ax.set_ylabel(y_axis_label,fontsize=18)
  #  ax.set_yticklabels([0, 0.5, 1])
    plt.yticks(fontsize = 18)
    plt.xlim([ind[0]-width,ind[-1]+len(DICS.keys() )*width])
        
    print(outfile)

    if lgd:
        fig.savefig(outfile+".pdf",format='pdf', bbox_extra_artists=(LGD,), bbox_inches='tight') 

    else:
        fig.savefig(outfile+".pdf",format='pdf', bbox_inches='tight' )


def linePlotter(DICS, outfile, yaxis_label='Objective', xaxis_label='Looseness coefficient $\kappa$', x_scale='linear', y_scale='linear'):

    #iterate over trjacetories for all leys (algorithms)
    for i, alg in enumerate( DICS ):

        #load trajectory
        data_dict = DICS[alg]

        vals = [float(val) for val in data_dict.values()]

        print(alg)
        #plot trajectory
        plt.plot(list( data_dict.keys() ), list( data_dict.values() ), lin_styles[i], label=alg, linewidth=3, markersize=18)

    #set sizes
    plt.xlabel(xaxis_label,fontsize = 18)
    plt.ylabel(yaxis_label, fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 16)

    lgd = plt.legend( loc='upper right',bbox_to_anchor=(1,1),ncol=1,borderaxespad=0.,fontsize= 16)

  #  plt.xlim(0.5,1.1)
    plt.xscale(x_scale)
    plt.yscale(y_scale )


    #save figure
    plt.savefig(outfile+'.pdf',  bbox_extra_artists=(lgd,), format='pdf', bbox_inches='tight' )

def doubleAxeslinePlotter(DIC, outfile, yaxis_label='Objective', y2axis_label='Satisfied Constraints Ratio', xaxis_label='Time (s)', x_scale='linear', y_scale='linear'):
    def formVals(DIC_alg):
        vals = []
        vals2 = []
        x_axis = []
        for key in sorted( [eval(val) for val in DIC_alg.keys()] ):
            vals.append( DIC_alg[str(key)][0]  )
            vals2.append( DIC_alg[str(key)][1]  )
            x_axis.append(key )
        return vals, vals2, x_axis
    fig, ax = plt.subplots()

    ax2 = ax.twinx()

   # fig.set_size_inches(8, 6)
    i=0
    vals, vals2, x_axis = formVals(DIC)
    ax.plot(x_axis, vals, lin_styles[i], label=yaxis_label, linewidth=3, markevery=1, markersize=18)
    ax2.plot(x_axis, vals2, lin_styles[i+1], label=y2axis_label,  linewidth=3, markevery=1, markersize=18)
    
    ax.set_xlabel(xaxis_label,fontsize = 16)
    ax.set_ylabel(yaxis_label, fontsize = 16)
    ax2.set_ylabel(y2axis_label, fontsize = 16)

    ax.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    plt.xticks(fontsize = 18)

    lgd = ax.legend( loc='upper left',bbox_to_anchor=(0.1, 1),ncol=1,borderaxespad=0.,fontsize= 14)
    lgd2 = ax2.legend( loc='upper left',bbox_to_anchor=(0.1,0.9),ncol=1,borderaxespad=0.,fontsize= 14)
    lgd.get_frame().set_edgecolor('w')
    lgd2.get_frame().set_edgecolor('w')

  #  plt.xlim(0.5,1.1)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale )
    ax2.set_yscale(y_scale)


    fig.savefig(outfile+'.pdf', bbox_extra_artists=(lgd,lgd2), format='pdf', bbox_inches='tight' )
def read_file(fname,normalize=0):
    f = open(fname,'r')
    l= eval(f.readline())
    f.close
    (Time, OBJ) = l[-1] 
    return {"TIME":Time,"OBJ":OBJ}
def CG_readfile(CG_file,rounded_file):
    dic_CG = read_file(CG_file)
    dic_round = read_file(rounded_file)
    return {"TIME":dic_CG["TIME"]+dic_round["TIME"],"OBJ":dic_round["OBJ"]}
def gen_dictionaries(random_f,greedy_f,CG_1,round_1,CG_2,round_2,CG_3,round_3):
    dic_time = {"Random":0.,"Greedy":0.,"CG (10 samples)":0.,"CG (100 samples)":0.}
    dic_obj = {"Random":0.,"Greedy":0.,"CG (10 samples)":0.,"CG (100 samples)":0.}
    dic_time["Random"] = read_file(random_f)["TIME"]
    dic_obj["Random"] = read_file(random_f)["OBJ"] 
    
    dic_time["Greedy"] = read_file(greedy_f)["TIME"]
    dic_obj["Greedy"] = read_file(greedy_f)["OBJ"] 
    
    dic_time["CG (10 samples)"] = CG_readfile(CG_1,round_1)["TIME"]
    dic_obj["CG (10 samples)"] = CG_readfile(CG_1,round_1)["OBJ"] 
    dic_time["CG (100 samples)"] = CG_readfile(CG_2,round_2)["TIME"]
    dic_obj["CG (100 samples)"] = CG_readfile(CG_2,round_2)["OBJ"]
    dic_time["CG (200 samples)"] = CG_readfile(CG_3,round_3)["TIME"]
    dic_obj["CG (200 samples)"] = CG_readfile(CG_3,round_3)["OBJ"]
    return dic_time,dic_obj

plt.show()
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
    
    
    DICS = {}
    ##########HARD CODE#############################
   # keywords = {'_1':'p=1', '_2':'p=2',  '_-2':'ell 2 squared','_3':'p=3', 'SGD':'ell 2 squared (SGD)'}
    keywords = {'MBO':'MBO', '_SGD':'SGD', 'MBOSGD': 'MBOSGD'}
    kw_ord = ['MBOSGD', '_SGD', 'MBO']

#    keywords = {}
    
  ##  for BS in ['8', '32', '128']:
  #  for lr in ["0.001", '0.001', '0.0001', '0.00001', "0.000001"]:
  #      for momentum in ["0.0", "0.5", "0.9"]:
  #          key = "MBOSGD_lr{}_momentum{}".format(lr, momentum)

  #          keywords[key] =  "lr {}, momentum {}".format(lr, momentum)
  ##              key = "BS{}_rho_inner5.0_SGD_lr{}_momentum{}".format(BS, lr, momentum)

  ##              keywords[key] = "batch-size {}, lr {}, momentum {}".format(BS, lr, momentum)

   # kw_ord = keywords.keys()
#####################################################
    

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
                DICS[Alg] = dict( [ (trace['time'][ind], val) for ind, val in enumerate( trace[ args.yaxis ] ) ] )
                x_axis_label = 'time(s)' 
        
    #        continue
        #y-axis label
        if args.yaxis == 'OBJ':
            y_axis_label = 'Objective'
        else:
            y_axis_label = 'Time(s)'

    print(DICS)
    #Plot 
    linePlotter(DICS, outfile=args.outfile, yaxis_label=y_axis_label, xaxis_label=x_axis_label, x_scale='linear', y_scale='linear') 
