
from plotter import linePlotter

ACC_DICT = { 'SGD $\|\cdot\|_2^2$':{0.0:0.876, 0.05:0.928, 0.1:0.907, 0.2:0.897, 0.3:0.867   }, 'MBOOADM ($p=1$)': {0.0: 0.936, 0.05:0.935, 0.1:0.935, 0.2: 0.931, 0.3: 0.927}, 'MBOSGD ($p=1$)': {0.0:0.904, 0.05:0.930, 0.1:0.926, 0.2:0.931, 0.3:0.922} }


linePlotter(ACC_DICT, '/scratch/armin_m/ModelFunctionFramework/plots/acc_plot.pdf', yaxis_label='Test Accuracy', xaxis_label='$P_{out}$', x_scale='linear', y_scale='linear', DICS_key_ord = ['SGD $\|\cdot\|_2^2$', 'MBOOADM ($p=1$)', 'MBOSGD ($p=1$)' ], normalize = False) 
