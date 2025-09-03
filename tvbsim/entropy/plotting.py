import os
import numpy as np
import plotly.graph_objects as go
from einops import rearrange

from tvbsim.plotting import plot_box
from tvbsim.common import find_file_seed
from tvbsim.entropy.measures import NAME2ENTROPY

def plot_entropy(entropy_name, set_params=None, **kwargs):
    """
        Plot boxplots for any given entropy.
    
        :param set_params: Dict of axis parameters for the plot.
        :param kwargs: Additional arguments forwarded to the underlying plotting function.
        :returns: None.
    """
    if set_params is None:
        set_params = {}
    entropy = NAME2ENTROPY[entropy_name]
    set_params['ylabel'] = entropy.label
    plot_box(file_prefix=entropy.file_prefix, set_params=set_params, **kwargs)
    
def plot_entropies(simconfigs, seeds, file_prefixes, data=None, save=0, path=None, file_name=None, names=None, save_path=None,
                   plot_params={}, layout_params={}, traces_params={}):
    """
        :param simconfigs: simconfigs of which to plot results
        :param seeds: seeds of which to plot results for each simconfigs
        :param file_prefix: beginning of files of which to get data from ('LempelZiv' or 'PCI' for instance)
        :param data: if None, searches for the data itself in path based on SimConfigs' params_to_report, (n_methods, n_simconfigs, n_seeds)
        :param save: 0 display & no save, 1 display & save, 2 no display & save
        :param path: where to look for the .npy files
        :param file_name: name of the saved plot
        :param x: how to group data, list of size number of simconfigs
                  if the first 2 simconfigs should be grouped in one box with label A, and the two last in another with label B:
                    x = ['A','A','B','B'], and all the seeds of the first two simconfigs will go in the first box plot and 
        :param save_path: where to save plot
        :param plot_params: dictionary of parameters given to plotly's box function
        :param layout_params: dictionary of parameters given to plotly's update_layout function
        :param traces_params: dictionary of parameters given to plotly's update_traces function
    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)        
    # Collect data
    n_seeds = len(seeds)
    if data is None:
        data = np.zeros((len(file_prefixes), len(simconfigs), n_seeds))
        for f,file_prefix in enumerate(file_prefixes):
                for i,simconfig in enumerate(simconfigs):
        #            print(simconfig.get_sim_name(n_seeds)[1])
                    file_path = find_file_seed(simconfig, path, file_prefix, n_minimal_seeds=len(seeds))
                    data[f,i] = np.load(file_path)[:n_seeds]
        #            Printer.print(data[i])
    else:
#        print(data.shape)
        n_seeds = data.shape[-1]
        
    # Plot
    if names is None:
        names = [simconfig.get_plot_title() for simconfig in simconfigs] * n_seeds
    else:
        names = names * n_seeds
    
    file_prefixes = file_prefixes * n_seeds * len(simconfigs)

#    fig = px.box(x=x, y=data.T.flatten(), points='all', height=height, width=width, title=title)
#    fig.update_layout(xaxis_title='Configuration', yaxis_title=yaxis_name,
#                      font=dict(size=30))
#    fig.update_traces(marker={'size': marker_size})
    
    fig = go.Figure()
    for i,simconfig in enumerate(simconfigs):
        fig.add_trace(go.Box(
                y=rearrange(data[:,i], 'e s -> (s e)'),
                x=file_prefixes,
                name=names[i],
                **plot_params))
    
    fig.update_layout(boxmode='group', **layout_params)
    fig.update_traces(**traces_params)
    
    if save >= 1:
        sim_names = [simconfig.get_sim_name(seeds[-1])[1] for simconfig in simconfigs]
        if file_name is None:
            file_name = f"{file_prefix}_{'_'.join(sim_names)}.png"
        if save_path is None:
            save_path = ''
        fig.write_image(os.path.join(save_path, file_name))
    if save < 2:
        fig.show()