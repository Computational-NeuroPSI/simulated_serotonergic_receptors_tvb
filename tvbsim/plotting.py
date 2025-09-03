import numpy as np
import os
from matplotlib import figure
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, f_oneway, gaussian_kde
import itertools
import pandas as pd
import seaborn as sns

from tvbsim.common import create_dicts, find_file_seed, pvalue_to_asterisks
from tvbsim.printer import Printer
           
# TODO : there might be something wrong with the begin time, end time things
# TODO : plot moments where b_e changes to make it clearer in the plot ? (only works for periodically changing parameters)
def plot_tvb_results(simconfigs, result, monitor, for_explan, var_select, seeds=[10], figsize=(8,5),
                     begin_time=None, save=False, color_regions=None, label_regions=None, priority_regions=[], end_time=None, with_legend_title=True,
                     save_path=None, regions=None, with_title=True, extension='png'):
    """
        Used to plot the results of simulations, like the excitatory firing rates for instance 
    
        :param simconfigs: simconfigs of which to plot results
        :param result: list with dimensions (len(simconfigs), len(seeds), result obtained via get_result)
        :param seeds: seeds of which to plot results for each simconfigs
        :param monitor: 'Bold', 'Raw', etc
        :param for_explan: one of the output of get_result
        :param var_select: list of vars to include in the plots, 'E', 'I', etc see get_result for list
        :param figsize: size of the figure
        :param begin_time: where plot starts, default is each simconfig's cut_transient
        :param save: 0 display & no save, 1 display & save, 2 no display & save
        :param color_regions: colors of each region, default is same if var_select=['E','I'], or all different colors if only ['E']
                              for example [(1,0,0,1)]*34+[(0,1,0,1)*34] for right hemishpere regions in red, left in green)
        :param label_regions: list of label of each region, default labels Exc and Inh if var_select=['E','I'] else no label
        :param priority_regions: regions to display at the end, will appear on top of the others
        :param end_time: when to end plot, default is each simconfig's run_sim
        :param with_legend_title: whether to include title & legend
        :param save_path: where to save plots
        :param regions: which regions to plot (default: None, plot all)
    """
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = ''

    rows =int(len(simconfigs)*len(seeds))
    cols = len(var_select) 
    if 'E' and 'I' in var_select:
        cols = cols-1 #put E and I in the same plot
    single_plot=False
    if cols == 1:
        if rows>1:
            pass
#            cols=rows
#            rows=1
        else:
            cols=2
            single_plot=True

    if save < 2:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
    else:
        fig = figure.Figure(figsize=figsize)
        axes = fig.subplots(rows,cols)
        
    for i,simconfig in enumerate(simconfigs):
        for j,seed in enumerate(seeds):
            
            title = simconfig.get_plot_title()
            print(i)
            result_fin = create_dicts(
                    simconfig, result[i][j], monitor, for_explan, var_select)
            #create list with the indices for each variable
            var_ind_list = {}
            j =0
            for var in var_select:
                if len(axes.shape) == 1:
                    ax_index = i
                    i +=1 
                else: 
                    ax_index= (i, j )#that means that you have one row or one col
                if (var == 'E' and 'I' in var_ind_list.keys()) or (var == 'I' and 'E' in var_ind_list.keys()):
                    continue
                else:
                    var_ind_list[var] = ax_index
                    j+=1
                    
            fontsize = 10
            
            if end_time is None:
                if monitor != 'Bold':
                    end_time_simconfig = min(
                            simconfig.run_sim,
                            result_fin['E'].shape[0]*simconfig.general_parameters.parameter_integrator['dt'])
                    print(result_fin['E'].shape[0]*simconfig.general_parameters.parameter_integrator['dt'])
                    print('END time', end_time_simconfig)
                else:
                    end_time_simconfig = min(
                        simconfig.run_sim,
                        result_fin['E'].shape[0]*simconfig.general_parameters.parameter_monitor['parameter_Bold']['period'])/10
                    print(result_fin['E'].shape[0]*simconfig.general_parameters.parameter_monitor['parameter_Bold']['period'])
                    print('END time', end_time_simconfig)
            else:
               end_time_simconfig = end_time
            if begin_time is None:
                begin_time = simconfig.cut_transient
            
            #if E and I in the vars, plot them in the same subplot
            if 'E' and 'I' in result_fin.keys():
                try:
                    ax_ind= var_ind_list['E']
                    del var_ind_list['E']
                except KeyError:
                    ax_ind = var_ind_list['I']
                    del var_ind_list['I']
                
                if regions is None:
                    regions = range(result_fin['I'].shape[1])
                
                time_s = np.linspace(begin_time, end_time_simconfig, result_fin['E'].shape[0])/1000
                closest_index_begin = np.argmin(np.abs(time_s - begin_time)) # index of the time point closest to the desired time
                closest_index_end = np.argmin(np.abs(time_s - end_time_simconfig))
                Li = axes[ax_ind].plot(time_s[closest_index_begin:closest_index_end],result_fin['I'][closest_index_begin:closest_index_end, regions],color='red', label='Inh') # [times, regions]
                Le = axes[ax_ind].plot(time_s[closest_index_begin:closest_index_end],result_fin['E'][closest_index_begin:closest_index_end, regions],color='blue', label='Exc') # [times, regions]
                axes[ax_ind].set_ylabel('Firing rate (Hz)', fontsize=fontsize)
                axes[ax_ind].set_xlabel('Time (s)', fontsize=fontsize)
                if with_title:
                    axes[ax_ind].set_title(title, fontsize=fontsize)
                axes[ax_ind].legend([Li[0], Le[0]], ['Inh.','Exc.'], loc='upper right', fontsize='xx-small')
    
                for var in var_ind_list.keys():
                    plot_vars_separately(
                            simconfig, fig, axes, fontsize, var_ind_list, result_fin, begin_time, save, 
                            color_regions=None, label_regions=None, priority_regions=[], end_time=end_time_simconfig, with_legend_title=with_legend_title,
                            save_path=save_path, seed=seed, regions=regions, extension=extension)
    
                if save >= 1:
                    Printer.print('Saving plot')
                    extent = axes[ax_ind].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    Printer.print(title)
                    fig.savefig(os.path.join(save_path, title + f'_seed_{seed}.{extension}'), bbox_inches=extent.expanded(1.3, 1.2))
               
            
            #else plot all the variables separately
            else:
                plot_vars_separately(simconfig, fig, axes, fontsize, var_ind_list, result_fin, begin_time, save,
                                     color_regions, label_regions, priority_regions, end_time=end_time_simconfig, 
                                     with_legend_title=with_legend_title, save_path=save_path, seed=seed, 
                                     title=title, regions=regions, extension=extension)

    
    for ax in axes.reshape(-1):
        if with_legend_title:
            ax.set_xlabel('Time (s)')

    if single_plot:
        fig.delaxes(axes[1])
    
    if save < 2:
        plt.tight_layout()
        plt.show()

def plot_vars_separately(simconfig, fig, axes, fontsize, var_ind_list, result_fin, begin_time, save, 
                         color_regions, label_regions, priority_regions, end_time, with_legend_title, 
                         title, save_path, seed, regions, extension):
    for var in var_ind_list.keys():
        title = simconfig.get_plot_title()
        ax_ind= var_ind_list[var]
        time_s = np.linspace(begin_time, end_time, result_fin[var].shape[0])#/1000
        closest_index_begin = np.argmin(np.abs(time_s - begin_time)) # index of the time point closest to the desired time
        closest_index_end = np.argmin(np.abs(time_s - end_time))
        color = None
        label = var
        for i in range(result_fin[var].shape[1]):
            if regions is not None and i not in regions:
                continue
            if i in priority_regions:
                continue
            if color_regions is not None:
                color = color_regions[i]
            if label_regions is not None:
                label = label_regions[i]
            Li = axes[ax_ind].plot(
                    time_s[closest_index_begin:closest_index_end],
                    result_fin[var][closest_index_begin:closest_index_end,i], 
                    label=label, color=color) # [times, regions]
        for i in priority_regions:
            if color_regions is not None:
                color = color_regions[i]
            if label_regions is not None:
                label = label_regions[i]
            Li = axes[ax_ind].plot(time_s[closest_index_begin:closest_index_end],result_fin[var][closest_index_begin:closest_index_end,i], label=label, color=color) # [times, regions]            
        if with_legend_title:
            axes[ax_ind].set_ylabel('Firing rate (Hz)', fontsize=fontsize)#var, fontsize=fontsize)
            axes[ax_ind].set_xlabel('Time (s)', fontsize=fontsize)
#            axes[ax_ind].set_title(title, fontsize=fontsize)
#            handles, labels = axes[ax_ind].get_legend_handles_labels()
#            by_label = dict(zip(labels, handles))
#            axes[ax_ind].legend(by_label.values(), by_label.keys())
        else:
            axes[ax_ind].set_xticks([])

#        axes[ax_ind].legend(Li, label_regions, loc='upper right', fontsize=fontsize)
        
    if save >= 1:
        Printer.print('Saving plot')
        extent = axes[ax_ind].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        Printer.print(title)
        fig.savefig(os.path.join(save_path, title+'seed_'+str(seed)+f'.{extension}'), bbox_inches=extent.expanded(1.3, 1.2))

def plot_box_points_violin(
        simconfigs, file_prefix, n_seeds=None, data=None, save=0, path=None, file_name=None, x=None, save_path=None,
        boxplot_params={}, set_params={}, box_colors=None, figsize=(8,8), pvals_pairs=None):
    """
    Plots a combination of boxplots, individual data points, and violin density estimates for given simulation configurations or data.
    Optionally performs statistical tests between groups and annotates significance.
    
    :param simconfigs: List of simulation configuration objects used to extract and label data if data is not provided.
    :param file_prefix: String prefix used when searching for data files.
    :param n_seeds: Number of seeds (samples) to load per configuration. Ignored if data is provided.
    :param data: Optional NumPy array of shape (n_groups, n_seeds). If None, data will be loaded from files using simconfigs.
    :param save: Integer flag controlling saving and displaying of the plot.
    - 0: Show only
    - 1: Save and show
    - 2: Save and close (do not show)
    :param path: Path where seed data files are stored, used if data is None.
    :param file_name: File name to save the plot as, defaults to "boxplot.png".
    :param x: Optional list of custom labels for the groups on the x-axis.
    :param save_path: Directory to save the plot. If it does not exist, it is created.
    :param boxplot_params: Dictionary of additional parameters passed to the seaborn boxplot.
    :param set_params: Dictionary of matplotlib axis parameters (e.g., title, labels, limits).
    :param box_colors: List or palette of colors used for the groups. Defaults to seaborn palette.
    :param figsize: Tuple defining the figure size (width, height). Defaults to (8, 8).
    :param pvals_pairs: Optional list of group index pairs for which significance tests are performed and annotated.
    
    :returns: None. Displays and/or saves the generated plot depending on save.
    """
    
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    if data is None:
        data = np.zeros((len(simconfigs), n_seeds))
        for i, simconfig in enumerate(simconfigs):
            file_path = find_file_seed(simconfig, path, file_prefix, n_minimal_seeds=n_seeds)
            data[i] = np.load(file_path)[:n_seeds]
    else:
        n_seeds = data.shape[1]

    labels = [simconfig.get_plot_title() for simconfig in simconfigs]
    if x is not None:
        labels = x

    df_list = []
    for i, label in enumerate(labels):
        for value in data[i, :]:
            df_list.append({'Group': label, 'Value': value, 'Index': i})
    df = pd.DataFrame(df_list)

    fig, ax = plt.subplots(figsize=figsize)
    palette = box_colors if box_colors else sns.color_palette()

    violin_shift = 0.25
    point_shift = 0.5

    sns.boxplot(data=df, x='Group', y='Value', palette=palette, ax=ax,
                width=0.3, showfliers=False, boxprops=dict(alpha=0.7), **boxplot_params)
    
    means = df.groupby('Group')['Value'].mean()
    for i, label in enumerate(labels):
        mean_val = means[label]
        ax.hlines(mean_val, i - 0.15, i + 0.15, colors='red', linewidth=2.5, zorder=10)

    for i, label in enumerate(labels):
        group_data = df[df['Group'] == label]['Value'].values
        color = palette[i % len(palette)]

        jitter = np.random.normal(loc=0, scale=0.02, size=len(group_data))
        ax.scatter(np.full_like(group_data, i + point_shift) + jitter, group_data,
                   color=color, alpha=0.6, edgecolors='gray', linewidths=0.5, zorder=2)

        kde = gaussian_kde(group_data, bw_method=0.3)
        y_vals = np.linspace(min(group_data), max(group_data), 200)
        density = kde(y_vals)

        max_density = np.max(density)
        scale = 0.2 / max_density  # adjust 0.2 for width
        x_vals = i + violin_shift + density * scale  # right half only

        ax.fill_betweenx(y_vals, i + violin_shift, x_vals, facecolor=color, alpha=0.4)

    y_max = df['Value'].max()
    y_min = df['Value'].min()
    h = (y_max - y_min) * 0.05
    current_y = y_max + h

    grouped_values = [df[df['Group'] == label]['Value'].values for label in labels]
    pairs = list(itertools.combinations(range(len(labels)), 2))
    for i, j in pairs:
        if pvals_pairs is not None and (i,j) not in pvals_pairs:
            continue
        group1, group2 = grouped_values[i], grouped_values[j]
        stat, pval = ttest_ind(group1, group2, equal_var=False)
        sig = pvalue_to_asterisks(pval)

        x1, x2 = i, j
        ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y],
                lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, current_y + h, sig,
                ha='center', va='bottom', color='k', fontsize=12)
        current_y += h * 1.5

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set(**set_params)
    plt.tight_layout()

    if save:
        if file_name is None:
            file_name = 'boxplot.png'
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path)
        if save == 1:
            plt.show()
        else:
            plt.close()
    else:
        plt.show()
        
def plot_box(simconfigs, file_prefix, n_seeds=None, data=None, save=0, path=None, file_name=None, x=None, save_path=None,
             boxplot_params={}, set_params={}, box_colors=None, figsize=(8,8)):
    """
    Plots boxplots for simulation configurations or provided data, overlays group means,
    and computes pairwise statistical significance with annotated bars.
    
    :param simconfigs: List of simulation configuration objects used to extract and label data if data is not provided.
    :param file_prefix: String prefix used when searching for data files.
    :param n_seeds: Number of seeds (samples) to load per configuration. Ignored if data is provided.
    :param data: Optional NumPy array of shape (n_groups, n_seeds). If None, data will be loaded from files using simconfigs.
    :param save: Integer flag controlling saving and displaying of the plot.
    - 0: Show only
    - 1: Save and show
    - 2: Save and close (do not show)
    :param path: Path where seed data files are stored, used if data is None.
    :param file_name: File name to save the plot as, defaults to "boxplot.png".
    :param x: Optional list of custom labels for the groups on the x-axis.
    :param save_path: Directory to save the plot. If it does not exist, it is created.
    :param boxplot_params: Dictionary of additional parameters passed to the seaborn boxplot.
    :param set_params: Dictionary of matplotlib axis parameters (e.g., title, labels, limits).
    :param box_colors: List or palette of colors used for the groups. Defaults to seaborn palette.
    :param figsize: Tuple defining the figure size (width, height). Defaults to (8, 8).
    
    :returns: None. Displays and/or saves the generated plot depending on save.
    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    # Collect data
    if data is None:
        data = np.zeros((len(simconfigs), n_seeds))
        for i, simconfig in enumerate(simconfigs):
            file_path = find_file_seed(simconfig, path, file_prefix, n_minimal_seeds=n_seeds)
            data[i] = np.load(file_path)[:n_seeds]
    else:
        n_seeds = data.shape[1]

    # Prepare long-format DataFrame for seaborn
    df_list = []
    labels = [simconfig.get_plot_title() for simconfig in simconfigs]
    if x is not None:
        labels = x

    for i, label in enumerate(labels):
        for value in data[i, :]:
            df_list.append({'Group': label, 'Value': value})
    
    df = pd.DataFrame(df_list)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot seaborn boxplot
    palette = box_colors if box_colors else sns.color_palette()

    sns.boxplot(data=df, x='Group', y='Value', palette=palette, ax=ax, **boxplot_params)

    # Draw mean as red horizontal lines over each box
    means = df.groupby('Group')['Value'].mean()
    # Get positions of each box on the x-axis
    positions = range(len(means))
    for pos, mean_val in zip(positions, means):
        ax.hlines(mean_val, pos - 0.4, pos + 0.4, colors='red', linewidth=2, zorder=10)

    # Compute p-values and assign significance stars
    unique_groups = df['Group'].unique()
    grouped_values = [df[df['Group'] == grp]['Value'].values for grp in unique_groups]

    y_max = df['Value'].max()
    y_min = df['Value'].min()
    h = (y_max - y_min) * 0.05
    current_y = y_max + h

    pairs = list(itertools.combinations(range(len(unique_groups)), 2))
    for i, j in pairs:
        group1 = grouped_values[i]
        group2 = grouped_values[j]
        stat, pval = ttest_ind(group1, group2, equal_var=False)
        
        # Determine significance level
        significance = pvalue_to_asterisks(pval)
        
        # Draw significance bar
        x1, x2 = i, j
        ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, current_y + h, significance, ha='center', va='bottom', color='k', fontsize=12)
        current_y += h * 1.5

    # Apply general axis settings
    ax.set(**set_params)

    plt.tight_layout()

    # Save or show
    if save:
        if file_name is None:
            file_name = 'boxplot.png'
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path)
        if save == 1:
            plt.show()
        else:
            plt.close()
    else:
        plt.show()

## from Manuel Carrasco Yague (https://stackoverflow.com/questions/67505252/plotly-box-p-value-significant-annotation)
#def add_p_value_annotation(fig, array_columns, subplot=None, _format=dict(interline=0.07, text_height=1.07, color='black')):
#    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
#    
#    Parameters:
#    ----------
#    fig: figure
#        plotly boxplot figure
#    array_columns: np.array
#        array of which columns to compare 
#        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
#    subplot: None or int
#        specifies if the figures has subplots and what subplot to add the notation to
#    _format: dict
#        format characteristics for the lines
#
#    Returns:
#    -------
#    fig: figure
#        figure with the added notation
#    '''
#    # Specify in what y_range to plot for each pair of columns
#    y_range = np.zeros([len(array_columns), 2])
#    for i in range(len(array_columns)):
#        y_range[i] = [1.01+i*_format['interline'], 1.02+i*_format['interline']]
#
#    # Get values from figure
#    fig_dict = fig.to_dict()
#
#    # Get indices if working with subplots
#    if subplot:
#        if subplot == 1:
#            subplot_str = ''
#        else:
#            subplot_str =str(subplot)
#        indices = [] #Change the box index to the indices of the data for that subplot
#        for index, data in enumerate(fig_dict['data']):
#            #print(index, data['xaxis'], 'x' + subplot_str)
#            if data['xaxis'] == 'x' + subplot_str:
#                indices = np.append(indices, index)
#        indices = [int(i) for i in indices]
#        print((indices))
#    else:
#        subplot_str = ''
#
#    # Print the p-values
#    for index, column_pair in enumerate(array_columns):
#        if subplot:
#            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
#        else:
#            data_pair = column_pair
#
#        # Mare sure it is selecting the data and subplot you want
#        print(fig_dict['data'])
#        print(data_pair)
#        print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
#        print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])
#
#        # Get the p-value
#        pvalue = ttest_ind(
#            fig_dict['data'][data_pair[0]]['y'],
#            fig_dict['data'][data_pair[1]]['y'],
#            equal_var=False,
#        )[1]
#        if pvalue >= 0.05:
#            symbol = 'ns'
#        elif pvalue >= 0.01: 
#            symbol = '*'
#        elif pvalue >= 0.001:
#            symbol = '**'
#        else:
#            symbol = '***'
#        # Vertical line
#        fig.add_shape(type="line",
#            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
#            x0=column_pair[0], y0=y_range[index][0], 
#            x1=column_pair[0], y1=y_range[index][1],
#            line=dict(color=_format['color'], width=2,)
#        )
#        # Horizontal line
#        fig.add_shape(type="line",
#            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
#            x0=column_pair[0], y0=y_range[index][1], 
#            x1=column_pair[1], y1=y_range[index][1],
#            line=dict(color=_format['color'], width=2,)
#        )
#        # Vertical line
#        fig.add_shape(type="line",
#            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
#            x0=column_pair[1], y0=y_range[index][0], 
#            x1=column_pair[1], y1=y_range[index][1],
#            line=dict(color=_format['color'], width=2,)
#        )
#        ## add text at the correct x, y coordinates
#        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
#        fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
#            x=(column_pair[0] + column_pair[1])/2,
#            y=y_range[index][1]*_format['text_height'],
#            showarrow=False,
#            text=symbol,
#            textangle=0,
#            xref="x"+subplot_str,
#            yref="y"+subplot_str+" domain"
#        ))
#    return fig
#
#           
##def plot_box(simconfigs, seeds, file_prefix, yaxis_name, data=None, title=None,
##             height=None, width=None, save=0, path=None, file_name=None, x=None, save_path=None,
##             marker_size=10):
#def plot_box(simconfigs, seeds, file_prefix, data=None, save=0, path=None, file_name=None, x=None, save_path=None,
#             plot_params={}, layout_params={}, traces_params={}):
#    """
#        :param simconfigs: simconfigs of which to plot results
#        :param seeds: seeds of which to plot results for each simconfigs
#        :param file_prefix: beginning of files of which to get data from ('LempelZiv' or 'PCI' for instance)
#        :param data: if None, searches for the data itself in path based on SimConfigs' params_to_report
#        :param save: 0 display & no save, 1 display & save, 2 no display & save
#        :param path: where to look for the .npy files
#        :param file_name: name of the saved plot
#        :param x: how to group data, list of size number of simconfigs
#                  if the first 2 simconfigs should be grouped in one box with label A, and the two last in another with label B:
#                    x = ['A','A','B','B'], and all the seeds of the first two simconfigs will go in the first box plot 
#        :param save_path: where to save plot
#        :param plot_params: dictionary of parameters given to plotly's box function
#        :param layout_params: dictionary of parameters given to plotly's update_layout function
#        :param traces_params: dictionary of parameters given to plotly's update_traces function
#    """
#    if save_path is not None:
#        if not os.path.exists(save_path):
#            os.makedirs(save_path)        
#    # Collect data
#    if data is None:
#        n_seeds = len(seeds)
#        data = np.zeros((len(simconfigs), n_seeds))
#        for i,simconfig in enumerate(simconfigs):
##            print(simconfig.get_sim_name(n_seeds)[1])
#            file_path = find_file_seed(simconfig, path, file_prefix, n_minimal_seeds=n_seeds)
#            data[i] = np.load(file_path)[:n_seeds]
##            Printer.print(data[i])
#    else:
##        print(data.shape)
#        n_seeds = data.shape[1]
#    
#    # Plot
#    if x is None:
#        x = [simconfig.get_plot_title() for simconfig in simconfigs] * n_seeds
#    else:
#        x = [x_ for x_ in x] * n_seeds        
#
#    fig = px.box(x=x, y=data.T.flatten(), **plot_params)
#    fig.update_layout(**layout_params)
#    fig.update_traces(**traces_params)
#    
##    fig = go.Figure()
##    for i,(x_, simconfig) in enumerate(x, simconfigs):
##        fig.add_trace(go.Box(
##                y=data[i],
##                name=x_,
##                **plot_params))
##    fig.update_layout(**layout_params)
##    fig.update_traces(**traces_params)
##    
##    
##    fig = add_p_value_annotation(fig, [[0,1]])
#    
#    if save >= 1:
#        if file_name is None:
#            sim_names = [simconfig.get_sim_name(seeds[-1])[1] for simconfig in simconfigs]
#            file_name = f"{file_prefix}_{'_'.join(sim_names)}.png"
#        if save_path is None:
#            save_path = ''
#        fig.write_image(os.path.join(save_path, file_name))
#    if save < 2:
#        fig.show()
